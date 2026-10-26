"""Reading tables and pairs from Parquet, CSV, TSV or JSON Lines files into pandas or Arrow,
with the column names of Iris."""
import contextlib
import csv
import os
import re
import tempfile

import numba as nb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv
import pyarrow.dataset as ds
import pyarrow.json as pjson
import pyarrow.parquet as pq

from . import inputs

ROW_GROUP_ROWS = 65_536

SCAN_BYTES = 1 << 24

MIN_BLOCK = 1 << 20

MAX_BLOCK = 2**31 - 1

TEXT_FORMATS = {'csv': 'CSV', 'tsv': 'TSV', 'jsonl': 'JSON Lines'}

DELIMITER_NAMES = {'csv': 'a comma', 'tsv': 'a tab'}

TYPE_WORDS = {'int64': 'whole numbers', 'double': 'numbers', 'float': 'numbers'}

SHOWN_CHARS = 80

_CSV_VALUE = re.compile(r"In CSV column #(\d+): (?:Row #(\d+): )?CSV conversion error to "
                        r"(\w+): invalid value '(.*)'", re.S)
_CSV_UTF8 = re.compile(r'In CSV column #(\d+): (?:Row #(\d+): )?CSV conversion error to '
                       r'\w+: invalid UTF8 data')
_CSV_FIELDS = re.compile(r'CSV parse error: (?:Row #(\d+): )?Expected (\d+) columns, got '
                         r'(\d+): (.*)', re.S)
_JSON_CHANGED = re.compile(r'Column\((.*?)\) changed from (\w+) to (\w+)')
_JSON_CONVERT = re.compile(r"Failed to convert JSON to (\w+), couldn't parse:(.*)", re.S)
_JSON_PARSE = re.compile(r'JSON parse error: (.*?)\.?(?: in row \d+)?$', re.S)


def source(path, fmt, types):
    """Return a lazy dataset over a file, a text file with the given column types and
    every other column read as text, or as JSON gives it for JSON Lines."""
    if fmt == 'parquet':
        return inputs.dataset(path)
    if fmt == 'jsonl':
        with _reading(path, fmt, pa.schema(list(types.items()))):
            found = _jsonl_schema(path)
        schema = pa.schema([pa.field(f.name, types.get(f.name, f.type)) for f in found])
        if os.path.getsize(path) == 0:
            return ds.dataset(schema.empty_table())
        with _reading(path, fmt, schema):
            return ds.dataset(path, schema=schema, format=ds.JsonFileFormat(
                parse_options=pjson.ParseOptions(explicit_schema=schema),
                read_options=pjson.ReadOptions(block_size=_block(path, None))))
    delimiter = '\t' if fmt == 'tsv' else ','
    try:
        with open(path, newline='', encoding='utf-8') as fh:
            names = next(csv.reader(fh, delimiter=delimiter), None)
    except UnicodeDecodeError as e:
        raise ValueError(f'{path}: the file is not UTF-8 text ({e.reason} at byte '
                         f'{e.start}); write it as UTF-8, or as Parquet') from None
    if not names:
        raise ValueError(f'{path}: the file has no header row')
    column_types = {n: types.get(n, pa.string()) for n in names}
    with _reading(path, fmt, pa.schema([(n, column_types[n]) for n in names])):
        return ds.dataset(path, format=ds.CsvFileFormat(
            parse_options=pcsv.ParseOptions(delimiter=delimiter, newlines_in_values=True),
            convert_options=pcsv.ConvertOptions(column_types=column_types,
                                                strings_can_be_null=True),
            read_options=pcsv.ReadOptions(block_size=_block(path, delimiter))))


def _text_format(src):
    found = getattr(src, 'format', None)
    if isinstance(found, ds.CsvFileFormat):
        return 'tsv' if found.parse_options.delimiter == '\t' else 'csv'
    if isinstance(found, ds.JsonFileFormat):
        return 'jsonl'
    return None


@contextlib.contextmanager
def _reading(path, fmt, schema):
    try:
        yield
    except pa.ArrowInvalid as e:
        if fmt not in TEXT_FORMATS:
            raise
        raise ValueError(_read_error(path, fmt, schema, str(e))) from None


def _shown(text):
    return text if len(text) <= SHOWN_CHARS else text[:SHOWN_CHARS] + '...'


def _a(word):
    return f"{'an' if word[:1] in 'aeiou' else 'a'} {word}"


def _row(row):
    return f'row {row} (the header is row 1)' if row else 'a row'


def _column(schema, index):
    k = int(index)
    return repr(schema.names[k]) if k < len(schema.names) else f'#{k + 1}'


def _read_error(path, fmt, schema, message):
    m = _CSV_VALUE.search(message)
    if m:
        col, row, to, value = m.groups()
        return (f'{path}: {_row(row)} holds {_shown(value)!r} in the column '
                f'{_column(schema, col)}, which Iris reads as {TYPE_WORDS.get(to, to)}; '
                f'correct or remove that row')
    m = _CSV_UTF8.search(message)
    if m:
        col, row = m.groups()
        return (f'{path}: {_row(row)} holds text that is not UTF-8 in the column '
                f'{_column(schema, col)}; write the file as UTF-8, or as Parquet')
    m = _CSV_FIELDS.search(message)
    if m:
        row, expected, got, text = m.groups()
        return (f'{path}: {_row(row)} has {got} field(s) where the header has {expected}: '
                f'{_shown(text)!r}; quote the values that hold {DELIMITER_NAMES[fmt]}, or '
                f'correct that row')
    if 'Invalid UTF8' in message:
        return (f'{path}: the file holds text that is not UTF-8; write it as UTF-8, or as '
                f'Parquet')
    m = _JSON_CHANGED.search(message)
    if m:
        col, was, now = m.groups()
        name = col.lstrip('/').replace('/', '.')
        if not name:
            return (f'{path}: a line holds {_a(now)}, and each line of JSON Lines holds one '
                    f'object; write one JSON object on each line')
        whole = name in schema.names and pa.types.is_integer(schema.field(name).type)
        return (f"{path}: the column '{name}' holds {_a(now)} on some line, where Iris reads "
                f"{'a whole number' if whole else _a(was)}; give the column one JSON type on "
                f'every line, or write the file as Parquet')
    m = _JSON_CONVERT.search(message)
    if m:
        to, value = m.groups()
        cols = [f.name for f in schema if str(f.type) == to]
        where = f'the column {cols[0]!r}' if len(cols) == 1 else f'one of the columns {cols}'
        return (f'{path}: a line holds {_shown(value.strip())} in {where}, which Iris reads as '
                f'{TYPE_WORDS.get(to, to)}; correct that line, or write the file as Parquet')
    m = _JSON_PARSE.search(message)
    if m:
        return (f'{path}: a line is not valid JSON ({m.group(1)}); write one JSON object on '
                f'each line')
    detail = message.rsplit('Invalid: ', 1)[-1]
    return (f'{path}: cannot be read as {TEXT_FORMATS[fmt]}: {detail}; correct the file, or '
            f'write it as Parquet')


@nb.njit(cache=False)
def _records(data, delimiter, state, start, offset, longest):
    for k in range(len(data)):
        c = data[k]
        if state == 2:
            if c == 34:
                state = 3
            continue
        if state == 3 and c == 34:
            state = 2
            continue
        if c == 10 or (delimiter >= 0 and c == 13):
            end = offset + k + 1
            if end - start > longest:
                longest = end - start
            start = end
            state = 0
        elif c == delimiter:
            state = 0
        elif state == 0 and c == 34 and delimiter >= 0:
            state = 2
        else:
            state = 1
    return state, start, longest


def _longest_record(path, delimiter):
    code = -1 if delimiter is None else ord(delimiter)
    state = start = offset = longest = 0
    buf = bytearray(max(1, min(SCAN_BYTES, os.path.getsize(path))))
    with open(path, 'rb') as fh:
        while True:
            n = fh.readinto(buf)
            if not n:
                break
            state, start, longest = _records(np.frombuffer(buf, np.uint8, n), code, state,
                                             start, offset, longest)
            offset += n
    return max(longest, offset - start)


def _block(path, delimiter):
    longest = _longest_record(path, delimiter)
    return int(min(MAX_BLOCK, max(MIN_BLOCK, 2 * longest + MIN_BLOCK)))


def _jsonl_schema(path, head_bytes=1 << 20):
    head = bytearray()
    with open(path, 'rb') as fh:
        while True:
            more = fh.read(head_bytes)
            head += more
            if not more or b'\n' in more:
                break
    head = bytes(head)
    if not head.strip():
        return pa.schema([('id1', pa.int64()), ('id2', pa.int64())])
    if len(head) > head_bytes:
        cut = head.find(b'\n') + 1 or len(head)
    else:
        cut = head.rfind(b'\n') + 1 or len(head)
    return pjson.read_json(pa.BufferReader(head[:cut]), read_options=pjson.ReadOptions(
        block_size=int(min(MAX_BLOCK, max(MIN_BLOCK, 2 * cut))))).schema


def output(path, src, projection, df_type, as_is, transform=None):
    """Return the projected rows as pandas or as an Arrow dataset; an Arrow result is
    the file itself when `as_is`, else a copy written to scratch."""
    fmt = _text_format(src)
    with _reading(path, fmt, src.schema):
        if df_type == 'arrow':
            return src if as_is else _scratch_copy(path, src, projection, transform, fmt)
        table = src.to_table(columns=projection)
        if fmt == 'jsonl':
            table.validate(full=True)
        if transform is not None:
            table = transform(table)
        return table.to_pandas()


def _scratch_copy(path, src, projection, transform, fmt=None):
    from .._core.storage import scratch_prefix, spill_dir
    from .featurize import _remove_at_exit

    directory = tempfile.mkdtemp(prefix=scratch_prefix('read'), dir=spill_dir())
    _remove_at_exit(directory)
    inputs.require_disk(_source_bytes(path), directory, f'a Parquet copy of {path}')
    target = os.path.join(directory, 'part-0.parquet')
    scanner = src.scanner(columns=projection, batch_size=ROW_GROUP_ROWS, use_threads=False,
                          **inputs._scan_options(src))
    writer = None
    try:
        for batch in scanner.to_batches():
            if fmt == 'jsonl':
                batch.validate(full=True)
            if transform is not None:
                batch = transform(batch)
            if batch.num_rows == 0:
                continue
            if writer is None:
                writer = pq.ParquetWriter(target, batch.schema, compression='zstd')
            writer.write_batch(batch, row_group_size=ROW_GROUP_ROWS)
        if writer is None:
            empty = scanner.projected_schema.empty_table()
            pq.write_table(transform(empty) if transform is not None else empty, target)
    finally:
        if writer is not None:
            writer.close()
    return inputs.dataset(directory)


def _source_bytes(path):
    path = os.fspath(path)
    if os.path.isdir(path):
        return sum(os.path.getsize(f) for f in inputs.parquet_files(path))
    return os.path.getsize(path)


def flatten(list_col, other_col, list_is_a, with_label):
    """Return a function that turns grouped rows into one row per listed id, as `id1`,
    `id2` and, with `with_label`, `label`."""
    def run(rows):
        lists = rows.column(list_col)
        parents = pc.list_parent_indices(lists)
        listed = pc.list_flatten(lists).cast(pa.int64())
        other = pc.take(rows.column(other_col), parents).cast(pa.int64())
        cols = {'id1': listed, 'id2': other} if list_is_a else {'id1': other, 'id2': listed}
        if with_label:
            cols['label'] = pc.list_flatten(rows.column('label')).cast(pa.float64())
        if isinstance(rows, pa.RecordBatch):
            return pa.RecordBatch.from_pydict(cols)
        return pa.table(cols)
    return run
