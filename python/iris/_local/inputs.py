"""Reading the inputs of the local engine, pandas, Arrow tables and Parquet datasets, a
batch at a time."""
import contextlib
import os
from typing import Any, NamedTuple
import shutil
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .._core.frames import has_filter
from . import disk

BATCH_ROWS = 1 << 20

LOCAL_TYPES = (pd.DataFrame, pa.Table, ds.Dataset)

READ_BUFFER = 1 << 20

NO_PREBUFFER = ds.ParquetFragmentScanOptions(pre_buffer=False, use_buffered_stream=True,
                                             buffer_size=READ_BUFFER)
PARQUET = ds.ParquetFileFormat(default_fragment_scan_options=NO_PREBUFFER)

NO_READAHEAD = {'batch_readahead': 0, 'fragment_readahead': 0}


def is_arrow(obj):
    """Return whether `obj` is an Arrow table or dataset."""
    return isinstance(obj, (pa.Table, ds.Dataset))


def is_local(obj):
    """Return whether `obj` is an input of the local engine: pandas, an Arrow table or
    an Arrow dataset."""
    return isinstance(obj, LOCAL_TYPES)


def dataset(path):
    """Return a Parquet file or directory as an Arrow dataset, without reading it."""
    path = str(path)
    if os.path.isdir(path):
        files = parquet_files(path)
        if files:
            return ds.dataset(files, format=PARQUET)
    return ds.dataset(path, format=PARQUET)


def parquet_files(directory):
    """Return the data files under `directory`, sorted, without names that start with
    `_` or `.`."""
    out = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = sorted(d for d in dirs if not d.startswith(('_', '.')))
        out += [os.path.join(root, f) for f in files if not f.startswith(('_', '.'))]
    return sorted(out)


def columns(obj):
    """Return the column names of an input, in order."""
    if isinstance(obj, pd.DataFrame):
        return list(obj.columns)
    if isinstance(obj, pa.Table):
        return list(obj.column_names)
    return list(obj.schema.names)


def batches(obj, cols=None, rows=None):
    """Yield an input as pandas frames of at most `rows` rows, in input order."""
    rows = rows or BATCH_ROWS
    if isinstance(obj, pd.DataFrame):
        frame = obj if cols is None else obj[list(cols)]
        for lo in range(0, max(len(frame), 1), rows):
            yield frame.iloc[lo:lo + rows]
        return
    if isinstance(obj, pa.Table):
        table = obj if cols is None else obj.select(list(cols))
        parts = table.to_batches(max_chunksize=rows)
        schema = table.schema
    else:
        parts = _coalesce(_source_batches(obj, cols), rows, SLICE_BYTES)
        schema = obj.schema if cols is None else pa.schema(
            [obj.schema.field(c) for c in cols])
    seen = False
    for batch in parts:
        seen = True
        yield batch.to_pandas()
    if not seen:
        yield schema.empty_table().to_pandas()


CHUNK_ROWS = 1 << 20

CHUNK_BYTES = 64 * 2**20

MIN_CHUNK_ROWS = 1 << 14

GROUP_ROWS = 1 << 14

GROUP_BYTES = 16 * 2**20

FILE_ROWS = 1 << 20

SLICE_ROWS = 10_000

SLICE_BYTES = 4 * 2**20

READ_ROWS = 1024

TAKE_BYTES = 64 * 2**20

VALUE_BYTES = {'BOOLEAN': 1, 'INT32': 4, 'INT64': 8, 'INT96': 12, 'FLOAT': 4, 'DOUBLE': 8}


class Chunk(NamedTuple):
    """A run of whole row groups of one Parquet file, read by one worker in the schema of
    its dataset."""
    filesystem: Any
    path: str
    row_groups: tuple
    rows: int
    schema: Any = None


@contextlib.contextmanager
def scratch():
    """Yield a scratch directory that is removed afterwards."""
    from .._core.storage import scratch_prefix, spill_dir, sweep

    sweep()
    path = tempfile.mkdtemp(prefix=scratch_prefix('scratch'), dir=spill_dir())
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def to_arrow(obj):
    """Return an input held in memory as an Arrow table."""
    if isinstance(obj, pa.Table):
        return obj
    if isinstance(obj, pd.DataFrame):
        return pa.Table.from_pandas(obj, preserve_index=False)
    return obj.to_table()


def _parquet_dataset(obj):
    return (isinstance(obj, ds.FileSystemDataset)
            and isinstance(obj.format, ds.ParquetFileFormat))


def _parquet_files(obj):
    return _read_by_file(obj) and not _partition_fields(obj)


def _read_by_file(obj):
    return _parquet_dataset(obj) and not has_filter(obj)


def _partition_fields(obj):
    found = getattr(obj, 'partitioning', None)
    return [] if found is None else found.schema.names


def _partition_values(obj, frag, cols):
    names = obj.schema.names if cols is None else list(cols)
    keys = [c for c in ds.get_partition_keys(frag.partition_expression) if c in names]
    if not keys:
        return {}
    head = ds.Scanner.from_fragment(frag, schema=obj.schema, columns=keys, batch_size=1,
                                    use_threads=False).head(1)
    return {c: head.column(c)[0] for c in keys} if head.num_rows else {}


def _group_bytes(meta, k, cols):
    group = meta.row_group(k)
    names = None if cols is None else set(cols)
    total = 0
    for i in range(group.num_columns):
        column = group.column(i)
        path = column.path_in_schema
        if names is None or path in names or any(path.startswith(c + '.') for c in names):
            width = VALUE_BYTES.get(column.physical_type, 0)
            if column.physical_type == 'FIXED_LEN_BYTE_ARRAY':
                width = meta.schema.column(i).length
            total += max(column.total_uncompressed_size, column.num_values * width)
    return total


def _groups(source, cols=None):
    for frag in source.get_fragments():
        meta = frag.metadata
        for k in range(meta.num_row_groups):
            yield frag, k, meta.row_group(k).num_rows, _group_bytes(meta, k, cols)


def chunk_bytes():
    """Return the most bytes of input a chunk holds on this machine."""
    from .. import resources
    return resources.piece_bytes(CHUNK_BYTES)


def _needs_staging(source, cols, workers):
    groups = [(frag.path, n, b) for frag, _, n, b in _groups(source, cols)]
    most = chunk_bytes()
    if any(n > CHUNK_ROWS or b > most for _, n, b in groups):
        return True
    rows = sum(n for _, n, _ in groups)
    if len(groups) < min(workers, rows // MIN_CHUNK_ROWS):
        return True
    files = len({path for path, _, _ in groups})
    staged = max(1, -(-rows // _chunk_rows(rows, CHUNK_ROWS, workers)),
                 -(-sum(b for _, _, b in groups) // most))
    return files * min(CHUNK_ROWS, MIN_CHUNK_ROWS) > staged * CHUNK_ROWS


def _source_batches(obj, cols):
    if isinstance(obj, pd.DataFrame):
        frame = obj if cols is None else obj[list(cols)]
        schema = pa.Schema.from_pandas(frame, preserve_index=False)
        lo, step = 0, 1
        while lo < len(frame):
            part = pa.Table.from_pandas(frame.iloc[lo:lo + step], schema=schema,
                                        preserve_index=False)
            lo += part.num_rows
            step = int(max(1, min(GROUP_ROWS, step * READ_BUFFER // max(part.nbytes, 1))))
            yield from part.to_batches()
    elif isinstance(obj, pa.Table):
        yield from (obj if cols is None else obj.select(list(cols))).to_batches(
            max_chunksize=READ_ROWS)
    elif _read_by_file(obj):
        for frag in obj.get_fragments():
            groups = tuple(range(frag.metadata.num_row_groups))
            if groups:
                yield from _chunk_batches(Chunk(obj.filesystem, frag.path, groups,
                                                frag.metadata.num_rows, obj.schema), cols,
                                          _partition_values(obj, frag, cols))
    else:
        yield from obj.to_batches(columns=None if cols is None else list(cols),
                                  batch_size=READ_ROWS, use_threads=False,
                                  **_scan_options(obj))


def _gauge(obj, cols, path, what):
    if isinstance(obj, (pd.DataFrame, pa.Table)):
        return (disk.Meter(what, path, column_bytes(obj, cols), kept=False),
                lambda rows, nbytes: nbytes)
    if not _read_by_file(obj):
        return None
    groups = list(_groups(obj, cols))
    rows = np.cumsum([0] + [n for _, _, n, _ in groups])
    held = np.cumsum([0] + [b for _, _, _, b in groups])
    return (disk.Meter(what, path, int(held[-1]), kept=False),
            lambda done, nbytes: int(np.interp(done, rows, held)))


def _write(obj, path, cols, gauge=None):
    schema = schema_of(obj, cols)
    state = {'writer': None, 'files': 0, 'in_file': 0, 'rows': 0, 'bytes': 0}

    def put(held):
        if state['writer'] is None:
            name = os.path.join(path, f"part-{state['files']:05d}.parquet")
            state['writer'] = pq.ParquetWriter(name, schema, use_dictionary=False)
            state['files'] += 1
        table = pa.Table.from_batches(held, schema=schema) if held else schema.empty_table()
        state['writer'].write_table(table)
        state['rows'] += table.num_rows
        state['bytes'] += table.nbytes
        if gauge is not None:
            gauge[0].wrote(gauge[1](state['rows'], state['bytes']),
                           disk.written_in(path)[0])
        state['in_file'] += table.num_rows
        if state['in_file'] >= FILE_ROWS:
            state['writer'].close()
            state['writer'], state['in_file'] = None, 0

    held, rows, nbytes = [], 0, 0
    for batch in _source_batches(obj, cols):
        if batch.schema != schema:
            batch = pa.Table.from_batches([batch]).cast(schema).to_batches()[0] \
                if batch.num_rows else None
        if batch is None:
            continue
        held.append(batch)
        rows += batch.num_rows
        nbytes += batch.nbytes
        if rows >= GROUP_ROWS or nbytes >= GROUP_BYTES:
            put(held)
            held, rows, nbytes = [], 0, 0
    if held or state['files'] == 0:
        put(held)
    if state['writer'] is not None:
        state['writer'].close()
    if gauge is not None:
        gauge[0].finish()


def schema_of(obj, cols=None):
    """Return the Arrow schema of the columns `cols` of an input (all when None)."""
    full = schema(obj)
    return full if cols is None else pa.schema([full.field(c) for c in cols])


def stage(obj, directory, name, cols=None, workers=1):
    """Return an input as a Parquet dataset, first writing columns `cols` to `directory` when
    it is in memory or its row groups or files are badly sized for chunks of `workers`."""
    if _parquet_files(obj) and not _needs_staging(obj, cols, workers):
        return obj
    path = os.path.join(directory, name)
    os.makedirs(path)
    _write(obj, path, cols, _gauge(obj, cols, path, 'a copy of the input in small row groups'
                                   if _parquet_files(obj) else 'a copy of the input as Parquet'))
    return dataset(path)


def _chunk_rows(total, rows, workers):
    return min(rows, max(MIN_CHUNK_ROWS, -(-total // (2 * workers))))


def chunks(source, rows=None, workers=None):
    """Return a Parquet dataset as chunks of whole row groups, of at most `rows` rows and
    about `chunk_bytes()`, enough for `workers` when the rows allow, in input order."""
    rows = rows or CHUNK_ROWS
    most = chunk_bytes()
    groups = list(_groups(source))
    if workers:
        rows = _chunk_rows(sum(n for _, _, n, _ in groups), rows, workers)
    out, held = [], None
    for frag, k, n, b in groups:
        if held is not None and (held[0] is not frag or held[2] + n > rows
                                 or held[3] + b > most):
            out.append(Chunk(source.filesystem, held[0].path, tuple(held[1]), held[2],
                             source.schema))
            held = None
        if held is None:
            held = [frag, [], 0, 0]
        held[1].append(k)
        held[2] += n
        held[3] += b
    if held is not None:
        out.append(Chunk(source.filesystem, held[0].path, tuple(held[1]), held[2],
                         source.schema))
    return out


def _conform(batch, schema, known=None):
    if schema is None or batch.schema.equals(schema):
        return batch
    arrays = []
    for field in schema:
        k = batch.schema.get_field_index(field.name)
        if known and field.name in known:
            arrays.append(pa.repeat(known[field.name], batch.num_rows))
        elif k < 0:
            arrays.append(pa.nulls(batch.num_rows, field.type))
        else:
            column = batch.column(k)
            arrays.append(column if column.type == field.type else column.cast(field.type))
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def _chunk_batches(chunk, cols, known=None):
    want = None if chunk.schema is None else (
        chunk.schema if cols is None else pa.schema([chunk.schema.field(c) for c in cols]))
    with chunk.filesystem.open_input_file(chunk.path) as fh:
        reader = pq.ParquetFile(fh, pre_buffer=False, buffer_size=READ_BUFFER)
        meta = reader.metadata
        present = set(reader.schema_arrow.names) - set(known or ())
        read = None if cols is None and not known else [
            c for c in (want.names if cols is None else cols) if c in present]
        widest = max(_group_bytes(meta, k, read) / max(meta.row_group(k).num_rows, 1)
                     for k in chunk.row_groups)
        rows = int(max(1, min(READ_ROWS, READ_BUFFER // max(widest, 1))))
        for batch in reader.iter_batches(batch_size=rows, row_groups=list(chunk.row_groups),
                                         columns=read, use_threads=False):
            yield _conform(batch, want, known)


def _coalesce(source, rows, nbytes):
    held, n, size = [], 0, 0
    for batch in source:
        held.append(batch)
        n += batch.num_rows
        size += batch.nbytes
        if n >= rows or size >= nbytes:
            table = pa.Table.from_batches(held, schema=batch.schema)
            whole = min(n, rows) if size < nbytes else n
            yield table.slice(0, whole) if n > whole else table
            rest = table.slice(whole) if n > whole else None
            while rest is not None and rest.num_rows >= rows:
                yield rest.slice(0, rows)
                rest = rest.slice(rows)
            held = rest.to_batches() if rest is not None and rest.num_rows else []
            n = sum(b.num_rows for b in held)
            size = sum(b.nbytes for b in held)
    if held:
        yield pa.Table.from_batches(held, schema=held[0].schema)


def read_slices(chunk, cols, rows=None, nbytes=None):
    """Yield the rows of a chunk as Arrow tables of at most `rows` rows and about `nbytes`
    bytes, read a little at a time."""
    yield from _coalesce(_chunk_batches(chunk, cols), rows or SLICE_ROWS,
                         nbytes or SLICE_BYTES)


def slices(table, rows=None):
    """Yield an Arrow table in slices of at most `rows` rows."""
    rows = rows or SLICE_ROWS
    for lo in range(0, table.num_rows, rows):
        yield table.slice(lo, rows)


def read_chunk(chunk, cols):
    """Return the columns `cols` of a chunk as an Arrow table."""
    with chunk.filesystem.open_input_file(chunk.path) as fh:
        return pq.ParquetFile(fh).read_row_groups(
            list(chunk.row_groups), columns=None if cols is None else list(cols))


def schema(obj):
    """Return the Arrow schema of an input."""
    if isinstance(obj, pd.DataFrame):
        try:
            return pa.Schema.from_pandas(obj, preserve_index=False)
        except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError) as e:
            for c in obj.columns:
                try:
                    pa.Schema.from_pandas(obj[[c]], preserve_index=False)
                except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError):
                    kinds = sorted({type(v).__name__ for v in obj[c].dropna().head(10_000)})
                    raise ValueError(f'column {c!r} cannot be read as one type ({e}); it '
                                     f'holds {kinds}: convert it, for example with '
                                     f'.astype(str)') from None
            raise
    return obj.schema


def gather(parts, starts, rows):
    """Return the rows `rows` of the tables `parts`, which begin at the row numbers
    `starts`, in that order, taken from one part at a time."""
    part = np.searchsorted(starts, rows, side='right') - 1
    by_part = np.argsort(part, kind='stable')
    pieces = []
    for k in np.unique(part).tolist():
        at = by_part[part[by_part] == k]
        pieces.append(parts[k].take(pa.array(rows[at] - starts[k])))
    gathered = pa.concat_tables(pieces)
    placed = np.empty(len(rows), np.int64)
    placed[np.concatenate([by_part[part[by_part] == k]
                           for k in np.unique(part).tolist()])] = np.arange(len(rows))
    return gathered.take(pa.array(placed)).combine_chunks()


def take_rows(row_bytes):
    """Return how many rows of `row_bytes` bytes one piece of a take holds on this
    machine."""
    from .. import resources
    return max(1, resources.piece_bytes(TAKE_BYTES) // max(int(row_bytes), 1))


def take(table, rows):
    """Return the rows `rows` of a table or dataset in that order; a table's are taken a
    piece at a time, since Arrow joins a chunked column whole to take from it."""
    rows = np.asarray(rows, dtype=np.int64)
    if not isinstance(table, pa.Table):
        return table.take(pa.array(rows), batch_size=READ_ROWS, use_threads=False,
                          **_scan_options(table))
    if not len(rows):
        return table.slice(0, 0)
    parts = [pa.Table.from_batches([b]) for b in table.to_batches()]
    starts = np.cumsum([0] + [p.num_rows for p in parts])
    step = take_rows(table.nbytes / max(table.num_rows, 1))
    return pa.concat_tables([gather(parts, starts, rows[lo:lo + step])
                             for lo in range(0, len(rows), step)])


def arrow_batches(obj, cols):
    """Yield the columns `cols` of an input as Arrow record batches."""
    if isinstance(obj, pd.DataFrame):
        yield from _source_batches(obj, cols)
    elif isinstance(obj, pa.Table):
        yield from obj.select(list(cols)).to_batches(max_chunksize=BATCH_ROWS)
    else:
        for part in _coalesce(_source_batches(obj, cols), BATCH_ROWS, READ_BUFFER):
            yield from part.to_batches()


def _scan_options(obj):
    if _parquet_dataset(obj):
        return {'fragment_scan_options': NO_PREBUFFER, **NO_READAHEAD}
    return dict(NO_READAHEAD)


def read_column(source, name):
    """Return one column of a Parquet dataset, in input order."""
    parts = []
    for frag in source.get_fragments():
        with source.filesystem.open_input_file(frag.path) as fh:
            parts.append(pq.ParquetFile(fh).read(columns=[name]).column(name))
    if not parts:
        return pa.chunked_array([], type=source.schema.field(name).type)
    return pa.chunked_array([c for p in parts for c in p.chunks],
                            type=source.schema.field(name).type)


def require_disk(nbytes, path, what, remedy=disk.REMEDY, kept=None):
    """Raise an error before any work when the volume of `path` cannot hold `nbytes`, of
    which `kept` stay once the running phase ends, with what the later phases hold there."""
    disk.require(nbytes, path, what, remedy, kept)


def column_bytes(obj, cols=None):
    """Return about the bytes of the columns `cols` of an input (all when None): from
    Parquet metadata, from Arrow's buffers, or from the size of its files."""
    if isinstance(obj, pa.Table):
        return (obj if cols is None else obj.select(list(cols))).nbytes
    if isinstance(obj, pd.DataFrame):
        schema_of(obj, cols)
        return sum(b.nbytes for b in _source_batches(obj, cols))
    if has_filter(obj):
        return 0
    if _parquet_dataset(obj):
        return sum(b for _, _, _, b in _groups(obj, cols))
    files = getattr(obj, 'files', None)
    if not files:
        return 0
    return sum(f.size or 0 for f in obj.filesystem.get_file_info(files))
