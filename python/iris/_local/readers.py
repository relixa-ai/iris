"""Reading tables and pairs from Parquet, CSV or TSV files into pandas or Arrow, with the
column names of Iris."""
import csv
import os
import tempfile

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from . import inputs

ROW_GROUP_ROWS = 65_536


def source(path, fmt, types):
    """Return a lazy dataset over a file, a text file with the given column types and
    every other column read as text."""
    if fmt == 'parquet':
        return inputs.dataset(path)
    delimiter = '\t' if fmt == 'tsv' else ','
    with open(path, newline='', encoding='utf-8') as fh:
        names = next(csv.reader(fh, delimiter=delimiter), None)
    if not names:
        raise ValueError(f'{path}: the file has no header row')
    column_types = {n: types.get(n, pa.string()) for n in names}
    return ds.dataset(path, format=ds.CsvFileFormat(
        parse_options=pcsv.ParseOptions(delimiter=delimiter),
        convert_options=pcsv.ConvertOptions(column_types=column_types,
                                            strings_can_be_null=True)))


def output(path, src, projection, df_type, as_is, transform=None):
    """Return the projected rows as pandas or as an Arrow dataset; an Arrow result is
    the file itself when `as_is`, else a copy written to scratch."""
    if df_type == 'arrow':
        return src if as_is else _scratch_copy(path, src, projection, transform)
    table = src.to_table(columns=projection)
    if transform is not None:
        table = transform(table)
    return table.to_pandas()


def _scratch_copy(path, src, projection, transform):
    from .._core.storage import spill_dir
    from .featurize import _remove_at_exit

    directory = tempfile.mkdtemp(prefix='iris-read-', dir=spill_dir())
    _remove_at_exit(directory)
    inputs.require_disk(_source_bytes(path), directory, f'a Parquet copy of {path}')
    target = os.path.join(directory, 'part-0.parquet')
    scanner = src.scanner(columns=projection, batch_size=ROW_GROUP_ROWS, use_threads=False,
                          **inputs._scan_options(src))
    writer = None
    try:
        for batch in scanner.to_batches():
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
