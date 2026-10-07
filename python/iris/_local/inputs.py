"""Reading the inputs of the local engine, pandas, Arrow tables and Parquet datasets, a
batch at a time."""
import contextlib
import os
from typing import Any, NamedTuple
import shutil
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

BATCH_ROWS = 1 << 20

LOCAL_TYPES = (pd.DataFrame, pa.Table, ds.Dataset)

NO_PREBUFFER = ds.ParquetFragmentScanOptions(pre_buffer=False)
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
        parts = obj.to_batches(columns=None if cols is None else list(cols),
                               batch_size=rows, use_threads=False, **_scan_options(obj))
        schema = obj.schema if cols is None else pa.schema(
            [obj.schema.field(c) for c in cols])
    seen = False
    for batch in parts:
        seen = True
        yield batch.to_pandas()
    if not seen:
        yield schema.empty_table().to_pandas()


CHUNK_ROWS = 1 << 20

SLICE_ROWS = 10_000


class Chunk(NamedTuple):
    """A run of whole row groups of one Parquet file, read by one worker."""
    filesystem: Any
    path: str
    row_groups: tuple
    rows: int


@contextlib.contextmanager
def scratch():
    """Yield a scratch directory that is removed afterwards."""
    from .._core.storage import spill_dir

    path = tempfile.mkdtemp(prefix='iris-', dir=spill_dir())
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


def _parquet_files(obj):
    return (isinstance(obj, ds.FileSystemDataset)
            and isinstance(obj.format, ds.ParquetFileFormat))


def stage(obj, directory, name):
    """Return an input as a dataset of Parquet files, writing it to `directory` first
    when it is held in memory."""
    if _parquet_files(obj):
        return obj
    path = os.path.join(directory, name)
    os.makedirs(path)
    pq.write_table(to_arrow(obj), os.path.join(path, 'part-0.parquet'),
                   row_group_size=CHUNK_ROWS)
    return dataset(path)


def chunks(source, rows=None):
    """Return a dataset of Parquet files as chunks of about `rows` rows, in input
    order."""
    rows = rows or CHUNK_ROWS
    out = []
    for frag in source.get_fragments():
        meta = frag.metadata
        sizes = [meta.row_group(k).num_rows for k in range(meta.num_row_groups)]
        start, held = 0, 0
        for k, n in enumerate(sizes):
            held += n
            if held >= rows:
                out.append(Chunk(source.filesystem, frag.path, tuple(range(start, k + 1)),
                                 held))
                start, held = k + 1, 0
        if start < len(sizes):
            out.append(Chunk(source.filesystem, frag.path, tuple(range(start, len(sizes))),
                             held))
    return out


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
        return pa.Schema.from_pandas(obj, preserve_index=False)
    return obj.schema


def arrow_batches(obj, cols):
    """Yield the columns `cols` of an input as Arrow record batches."""
    if isinstance(obj, (pd.DataFrame, pa.Table)):
        yield from to_arrow(obj).select(list(cols)).to_batches()
    else:
        yield from obj.to_batches(columns=list(cols), use_threads=False,
                                  **_scan_options(obj))


def _scan_options(obj):
    if _parquet_files(obj):
        return {'fragment_scan_options': NO_PREBUFFER, **NO_READAHEAD}
    return dict(NO_READAHEAD)


def _row_groups(source):
    for frag in source.get_fragments():
        meta = frag.metadata
        for g in range(meta.num_row_groups):
            yield frag, g, meta.row_group(g).num_rows


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


def read_rows(source, start, stop, cols):
    """Return the rows `start` to `stop` of a Parquet dataset, columns `cols` only."""
    pieces, pos = [], 0
    for frag, g, n in _row_groups(source):
        if pos + n > start and pos < stop:
            with source.filesystem.open_input_file(frag.path) as fh:
                t = pq.ParquetFile(fh).read_row_group(g, columns=list(cols))
            lo, hi = max(start - pos, 0), min(stop - pos, n)
            pieces.append(t.slice(lo, hi - lo))
        pos += n
        if pos >= stop:
            break
    if not pieces:
        return pa.schema([source.schema.field(c) for c in cols]).empty_table()
    return pa.concat_tables(pieces)


def require_disk(nbytes, path, what, remedy='Set IRIS_SPILL_DIR to a directory on a '
                                             'larger volume, or free space.'):
    """Raise an error before any work when the volume of `path` cannot hold `nbytes`."""
    probe = os.path.abspath(os.fspath(path))
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            return
        probe = parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        return
    if free < nbytes:
        raise RuntimeError(f'not enough free disk for {what} in {probe!r}: need about '
                           f'{nbytes / 2**30:.2f} GiB, {free / 2**30:.2f} GiB free. '
                           f'{remedy}')
