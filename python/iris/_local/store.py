"""The record store of the local engine: the packed records of a table, in memory-mapped
segments."""
import json
import os
from collections import OrderedDict
from functools import partial

import numba as nb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .. import resources
from .._core import recpack, records
from .._core.storage import map_read_only, unknown_id_error
from .._core.textform import to_text
from . import inputs
from .procpool import atomic_output, memory_error, run_tasks, workers_for

SEGMENT_BYTES = 256 * 2**20

MIN_SEGMENT_ROWS, MAX_SEGMENT_ROWS = 1_000, 4_000_000

ROUTE_BYTES = 128 * 2**20

SLICE_PACKED_BYTES = 64 * 2**20

ROUTE_SAMPLE = 1 << 20

ROUTE_FLUSH_BYTES = 1 << 16

SAMPLE_CHUNKS, SAMPLE_ROWS, SAMPLE_BYTES = 8, 250, 2**20

MAX_OPEN_SEGMENTS = 64


MANIFEST = 'manifest.json'


def record_columns(table, what, features, for_a, for_b):
    """Return the columns a record holds for the given features and sides, in table
    order."""
    names = inputs.columns(table)
    wanted = set()
    if for_a:
        wanted |= {f.a_attr for f in features}
    if for_b:
        wanted |= {f.b_attr for f in features}
    missing = sorted(c for c in wanted if c not in names)
    if missing:
        raise ValueError(f'features read column(s) {missing} that {what} does not have '
                         f'(columns: {names})')
    return [c for c in names if c in wanted and c != '_id']


def _text_frame(table, cols, tz):
    frame = {'_id': table.column('_id').to_numpy()}
    frame.update({c: to_text(table.column(c), tz, c).to_pandas() for c in cols})
    return pd.DataFrame(frame)


def _segment_path(directory, k, ext):
    return os.path.join(directory, f'seg-{k}.{ext}')


class _Segments:

    def __init__(self, directory, task, limits):
        self.directory, self.task, self.done = directory, task, []
        self.seg_bytes, self.min_rows, self.max_rows = limits
        self._open()

    def _open(self):
        self.key = f'{self.task:05d}-{len(self.done):05d}'
        self.tmp = os.path.join(self.directory, f'.seg-{self.key}.bin.tmp')
        self.fh = open(self.tmp, 'wb')
        self.ids, self.ends, self.written, self.rows = [], [], 0, 0

    def add(self, ids, data, ends):
        start = 0
        while start < len(ids):
            room = max(self.min_rows - self.rows, 0)
            base = int(ends[start - 1]) if start else 0
            fit = int(np.searchsorted(ends[start:], base + self.seg_bytes - self.written,
                                      side='right'))
            take = max(1, room, fit)
            take = min(take, self.max_rows - self.rows, len(ids) - start)
            stop = start + take
            lo = int(ends[start - 1]) if start else 0
            self.fh.write(memoryview(data[lo:int(ends[stop - 1])]))
            self.ends.append(ends[start:stop].astype(np.uint64) - np.uint64(lo)
                             + np.uint64(self.written))
            self.ids.append(np.ascontiguousarray(ids[start:stop], np.int64))
            self.written += int(ends[stop - 1]) - lo
            self.rows += take
            start = stop
            if self.rows >= self.max_rows or (self.rows >= self.min_rows
                                              and self.written >= self.seg_bytes):
                self._close()
                self._open()

    def _close(self):
        self.fh.close()
        if not self.rows:
            os.remove(self.tmp)
            return
        ids = np.concatenate(self.ids)
        os.replace(self.tmp, _segment_path(self.directory, self.key, 'bin'))
        with atomic_output(_segment_path(self.directory, self.key, 'end')) as tmp:
            np.concatenate(self.ends).tofile(tmp)
        with atomic_output(_segment_path(self.directory, self.key, 'ids')) as tmp:
            ids.tofile(tmp)
        self.done.append({'k': self.key, 'min_id': int(ids[0]), 'max_id': int(ids[-1]),
                          'rows': int(len(ids)), 'bytes': int(self.written)})

    def finish(self):
        self._close()
        return self.done


def _unique_error(what, ids):
    return ValueError(f"{what} '_id' must be unique; e.g. ids {ids[:5]} repeat")


def _slices_in_order(kind, where, cols, rows):
    if kind == 'rows':
        yield from inputs.read_slices(where, ['_id'] + cols, rows=rows)
        return
    table = inputs.dataset(where).to_table(columns=['_id'] + cols).sort_by('_id')
    yield from inputs.slices(table, rows)


def _build_segments(task, cols, layout, features, for_a, for_b, tz, directory, what,
                    limits, rows):
    kind, k, where = task
    out = _Segments(directory, k, limits)
    last = part = None
    try:
        for part in _slices_in_order(kind, where, cols, rows):
            ids = part.column('_id').to_numpy()
            if len(ids) and last is not None and ids[0] <= last:
                raise _unique_error(what, [int(ids[0])])
            if len(ids) > 1 and not np.all(ids[1:] > ids[:-1]):
                raise _unique_error(what, ids[1:][ids[1:] <= ids[:-1]].tolist())
            if len(ids):
                last = ids[-1]
            for slice_ids, data, ends in records.pack_rows(_text_frame(part, cols, tz),
                                                           layout, features, for_a, for_b):
                out.add(slice_ids, data, ends)
            part = None
        return out.finish()
    except MemoryError as e:
        raise memory_error(e, _largest(part, cols, what)) from e
    finally:
        if not out.fh.closed:
            out.fh.close()
        if os.path.exists(out.tmp):
            os.remove(out.tmp)


def _largest(part, cols, what):
    if part is None or not part.num_rows:
        return f'it ran out while reading the records of {what}'
    sizes = _row_bytes(part.select(cols)) if cols else np.zeros(part.num_rows, np.int64)
    at = int(np.argmax(sizes))
    return (f'the largest record of the {part.num_rows:,} it was packing is id '
            f"{part.column('_id')[at].as_py()} of {what}, {sizes[at] / 2**20:,.1f} MiB as "
            f'read')


def _raise_open_files(wanted):
    try:
        import resource
    except ImportError:
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = wanted if hard == resource.RLIM_INFINITY else min(wanted, hard)
    if target > soft:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        except (ValueError, OSError):
            pass


def _route(task, cols, boundaries, out_dir):
    k, chunks = task
    _raise_open_files(len(boundaries) + 1 + 256)
    writers, held, sizes = {}, {}, {}

    def flush(r):
        table = pa.concat_tables(held.pop(r))
        sizes[r] = 0
        if r not in writers:
            part = os.path.join(out_dir, f'range-{r:05d}')
            os.makedirs(part, exist_ok=True)
            path = os.path.join(part, f'part-{k:05d}.parquet')
            writers[r] = (pq.ParquetWriter(f'{path}.tmp', table.schema), path)
        writers[r][0].write_table(table)

    try:
        for chunk in chunks:
            for part in inputs.read_slices(chunk, ['_id'] + cols):
                ranges = np.searchsorted(boundaries, part.column('_id').to_numpy(),
                                         side='right')
                for r in np.unique(ranges).tolist():
                    rows = part.filter(pa.array(ranges == r))
                    held.setdefault(r, []).append(rows)
                    sizes[r] = sizes.get(r, 0) + rows.nbytes
                    if sizes[r] >= ROUTE_FLUSH_BYTES:
                        flush(r)
        for r in list(held):
            flush(r)
    finally:
        for writer, path in writers.values():
            writer.close()
    for _, path in writers.values():
        os.replace(f'{path}.tmp', path)
    return k


def _row_bytes(table):
    total = np.zeros(table.num_rows, np.int64)
    for col in table.columns:
        t = col.type
        if pa.types.is_dictionary(t):
            col, t = col.cast(t.value_type), t.value_type
        if pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_binary(t) \
                or pa.types.is_large_binary(t):
            total += pc.fill_null(pc.binary_length(col), 0).to_numpy().astype(np.int64) + 4
        else:
            try:
                width = t.bit_width
            except ValueError:
                width = 64
            total += max(1, width // 8)
    return total


def _summary(chunk):
    rows = nulls = 0
    first = last = None
    ordered = True
    for part in inputs.read_slices(chunk, ['_id']):
        column = part.column('_id')
        rows += part.num_rows
        nulls += column.null_count
        if column.null_count or not part.num_rows:
            continue
        ids = column.to_numpy().astype(np.int64, copy=False)
        if (last is not None and ids[0] <= last) or np.any(ids[1:] <= ids[:-1]):
            ordered = False
        first = ids[0] if first is None else first
        last = ids[-1]
    return rows, nulls, None if first is None else int(first), \
        None if last is None else int(last), ordered


def _sample(task, cols):
    chunk, every = task
    ids, sizes, at = [], [], 0
    for part in inputs.read_slices(chunk, ['_id'] + cols):
        keep = np.flatnonzero((at + np.arange(part.num_rows)) % every == 0)
        at += part.num_rows
        if len(keep):
            ids.append(part.column('_id').to_numpy()[keep].astype(np.int64, copy=False))
            sizes.append((_row_bytes(part.select(cols)) if cols
                          else np.ones(part.num_rows, np.int64))[keep] * every)
    empty = np.empty(0, np.int64)
    return (np.concatenate(ids) if ids else empty, np.concatenate(sizes) if sizes else empty)


def _ranges(chunks, cols, rows, workers):
    every = max(1, rows // ROUTE_SAMPLE)
    found = run_tasks(partial(_sample, cols=cols), [(c, every) for c in chunks], workers,
                      task_bytes=inputs.chunk_bytes())
    ids = np.concatenate([i for i, _ in found])
    sizes = np.concatenate([s for _, s in found])
    order = np.argsort(ids, kind='stable')
    ids, cum = ids[order], np.cumsum(sizes[order])
    total = int(cum[-1]) if len(cum) else 0
    most = resources.piece_bytes(ROUTE_BYTES)
    per_range = max(1, min(most, total // max(1, 2 * workers) + 1))
    cuts = np.searchsorted(cum, np.arange(per_range, total, per_range), side='right')
    cuts = np.unique(cuts[(cuts > 0) & (cuts < len(ids))])
    return np.unique(ids[cuts])


def build(table, features, for_a, for_b, tz, directory, what='table', workers=None):
    """Return a record store of `table` built in `directory` by worker processes."""
    cols = record_columns(table, what, features, for_a, for_b)
    layout = recpack.layout(features, for_a, for_b)
    os.makedirs(directory)
    workers = workers or workers_for()
    with inputs.scratch() as tmp:
        src = inputs.stage(table, tmp, 'in', ['_id'] + cols, workers)
        chunks = inputs.chunks(src, workers=workers)
        found = run_tasks(_summary, chunks, workers, task_bytes=inputs.chunk_bytes())
        n = sum(rows for rows, _, _, _, _ in found)
        nulls = sum(x for _, x, _, _, _ in found)
        if nulls:
            raise ValueError(f"{what} '_id' has {nulls} null value(s); every record needs "
                             f"an id")
        edges = [(first, last) for _, _, first, last, _ in found if first is not None]
        in_order = all(ordered for *_, ordered in found) and all(
            prev[1] < nxt[0] for prev, nxt in zip(edges, edges[1:]))
        sample = _sample_sizes(src, n, cols, layout, features, for_a, for_b, tz, workers)
        mean = max(1.0, sample[1] / max(sample[0], 1))
        measured = inputs.column_bytes(src, ['_id'] + cols)
        planned = _store_bytes(n, measured, *sample)
        inputs.require_disk(planned, directory, f'the record store of {what}')
        if in_order:
            tasks = [('rows', k, c) for k, c in enumerate(chunks)]
        else:
            boundaries = _ranges(chunks, cols, n, workers)
            routed = os.path.join(tmp, 'routed')
            os.makedirs(routed)
            groups = max(1, min(len(chunks), 2 * workers))
            run_tasks(partial(_route, cols=cols, boundaries=boundaries, out_dir=routed),
                      [(k, chunks[k::groups]) for k in range(groups)], workers,
                      task_bytes=inputs.chunk_bytes())
            tasks = [('routed', k, os.path.join(routed, name))
                     for k, name in enumerate(sorted(os.listdir(routed)))]
        fn = partial(_build_segments, cols=cols, layout=layout, features=features,
                     for_a=for_a, for_b=for_b, tz=tz, directory=directory, what=what,
                     limits=(SEGMENT_BYTES, MIN_SEGMENT_ROWS, MAX_SEGMENT_ROWS),
                     rows=int(max(1, min(records.PREPROCESS_ROWS,
                                         resources.piece_bytes(SLICE_PACKED_BYTES) // mean))))
        piece = inputs.chunk_bytes() if in_order else resources.piece_bytes(ROUTE_BYTES)
        segments = [seg for done in run_tasks(fn, tasks, workers, task_bytes=piece)
                    for seg in done]
    for prev, seg in zip(segments, segments[1:]):
        if prev['max_id'] >= seg['min_id']:
            raise _unique_error(what, [seg['min_id']])
    with atomic_output(os.path.join(directory, MANIFEST)) as tmp, open(tmp, 'w') as fh:
        json.dump({'layout': [list(x) for x in layout], 'rows': n,
                   'segments': segments}, fh)
    return Store(directory)


def _samples(src, cols):
    if not inputs._parquet_files(src):
        left = SAMPLE_CHUNKS * SAMPLE_ROWS
        for batch in inputs.arrow_batches(src, ['_id'] + cols):
            if left <= 0:
                return
            part = pa.Table.from_batches([batch]).slice(0, left)
            left -= part.num_rows
            yield part
        return
    chunks = inputs.chunks(src)
    step = max(1, len(chunks) // SAMPLE_CHUNKS)
    for chunk in chunks[::step][:SAMPLE_CHUNKS]:
        part = next(inputs.read_slices(chunk, ['_id'] + cols, rows=SAMPLE_ROWS,
                                       nbytes=SAMPLE_BYTES), None)
        if part is not None:
            yield part.slice(0, max(1, min(part.num_rows, SAMPLE_ROWS)))


def _sample_size(part, cols, layout, features, for_a, for_b, tz):
    return (part.num_rows, sum(len(data) for _, data, _ in
                               records.pack_rows(_text_frame(part, cols, tz), layout,
                                                 features, for_a, for_b)),
            int(_row_bytes(part).sum()))


def _sample_sizes(src, n, cols, layout, features, for_a, for_b, tz, workers=None):
    if n == 0:
        return 0, 0, 0
    found = run_tasks(partial(_sample_size, cols=cols, layout=layout, features=features,
                              for_a=for_a, for_b=for_b, tz=tz),
                      list(_samples(src, cols)), workers or workers_for(),
                      task_bytes=inputs.chunk_bytes())
    return tuple(sum(f[k] for f in found) for k in range(3))


def _store_bytes(n, measured, rows, size, raw):
    if measured and raw:
        return int(measured * size / raw) + 16 * n
    return int(n * size / max(rows, 1)) + 16 * n


def estimate(table, features, for_a, for_b, tz, what='table'):
    """Return the bytes the record store of `table` is planned to take on disk, from a
    sample of its records and the bytes of its columns."""
    cols = record_columns(table, what, features, for_a, for_b)
    layout = recpack.layout(features, for_a, for_b)
    n = (len(table) if isinstance(table, pd.DataFrame) else
         table.num_rows if isinstance(table, pa.Table) else table.count_rows())
    sample = _sample_sizes(table, n, cols, layout, features, for_a, for_b, tz)
    return _store_bytes(n, inputs.column_bytes(table, ['_id'] + cols), *sample)


@nb.njit(nogil=True, cache=False)
def _copy(src, starts, ends, dst, at):
    for i in range(len(starts)):
        s = starts[i]
        o = at[i]
        for j in range(ends[i] - s):
            dst[o + j] = src[s + j]


class Store:
    """A record store opened for reading."""

    def __init__(self, directory):
        self.directory = directory
        with open(os.path.join(directory, MANIFEST)) as fh:
            manifest = json.load(fh)
        self.layout = [tuple(x) for x in manifest['layout']]
        self.rows = manifest['rows']
        self.segments = manifest['segments']
        self._min = np.array([s['min_id'] for s in self.segments], np.int64)
        self._max = np.array([s['max_id'] for s in self.segments], np.int64)
        self._open = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_open'] = OrderedDict()
        return state

    def _segment(self, i):
        hit = self._open.pop(i, None)
        if hit is None:
            k = self.segments[i]['k']
            rows = self.segments[i]['rows']
            path = partial(_segment_path, self.directory, k)
            hit = (map_read_only(path('ids'), np.int64, rows),
                   map_read_only(path('end'), np.uint64, rows),
                   map_read_only(path('bin'), np.uint8, self.segments[i]['bytes'])
                   if self.segments[i]['bytes'] else np.empty(0, np.uint8))
            while len(self._open) >= MAX_OPEN_SEGMENTS:
                self._open.popitem(last=False)
        self._open[i] = hit
        return hit

    def _locate(self, ids, side):
        n = len(ids)
        seg = np.searchsorted(self._min, ids, side='right') - 1
        outside = (seg < 0) | (ids > self._max[np.maximum(seg, 0)]) if len(self._min) \
            else np.ones(n, bool)
        if outside.any():
            raise unknown_id_error(ids[outside], side)
        cuts = np.r_[0, np.flatnonzero(np.diff(seg)) + 1, n]
        src_start = np.empty(n, np.int64)
        src_end = np.empty(n, np.int64)
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            seg_ids, seg_end, _ = self._segment(int(seg[lo]))
            want = ids[lo:hi]
            pos = np.searchsorted(seg_ids, want)
            found = (pos < len(seg_ids)) & (seg_ids[np.minimum(pos, len(seg_ids) - 1)] == want)
            if not found.all():
                raise unknown_id_error(want[~found], side)
            src_end[lo:hi] = seg_end[pos]
            src_start[lo:hi] = np.where(pos > 0, seg_end[np.maximum(pos - 1, 0)], 0)
        return seg, cuts, src_start, src_end

    def sizes(self, ids, side=None):
        """Return the packed size in bytes of the record of each of `ids`."""
        distinct, at = np.unique(np.asarray(ids, np.int64), return_inverse=True)
        if len(distinct) == 0:
            return np.empty(0, np.int64)
        _, _, src_start, src_end = self._locate(distinct, side)
        return (src_end - src_start)[at]

    def gather(self, ids, side=None):
        """Return the packed records of `ids` and their start and end offsets."""
        ids = np.ascontiguousarray(ids, np.int64)
        n = len(ids)
        if n == 0:
            return np.empty(0, np.uint8), np.empty(0, np.int64), np.empty(0, np.int64)
        seg, cuts, src_start, src_end = self._locate(ids, side)
        sizes = src_end - src_start
        ends = np.cumsum(sizes)
        starts = ends - sizes
        buf = np.empty(int(ends[-1]), np.uint8)
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            _, _, seg_bin = self._segment(int(seg[lo]))
            _copy(seg_bin, src_start[lo:hi], src_end[lo:hi], buf, starts[lo:hi])
        return buf, starts, ends

    def frame(self, ids, side=None):
        """Return the records of `ids` decoded into a frame, row `i` holding
        `ids[i]`."""
        buf, starts, ends = self.gather(ids, side)
        return recpack.decode_frame(self.layout, buf, starts, ends)
