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
import pyarrow.parquet as pq

from .._core import recpack, records
from .._core.storage import unknown_id_error
from .._core.textform import to_text
from . import inputs
from .procpool import atomic_output, run_tasks, workers_for

SEGMENT_BYTES = 256 * 2**20

MIN_SEGMENT_ROWS, MAX_SEGMENT_ROWS = 1_000, 4_000_000

MAX_OPEN_SEGMENTS = 64

WORKER_GB = 1.0

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
    return os.path.join(directory, f'seg-{k:05d}.{ext}')


def _build_segment(task, src, cols, layout, features, for_a, for_b, tz, directory, what):
    kind, k, where = task
    if kind == 'rows':
        table = inputs.read_rows(src, where[0], where[1], ['_id'] + cols)
    else:
        table = inputs.dataset(where).to_table(columns=['_id'] + cols)
        table = table.sort_by('_id')
    ids = table.column('_id').to_numpy()
    if len(ids) > 1 and not np.all(ids[1:] > ids[:-1]):
        dup = ids[1:][ids[1:] == ids[:-1]][:5].tolist()
        raise ValueError(f"{what} '_id' must be unique; e.g. ids {dup} repeat")
    frame = _text_frame(table, cols, tz)
    del table
    ends = []
    written = 0
    with atomic_output(_segment_path(directory, k, 'bin')) as tmp, open(tmp, 'wb') as fh:
        for _, data, slice_ends in records.pack_rows(frame, layout, features, for_a, for_b):
            fh.write(memoryview(data))
            ends.append(slice_ends.astype(np.uint64) + np.uint64(written))
            written += len(data)
    ends = np.concatenate(ends) if ends else np.empty(0, np.uint64)
    with atomic_output(_segment_path(directory, k, 'end')) as tmp:
        ends.tofile(tmp)
    with atomic_output(_segment_path(directory, k, 'ids')) as tmp:
        np.ascontiguousarray(ids, np.int64).tofile(tmp)
    return {'k': k, 'min_id': int(ids[0]), 'max_id': int(ids[-1]), 'rows': int(len(ids)),
            'bytes': int(written)}


def _route_chunk(task, cols, boundaries, out_dir):
    c, chunk = task
    table = inputs.read_chunk(chunk, ['_id'] + cols)
    seg = np.searchsorted(boundaries, table.column('_id').to_numpy(), side='right')
    for k in np.unique(seg).tolist():
        part = os.path.join(out_dir, f'seg-{k:05d}')
        os.makedirs(part, exist_ok=True)
        with atomic_output(os.path.join(part, f'part-{c:05d}.parquet')) as tmp:
            pq.write_table(table.filter(pa.array(seg == k)), tmp)
    return c


def build(table, features, for_a, for_b, tz, directory, what='table', workers=None):
    """Return a record store of `table` built in `directory` by worker processes."""
    cols = record_columns(table, what, features, for_a, for_b)
    layout = recpack.layout(features, for_a, for_b)
    os.makedirs(directory)
    workers = workers or workers_for(WORKER_GB)
    with inputs.scratch() as tmp:
        src = inputs.stage(table, tmp, 'in')
        ids = inputs.read_column(src, '_id')
        if ids.null_count:
            raise ValueError(f"{what} '_id' has {ids.null_count} null value(s); every "
                             f"record needs an id")
        ids = ids.to_numpy().astype(np.int64, copy=False)
        n = len(ids)
        mean = _mean_record(src, n, cols, layout, features, for_a, for_b, tz)
        rows_per_segment = int(min(max(SEGMENT_BYTES // mean, MIN_SEGMENT_ROWS),
                                   MAX_SEGMENT_ROWS)) if n else 1
        inputs.require_disk(int(n * mean * 2), directory, f'the record store of {what}')
        if n < 2 or bool(np.all(ids[1:] > ids[:-1])):
            tasks = [('rows', k, (lo, min(lo + rows_per_segment, n)))
                     for k, lo in enumerate(range(0, n, rows_per_segment))]
        else:
            boundaries = np.sort(ids)[rows_per_segment::rows_per_segment]
            routed = os.path.join(tmp, 'routed')
            os.makedirs(routed)
            run_tasks(partial(_route_chunk, cols=cols, boundaries=boundaries,
                              out_dir=routed),
                      list(enumerate(inputs.chunks(src))), workers)
            tasks = [('routed', k, os.path.join(routed, name))
                     for k, name in enumerate(sorted(os.listdir(routed)))]
        del ids
        fn = partial(_build_segment, src=src, cols=cols, layout=layout, features=features,
                     for_a=for_a, for_b=for_b, tz=tz, directory=directory, what=what)
        segments = run_tasks(fn, tasks, workers)
    for prev, seg in zip(segments, segments[1:]):
        if prev['max_id'] >= seg['min_id']:
            raise ValueError(f"{what} '_id' must be unique; e.g. ids [{seg['min_id']}] "
                             f"repeat")
    with atomic_output(os.path.join(directory, MANIFEST)) as tmp, open(tmp, 'w') as fh:
        json.dump({'layout': [list(x) for x in layout], 'rows': n,
                   'segments': segments}, fh)
    return Store(directory)


def _mean_record(src, n, cols, layout, features, for_a, for_b, tz):
    if n == 0:
        return 1.0
    sample = inputs.read_rows(src, 0, min(n, records.PREPROCESS_ROWS), ['_id'] + cols)
    size = sum(len(data) for _, data, _ in
               records.pack_rows(_text_frame(sample, cols, tz), layout, features, for_a,
                                 for_b))
    return max(1.0, size / len(sample))


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
            hit = (np.memmap(path('ids'), np.int64, 'r', shape=(rows,)),
                   np.memmap(path('end'), np.uint64, 'r', shape=(rows,)),
                   np.memmap(path('bin'), np.uint8, 'r',
                             shape=(self.segments[i]['bytes'],))
                   if self.segments[i]['bytes'] else np.empty(0, np.uint8))
            while len(self._open) >= MAX_OPEN_SEGMENTS:
                self._open.popitem(last=False)
        self._open[i] = hit
        return hit

    def gather(self, ids, side=None):
        """Return the packed records of `ids` and their start and end offsets."""
        ids = np.ascontiguousarray(ids, np.int64)
        n = len(ids)
        if n == 0:
            return np.empty(0, np.uint8), np.empty(0, np.int64), np.empty(0, np.int64)
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
