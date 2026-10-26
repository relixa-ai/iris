"""Candidate pairs made distinct and flat, in parts ordered by `(id2, id1)`, on the
local engine."""
import glob
import math
import os
from functools import partial

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .._core.utils import get_logger
from . import disk, inputs
from .procpool import atomic_output, run_tasks, workers_for

PAIRS_PER_PART = 1 << 20

SAMPLE_PAIRS = 1 << 20


PAIR_SCHEMA = pa.schema([('id1', pa.int64()), ('id2', pa.int64())])

log = get_logger(__name__)


def is_flat(cols):
    """Return whether the columns describe a flat candidate set."""
    return 'id1' in cols and 'id1_list' not in cols


def _columns(cols):
    return ['id1', 'id2'] if is_flat(cols) else ['id2', 'id1_list']


def _pairs(table, flat):
    if flat:
        id1, id2 = table.column('id1'), table.column('id2')
        nulls = id1.null_count + id2.null_count
    else:
        lists = table.column('id1_list').combine_chunks()
        id2 = table.column('id2')
        values = pc.list_flatten(lists)
        nulls = values.null_count + id2.null_count
        sizes = pc.fill_null(pc.list_value_length(lists), 0).to_numpy(zero_copy_only=False)
    if nulls:
        raise ValueError(f'candidates contain {nulls} null id(s): every pair needs two '
                         f'ids (iris.check_candidates reports them)')
    if flat:
        return (id1.to_numpy().astype(np.int64, copy=False),
                id2.to_numpy().astype(np.int64, copy=False))
    return (values.to_numpy().astype(np.int64, copy=False),
            np.repeat(id2.to_numpy().astype(np.int64, copy=False), sizes))


def count(cands):
    """Return the number of pairs a candidate set lists, repeats included."""
    cols = inputs.columns(cands)
    if is_flat(cols):
        return sum(b.num_rows for b in inputs.arrow_batches(cands, ['id1']))
    return sum(int(pc.sum(pc.list_value_length(b.column('id1_list'))).as_py() or 0)
               for b in inputs.arrow_batches(cands, ['id1_list']))


def _chunk_pairs(chunk, flat, drop_self, orient):
    id1, id2 = _pairs(inputs.read_chunk(chunk, ['id1', 'id2'] if flat else ['id2', 'id1_list']),
                      flat)
    if drop_self:
        keep = id1 != id2
        id1, id2 = id1[keep], id2[keep]
    if orient:
        id1, id2 = np.minimum(id1, id2), np.maximum(id1, id2)
    return id1, id2


def part_of(major, minor, bounds):
    """Return the part of each pair: how many of the sorted `(major, minor)` boundaries
    are at or below it."""
    if not len(bounds):
        return np.zeros(len(major), np.int64)
    first, second = bounds[:, 0], bounds[:, 1]
    lo = np.searchsorted(first, major, side='left')
    hi = np.searchsorted(first, major, side='right')
    part = lo.astype(np.int64)
    tied = np.flatnonzero(hi > lo)
    if len(tied):
        for value in np.unique(major[tied]).tolist():
            at = tied[major[tied] == value]
            a, z = lo[at[0]], hi[at[0]]
            part[at] = a + np.searchsorted(second[a:z], minor[at], side='right')
    return part


def _sample(task, flat, drop_self, orient, by):
    chunk, every = task
    id1, id2 = _chunk_pairs(chunk, flat, drop_self, orient)
    pick = slice(0, None, every)
    major, minor = (id1, id2) if by == 'id1' else (id2, id1)
    return np.stack([major[pick], minor[pick]], axis=1)


def boundaries(chunks, n, parts, workers, flat=True, drop_self=False, orient=False,
               by='id2'):
    """Return the `(major, minor)` keys that cut a sample of the pairs into `parts`
    parts of about as many pairs each."""
    if parts <= 1 or not chunks:
        return np.empty((0, 2), np.int64)
    every = max(1, n // SAMPLE_PAIRS)
    found = run_tasks(partial(_sample, flat=flat, drop_self=drop_self, orient=orient, by=by),
                      [(c, every) for c in chunks], workers,
                      task_bytes=inputs.chunk_bytes())
    sample = np.concatenate(found) if found else np.empty((0, 2), np.int64)
    if not len(sample):
        return np.empty((0, 2), np.int64)
    sample = sample[np.lexsort((sample[:, 1], sample[:, 0]))]
    return np.unique(sample[(np.arange(1, parts) * len(sample)) // parts], axis=0)


def _route(task, flat, boundaries, out_dir, drop_self=False, orient=False, by='id2'):
    c, chunk = task
    id1, id2 = _chunk_pairs(chunk, flat, drop_self, orient)
    major, minor = (id1, id2) if by == 'id1' else (id2, id1)
    part = part_of(major, minor, boundaries)
    order = np.argsort(part, kind='stable')
    part, id1, id2 = part[order], id1[order], id2[order]
    cuts = np.r_[0, np.flatnonzero(np.diff(part)) + 1, len(part)]
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        if hi > lo:
            k = int(part[lo])
            path = os.path.join(out_dir, f'part-{k:05d}', f'chunk-{c:05d}.npy')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with atomic_output(path) as tmp, open(tmp, 'wb') as fh:
                np.save(fh, np.stack([id1[lo:hi], id2[lo:hi]], axis=1))
    return len(id1)


def _sort(task, out_dir, by='id2'):
    k, routed = task
    files = sorted(glob.glob(os.path.join(routed, '*.npy')))
    pairs = (np.concatenate([np.load(f) for f in files]) if files
             else np.empty((0, 2), np.int64))
    n_in = len(pairs)
    major, minor = (0, 1) if by == 'id1' else (1, 0)
    pairs = pairs[np.lexsort((pairs[:, minor], pairs[:, major]))]
    keep = np.ones(len(pairs), bool)
    keep[1:] = (pairs[1:, 0] != pairs[:-1, 0]) | (pairs[1:, 1] != pairs[:-1, 1])
    pairs = pairs[keep]
    path = os.path.join(out_dir, f'pairs-{k:05d}.parquet')
    with atomic_output(path) as tmp:
        pq.write_table(pa.table({'id1': pairs[:, 0], 'id2': pairs[:, 1]},
                                schema=PAIR_SCHEMA), tmp)
    return path, n_in, len(pairs)


def canonical(cands, directory, workers=None, one_table=False):
    """Write the candidates as distinct pairs in parts ordered by `(id2, id1)`, and
    return the parts, the repeats dropped and, for one table, the self-pairs dropped."""
    workers = workers or workers_for()
    flat = is_flat(inputs.columns(cands))
    n = count(cands)
    parts = max(1, math.ceil(n / PAIRS_PER_PART))
    routed = os.path.join(directory, 'routed')
    os.makedirs(routed)
    with inputs.scratch() as tmp:
        src = inputs.stage(cands, tmp, 'candidates', workers=workers)
        inputs.require_disk(16 * n, routed, 'the routed candidate pairs', kept=0)
        chunks = inputs.chunks(src, workers=workers)
        cuts = boundaries(chunks, n, parts, workers, flat, one_table, one_table)
        kept = run_tasks(partial(_route, flat=flat, boundaries=cuts, out_dir=routed,
                                 drop_self=one_table, orient=one_table),
                         list(enumerate(chunks)), workers, task_bytes=inputs.chunk_bytes())
    tasks, out = _sorted(routed, directory, sum(kept), workers)
    for _, routed_part in tasks:
        for f in glob.glob(os.path.join(routed_part, '*.npy')):
            os.remove(f)
    repeats = sum(n_in - n_out for _, n_in, n_out in out)
    return [(path, n_out) for path, _, n_out in out], repeats, n - sum(kept)


def _sorted(routed, directory, total, workers, by='id2'):
    tasks = [(k, os.path.join(routed, name))
             for k, name in enumerate(sorted(os.listdir(routed)))]
    meter = disk.Meter('the sorted candidate pairs', directory, total)
    finished = {}

    def done(k, result):
        finished[os.path.basename(result[0])] = result[1]
        meter.wrote(sum(finished.values()), *disk.written_in(directory, finished))
    out = disk.measured(run_tasks, partial(_sort, out_dir=directory, by=by), tasks, workers,
                        meter, 16 * total, done, task_bytes=16 * PAIRS_PER_PART)
    return tasks, out


@disk.disk_step('clean_candidates')
def clean(cands, drop_self_pairs=False, canonicalize_mirrors=False, directory=None,
          workers=None):
    """Return the candidates as distinct flat pairs, without self-pairs and mirrored
    pairs when requested."""
    import tempfile

    from .._core.storage import scratch_prefix, spill_dir
    from .featurize import _remove_at_exit

    workers = workers or workers_for()
    if directory is None:
        directory = tempfile.mkdtemp(prefix=scratch_prefix('pairs'), dir=spill_dir())
        _remove_at_exit(directory)
        disk.output(directory)
    os.makedirs(directory, exist_ok=True)
    flat = is_flat(inputs.columns(cands))
    n = count(cands)
    parts = max(1, math.ceil(n / PAIRS_PER_PART))
    routed = os.path.join(directory, 'routed')
    os.makedirs(routed)
    with inputs.scratch() as tmp:
        src = inputs.stage(cands, tmp, 'candidates', workers=workers)
        inputs.require_disk(16 * n, routed, 'the routed candidate pairs', kept=0)
        chunks = inputs.chunks(src, workers=workers)
        cuts = boundaries(chunks, n, parts, workers, flat, drop_self_pairs,
                          canonicalize_mirrors)
        kept = run_tasks(partial(_route, flat=flat, boundaries=cuts, out_dir=routed,
                                 drop_self=drop_self_pairs, orient=canonicalize_mirrors),
                         list(enumerate(chunks)), workers, task_bytes=inputs.chunk_bytes())
    _, out = _sorted(routed, directory, sum(kept), workers)
    import shutil
    shutil.rmtree(routed, ignore_errors=True)
    kept = sum(n_out for _, _, n_out in out)
    log.info(f'clean_candidates: {n - kept} of {n} pair(s) removed')
    if not out:
        with atomic_output(os.path.join(directory, 'pairs-00000.parquet')) as tmp:
            pq.write_table(PAIR_SCHEMA.empty_table(), tmp)
    return inputs.dataset(directory)

