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
from . import inputs
from .procpool import atomic_output, run_tasks, workers_for

PAIRS_PER_PART = 1 << 20

WORKER_GB = 1.0

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


def _route(task, flat, boundaries, out_dir, parts=None, drop_self=False, orient=False,
           by='id2'):
    c, chunk = task
    id1, id2 = _pairs(inputs.read_chunk(chunk, ['id1', 'id2'] if flat else ['id2', 'id1_list']),
                      flat)
    if drop_self:
        keep = id1 != id2
        id1, id2 = id1[keep], id2[keep]
    if orient:
        id1, id2 = np.minimum(id1, id2), np.maximum(id1, id2)
    key = id1 if by == 'id1' else id2
    part = (np.searchsorted(boundaries, key, side='right') if boundaries is not None
            else np.mod(key, parts))
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


def canonical(cands, b_ids, directory, workers=None):
    """Write the candidates as distinct pairs in parts ordered by `(id2, id1)`, and
    return the parts and the repeats dropped."""
    workers = workers or workers_for(WORKER_GB)
    flat = is_flat(inputs.columns(cands))
    n = count(cands)
    inputs.require_disk(32 * n, directory, 'the candidate pairs')
    parts = max(1, math.ceil(n / PAIRS_PER_PART))
    cut_at = (np.arange(1, parts) * len(b_ids)) // parts
    boundaries = np.unique(np.asarray(b_ids, np.int64)[cut_at]) if len(b_ids) \
        else np.empty(0, np.int64)
    routed = os.path.join(directory, 'routed')
    os.makedirs(routed)
    with inputs.scratch() as tmp:
        src = inputs.stage(cands, tmp, 'candidates')
        run_tasks(partial(_route, flat=flat, boundaries=boundaries, out_dir=routed),
                  list(enumerate(inputs.chunks(src))), workers)
    tasks = [(k, os.path.join(routed, name))
             for k, name in enumerate(sorted(os.listdir(routed)))]
    out = run_tasks(partial(_sort, out_dir=directory), tasks, workers)
    for _, routed_part in tasks:
        for f in glob.glob(os.path.join(routed_part, '*.npy')):
            os.remove(f)
    repeats = sum(n_in - n_out for _, n_in, n_out in out)
    return [(path, n_out) for path, _, n_out in out], repeats


def clean(cands, drop_self_pairs=False, canonicalize_mirrors=False, directory=None,
          workers=None):
    """Return the candidates as distinct flat pairs, without self-pairs and mirrored
    pairs when requested."""
    import tempfile

    from .._core.storage import spill_dir
    from .featurize import _remove_at_exit

    workers = workers or workers_for(WORKER_GB)
    if directory is None:
        directory = tempfile.mkdtemp(prefix='iris-pairs-', dir=spill_dir())
        _remove_at_exit(directory)
    os.makedirs(directory, exist_ok=True)
    flat = is_flat(inputs.columns(cands))
    n = count(cands)
    inputs.require_disk(32 * n, directory, 'the candidate pairs')
    parts = max(1, math.ceil(n / PAIRS_PER_PART))
    routed = os.path.join(directory, 'routed')
    os.makedirs(routed)
    with inputs.scratch() as tmp:
        src = inputs.stage(cands, tmp, 'candidates')
        run_tasks(partial(_route, flat=flat, boundaries=None, out_dir=routed, parts=parts,
                          drop_self=drop_self_pairs, orient=canonicalize_mirrors),
                  list(enumerate(inputs.chunks(src))), workers)
    tasks = [(k, os.path.join(routed, name))
             for k, name in enumerate(sorted(os.listdir(routed)))]
    out = run_tasks(partial(_sort, out_dir=directory), tasks, workers)
    import shutil
    shutil.rmtree(routed, ignore_errors=True)
    kept = sum(n_out for _, _, n_out in out)
    log.info(f'clean_candidates: {n - kept} of {n} pair(s) removed')
    if not out:
        with atomic_output(os.path.join(directory, 'pairs-00000.parquet')) as tmp:
            pq.write_table(PAIR_SCHEMA.empty_table(), tmp)
    return inputs.dataset(directory)

