"""The labeling pool on the local engine, drawn from feature vectors on disk."""
import glob
import math
import os
from functools import partial

import numba as nb
import numpy as np
import pyarrow as pa

from .._core import pool
from .._core.pairhash import pair_hash, seeded_hash
from . import inputs
from .procpool import atomic_output, run_tasks, workers_for

GROUP_ROWS = 1 << 21

WORKER_GB = 1.5


def _route(task, score_column, B, B_keep, groups, out_dir):
    c, chunk = task
    t = inputs.read_chunk(chunk, ['id1', 'id2', score_column])
    id1 = t.column('id1').to_numpy().astype(np.int64, copy=False)
    id2 = t.column('id2').to_numpy().astype(np.int64, copy=False)
    score = t.column(score_column).to_numpy(zero_copy_only=False).astype(np.float64)
    bucket = np.mod(pair_hash(id1, id2), B)
    live = bucket < B_keep
    id1, id2, score, bucket = id1[live], id2[live], score[live], bucket[live]
    group = bucket * groups // B_keep
    for g in np.unique(group).tolist():
        m = group == g
        path = os.path.join(out_dir, f'group-{g:05d}', f'chunk-{c:05d}.npz')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with atomic_output(path) as tmp, open(tmp, 'wb') as fh:
            np.savez(fh, id1=id1[m], id2=id2[m], score=score[m], bucket=bucket[m])
    return int(live.sum())


def _select(task, percent, n, seed):
    files = sorted(glob.glob(os.path.join(task, '*.npz')))
    parts = [np.load(f) for f in files]
    cols = {k: np.concatenate([p[k] for p in parts]) for k in ('id1', 'id2', 'score',
                                                              'bucket')}
    rnd = seeded_hash(seed, cols['id1'], cols['id2'])
    keep = pool.select_in_buckets(cols['bucket'], cols['id1'], cols['id2'],
                                  cols['score'], rnd, percent, n)
    return cols['id1'][keep], cols['id2'][keep], rnd[keep]


@nb.njit(cache=False)
def _members(sel2, sel1, q2, q1):
    out = np.zeros(len(q2), np.bool_)
    for i in range(len(q2)):
        lo, hi = 0, len(sel2)
        while lo < hi:
            mid = (lo + hi) // 2
            if sel2[mid] < q2[i] or (sel2[mid] == q2[i] and sel1[mid] < q1[i]):
                lo = mid + 1
            else:
                hi = mid
        out[i] = lo < len(sel2) and sel2[lo] == q2[i] and sel1[lo] == q1[i]
    return out


def _gather(task, sel2, sel1, out_dir):
    c, chunk = task
    t = inputs.read_chunk(chunk, None)
    m = _members(sel2, sel1, t.column('id2').to_numpy().astype(np.int64, copy=False),
                 t.column('id1').to_numpy().astype(np.int64, copy=False))
    if not m.any():
        return 0
    path = os.path.join(out_dir, f'rows-{c:05d}.arrow')
    with atomic_output(path) as tmp:
        with pa.OSFile(tmp, 'wb') as sink, pa.ipc.new_file(sink, t.schema) as w:
            w.write_table(t.filter(pa.array(m)))
    return int(m.sum())


def down_sample(fvs, percent, score_column='score', bucket_size=1000,
                max_pool_size=pool.DEFAULT_MAX_POOL_SIZE, seed=42, directory=None,
                workers=None):
    """Return the labeling pool as a memory-mapped Arrow table sorted by `(id2,
    id1)`."""
    workers = workers or workers_for(WORKER_GB)
    if directory is None:
        from .featurize import _remove_at_exit
        from .._core.storage import spill_dir
        import tempfile
        directory = tempfile.mkdtemp(prefix='iris-pool-', dir=spill_dir())
        _remove_at_exit(directory)
    os.makedirs(directory, exist_ok=True)
    with inputs.scratch() as tmp:
        src = inputs.stage(fvs, tmp, 'fvs')
        N = src.count_rows()
        B, n, B_keep = pool.plan(N, percent, bucket_size, max_pool_size)
        groups = max(1, math.ceil(N * B_keep / B / GROUP_ROWS))
        chunks = list(enumerate(inputs.chunks(src)))
        routed = os.path.join(tmp, 'routed')
        os.makedirs(routed)
        run_tasks(partial(_route, score_column=score_column, B=B, B_keep=B_keep,
                          groups=groups, out_dir=routed), chunks, workers)
        group_dirs = [os.path.join(routed, d) for d in sorted(os.listdir(routed))]
        picked = run_tasks(partial(_select, percent=percent, n=n, seed=seed), group_dirs,
                           workers)
        i1 = np.concatenate([p[0] for p in picked]) if picked else np.empty(0, np.int64)
        i2 = np.concatenate([p[1] for p in picked]) if picked else np.empty(0, np.int64)
        rnd = np.concatenate([p[2] for p in picked]) if picked else np.empty(0, np.int64)
        keep = pool.cap(i1, i2, rnd, max_pool_size)
        order = np.lexsort((i1[keep], i2[keep]))
        sel1, sel2 = i1[keep][order], i2[keep][order]
        rows = os.path.join(tmp, 'rows')
        os.makedirs(rows)
        run_tasks(partial(_gather, sel2=sel2, sel1=sel1, out_dir=rows), chunks, workers)
        parts = [pa.ipc.open_file(pa.memory_map(p)).read_all()
                 for p in sorted(glob.glob(os.path.join(rows, '*.arrow')))]
        table = pa.concat_tables(parts) if parts else src.schema.empty_table()
        order = np.lexsort((table.column('id1').to_numpy(),
                            table.column('id2').to_numpy())) if table.num_rows \
            else np.empty(0, np.int64)
        path = os.path.join(directory, 'pool.arrow')
        with atomic_output(path) as out:
            with pa.OSFile(out, 'wb') as sink, pa.ipc.new_file(sink, table.schema) as w:
                for lo in range(0, len(order), 1 << 20):
                    w.write_table(table.take(pa.array(order[lo:lo + (1 << 20)])))
        del table, parts
    return pa.ipc.open_file(pa.memory_map(path)).read_all()
