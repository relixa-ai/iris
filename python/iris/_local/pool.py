"""The labeling pool on the local engine, drawn from feature vectors on disk."""
import glob
import math
import os
from functools import partial

import numba as nb
import numpy as np
import pyarrow as pa

from .. import resources
from .._core import pool
from .._core.pairhash import pair_hash, seeded_hash
from . import disk, inputs
from .candidates import part_of
from .procpool import atomic_output, run_tasks, workers_for

GROUP_ROWS = 1 << 21

ORDER_BYTES = 128 * 2**20

ORDER_SAMPLE = 1 << 20

FLUSH_BYTES = 1 << 16


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


def _selection(out_dir, g, name):
    return os.path.join(out_dir, f'group-{g:05d}.{name}.npy')


def _save_selection(out_dir, g, id1, id2, rnd):
    order = np.lexsort((id1, id2))
    for name, values in (('id1', id1), ('id2', id2), ('rnd', rnd)):
        with atomic_output(_selection(out_dir, g, name)) as tmp, open(tmp, 'wb') as fh:
            np.save(fh, values[order])


def _load_selection(out_dir, g):
    return [np.load(_selection(out_dir, g, name)) for name in ('id1', 'id2', 'rnd')]


def _select(task, percent, n, seed, out_dir):
    g, group_dir = task
    files = sorted(glob.glob(os.path.join(group_dir, '*.npz')))
    parts = [np.load(f) for f in files]
    cols = {k: np.concatenate([p[k] for p in parts]) for k in ('id1', 'id2', 'score',
                                                              'bucket')}
    rnd = seeded_hash(seed, cols['id1'], cols['id2'])
    keep = pool.select_in_buckets(cols['bucket'], cols['id1'], cols['id2'],
                                  cols['score'], rnd, percent, n)
    _save_selection(out_dir, g, cols['id1'][keep], cols['id2'][keep], rnd[keep])
    return len(keep)


def _cap(out_dir, groups, max_pool_size):
    found = [_load_selection(out_dir, g) for g in groups]
    i1, i2, rnd = (np.concatenate([f[k] for f in found]) for k in range(3))
    keep = np.zeros(len(i1), bool)
    keep[pool.cap(i1, i2, rnd, max_pool_size)] = True
    at, kept = 0, []
    for g, (a, b, r) in zip(groups, found):
        m = keep[at:at + len(a)]
        _save_selection(out_dir, g, a[m], b[m], r[m])
        kept.append(int(m.sum()))
        at += len(a)
    return kept


def _assemble(out_dir, found, groups):
    starts = np.zeros(groups + 1, np.int64)
    for g, n in found:
        starts[g + 1] = n
    starts = np.cumsum(starts)
    arrays = [np.lib.format.open_memmap(os.path.join(out_dir, f'{name}.npy'), 'w+',
                                        np.int64, (int(starts[-1]),))
              for name in ('id1', 'id2')]
    for g, _ in found:
        for arr, name in zip(arrays, ('id1', 'id2')):
            arr[starts[g]:starts[g + 1]] = np.load(_selection(out_dir, g, name))
    for arr in arrays:
        arr.flush()
    return starts


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


def _gather(task, selected, starts, B, B_keep, groups, out_dir):
    c, chunk = task
    sel1 = np.load(os.path.join(selected, 'id1.npy'), mmap_mode='r')
    sel2 = np.load(os.path.join(selected, 'id2.npy'), mmap_mode='r')
    t = inputs.read_chunk(chunk, None)
    id1 = t.column('id1').to_numpy().astype(np.int64, copy=False)
    id2 = t.column('id2').to_numpy().astype(np.int64, copy=False)
    bucket = np.mod(pair_hash(id1, id2), B)
    live = np.flatnonzero(bucket < B_keep)
    group = bucket[live] * groups // B_keep
    by_group = np.argsort(group, kind='stable')
    live, group = live[by_group], group[by_group]
    cuts = np.r_[0, np.flatnonzero(np.diff(group)) + 1, len(group)]
    m = np.zeros(len(id1), np.bool_)
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        if hi > lo:
            g = int(group[lo])
            at = live[lo:hi]
            a, z = starts[g], starts[g + 1]
            m[at] = _members(sel2[a:z], sel1[a:z], id2[at], id1[at])
    if not m.any():
        return None
    rows = t.filter(pa.array(m)).combine_chunks()
    path = os.path.join(out_dir, f'rows-{c:05d}.arrow')
    with atomic_output(path) as tmp:
        with pa.OSFile(tmp, 'wb') as sink, pa.ipc.new_file(sink, t.schema) as w:
            w.write_table(rows)
    return path, _summary(rows.to_batches())


def _keys(batch):
    return tuple(batch.column(c).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
                 for c in ('id1', 'id2'))


def _ordered(id1, id2):
    later = id2[1:] > id2[:-1]
    return bool(np.all(later | ((id2[1:] == id2[:-1]) & (id1[1:] >= id1[:-1]))))


def _summary(batches):
    rows, first, last, ordered, count = 0, None, None, True, 0
    for batch in batches:
        count += 1
        if not batch.num_rows:
            continue
        id1, id2 = _keys(batch)
        if not _ordered(id1, id2) or (last is not None
                                      and (int(id2[0]), int(id1[0])) < last):
            ordered = False
        rows += batch.num_rows
        first = first or (int(id2[0]), int(id1[0]))
        last = (int(id2[-1]), int(id1[-1]))
    return rows, first, last, ordered, count


def _in_order(summaries):
    last = None
    for rows, first, end, ordered, _ in summaries:
        if not rows:
            continue
        if not ordered or (last is not None and first < last):
            return False
        last = end
    return True


def in_pair_order(table, step):
    """Return whether the rows of a table are in `(id2, id1)` order, read `step` rows at
    a time."""
    return _summary(table.slice(lo, step) for lo in range(0, table.num_rows, step))[3]


def _batches(path, at=None):
    with pa.memory_map(path) as source:
        reader = pa.ipc.open_file(source)
        for i in range(reader.num_record_batches) if at is None else at:
            yield reader.get_batch(i)


def _concat(paths, path, schema, remove=False):
    with atomic_output(path) as out:
        with pa.OSFile(out, 'wb') as sink, pa.ipc.new_file(sink, schema) as w:
            for p in paths:
                for batch in _batches(p):
                    w.write_batch(batch)
                if remove:
                    os.remove(p)


def _sample(task, every):
    path, start = task
    keys, sizes = [], []
    for batch in _batches(path):
        keep = np.flatnonzero((start + np.arange(batch.num_rows)) % every == 0)
        start += batch.num_rows
        if len(keep):
            id1, id2 = _keys(batch)
            keys.append(np.stack([id2[keep], id1[keep]], axis=1))
            sizes.append(np.full(len(keep), every * batch.nbytes / batch.num_rows))
    if not keys:
        return np.empty((0, 2), np.int64), np.empty(0)
    return np.concatenate(keys), np.concatenate(sizes)


def _bounds(parts, workers):
    starts = np.cumsum([0] + [s[0] for _, s in parts])
    every = max(1, int(starts[-1]) // ORDER_SAMPLE)
    found = run_tasks(partial(_sample, every=every),
                      [(p, int(start)) for (p, _), start in zip(parts, starts)], workers)
    keys = np.concatenate([k for k, _ in found])
    sizes = np.concatenate([s for _, s in found])
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    keys, cum = keys[order], np.cumsum(sizes[order])
    total = float(cum[-1]) if len(cum) else 0.0
    per_range = max(1.0, min(resources.piece_bytes(ORDER_BYTES), total / (2 * workers)))
    cuts = np.searchsorted(cum, np.arange(per_range, total, per_range), side='right')
    cuts = np.unique(cuts[(cuts > 0) & (cuts < len(keys))])
    return np.unique(keys[cuts], axis=0) if len(cuts) else np.empty((0, 2), np.int64)


def _route_rows(task, schema, bounds, out_dir):
    from .store import _raise_open_files

    k, units = task
    _raise_open_files(len(bounds) + 1 + 256)
    writers, held, sizes = {}, {}, {}

    def flush(r):
        table = pa.Table.from_batches(held.pop(r), schema=schema)
        sizes[r] = 0
        if r not in writers:
            part = os.path.join(out_dir, f'range-{r:05d}')
            os.makedirs(part, exist_ok=True)
            path = os.path.join(part, f'part-{k:05d}.arrow')
            sink = pa.OSFile(f'{path}.tmp', 'wb')
            writers[r] = (pa.ipc.new_file(sink, schema), sink, path)
        writers[r][0].write_table(table)

    try:
        for path, at in units:
            for batch in _batches(path, at):
                if not batch.num_rows:
                    continue
                id1, id2 = _keys(batch)
                ranges = part_of(id2, id1, bounds)
                for r in np.unique(ranges).tolist():
                    rows = batch.filter(pa.array(ranges == r))
                    held.setdefault(r, []).append(rows)
                    sizes[r] = sizes.get(r, 0) + rows.nbytes
                    if sizes[r] >= FLUSH_BYTES:
                        flush(r)
        for r in list(held):
            flush(r)
    finally:
        for writer, sink, _ in writers.values():
            writer.close()
            sink.close()
    for _, _, path in writers.values():
        os.replace(f'{path}.tmp', path)
    return k


def _sort_range(task, out_dir):
    r, directory = task
    files = sorted(glob.glob(os.path.join(directory, '*.arrow')))
    table = pa.concat_tables([pa.ipc.open_file(pa.memory_map(f)).read_all() for f in files])
    id1, id2 = _keys(table)
    path = os.path.join(out_dir, f'range-{r:05d}.arrow')
    with atomic_output(path) as tmp:
        with pa.OSFile(tmp, 'wb') as sink, pa.ipc.new_file(sink, table.schema) as w:
            w.write_table(inputs.take(table, np.lexsort((id1, id2))))
    del table
    for f in files:
        os.remove(f)
    return path


def write_in_order(parts, path, schema, workers=None):
    """Write the rows of the Arrow files `parts`, each given with its summary, to the
    Arrow file `path` in `(id2, id1)` order, holding a piece of them at a time."""
    if _in_order([s for _, s in parts]):
        _concat([p for p, _ in parts], path, schema)
        return
    workers = workers or workers_for()
    units = [(p, i) for p, s in parts for i in range(s[4])]
    groups = max(1, min(len(units), 2 * workers))
    tasks = []
    for k in range(groups):
        runs = {}
        for p, i in units[k * len(units) // groups:(k + 1) * len(units) // groups]:
            runs.setdefault(p, []).append(i)
        tasks.append((k, list(runs.items())))
    with inputs.scratch() as tmp:
        inputs.require_disk(sum(os.path.getsize(p) for p, _ in parts), tmp,
                            'the pool put in pair order', kept=0)
        bounds = _bounds(parts, workers)
        routed = os.path.join(tmp, 'routed')
        ranked = os.path.join(tmp, 'sorted')
        os.makedirs(routed)
        os.makedirs(ranked)
        run_tasks(partial(_route_rows, schema=schema, bounds=bounds, out_dir=routed), tasks,
                  workers, task_bytes=inputs.chunk_bytes())
        ranges = sorted((int(d.split('-')[1]), os.path.join(routed, d))
                        for d in os.listdir(routed))
        done = run_tasks(partial(_sort_range, out_dir=ranked), ranges, workers,
                         task_bytes=resources.piece_bytes(ORDER_BYTES))
        _concat(done, path, schema, remove=True)


def write_file_in_order(path, out, workers=None):
    """Write the rows of the Arrow file `path` to the Arrow file `out` in `(id2, id1)`
    order."""
    with pa.memory_map(path) as source:
        schema = pa.ipc.open_file(source).schema
    write_in_order([(path, _summary(_batches(path)))], out, schema, workers)


@disk.disk_step('down_sample')
def down_sample(fvs, percent, score_column='score', bucket_size=1000,
                max_pool_size=pool.DEFAULT_MAX_POOL_SIZE, seed=42, directory=None,
                workers=None):
    """Return the labeling pool as a memory-mapped Arrow table sorted by `(id2,
    id1)`."""
    workers = workers or workers_for()
    if directory is None:
        from .featurize import _remove_at_exit
        from .._core.storage import scratch_prefix, spill_dir
        import tempfile
        directory = tempfile.mkdtemp(prefix=scratch_prefix('pool'), dir=spill_dir())
        _remove_at_exit(directory)
        disk.output(directory)
    os.makedirs(directory, exist_ok=True)
    with inputs.scratch() as tmp:
        src = inputs.stage(fvs, tmp, 'fvs', workers=workers)
        N = src.count_rows()
        B, n, B_keep = pool.plan(N, percent, bucket_size, max_pool_size)
        live, pooled = N * B_keep // B, pool.pool_size(N, percent, max_pool_size)
        inputs.require_disk(32 * live + pooled * (40 + 2 * inputs.column_bytes(src) // max(N, 1)),
                            tmp, 'the routed pairs and the pool')
        groups = max(1, math.ceil(N * B_keep / B / GROUP_ROWS))
        chunks = list(enumerate(inputs.chunks(src, workers=workers)))
        routed = os.path.join(tmp, 'routed')
        os.makedirs(routed)
        run_tasks(partial(_route, score_column=score_column, B=B, B_keep=B_keep,
                          groups=groups, out_dir=routed), chunks, workers)
        group_dirs = [(int(d.split('-')[1]), os.path.join(routed, d))
                      for d in sorted(os.listdir(routed))]
        selected = os.path.join(tmp, 'selected')
        os.makedirs(selected)
        picked = run_tasks(partial(_select, percent=percent, n=n, seed=seed,
                                   out_dir=selected), group_dirs, workers)
        if max_pool_size is not None and sum(picked) > max_pool_size:
            picked = _cap(selected, [g for g, _ in group_dirs], max_pool_size)
        rows = os.path.join(tmp, 'rows')
        os.makedirs(rows)
        parts = []
        if sum(picked):
            starts = _assemble(selected, [(g, k) for (g, _), k in zip(group_dirs, picked)],
                               groups)
            parts = [p for p in run_tasks(partial(_gather, selected=selected, starts=starts,
                                                  B=B, B_keep=B_keep, groups=groups,
                                                  out_dir=rows), chunks, workers)
                     if p is not None]
        schema = pa.ipc.open_file(pa.memory_map(parts[0][0])).schema if parts \
            else src.schema
        path = os.path.join(directory, 'pool.arrow')
        write_in_order(parts, path, schema, workers)
    return pa.ipc.open_file(pa.memory_map(path)).read_all()
