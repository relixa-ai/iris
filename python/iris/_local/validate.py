"""The input checks on local inputs, streamed, with large inputs checked by worker
processes."""
import glob
import math
import os
import tempfile
from functools import partial
from itertools import chain

import numpy as np
import pandas as pd
import pyarrow as pa

from .. import resources
from ..io import (_candidate_shape, _dangling_error, _listlike, _misaligned_error,
                  _warn_duplicate_ids, _warn_self_pairs)
from . import inputs
from .candidates import part_of
from .inputs import batches, columns
from .procpool import atomic_output, run_tasks, workers_for

CHUNK_ROWS = inputs.CHUNK_ROWS


SAMPLE = 5

RANGE_BYTES = 128 * 2**20

SAMPLE_IDS = 1 << 20

FLUSH_BYTES = 1 << 16


def table(df, name):
    """Check the `_id` column of one table and return its row count."""
    if '_id' not in columns(df):
        raise ValueError(f"{name} must have an '_id' column; got {columns(df)}")
    large = _large(df, '_id')
    n = n_null = n_negative = 0
    dtype = None
    parts, negatives = [], []
    for b in batches(df, ['_id']):
        ids = b['_id']
        n += len(ids)
        n_null += int(ids.isna().sum())
        if ids.dtype.kind not in 'iu' and dtype is None and len(ids):
            dtype = ids.dtype
        if not n_null and dtype is None:
            if not large:
                parts.append(ids.to_numpy())
                continue
            values = ids.to_numpy().astype(np.int64, copy=False)
            below = values < 0
            n_negative += int(below.sum())
            negatives.extend(values[below][:SAMPLE - len(negatives)].tolist())
    if n == 0:
        raise ValueError(f'{name} is empty')
    if n_null:
        raise ValueError(
            f"{name} '_id' has {n_null} null value(s); every record needs an id")
    if dtype is not None:
        raise ValueError(
            f"{name} '_id' must be an integer column (ids are int64 "
            f"join keys), got {dtype}; a table keyed by strings is given integer ids as 'Tables keyed by strings' in the technical guide describes")
    if large:
        if n_negative:
            raise _negative_error(name, n_negative, negatives)
        repeats, repeated = _merged_runs(_by_ranges(df, '_id'))
        if repeats:
            raise _repeat_error(name, repeats, repeated)
        return n
    ids = np.concatenate(parts).astype(np.int64, copy=False)
    negative = ids < 0
    if negative.any():
        raise _negative_error(name, int(negative.sum()), ids[negative][:5].tolist())
    del negative
    repeats = _repeats(ids)
    if repeats:
        raise _repeat_error(name, repeats, _repeated(ids))
    return n


def _negative_error(name, n, sample):
    return ValueError(f"{name} '_id' must be non-negative; {n} negative id(s), e.g. "
                      f"ids {sample}")


def _repeat_error(name, n, sample):
    return ValueError(f"{name} '_id' must be unique; {n} duplicate row(s), e.g. ids "
                      f"{sample}")


def _large(df, col):
    if col not in columns(df):
        return False
    if isinstance(df, pd.DataFrame):
        return len(df) > CHUNK_ROWS and df[col].dtype.kind in 'iu'
    if not pa.types.is_integer(inputs.schema(df).field(col).type):
        return False
    rows, nbytes = _size(df, [col])
    return rows > CHUNK_ROWS or nbytes > inputs.chunk_bytes()


def _id_slices(chunk, col, start):
    for part in inputs.read_slices(chunk, [col], rows=inputs.CHUNK_ROWS):
        column = part.column(col)
        pos = start + np.arange(part.num_rows, dtype=np.int64)
        start += part.num_rows
        if column.null_count:
            valid = column.is_valid().to_numpy(zero_copy_only=False)
            column, pos = column.filter(pa.array(valid)), pos[valid]
        yield column.to_numpy().astype(np.int64, copy=False), pos


def _id_sample(task, col, every):
    chunk, start = task
    found = [np.stack([ids, pos], axis=1)[pos % every == 0]
             for ids, pos in _id_slices(chunk, col, start)]
    return np.concatenate(found) if found else np.empty((0, 2), np.int64)


def _id_route(task, col, bounds, out_dir):
    from .store import _raise_open_files

    k, runs = task
    _raise_open_files(len(bounds) + 1 + 256)
    files, held, sizes = {}, {}, {}

    def flush(r):
        if r not in files:
            part = os.path.join(out_dir, f'range-{r:05d}')
            os.makedirs(part, exist_ok=True)
            path = os.path.join(part, f'task-{k:05d}.bin')
            files[r] = (open(f'{path}.tmp', 'wb'), path)
        for block in held.pop(r):
            files[r][0].write(block.tobytes())
        sizes[r] = 0

    try:
        for chunk, start in runs:
            for ids, pos in _id_slices(chunk, col, start):
                part = part_of(ids, pos, bounds)
                order = np.argsort(part, kind='stable')
                part = part[order]
                cuts = np.r_[0, np.flatnonzero(np.diff(part)) + 1, len(part)]
                for lo, hi in zip(cuts[:-1], cuts[1:]):
                    if hi > lo:
                        r = int(part[lo])
                        at = order[lo:hi]
                        block = np.stack([ids[at], pos[at]], axis=1)
                        held.setdefault(r, []).append(block)
                        sizes[r] = sizes.get(r, 0) + block.nbytes
                        if sizes[r] >= FLUSH_BYTES:
                            flush(r)
        for r in list(held):
            flush(r)
    finally:
        for fh, _ in files.values():
            fh.close()
    for _, path in files.values():
        os.replace(f'{path}.tmp', path)
    return k


def _id_range(task, keep):
    r, directory = task
    files = sorted(glob.glob(os.path.join(directory, 'task-*.bin')))
    pairs = np.concatenate([np.fromfile(f, np.int64).reshape(-1, 2) for f in files])
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    ids, pos = pairs[:, 0], pairs[:, 1]
    path = None
    if keep:
        path = os.path.join(directory, 'sorted.bin')
        with atomic_output(path) as tmp:
            np.ascontiguousarray(ids).tofile(tmp)
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    counts = np.diff(np.r_[starts, len(ids)])
    second = np.where(counts > 1, pos[np.minimum(starts + 1, len(ids) - 1)], -1)
    edge = lambda j: (int(ids[starts[j]]), int(counts[j]), int(pos[starts[j]]),
                      int(second[j]))
    inner = np.arange(1, len(starts) - 1)
    inner = inner[counts[inner] > 1]
    inner = inner[np.argsort(second[inner], kind='stable')[:SAMPLE]]
    return {'rows': len(ids), 'distinct': len(starts), 'first': edge(0),
            'last': edge(len(starts) - 1), 'path': path,
            'inner': [(int(second[j]), int(ids[starts[j]])) for j in inner]}


def _by_ranges(df, col, out=None):
    workers = workers_for()
    with inputs.scratch() as tmp:
        src = inputs.stage(df, tmp, 'ids', [col], workers)
        chunks = inputs.chunks(src, workers=workers)
        starts = np.cumsum([0] + [c.rows for c in chunks]).tolist()
        n = starts[-1]
        sample = run_tasks(partial(_id_sample, col=col, every=max(1, n // SAMPLE_IDS)),
                           list(zip(chunks, starts)), workers,
                           task_bytes=inputs.chunk_bytes())
        sample = np.concatenate(sample) if sample else np.empty((0, 2), np.int64)
        sample = sample[np.lexsort((sample[:, 1], sample[:, 0]))]
        parts = max(2 * workers, math.ceil(16 * n / resources.piece_bytes(RANGE_BYTES)))
        bounds = np.unique(sample[(np.arange(1, parts) * len(sample)) // parts], axis=0) \
            if parts > 1 and len(sample) else np.empty((0, 2), np.int64)
        routed = os.path.join(tmp, 'routed')
        os.makedirs(routed)
        groups = max(1, min(len(chunks), 2 * workers))
        run_tasks(partial(_id_route, col=col, bounds=bounds, out_dir=routed),
                  [(k, list(zip(chunks[k::groups], starts[k:-1:groups])))
                   for k in range(groups)], workers, task_bytes=inputs.chunk_bytes())
        ranges = sorted((int(d.split('-')[1]), os.path.join(routed, d))
                        for d in os.listdir(routed))
        found = run_tasks(partial(_id_range, keep=out is not None), ranges, workers,
                          task_bytes=resources.piece_bytes(RANGE_BYTES))
        if out is not None:
            _assemble(found, out)
    return found


def _assemble(found, out):
    total = sum(f['rows'] for f in found)
    if not total:
        return
    ids = np.lib.format.open_memmap(out, 'w+', np.int64, (total,))
    step = inputs.SLICE_BYTES // 8
    at = 0
    for f in found:
        for lo in range(0, f['rows'], step):
            part = np.fromfile(f['path'], np.int64, count=step, offset=8 * lo)
            ids[at:at + len(part)] = part
            at += len(part)
    ids.flush()
    del ids


def _merged_runs(found):
    rows = distinct = 0
    pending, repeated = None, []

    def done(run):
        if run[1] > 1:
            repeated.append((run[3], run[0]))

    for f in found:
        rows += f['rows']
        distinct += f['distinct']
        first = f['first']
        if pending is not None and pending[0] == first[0]:
            distinct -= 1
            first = (first[0], pending[1] + first[1], pending[2],
                     pending[3] if pending[1] > 1 else first[2])
        elif pending is not None:
            done(pending)
        if f['distinct'] == 1:
            pending = first
        else:
            done(first)
            pending = f['last']
        repeated += f['inner']
    if pending is not None:
        done(pending)
    return rows - distinct, [i for _, i in sorted(repeated)[:SAMPLE]]


def _repeats(ids):
    ordered = np.sort(ids)
    return int(np.count_nonzero(ordered[1:] == ordered[:-1]))


def _repeated(ids, k=SAMPLE):
    order = np.argsort(ids, kind='stable')
    ordered = ids[order]
    later = np.r_[False, ordered[1:] == ordered[:-1]]
    return pd.unique(ids[np.sort(order[later])])[:k].tolist()


def sorted_ids(df, directory=None):
    """Return a table's non-null ids as a sorted int64 array, sorted by workers into a
    memory-mapped file in `directory` when the table is large."""
    if directory is not None and _large(df, '_id'):
        path = os.path.join(tempfile.mkdtemp(dir=directory), 'ids.npy')
        found = _by_ranges(df, '_id', path)
        if sum(f['rows'] for f in found):
            return np.load(path, mmap_mode='r')
        return np.empty(0, np.int64)
    parts = [b['_id'].dropna().to_numpy(np.int64) for b in batches(df, ['_id'])]
    return np.sort(np.concatenate(parts)) if parts else np.empty(0, np.int64)


def _missing(sample, known, ids):
    if len(sample) >= SAMPLE:
        return
    ids = np.asarray(ids)
    if ids.dtype.kind in 'iu':
        null = np.zeros(len(ids), bool)
        vals = ids.astype(np.int64, copy=False)
    else:
        null = np.asarray(pd.isna(ids), bool)
        try:
            vals = np.where(null, 0, ids).astype(np.int64)
        except (TypeError, ValueError):
            return
    if not len(vals):
        return
    pos = np.searchsorted(known, vals)
    hit = np.zeros(len(vals), bool) if not len(known) else \
        known[np.minimum(pos, len(known) - 1)] == vals
    for i in np.flatnonzero(null | ~hit).tolist():
        sample.setdefault(None if null[i] else int(vals[i]), None)
        if len(sample) >= SAMPLE:
            return


def _extend(sample, more):
    for x in more:
        if len(sample) >= SAMPLE:
            return
        sample.setdefault(x, None)


def _flatten(lists):
    cells = [np.asarray(x) for x in lists if len(x)]
    if not cells:
        return np.empty(0, np.int64)
    flat = np.concatenate(cells)
    if flat.dtype.kind in 'iu':
        return flat
    return np.array(list(chain.from_iterable(lists)), dtype=object)


def _size(source, cols):
    if inputs._parquet_files(source):
        groups = [(n, b) for _, _, n, b in inputs._groups(source, cols)]
        return sum(n for n, _ in groups), sum(b for _, b in groups)
    if isinstance(source, pa.Table):
        table = source if cols is None else source.select(list(cols))
        return table.num_rows, table.nbytes
    if isinstance(source, pd.DataFrame):
        frame = source if cols is None else source[list(cols)]
        items = 0
        for c in frame.columns:
            if frame[c].dtype == object:
                items += int(frame[c].map(lambda x: len(x) if _listlike(x) else 1).sum())
            else:
                items += len(frame)
        return len(frame), 8 * items
    return CHUNK_ROWS + 1, 0


def _tally(source, cols, fn, merge, known_a, known_b):
    rows, nbytes = _size(source, cols)
    if rows <= CHUNK_ROWS and nbytes <= inputs.chunk_bytes():
        return fn(batches(source, cols), known_a, known_b)
    workers = workers_for()
    with inputs.scratch() as tmp:
        try:
            source = inputs.stage(source, tmp, 'checked', cols, workers)
        except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError):
            return fn(batches(source, cols), known_a, known_b)
        chunks = inputs.chunks(source, CHUNK_ROWS, workers)
        schema = source.schema if cols is None else \
            pa.schema([source.schema.field(c) for c in cols])
        paths = []
        for k, known in enumerate((known_a, known_b)):
            if known is None:
                paths.append(None)
            elif k == 1 and known is known_a:
                paths.append(paths[0])
            elif isinstance(known, np.memmap):
                paths.append(known.filename)
            else:
                paths.append(os.path.join(tmp, f'known-{k}.npy'))
                np.save(paths[-1], known)
        tallies = run_tasks(partial(_chunk_tally, cols=cols, schema=schema, fn=fn,
                                    known=tuple(paths)),
                            chunks, workers, task_bytes=inputs.chunk_bytes())
    return merge(tallies)


def _chunk_tally(chunk, cols, schema, fn, known):
    frames = (part.to_pandas() for part in
              inputs.read_slices(chunk._replace(schema=schema), schema.names,
                                 rows=inputs.BATCH_ROWS))
    a, b = (None if p is None else np.load(p, mmap_mode='r') for p in known)
    return fn(frames, a, b)


def candidates(cands, table_a, table_b, dedup, membership, what):
    """Check a candidate set against the tables and return its number of pairs."""
    shape = _candidate_shape(columns(cands), what)
    with inputs.scratch() as tmp:
        known_a = sorted_ids(table_a, tmp) if membership else None
        known_b = (known_a if table_b is table_a else sorted_ids(table_b, tmp)) \
            if membership else None
        if shape == 'flat':
            return _flat(cands, known_a, known_b, dedup, membership, what)
        return _grouped(cands, known_a, known_b, dedup, membership, what)


def _flat_tally(frames, known_a, known_b, dedup, membership):
    t = {'n': 0, 'n1': 0, 'n2': 0, 'selfp': 0, 'dtypes': {}, 'miss_a': {}, 'miss_b': {}}
    for b in frames:
        t['n'] += len(b)
        t['n1'] += int(b['id1'].isna().sum())
        t['n2'] += int(b['id2'].isna().sum())
        for c in ('id1', 'id2'):
            if b[c].dtype.kind not in 'iu' and len(b):
                t['dtypes'].setdefault(c, b[c].dtype)
        if dedup:
            t['selfp'] += int((b['id1'] == b['id2']).sum())
        if membership:
            _missing(t['miss_a'], known_a, b['id1'].to_numpy())
            _missing(t['miss_b'], known_b, b['id2'].to_numpy())
    return t


def _merge_flat(tallies):
    out = {'n': 0, 'n1': 0, 'n2': 0, 'selfp': 0, 'dtypes': {}, 'miss_a': {}, 'miss_b': {}}
    for t in tallies:
        for k in ('n', 'n1', 'n2', 'selfp'):
            out[k] += t[k]
        for c, dtype in t['dtypes'].items():
            out['dtypes'].setdefault(c, dtype)
        _extend(out['miss_a'], t['miss_a'])
        _extend(out['miss_b'], t['miss_b'])
    return out


def _flat(cands, known_a, known_b, dedup, membership, what):
    t = _tally(cands, ['id1', 'id2'],
               partial(_flat_tally, dedup=dedup, membership=membership), _merge_flat,
               known_a, known_b)
    n = t['n']
    if n == 0:
        raise ValueError(f'{what} set is empty: there is nothing to match')
    if t['n1'] or t['n2']:
        raise ValueError(f"{what} contain null ids ({t['n1']} in id1, {t['n2']} in id2)")
    for c, t_name in (('id1', 'table_a'), ('id2', 'table_b')):
        if c in t['dtypes']:
            raise ValueError(
                f"{what} '{c}' must be an integer column matching "
                f"{t_name}._id, got {t['dtypes'][c]}")
    if dedup and t['selfp']:
        _warn_self_pairs(what, t['selfp'], grouped=False)
    if membership:
        if t['miss_a']:
            raise _dangling_error(what, 'id1', 'table_a', list(t['miss_a']))
        if t['miss_b']:
            raise _dangling_error(what, 'id2', 'table_b', list(t['miss_b']))
    return n


def _grouped_tally(frames, known_a, known_b, dedup, membership, aligned_cols):
    t = {'n': 0, 'pairs': 0, 'n_null': 0, 'dups': 0, 'selfp': 0, 'not_lists': False,
         'dtype': None, 'miss_a': {}, 'miss_b': {},
         'aligned': {c: [False, 0, []] for c in aligned_cols}}
    for b in frames:
        lists = b['id1_list']
        is_list = lists.map(_listlike)
        if not is_list.all():
            t['not_lists'] = True
            continue
        t['n'] += len(b)
        lengths = lists.map(len)
        t['pairs'] += int(lengths.sum())
        t['n_null'] += int(b['id2'].isna().sum())
        if b['id2'].dtype.kind not in 'iu' and len(b) and t['dtype'] is None:
            t['dtype'] = b['id2'].dtype
        for c, state in t['aligned'].items():
            vals = b[c]
            if vals.map(_listlike).any():
                state[0] = True
            bad = vals.map(lambda x: len(x) if _listlike(x) else -1) != lengths
            state[1] += int(bad.sum())
            if len(state[2]) < SAMPLE:
                state[2].extend(b.loc[bad, 'id2'].head(SAMPLE - len(state[2])).tolist())
        flat = _flatten(lists)
        if flat.dtype.kind in 'iu' and b['id2'].dtype.kind in 'iu':
            parents = np.repeat(np.arange(len(b)), lengths.to_numpy())
            order = np.lexsort((flat, parents))
            same = ((flat[order][1:] == flat[order][:-1])
                    & (parents[order][1:] == parents[order][:-1]))
            t['dups'] += len(np.unique(parents[order][1:][same]))
            if dedup:
                own = flat == b['id2'].to_numpy()[parents]
                t['selfp'] += len(np.unique(parents[own]))
        else:
            t['dups'] += int(lists.map(lambda x: len(set(x)) != len(x)).sum())
            if dedup:
                t['selfp'] += int(sum(1 for id2, lst in zip(b['id2'], lists)
                                      if id2 in set(lst)))
        if membership:
            _missing(t['miss_a'], known_a, flat)
            _missing(t['miss_b'], known_b, b['id2'].to_numpy())
    return t


def _merge_grouped(tallies):
    out = None
    for t in tallies:
        if out is None:
            out = t
            continue
        for k in ('n', 'pairs', 'n_null', 'dups', 'selfp'):
            out[k] += t[k]
        out['not_lists'] = out['not_lists'] or t['not_lists']
        if out['dtype'] is None:
            out['dtype'] = t['dtype']
        for c, (seen_list, n_bad, sample) in t['aligned'].items():
            state = out['aligned'][c]
            state[0] = state[0] or seen_list
            state[1] += n_bad
            state[2].extend(sample[:SAMPLE - len(state[2])])
        _extend(out['miss_a'], t['miss_a'])
        _extend(out['miss_b'], t['miss_b'])
    return out


def _repeat_count(df, col):
    if _large(df, col):
        return _merged_runs(_by_ranges(df, col))[0]
    ids = [b[col].to_numpy() for b in batches(df, [col])]
    return _repeats(np.concatenate(ids).astype(np.int64, copy=False))


def _grouped(cands, known_a, known_b, dedup, membership, what):
    aligned_cols = [c for c in columns(cands) if c not in ('id2', 'id1_list')]
    t = _tally(cands, None,
               partial(_grouped_tally, dedup=dedup, membership=membership,
                       aligned_cols=aligned_cols),
               _merge_grouped, known_a, known_b)
    n, pairs = t['n'], t['pairs']
    if t['not_lists']:
        raise ValueError(f"{what} 'id1_list' must be an array of ids")
    if n == 0 or pairs == 0:
        raise ValueError(f'{what} set is empty: there is nothing to match')
    if t['n_null']:
        raise ValueError(f"{what} contain {t['n_null']} null id2 row(s)")
    if t['dtype'] is not None:
        raise ValueError(
            f"{what} 'id2' must be an integer column matching table_b._id, got "
            f"{t['dtype']}")
    repeats = _repeat_count(cands, 'id2')
    if repeats:
        raise ValueError(
            f"{what} 'id2' must be unique (one row per B record); "
            f"{repeats} duplicate row(s)")
    for c, (seen_list, n_bad, sample) in t['aligned'].items():
        if seen_list and n_bad:
            raise _misaligned_error(what, c, n_bad, sample)
    if t['dups']:
        _warn_duplicate_ids(what, t['dups'])
    if dedup and t['selfp']:
        _warn_self_pairs(what, t['selfp'], grouped=True)
    if membership:
        if t['miss_a']:
            raise _dangling_error(what, 'id1_list', 'table_a', list(t['miss_a']))
        if t['miss_b']:
            raise _dangling_error(what, 'id2', 'table_b', list(t['miss_b']))
    return pairs


def pairs(prs, table_a, table_b, what):
    """Check a set of flat pairs against the tables and return its number of pairs."""
    with inputs.scratch() as tmp:
        known_a = sorted_ids(table_a, tmp)
        known_b = known_a if table_b is table_a else sorted_ids(table_b, tmp)
        t = _tally(prs, ['id1', 'id2'],
                   partial(_flat_tally, dedup=False, membership=True), _merge_flat,
                   known_a, known_b)
    if t['n1'] or t['n2']:
        raise ValueError(f"{what} contain null ids ({t['n1']} in id1, {t['n2']} in id2)")
    if t['miss_a']:
        raise _dangling_error(what, 'id1', 'table_a', list(t['miss_a']))
    if t['miss_b']:
        raise _dangling_error(what, 'id2', 'table_b', list(t['miss_b']))
    return t['n']
