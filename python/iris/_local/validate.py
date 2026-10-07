"""The input checks on local inputs, streamed, with large inputs checked by worker
processes."""
import os
from functools import partial
from itertools import chain

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..io import (_candidate_shape, _dangling_error, _listlike, _misaligned_error,
                  _warn_duplicate_ids, _warn_self_pairs)
from . import inputs
from .inputs import batches, columns
from .procpool import run_tasks, workers_for

CHUNK_ROWS = inputs.CHUNK_ROWS

WORKER_GB = 1.5

SAMPLE = 5


def table(df, name):
    """Check the `_id` column of one table and return its row count."""
    if '_id' not in columns(df):
        raise ValueError(f"{name} must have an '_id' column; got {columns(df)}")
    n = n_null = 0
    dtype = None
    parts = []
    for b in batches(df, ['_id']):
        ids = b['_id']
        n += len(ids)
        n_null += int(ids.isna().sum())
        if ids.dtype.kind not in 'iu' and dtype is None and len(ids):
            dtype = ids.dtype
        if not n_null and dtype is None:
            parts.append(ids.to_numpy())
    if n == 0:
        raise ValueError(f'{name} is empty')
    if n_null:
        raise ValueError(
            f"{name} '_id' has {n_null} null value(s); every record needs an id")
    if dtype is not None:
        raise ValueError(
            f"{name} '_id' must be an integer column (ids are int64 "
            f"join keys), got {dtype}")
    ids = pd.Series(np.concatenate(parts))
    n_distinct = int(ids.nunique())
    if n_distinct < n:
        dups = ids[ids.duplicated()].drop_duplicates().head(5).tolist()
        raise ValueError(
            f"{name} '_id' must be unique; {n - n_distinct} duplicate row(s), "
            f"e.g. ids {dups}")
    return n


def sorted_ids(df):
    """Return a table's non-null ids as a sorted int64 array."""
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


def _tally(source, cols, fn, merge, known_a, known_b):
    chunks = (inputs.chunks(source, CHUNK_ROWS) if inputs._parquet_files(source)
              else [])
    if len(chunks) <= 1 or sum(c.rows for c in chunks) <= CHUNK_ROWS:
        return fn(batches(source, cols), known_a, known_b)
    schema = source.schema if cols is None else \
        pa.schema([source.schema.field(c) for c in cols])
    with inputs.scratch() as tmp:
        paths = []
        for k, known in enumerate((known_a, known_b)):
            if known is None:
                paths.append(None)
            elif k == 1 and known is known_a:
                paths.append(paths[0])
            else:
                paths.append(os.path.join(tmp, f'known-{k}.npy'))
                np.save(paths[-1], known)
        tallies = run_tasks(partial(_chunk_tally, cols=cols, schema=schema, fn=fn,
                                    known=tuple(paths)),
                            chunks, workers_for(WORKER_GB))
    return merge(tallies)


def _chunk_tally(chunk, cols, schema, fn, known):
    with chunk.filesystem.open_input_file(chunk.path) as fh:
        f = pq.ParquetFile(fh)
        present = [c for c in schema.names if c in f.schema_arrow.names]
        table = f.read_row_groups(list(chunk.row_groups), columns=present)
    if table.schema != schema:
        table = pa.table([table.column(fld.name).cast(fld.type) if fld.name in present
                          else pa.nulls(table.num_rows, fld.type) for fld in schema],
                         schema=schema)
    frames = (b.to_pandas() for b in table.to_batches(max_chunksize=inputs.BATCH_ROWS))
    a, b = (None if p is None else np.load(p, mmap_mode='r') for p in known)
    return fn(frames, a, b)


def candidates(cands, table_a, table_b, dedup, membership, what):
    """Check a candidate set against the tables and return its number of pairs."""
    shape = _candidate_shape(columns(cands), what)
    known_a = sorted_ids(table_a) if membership else None
    known_b = (known_a if table_b is table_a else sorted_ids(table_b)) if membership \
        else None
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
         'dtype': None, 'id2': [], 'miss_a': {}, 'miss_b': {},
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
        t['id2'].append(b['id2'].to_numpy())
        for c, state in t['aligned'].items():
            vals = b[c]
            if vals.map(_listlike).any():
                state[0] = True
            bad = vals.map(lambda x: len(x) if _listlike(x) else -1) != lengths
            state[1] += int(bad.sum())
            if len(state[2]) < SAMPLE:
                state[2].extend(b.loc[bad, 'id2'].head(SAMPLE - len(state[2])).tolist())
        t['dups'] += int(lists.map(lambda x: len(set(x)) != len(x)).sum())
        if dedup:
            t['selfp'] += int(sum(1 for id2, lst in zip(b['id2'], lists) if id2 in set(lst)))
        if membership:
            _missing(t['miss_a'], known_a, _flatten(lists))
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
        out['id2'] += t['id2']
        for c, (seen_list, n_bad, sample) in t['aligned'].items():
            state = out['aligned'][c]
            state[0] = state[0] or seen_list
            state[1] += n_bad
            state[2].extend(sample[:SAMPLE - len(state[2])])
        _extend(out['miss_a'], t['miss_a'])
        _extend(out['miss_b'], t['miss_b'])
    return out


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
    id2 = pd.Series(np.concatenate(t['id2']))
    n_distinct = int(id2.nunique())
    if n_distinct < n:
        raise ValueError(
            f"{what} 'id2' must be unique (one row per B record); "
            f"{n - n_distinct} duplicate row(s)")
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
    known_a = sorted_ids(table_a)
    known_b = known_a if table_b is table_a else sorted_ids(table_b)
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
