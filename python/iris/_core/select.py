"""The choice of the pairs that batch-mode active learning labels next."""
from __future__ import annotations

import numpy as np

from .pairhash import seeded_hash

SELECTION_STRATEGIES = ('entropy', 'random', 'hybrid')

BASE_SEED = 42


def iteration_seed(i):
    """Return the seed of iteration `i`."""
    return BASE_SEED + int(i)


def entropy(probs):
    """Return the Shannon entropy in bits of each row of class probabilities."""
    p = np.asarray(probs, dtype=np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        e = (-p * np.log2(p)).sum(axis=1)
    return np.nan_to_num(e).astype(np.float64)


def _first(first_key, keys, n):
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    if n < len(first_key):
        cut = np.partition(first_key, n - 1)[n - 1]
        cand = np.flatnonzero(first_key <= cut)
        order = np.lexsort(tuple(k[cand] for k in keys))
        return cand[order][:n]
    return np.lexsort(keys)[:n]


def _by_entropy(id1, id2, ent, rnd, n):
    neg = -ent
    return _first(neg, (id1, id2, rnd, neg), n)


def _at_random(id1, id2, rnd, n):
    return _first(rnd, (id1, id2, rnd), n)


def choose(id1, id2, ent, n, selection, i):
    """Return the row indices of the next batch to label, in labeling order."""
    if selection not in SELECTION_STRATEGIES:
        raise ValueError(f'selection must be one of {SELECTION_STRATEGIES}, '
                         f'got {selection!r}')
    id1 = np.asarray(id1, dtype=np.int64)
    id2 = np.asarray(id2, dtype=np.int64)
    rnd = seeded_hash(iteration_seed(i), id1, id2)
    if selection == 'random':
        return _at_random(id1, id2, rnd, n)
    ent = np.asarray(ent, dtype=np.float64)
    if selection == 'entropy':
        return _by_entropy(id1, id2, ent, rnd, n)
    n_unc = (n + 1) // 2
    unc = _by_entropy(id1, id2, ent, rnd, n_unc)
    rest = np.setdiff1d(np.arange(len(id1)), unc, assume_unique=True)
    more = rest[_at_random(id1[rest], id2[rest], rnd[rest], n - len(unc))]
    return np.concatenate([unc, more])
