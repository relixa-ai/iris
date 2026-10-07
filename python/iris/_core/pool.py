"""The score by which pairs are ranked, and the rules that decide which pairs the
labeling pool keeps."""
from __future__ import annotations

import numba as nb
import numpy as np

from .pairhash import pair_hash, seeded_hash

POSITIVE_PREFIXES = ('exact_match', 'needleman_wunch', 'smith_waterman', 'jaccard',
                     'overlap_coeff', 'cosine', 'monge_elkan_jw', 'tf_idf', 'sif',
                     'edit_distance')


def positive_features(names):
    """Return, for each feature name, whether the feature counts toward the score."""
    return np.array([any(str(n).startswith(p) for p in POSITIVE_PREFIXES) for n in names],
                    dtype=np.bool_)


@nb.njit('float64[:](float32[:, :], boolean[:])')
def _score(mat, positive):
    n, k = mat.shape
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for j in range(k):
            if positive[j]:
                v = np.float64(mat[i, j])
                if not np.isnan(v):
                    s += v
        out[i] = s
    return out


def score(mat, positive):
    """Return the score of every row of a feature matrix."""
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError(f'expected a 2-D feature matrix, got shape {mat.shape}')
    positive = np.ascontiguousarray(positive, dtype=np.bool_)
    if positive.shape[0] != mat.shape[1]:
        raise ValueError(f'{positive.shape[0]} positivity flags for {mat.shape[1]} features')
    return _score(mat, positive)


DEFAULT_MAX_POOL_SIZE = 10_000_000


def pool_size(n_rows, percent, max_pool_size=DEFAULT_MAX_POOL_SIZE):
    """Return the number of rows the pool keeps: `percent` of `n_rows`, at most
    `max_pool_size`."""
    if n_rows <= 0:
        return 0
    target = int(n_rows * percent)
    if max_pool_size is None:
        return target
    if max_pool_size <= 0:
        raise ValueError(f'max_pool_size must be > 0 (got {max_pool_size})')
    return min(target, max_pool_size)


def plan(n_rows, percent, bucket_size=1000, max_pool_size=DEFAULT_MAX_POOL_SIZE):
    """Return the number of buckets, the rows kept and the buckets kept for a candidate
    set of `n_rows`."""
    nparts = max(n_rows // bucket_size, 1)
    B = 2 * nparts - 1
    S = pool_size(n_rows, percent, max_pool_size)
    n = max(1, -(-(S // 2) // B))
    B_keep = B if max_pool_size is None else max(1, min(B, max_pool_size // (2 * n)))
    return B, n, B_keep


def draw_seed(seed):
    """Return `seed`, or a new random seed when it is None."""
    if seed is not None:
        return int(seed)
    return int(np.random.default_rng().integers(0, 2**63 - 1))


def select(id1, id2, score, percent, bucket_size=1000,
           max_pool_size=DEFAULT_MAX_POOL_SIZE, seed=42):
    """Return the row indices the labeling pool keeps, in ascending order."""
    if not 0.0 < percent <= 1.0:
        raise ValueError('percent must be in the range (0.0, 1.0]')
    id1 = np.asarray(id1, dtype=np.int64)
    id2 = np.asarray(id2, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    N = len(id1)
    if N == 0:
        return np.empty(0, dtype=np.int64)
    B, n, B_keep = plan(N, percent, bucket_size, max_pool_size)
    bucket = np.mod(pair_hash(id1, id2), B)
    live = np.flatnonzero(bucket < B_keep)
    if not len(live):
        return live
    rnd = seeded_hash(draw_seed(seed), id1[live], id2[live])
    keep = select_in_buckets(bucket[live], id1[live], id2[live], score[live], rnd,
                             percent, n)
    keep = cap(id1[live][keep], id2[live][keep], rnd[keep], max_pool_size, keep)
    return np.sort(live[keep])


def select_in_buckets(b, i1, i2, s, rnd, percent, n):
    """Return the positions of the rows that each bucket and score band keeps."""
    if not len(b):
        return np.empty(0, dtype=np.int64)
    nan = np.isnan(s)
    key = np.where(nan, 0.0, s)
    order = np.lexsort((key, nan, b))
    bs, ns, ks = b[order], nan[order], key[order]
    new_bucket = np.r_[True, bs[1:] != bs[:-1]]
    new_group = new_bucket | np.r_[True, (ns[1:] != ns[:-1]) | (ks[1:] != ks[:-1])]
    starts = np.flatnonzero(new_bucket)
    ends = np.r_[starts[1:], len(bs)]
    bucket_end = np.repeat(ends, ends - starts)
    gstarts = np.flatnonzero(new_group)
    gends = np.r_[gstarts[1:], len(bs)]
    group_end = np.repeat(gends, gends - gstarts)
    size = np.repeat(ends - starts, ends - starts)
    greater = bucket_end - group_end
    pr = np.where(size > 1, greater / np.maximum(size - 1, 1), 0.0)
    band = np.empty(len(bs), dtype=bool)
    band[order] = pr <= percent

    order = np.lexsort((i1, i2, rnd, band, b))
    bb, nb_ = b[order], band[order]
    head = np.r_[True, (bb[1:] != bb[:-1]) | (nb_[1:] != nb_[:-1])]
    gstart = np.maximum.accumulate(np.where(head, np.arange(len(order)), 0))
    return order[(np.arange(len(order)) - gstart) < n]


def cap(i1, i2, rnd, max_pool_size, keep=None):
    """Return the positions of the `max_pool_size` rows kept when more were selected."""
    keep = np.arange(len(i1)) if keep is None else np.asarray(keep)
    if max_pool_size is not None and len(keep) > max_pool_size:
        keep = keep[np.lexsort((i1, i2, rnd))][:max_pool_size]
    return keep
