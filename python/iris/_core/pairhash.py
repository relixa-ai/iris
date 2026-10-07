"""Hashes of candidate pairs, equal to the Spark expressions that compute them."""
from __future__ import annotations

import numba as nb
import numpy as np

_P1 = np.uint64(0x9E3779B185EBCA87)
_P2 = np.uint64(0xC2B2AE3D27D4EB4F)
_P3 = np.uint64(0x165667B19E3779F9)
_P4 = np.uint64(0x85EBCA77C2B2AE63)
_P5 = np.uint64(0x27D4EB2F165667C5)
SPARK_SEED = 42


@nb.njit('uint64(uint64, uint64)', inline='always')
def _rotl(x, r):
    return (x << r) | (x >> (np.uint64(64) - r))


@nb.njit('uint64(int64, uint64)', inline='always')
def _hash_long(value, seed):
    h = seed + _P5 + np.uint64(8)
    h ^= _rotl(np.uint64(value) * _P2, np.uint64(31)) * _P1
    h = _rotl(h, np.uint64(27)) * _P1 + _P4
    h ^= h >> np.uint64(33)
    h *= _P2
    h ^= h >> np.uint64(29)
    h *= _P3
    h ^= h >> np.uint64(32)
    return h


@nb.njit('int64[:](int64[:], int64[:])')
def _pair(id1, id2):
    out = np.empty(len(id1), dtype=np.int64)
    for i in range(len(id1)):
        h = _hash_long(id1[i], np.uint64(SPARK_SEED))
        out[i] = np.int64(_hash_long(id2[i], h))
    return out


@nb.njit('int64[:](int64, int64[:], int64[:])')
def _seeded(seed, id1, id2):
    first = _hash_long(seed, np.uint64(SPARK_SEED))
    out = np.empty(len(id1), dtype=np.int64)
    for i in range(len(id1)):
        h = _hash_long(id1[i], first)
        out[i] = np.int64(_hash_long(id2[i], h))
    return out


def _ids(a):
    a = np.ascontiguousarray(a, dtype=np.int64)
    return a if a.flags.writeable else a.copy()


def pair_hash(id1, id2):
    """Return Spark's `xxhash64(id1, id2)` for arrays of ids."""
    return _pair(_ids(id1), _ids(id2))


def seeded_hash(seed, id1, id2):
    """Return Spark's `xxhash64(seed, id1, id2)` for arrays of ids."""
    return _seeded(np.int64(seed), _ids(id1), _ids(id2))
