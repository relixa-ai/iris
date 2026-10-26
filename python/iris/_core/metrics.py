"""Scoring predictions against gold pairs, shared by both engines."""
from __future__ import annotations

import math

import numba as nb
import numpy as np


def canonical_gold(id1, id2, dedup):
    """Return the distinct gold pairs sorted by `(id1, id2)`, oriented `id1 <= id2` when
    `dedup`."""
    id1 = np.asarray(id1, dtype=np.int64)
    id2 = np.asarray(id2, dtype=np.int64)
    if dedup:
        id1, id2 = np.minimum(id1, id2), np.maximum(id1, id2)
    if not len(id1):
        return id1, id2
    pairs = np.unique(np.stack([id1, id2], axis=1), axis=0)
    return pairs[:, 0].copy(), pairs[:, 1].copy()


def p_match(prediction, confidence):
    """Return the probability of a match from the predicted label and its confidence."""
    prediction = np.asarray(prediction, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    return np.where(prediction == 1.0, confidence, 1.0 - confidence)


def check_threshold(threshold):
    """Refuse a threshold that is not a probability from 0 to 1."""
    try:
        value = float(threshold)
    except (TypeError, ValueError):
        value = math.nan
    if isinstance(threshold, (str, bytes)) or not 0.0 <= value <= 1.0:
        raise ValueError(f'threshold must be from 0 to 1: it is the probability of a match '
                         f'at or above which a pair is predicted to match; got {threshold!r}')


def positives(prediction, confidence, threshold):
    """Return which rows are predicted matches at `threshold`."""
    if threshold == 0.5:
        return np.asarray(prediction, dtype=np.float64) == 1.0
    return p_match(prediction, confidence) >= threshold


def bucket(p):
    """Return the threshold-sweep bucket, 0 to 100, of each probability."""
    p = np.nan_to_num(np.asarray(p, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(np.floor(p * 100), 0, 100).astype(np.int64)


def prf(tp, n_predicted, n_gold):
    """Return precision, recall and F1 from the counts, 0.0 where undefined."""
    fp, fn = n_predicted - tp, n_gold - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def sweep(histogram, n_gold):
    """Return the threshold curve, a point every 0.05, and its point of best F1."""
    curve = []
    cum_n = cum_tp = 0
    for b in range(100, -1, -1):
        n_b, pos_b = histogram.get(b, (0, 0))
        cum_n += n_b
        cum_tp += pos_b
        if b % 5 or cum_n == 0:
            continue
        pr = cum_tp / cum_n
        rc = cum_tp / n_gold if n_gold else 0.0
        curve.append({'threshold': round(b / 100, 2), 'predicted': cum_n,
                      'precision': round(pr, 4), 'recall': round(rc, 4),
                      'f1': round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0})
    best = max(curve, key=lambda r: r['f1']) if curve else None
    return curve, best


def histogram(p, is_gold):
    """Return the number of rows and of gold rows in each bucket of the
    probabilities."""
    b = bucket(p)
    rows = np.bincount(b, minlength=101)
    gold = np.bincount(b, weights=np.asarray(is_gold, dtype=np.float64), minlength=101)
    return {int(i): (int(rows[i]), int(gold[i])) for i in np.flatnonzero(rows)}


@nb.njit('boolean[:](int64[:], int64[:], int64[:], int64[:])')
def _member(g1, g2, k1, k2):
    n = len(g1)
    out = np.zeros(len(k1), dtype=np.bool_)
    for i in range(len(k1)):
        a, b = k1[i], k2[i]
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if g1[mid] < a or (g1[mid] == a and g2[mid] < b):
                lo = mid + 1
            else:
                hi = mid
        out[i] = lo < n and g1[lo] == a and g2[lo] == b
    return out


def is_member(id1, id2, gold_id1, gold_id2):
    """Return whether each pair `(id1[i], id2[i])` is a gold pair."""
    g1 = np.asarray(gold_id1, dtype=np.int64)
    g2 = np.asarray(gold_id2, dtype=np.int64)
    order = np.lexsort((g2, g1))
    return _member(np.ascontiguousarray(g1[order]), np.ascontiguousarray(g2[order]),
                   _writable(id1), _writable(id2))


def sorted_gold(gold_id1, gold_id2):
    """Return the gold pairs sorted by `(id1, id2)` for `is_member_sorted`."""
    g1 = np.asarray(gold_id1, dtype=np.int64)
    g2 = np.asarray(gold_id2, dtype=np.int64)
    order = np.lexsort((g2, g1))
    return np.ascontiguousarray(g1[order]), np.ascontiguousarray(g2[order])


def is_member_sorted(id1, id2, sorted_id1, sorted_id2):
    """Return whether each pair is a gold pair, with gold already sorted by
    `sorted_gold`."""
    return _member(sorted_id1, sorted_id2, _writable(id1), _writable(id2))


def _writable(a):
    a = np.ascontiguousarray(a, dtype=np.int64)
    return a if a.flags.writeable else a.copy()
