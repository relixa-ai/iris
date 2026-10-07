"""The first labels of active learning, drawn from both ends of the score ranking."""
from __future__ import annotations

import logging

import numpy as np

SCAN_MULTIPLIER = 50

log = logging.getLogger(__name__)


def scan_size(nseeds, n_valid):
    """Return the number of rows examined from each end of the ranking."""
    return min(max(nseeds * SCAN_MULTIPLIER, nseeds), n_valid)


def rankings(id1, id2, score, nseeds):
    """Return the row indices of likely matches and of likely non-matches, best
    candidates first."""
    score = np.asarray(score, dtype=np.float64)
    valid = np.flatnonzero(~np.isnan(score))
    id1 = np.asarray(id1, dtype=np.int64)[valid]
    id2 = np.asarray(id2, dtype=np.int64)[valid]
    s = score[valid]
    k = scan_size(nseeds, len(valid))
    pos = valid[np.lexsort((id1, id2, -s))][:k]
    neg = valid[np.lexsort((id1, id2, s))][:k]
    return pos, neg


def label_seeds(likely_matches, likely_non_matches, labeler, nseeds, labeled=(),
                pos=0, neg=0, on_label=None):
    """Ask the labeler about rows until `nseeds` labels include both classes, and return
    the labeled rows."""
    seen = set(labeled)
    its = {True: iter(likely_matches), False: iter(likely_non_matches)}
    done = {True: False, False: False}
    out = []
    while pos + neg < nseeds or pos == 0 or neg == 0:
        if done[True] and done[False]:
            if pos == 0 or neg == 0:
                log.warning(f'ran out of candidates with {pos} positive / {neg} negative '
                            f'seeds; active learning needs at least one of each')
            else:
                log.warning('ran out of candidates before reaching nseeds')
            break
        take_pos = (pos <= neg and not done[True]) or done[False]
        try:
            row = next(its[take_pos])
        except StopIteration:
            done[take_pos] = True
            continue
        pair = (int(row['id1']), int(row['id2']))
        if pair in seen:
            continue
        label = float(labeler(row['id1'], row['id2']))
        if label == -1.0:
            break
        if label == 2.0:
            seen.add(pair)
            continue
        if label == 1.0:
            pos += 1
        else:
            neg += 1
        row = dict(row)
        row['label'] = label
        out.append(row)
        seen.add(pair)
        if on_label is not None:
            on_label(row, label)
    return out, pos, neg
