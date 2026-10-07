"""The vocabulary of a column and tokenizer: document frequencies of token hashes, built
the same way by both engines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numba as nb
import numpy as np


@dataclass(frozen=True)
class VocabRequest:
    """The column pair, tokenizer and bounds a vocabulary is built for. Equal requests
    share one vocabulary."""
    a_attr: str
    b_attr: str
    tokenizer: Any
    min_doc_freq: int = 1
    max_vocab_size: Optional[int] = None

    def describe(self):
        """Return a one-line description of the request."""
        return (f'{self.a_attr}/{self.b_attr} ({self.tokenizer}, min_doc_freq='
                f'{self.min_doc_freq}, max_vocab_size={self.max_vocab_size})')


@dataclass
class Vocabulary:
    """A built vocabulary: sorted token hashes and their document frequencies over a
    corpus of rows."""
    hashes: Any
    doc_freqs: Any
    corpus_size: int


def chunk_counts(row_hashes):
    """Return the distinct hashes of a chunk of rows and the number of rows each appears
    in."""
    rows = [np.asarray(r, dtype=np.int64) for r in row_hashes if r is not None and len(r)]
    if not rows:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.unique(np.concatenate(rows), return_counts=True)


@nb.njit(cache=False)
def _merge(h1, c1, h2, c2, h, c):
    i = j = k = 0
    while i < len(h1) and j < len(h2):
        if h1[i] < h2[j]:
            h[k] = h1[i]
            c[k] = c1[i]
            i += 1
        elif h2[j] < h1[i]:
            h[k] = h2[j]
            c[k] = c2[j]
            j += 1
        else:
            h[k] = h1[i]
            c[k] = c1[i] + c2[j]
            i += 1
            j += 1
        k += 1
    while i < len(h1):
        h[k] = h1[i]
        c[k] = c1[i]
        i += 1
        k += 1
    while j < len(h2):
        h[k] = h2[j]
        c[k] = c2[j]
        j += 1
        k += 1
    return k


def merge_counts(h1, c1, h2, c2):
    """Return two hash-count tables, each sorted by hash, merged into one with the
    counts added."""
    h1, c1 = np.ascontiguousarray(h1, np.int64), np.ascontiguousarray(c1, np.int64)
    h2, c2 = np.ascontiguousarray(h2, np.int64), np.ascontiguousarray(c2, np.int64)
    h = np.empty(len(h1) + len(h2), np.int64)
    c = np.empty(len(h1) + len(h2), np.int64)
    k = _merge(h1, c1, h2, c2, h, c)
    return h[:k].copy(), c[:k].copy()


def select(hashes, counts, min_doc_freq=1, max_vocab_size=None):
    """Return the hash and document-frequency table bounded by `min_doc_freq` and
    `max_vocab_size`, sorted by hash."""
    hashes = np.asarray(hashes, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    keep = counts >= min_doc_freq
    hashes, counts = hashes[keep], counts[keep]
    if max_vocab_size is not None and len(hashes) > max_vocab_size:
        top = np.lexsort((hashes, -counts))[:max_vocab_size]
        hashes, counts = hashes[top], counts[top]
    order = np.argsort(hashes, kind='stable')
    return hashes[order], counts[order]


def empty_error(request):
    """Return the error raised for a vocabulary left empty by its bounds."""
    return ValueError(
        f'the vocabulary for {request.a_attr!r}/{request.b_attr!r} with tokenizer '
        f'{request.tokenizer} is empty after filtering (min_doc_freq='
        f'{request.min_doc_freq}, max_vocab_size={request.max_vocab_size}). Lower '
        f'min_doc_freq or raise max_vocab_size.')
