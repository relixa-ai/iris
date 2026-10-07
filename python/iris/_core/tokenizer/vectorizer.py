"""Vectorizers that turn token sets into weighted sparse vectors."""

import numpy as np
from ..utils import is_null, get_logger, SparseVec, PerfectHashFunction, TOKEN_HASH_SEED
from ..storage import memmap_map
import numba as nb


log = get_logger(__name__)

class Vectorizer:
    """The base class of vectorizers, which weight tokens by a vocabulary."""
    def set_vocab(self, hashes, doc_freqs, corpus_size):
        """Set the vocabulary from its sorted token hashes, their document frequencies
        and the corpus size."""
        raise NotImplementedError
    def out_col_name(self, base):
        """Return the name of the column that holds the vectors of `base`."""
        raise NotImplementedError
    def init(self):
        """Open the vocabulary arrays before vectorizing."""
        raise NotImplementedError
    def memmaps(self):
        """Return the memory-mapped vocabulary arrays the workers read."""
        return []
    def vectorize(self, tokens):
        """Return the vector of a list of tokens."""
        raise NotImplementedError

    def vectorize_ids(self, uniq, counts):
        """Return the vector of sorted distinct token hashes and their counts."""
        raise NotImplementedError

class TFIDFVectorizer(Vectorizer):

    """A vectorizer that weights tokens by TF/IDF."""
    def __init__(self):
        self._N = None
        self._hash_func = None
        self._hashes = None
        self._idfs = None

    def set_vocab(self, hashes, doc_freqs, corpus_size):
        """Set the vocabulary and compute the IDF weights."""
        doc_freqs.init()
        self._idfs = memmap_map(
            doc_freqs.values,
            lambda chunk: _doc_freq_to_idf(chunk, corpus_size),
            np.float32,
        )

        self._hashes = hashes

        self._hash_func = PerfectHashFunction(seed=TOKEN_HASH_SEED)
        self._N = len(self._idfs)

    def memmaps(self):
        """Return the memory-mapped vocabulary and weight arrays the workers read."""
        return [self._hashes, self._idfs]

    def _hash(self, s):
        return self._hash_func.hash(s)
    
    def out_col_name(self, base):
        """Return the name of the column that holds the TF/IDF vectors of `base`."""
        return f'term_vec({base})'
    
    def init(self):
        """Open the vocabulary and weight arrays before vectorizing."""
        self._idfs.init()
        self._hashes.init()

    def vectorize(self, tokens):
        """Return the TF/IDF vector of a list of tokens."""
        if is_null(tokens):
            return None
        hashes = np.fromiter((self._hash(t) for t in tokens), dtype=np.int64, count=len(tokens))
        return self.vectorize_ids(*np.unique(hashes, return_counts=True))

    def vectorize_ids(self, uniq, counts):
        """Return the TF/IDF vector of sorted distinct token hashes and their counts."""
        if uniq is None:
            return None
        return SparseVec(self._N, *_vectorize_tfidf(
            self._hashes.values, self._idfs.values,
            np.ascontiguousarray(uniq, dtype=np.int64),
            np.ascontiguousarray(counts, dtype=np.int64)))

@nb.njit('float32(float32[:])')
def _l2(values):
    s = 0.0
    for x in values:
        d = np.float64(x)
        s += d * d
    return np.float32(np.sqrt(s))


@nb.njit('float32[:](int64[::1], int64)')
def _doc_freq_to_idf(doc_freq, corpus_size):
    return (np.log((corpus_size +1 ) / (doc_freq + 1)) + 1).astype(np.float32)


@nb.njit(
    nb.types.Tuple((
            nb.types.int32[:],
            nb.types.float32[:]
        ))(
            nb.types.Array(nb.types.int64, 1, 'C', readonly=True),
            nb.types.Array(nb.types.float32, 1, 'C', readonly=True),
            nb.types.Array(nb.types.int64, 1, 'C', readonly=False),
            nb.types.Array(nb.types.int64, 1, 'C', readonly=False)
    )
)
def _vectorize_tfidf(hash_idx, idfs, hashes, values):
        n = hash_idx.shape[0]
        if n == 0:
            return (np.empty(0, np.int32), np.empty(0, np.float32))
        idxes = np.searchsorted(hash_idx, hashes)
        idxes = np.minimum(idxes, n - 1)
        match = hash_idx[idxes] == hashes
        idxes = idxes[match]
        values = values[match]

        idf = idfs[idxes]
        values = (np.log(values).astype(np.float32) + 1) * idf
        if values.shape[0] > 0:
            norm = _l2(values)
            if norm > 0:
                values /= norm

        return (idxes.astype(np.int32), values)


class SIFVectorizer(Vectorizer):

    """A vectorizer that weights tokens by smooth inverse frequency."""
    def __init__(self):
        self._a_param = 0.001
        self._N = None
        self._sifs = None
        self._hash_func = None
        self._hashes = None

    def set_vocab(self, hashes, doc_freqs, corpus_size):
        """Set the vocabulary and compute the SIF weights."""
        a_param = self._a_param
        doc_freqs.init()
        self._sifs = memmap_map(
            doc_freqs.values,
            lambda chunk: _doc_freq_to_sif(chunk, corpus_size, a_param),
            np.float32,
        )

        self._hashes = hashes

        self._hash_func = PerfectHashFunction(seed=TOKEN_HASH_SEED)
        self._N = len(self._sifs)

    def memmaps(self):
        """Return the memory-mapped vocabulary and weight arrays the workers read."""
        return [self._hashes, self._sifs]
    
    def _hash(self, s):
        return self._hash_func.hash(s)
    
    def out_col_name(self, base):
        """Return the name of the column that holds the SIF vectors of `base`."""
        return f'sif_vec({base})'
    
    def init(self):
        """Open the vocabulary and weight arrays before vectorizing."""
        self._sifs.init()
        self._hashes.init()
        pass

    def vectorize(self, tokens):
        """Return the SIF vector of a list of tokens."""
        if is_null(tokens):
            return None
        hashes = np.fromiter((self._hash(t) for t in tokens), dtype=np.int64, count=len(tokens))
        return self.vectorize_ids(*np.unique(hashes, return_counts=True))

    def vectorize_ids(self, uniq, counts):
        """Return the SIF vector of sorted distinct token hashes and their counts."""
        if uniq is None:
            return None
        return SparseVec(self._N, *_vectorize_sif(
            self._hashes.values, self._sifs.values,
            np.ascontiguousarray(uniq, dtype=np.int64),
            np.ascontiguousarray(counts, dtype=np.int64)))


@nb.njit('float32[:](int64[::1], int64, float32)')
def _doc_freq_to_sif(doc_freq, corpus_size, a_param):
    return ((a_param / ((doc_freq / corpus_size) + a_param))).astype(np.float32)

@nb.njit(
    nb.types.Tuple((
            nb.types.int32[:],
            nb.types.float32[:]
        ))(
            nb.types.Array(nb.types.int64, 1, 'C', readonly=True),
            nb.types.Array(nb.types.float32, 1, 'C', readonly=True),
            nb.types.Array(nb.types.int64, 1, 'C', readonly=False),
            nb.types.Array(nb.types.int64, 1, 'C', readonly=False)
    )
)
def _vectorize_sif(hash_idx, sifs, hashes, values):
        n = hash_idx.shape[0]
        if n == 0:
            return (np.empty(0, np.int32), np.empty(0, np.float32))
        idxes = np.searchsorted(hash_idx, hashes)
        idxes = np.minimum(idxes, n - 1)
        match = hash_idx[idxes] == hashes
        idxes = idxes[match]
        values = values[match]

        sif = sifs[idxes]
        values = values.astype(np.float32) * sif
        if values.shape[0] > 0:
            norm = _l2(values)
            if norm > 0:
                values /= norm

        return (idxes.astype(np.int32), values)
