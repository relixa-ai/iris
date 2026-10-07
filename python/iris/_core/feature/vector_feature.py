"""Features computed from the TF/IDF and SIF vectors of tokenized strings."""
import pandas as pd
from ..tokenizer.vectorizer import TFIDFVectorizer, SIFVectorizer
from .token_feature import TokenFeature
import numpy as np
import numba as nb
from ..utils import get_logger, token_hashes
from ..vocab import VocabRequest
from threading import Lock

log = get_logger(__name__)

class _VocabEntry:

    def __init__(self, request):
        self.request = request
        self._built = None
        self._lock = Lock()

    def __eq__(self, o):
        return isinstance(o, _VocabEntry) and o.request == self.request

    def get(self, A, B):
        with self._lock:
            if self._built is None:
                from ..._spark.vocab import build_vocabularies
                v = build_vocabularies(A, B, [self.request])[0]
                self._built = (v.hashes, v.doc_freqs, v.corpus_size)
        return self._built


class TFIDFFeature(TokenFeature):

    """Cosine similarity of the TF/IDF vectors of two strings."""
    def __init__(self, a_attr, b_attr, tokenizer, min_doc_freq=1, max_vocab_size=None):
        super().__init__(a_attr, b_attr, tokenizer)
        self.vectorizer = TFIDFVectorizer()
        self.min_doc_freq = min_doc_freq
        self.max_vocab_size = max_vocab_size
        self._a_vec_col = self._get_vector_column(self.a_attr)
        self._b_vec_col = self._get_vector_column(self.b_attr)
    

    def sim_func(self, x,y):
        """Return the cosine similarity of the TF/IDF vectors of two token sets."""
        pass
    
    def __str__(self):
        return f'tf_idf_{str(self._tokenizer)}({self.a_attr}, {self.b_attr})'
    
    def _get_vector_column(self, input_col):
        return self.vectorizer.out_col_name(self._get_token_column(input_col))

    def _preprocess_output_column(self, attr):
        return self._get_vector_column(attr)

    def stored_columns(self, side):
        """Return the vector column a record store keeps for this feature on side
        `side`."""
        return [(self._get_vector_column(self.a_attr if side == 'a' else self.b_attr),
                 'vec')]
    

    def vocab_request(self):
        """Return the description of the vocabulary this feature needs."""
        return VocabRequest(self.a_attr, self.b_attr, self._tokenizer,
                            self.min_doc_freq, self.max_vocab_size)

    def set_vocab(self, hashes, doc_freqs, corpus_size):
        """Set the vocabulary from its sorted token hashes, their document frequencies
        and the corpus size."""
        self.vectorizer.set_vocab(hashes, doc_freqs, corpus_size)

    def memmaps(self):
        """Return the memory-mapped vocabulary arrays the feature's workers read."""
        return self.vectorizer.memmaps()

    def build(self, A, B, cache):
        """Set the vocabulary, built from tables A and B or taken from `cache`."""
        entry = cache.add_or_get(_VocabEntry(self.vocab_request()))
        self.set_vocab(*entry.get(A, B))

    def _preprocess(self, data, input_col):
        self.vectorizer.init()
        vz = self.vectorizer
        vecs = token_hashes(data, input_col, self._tokenizer).map(
            lambda hc: None if hc is None else vz.vectorize_ids(hc[0], hc[1]))
        vecs.name = self._get_vector_column(input_col)

        return vecs

    def __call__(self, rec, recs):
        vec = rec[self._b_vec_col]
        vecs = recs[self._a_vec_col]
        if vec is None:
            return pd.Series(np.nan, index=vecs.index)
        return vecs.apply(lambda x : vec.dot(x) if x is not None else np.nan).astype(np.float64)

    def batch_call(self, a_recs, b_recs):
        """Return the value for each pair of aligned rows of two record frames."""
        a = _packed_vec(a_recs, self._a_vec_col)
        b = _packed_vec(b_recs, self._b_vec_col)
        if a is not None and b is not None:
            a_idx, a_val, a_off, a_null = a
            b_idx, b_val, b_off, b_null = b
            out = np.zeros(len(a_off) - 1, dtype=np.float32)
            _batch_sparse_dot(a_idx, a_val, a_off, b_idx, b_val, b_off, out)
            return np.where(a_null | b_null, np.nan, out.astype(np.float64))

        a_vecs = a_recs[self._a_vec_col].to_numpy()
        b_vecs = b_recs[self._b_vec_col].to_numpy()
        return np.fromiter(
            (np.nan if (x is None or y is None) else y.dot(x)
             for x, y in zip(a_vecs, b_vecs)),
            dtype=np.float64, count=len(a_vecs))

class SIFFeature(TFIDFFeature):
    """Cosine similarity of the SIF-weighted vectors of two strings."""
    def __init__(self, a_attr, b_attr, tokenizer, min_doc_freq=1, max_vocab_size=None):
        super().__init__(a_attr, b_attr, tokenizer,
                         min_doc_freq=min_doc_freq, max_vocab_size=max_vocab_size)
        self.vectorizer = SIFVectorizer()
        self._a_vec_col = self._get_vector_column(self.a_attr)
        self._b_vec_col = self._get_vector_column(self.b_attr)
    
    def __str__(self):
        return f'sif_{str(self._tokenizer)}({self.a_attr}, {self.b_attr})'


VEC_IDX_SUFFIX = '#idx'
VEC_VAL_SUFFIX = '#val'


@nb.njit(cache=False)
def _batch_sparse_dot(a_idx, a_val, a_off, b_idx, b_val, b_off, out):
    n = a_off.shape[0] - 1
    for r in range(n):
        i, ia = a_off[r], a_off[r + 1]
        j, jb = b_off[r], b_off[r + 1]
        s = 0.0
        while i < ia and j < jb:
            if a_idx[i] == b_idx[j]:
                s += a_val[i] * b_val[j]
                i += 1
                j += 1
            elif a_idx[i] < b_idx[j]:
                i += 1
            else:
                j += 1
        out[r] = s


def _packed_vec(recs, col):
    from .token_feature import _arrow_list, _unpack_arrow_list

    idx_lst = _arrow_list(recs, col + VEC_IDX_SUFFIX)
    val_lst = _arrow_list(recs, col + VEC_VAL_SUFFIX)
    if idx_lst is None or val_lst is None:
        return None
    idx, offsets, null = _unpack_arrow_list(idx_lst, np.int32)
    val, _, _ = _unpack_arrow_list(val_lst, np.float32)
    return idx, val, offsets, null


@nb.njit(cache=False)
def _pairs_sparse_dot(a_idx, a_val, a_off, a_local, b_idx, b_val, b_off, b_local, out):
    for p in range(out.shape[0]):
        ai = a_local[p]
        bi = b_local[p]
        i, ia = a_off[ai], a_off[ai + 1]
        j, jb = b_off[bi], b_off[bi + 1]
        s = 0.0
        while i < ia and j < jb:
            if a_idx[i] == b_idx[j]:
                s += a_val[i] * b_val[j]
                i += 1
                j += 1
            elif a_idx[i] < b_idx[j]:
                i += 1
            else:
                j += 1
        out[p] = s


def _tfidf_pairs_call(self, a_frame, a_local, b_frame, b_local):
    a = _packed_vec(a_frame, self._a_vec_col)
    b = _packed_vec(b_frame, self._b_vec_col)
    if a is None or b is None:
        return TokenFeature.pairs_call(self, a_frame, a_local, b_frame, b_local)

    a_idx, a_val, a_off, a_null = a
    b_idx, b_val, b_off, b_null = b
    a_local = np.ascontiguousarray(a_local, dtype=np.int64)
    b_local = np.ascontiguousarray(b_local, dtype=np.int64)
    out = np.zeros(len(a_local), dtype=np.float32)
    _pairs_sparse_dot(a_idx, a_val, a_off, a_local, b_idx, b_val, b_off, b_local, out)
    return np.where(a_null[a_local] | b_null[b_local], np.nan, out.astype(np.float64))


TFIDFFeature.pairs_call = _tfidf_pairs_call
