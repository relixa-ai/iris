"""Features computed from the token sets of two strings."""
import pandas as pd
import numpy as np
import numba as nb
from .feature import Feature
from ..tokenizer.tokenizer import Tokenizer
from ..utils import TOKEN_HASH, is_null, get_logger, token_hashes, type_check
from py_stringmatching import MongeElkan
from abc import abstractmethod
import math

log = get_logger(__name__)

_TOKEN_HASH = TOKEN_HASH


@nb.njit(cache=False)
def _sorted_intersect_size(a, b):
    i = j = c = 0
    na, nb_ = a.shape[0], b.shape[0]
    while i < na and j < nb_:
        if a[i] == b[j]:
            c += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return c


@nb.njit(cache=False)
def _batch_intersect_sizes(a_flat, a_off, b_flat, b_off, out):
    n = a_off.shape[0] - 1
    for r in range(n):
        i, ia = a_off[r], a_off[r + 1]
        j, jb = b_off[r], b_off[r + 1]
        c = 0
        while i < ia and j < jb:
            if a_flat[i] == b_flat[j]:
                c += 1
                i += 1
                j += 1
            elif a_flat[i] < b_flat[j]:
                i += 1
            else:
                j += 1
        out[r] = c


@nb.njit(cache=False)
def _pairs_intersect_sizes(a_flat, a_off, a_local, b_flat, b_off, b_local, out):
    for p in range(out.shape[0]):
        ai = a_local[p]
        bi = b_local[p]
        i, ia = a_off[ai], a_off[ai + 1]
        j, jb = b_off[bi], b_off[bi + 1]
        c = 0
        while i < ia and j < jb:
            if a_flat[i] == b_flat[j]:
                c += 1
                i += 1
                j += 1
            elif a_flat[i] < b_flat[j]:
                i += 1
            else:
                j += 1
        out[p] = c


def _pack_id_column(values):
    n = len(values)
    lengths = np.zeros(n, dtype=np.int64)
    null = np.zeros(n, dtype=bool)
    for i, v in enumerate(values):
        if v is None:
            null[i] = True
        else:
            lengths[i] = len(v)
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    if offsets[-1]:
        flat = np.concatenate([v for v in values if v is not None and len(v)])
        flat = np.ascontiguousarray(flat, dtype=np.int64)
    else:
        flat = np.empty(0, dtype=np.int64)
    return flat, offsets, null


def _arrow_list(recs, col):
    arrow = recs.attrs.get('_mf_arrow')
    return None if arrow is None else arrow.get(col)


def _unpack_arrow_list(lst, dtype):
    import pyarrow as pa

    if isinstance(lst, pa.ChunkedArray):
        lst = lst.combine_chunks()
        if isinstance(lst, pa.ChunkedArray):
            lst = lst.chunk(0) if lst.num_chunks else pa.array([], type=lst.type)

    offsets = np.asarray(lst.offsets, dtype=np.int64)
    start, end = int(offsets[0]), int(offsets[-1])
    flat = np.asarray(lst.values)[start:end]
    offsets = np.ascontiguousarray(offsets - start, dtype=np.int64)
    flat = np.ascontiguousarray(flat, dtype=dtype)

    if offsets[-1] != len(flat) or offsets[0] != 0:
        raise RuntimeError(
            f'inconsistent Arrow list layout: offsets span [{offsets[0]}, {offsets[-1]}] '
            f'but the values window holds {len(flat)} elements')

    null = np.asarray(lst.is_null()).astype(bool, copy=False)
    return flat, offsets, null


def _packed(recs, col):
    lst = _arrow_list(recs, col)
    if lst is not None:
        return _unpack_arrow_list(lst, np.int64)

    cache = recs.attrs.get('_mf_pack_cache')
    if cache is not None and col in cache:
        return cache[col]
    packed = _pack_id_column(recs[col].to_numpy())
    if cache is not None:
        cache[col] = packed
    return packed


def _token_lists(recs, col):
    lst = _arrow_list(recs, col)
    if lst is not None:
        return np.array(lst.to_pylist(), dtype=object)
    return recs[col].to_numpy()


def _sorted_token_list(tokenizer, s):
    toks = tokenizer.tokenize(s)
    return None if toks is None else sorted(set(toks))


class TokenFeature(Feature):
    """A feature computed from the token sets that a tokenizer produces from two
    strings."""
    _TOKEN_REPR = 'ids'

    def __init__(self, a_attr, b_attr, tokenizer):
        super().__init__(a_attr, b_attr)
        type_check(tokenizer, 'tokenizer', Tokenizer)
        self._tokenizer = tokenizer
        self._a_toks_col = self._get_token_column(a_attr)
        self._b_toks_col = self._get_token_column(b_attr)
        self._a_store_col = self._get_store_column(a_attr)
        self._b_store_col = self._get_store_column(b_attr)

    @abstractmethod
    def sim_func(self, x, y):
        """Return the similarity of two token sets."""
        pass

    def _get_input_column(self, base):
        return base 

    def _get_token_column(self, base):
        return self._tokenizer.out_col_name(base)

    def _get_store_column(self, base):
        name = self._get_token_column(base)
        return name if self._TOKEN_REPR == 'set' else f'{name}_ids'

    def _preprocess_output_column(self, attr):
        return self._get_store_column(attr)

    def stored_columns(self, side):
        """Return the token column a record store keeps for this feature on side
        `side`."""
        col = self._a_store_col if side == 'a' else self._b_store_col
        return [(col, 'strset' if self._TOKEN_REPR == 'set' else 'ids')]

    def _preprocess(self, df, input_col):
        if self._TOKEN_REPR == 'set':
            tokenizer = self._tokenizer
            toks = df[input_col].apply(lambda s: _sorted_token_list(tokenizer, s))
        else:
            toks = token_hashes(df, input_col, self._tokenizer).map(
                lambda hc: None if hc is None else hc[0])
        toks.name = self._get_store_column(input_col)
        return toks

    def __call__(self, rec, recs):
        s = rec[self._b_store_col]
        if is_null(s):
            return pd.Series(np.nan, index=recs.index)
        sets = recs[self._a_store_col]
        return sets.apply(lambda x : self.sim_func(s, x)).astype(np.float64)

    def _combine(self, olap, la, lb):
        raise NotImplementedError

    def pairs_call(self, a_frame, a_local, b_frame, b_local):
        """Return the value for each pair, given two record frames and the row of each
        pair in them."""
        if self._TOKEN_REPR != 'ids' or _arrow_list(a_frame, self._a_store_col) is None:
            return super().pairs_call(a_frame, a_local, b_frame, b_local)

        a_flat, a_off, a_null = _packed(a_frame, self._a_store_col)
        b_flat, b_off, b_null = _packed(b_frame, self._b_store_col)
        a_local = np.ascontiguousarray(a_local, dtype=np.int64)
        b_local = np.ascontiguousarray(b_local, dtype=np.int64)

        olap = np.zeros(len(a_local), dtype=np.int64)
        _pairs_intersect_sizes(a_flat, a_off, a_local, b_flat, b_off, b_local, olap)

        la = (a_off[1:] - a_off[:-1]).astype(np.float64)[a_local]
        lb = (b_off[1:] - b_off[:-1]).astype(np.float64)[b_local]
        empty = (la == 0) | (lb == 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = self._combine(olap.astype(np.float64), la, lb)
        out = np.where(empty, 0.0, out)
        return np.where(a_null[a_local] | b_null[b_local], np.nan, out)

    def batch_call(self, a_recs, b_recs):
        """Return the value for each pair of aligned rows of two record frames."""
        if self._TOKEN_REPR != 'ids':
            a_sets = _token_lists(a_recs, self._a_store_col)
            b_sets = _token_lists(b_recs, self._b_store_col)
            sim = self.sim_func
            return np.fromiter(
                (np.nan if (a is None or b is None) else sim(b, a)
                 for a, b in zip(a_sets, b_sets)),
                dtype=np.float64, count=len(a_sets))

        a_flat, a_off, a_null = _packed(a_recs, self._a_store_col)
        b_flat, b_off, b_null = _packed(b_recs, self._b_store_col)

        n = len(a_off) - 1
        olap = np.zeros(n, dtype=np.int64)
        _batch_intersect_sizes(a_flat, a_off, b_flat, b_off, olap)

        la = (a_off[1:] - a_off[:-1]).astype(np.float64)
        lb = (b_off[1:] - b_off[:-1]).astype(np.float64)
        empty = (la == 0) | (lb == 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = self._combine(olap.astype(np.float64), la, lb)
        out = np.where(empty, 0.0, out)
        return np.where(a_null | b_null, np.nan, out)


def _overlap(x,y):
    if x is None or y is None or is_null(x) or is_null(y):
        return np.nan
    elif len(x) == 0 or len(y) == 0:
        return 0.0
    elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return float(_sorted_intersect_size(x, y))
    elif len(x) < len(y):
        return sum((e in y) for e in x)
    else:
        return sum((e in x) for e in y)

def _jaccard(x,y):
    olap = _overlap(x,y)
    if np.isnan(olap):
        return np.nan
    elif olap == 0:
        return 0.0
    else:
        return olap / (len(x) + len(y) - olap)

def _overlap_coeff(x,y):
    olap = _overlap(x,y)
    if np.isnan(olap):
        return np.nan
    elif olap == 0:
        return 0.0
    else:
        return olap / min(len(x), len(y))


class JaccardFeature(TokenFeature):
    
    """Jaccard similarity of two token sets."""
    def sim_func(self, x, y):
        """Return the Jaccard similarity of two token sets."""
        return _jaccard(x,y)

    def _combine(self, olap, la, lb):
        return olap / (la + lb - olap)

    def __str__(self):
        return f'jaccard({self._a_toks_col}, {self._b_toks_col})'

class OverlapCoeffFeature(TokenFeature):

    """Overlap coefficient of two token sets."""
    def sim_func(self, x, y):
        """Return the overlap coefficient of two token sets."""
        return _overlap_coeff(x,y)

    def _combine(self, olap, la, lb):
        return olap / np.minimum(la, lb)

    def __str__(self):
        return f'overlap_coeff({self._a_toks_col}, {self._b_toks_col})'


class CosineFeature(TokenFeature):
    """Cosine similarity of two token sets."""
    def sim_func(self, x, y):
        """Return the cosine similarity of two token sets."""
        olap = _overlap(x,y)
        if np.isnan(olap):
            return np.nan
        elif olap == 0:
            return 0.0
        else:
            return olap / math.sqrt(len(x) * len(y))

    def _combine(self, olap, la, lb):
        return olap / np.sqrt(la * lb)

    def __str__(self):
        return f'cosine({self._a_toks_col}, {self._b_toks_col})'

class MongeElkanFeature(TokenFeature):
    """Monge-Elkan similarity of two token lists, with Jaro-Winkler as the inner
    similarity."""
    _TOKEN_REPR = 'set'
    _me = MongeElkan().get_raw_score
    
    def sim_func(self, x, y):
        """Return the Monge-Elkan similarity of two token lists."""
        if not is_null(y):
            return self._me(x,y)
        else:
            return np.nan

    def __str__(self):
        return f'monge_elkan_jw({self.a_attr}, {self.b_attr})'
