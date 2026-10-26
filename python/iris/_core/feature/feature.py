"""The `Feature` base class and the features computed from whole strings and numbers."""

from ..frames import AttrsByRef
from ..numeric import parse_number
from abc import abstractmethod, ABC
from functools import partial
import numba as nb
import pandas as pd
import numpy as np
from ..utils import SparseVec, is_null, get_logger

log = get_logger(__name__)

def materialize_arrow_columns(recs):
    """Return a record frame with its Arrow-backed columns converted to Python
    objects."""
    arrow = recs.attrs.get('_mf_arrow')
    if not arrow:
        return recs

    import pyarrow as pa
    out = recs.copy()
    out.attrs = dict(recs.attrs)
    for name, lst in arrow.items():
        if name.endswith('#val'):
            continue
        if name.endswith('#idx'):
            base = name[:-4]
            vals = arrow.get(base + '#val')
            if vals is None:
                continue
            out[base] = [
                None if i is None or v is None
                else SparseVec(0, np.asarray(i, dtype=np.int32),
                               np.asarray(v, dtype=np.float32))
                for i, v in zip(lst.to_pylist(), vals.to_pylist())
            ]
        elif pa.types.is_string(lst.type.value_type):
            out[name] = lst.to_pylist()
        else:
            out[name] = [None if v is None else np.asarray(v, dtype=np.int64)
                         for v in lst.to_pylist()]
    out.attrs['_mf_arrow'] = {}
    return out


def _code_points(s):
    return np.frombuffer(s.encode('utf-32-le', 'surrogatepass'), dtype=np.uint32)


@nb.njit(cache=False)
def _levenshtein(a, b):
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = np.arange(m + 1, dtype=np.int64)
    cur = np.empty(m + 1, dtype=np.int64)
    for i in range(n):
        cur[0] = i + 1
        ca = a[i] & 0xFF
        for j in range(m):
            sub = prev[j] + (0 if ca == (b[j] & 0xFF) else 1)
            cur[j + 1] = min(cur[j] + 1, prev[j + 1] + 1, sub)
        prev, cur = cur, prev
    return prev[m]


@nb.njit(cache=False)
def _smith_waterman(a, b, gap):
    m = len(b)
    prev = np.zeros(m + 1)
    cur = np.zeros(m + 1)
    best = 0.0
    for i in range(len(a)):
        cur[0] = 0.0
        for j in range(m):
            match = prev[j] + (1.0 if a[i] == b[j] else 0.0)
            v = max(0.0, match, prev[j + 1] - gap, cur[j] - gap)
            cur[j + 1] = v
            if v > best:
                best = v
        prev, cur = cur, prev
    return best


@nb.njit(cache=False)
def _needleman_wunsch(a, b, gap):
    m = len(b)
    prev = np.empty(m + 1)
    cur = np.empty(m + 1)
    for j in range(m + 1):
        prev[j] = -(j * gap)
    for i in range(len(a)):
        cur[0] = -((i + 1) * gap)
        for j in range(m):
            match = prev[j] + (1.0 if a[i] == b[j] else 0.0)
            cur[j + 1] = max(match, prev[j + 1] - gap, cur[j] - gap)
        prev, cur = cur, prev
    return prev[m]


def _edit_similarity(x, y):
    if x == y:
        return 1.0
    longest = max(len(x), len(y))
    return 1 - _levenshtein(_code_points(x), _code_points(y)) / longest


def _smith_waterman_score(x, y):
    return _smith_waterman(_code_points(x), _code_points(y), 1.0)


def _needleman_wunsch_score(x, y):
    return _needleman_wunsch(_code_points(x), _code_points(y), 1.0)


def _own(recs, columns):
    names = [name for name, _ in columns]
    arrow = recs.attrs.get('_mf_arrow') or {}
    kept = {k: arrow[k] for name in names
            for k in (name, name + '#idx', name + '#val') if k in arrow}
    out = recs[[name for name in names if name in recs.columns]]
    out.attrs = {'_mf_arrow': AttrsByRef(kept)}
    return out


class Feature(ABC):
    """A function of a pair of records, computed from one column of table A and one
    column of table B."""
    def __init__(self, a_attr : str, b_attr : str):
        if not isinstance(a_attr, str) or not isinstance(b_attr, str):
            raise TypeError(f'a_attr and b_attr must be strings not {type(a_attr), type(b_attr)}')

        self._a_attr = a_attr
        self._b_attr = b_attr

    def build(self, A, B, cache):
        """Compute what the feature needs from tables A and B, before any record is
        preprocessed."""
        pass

    def memmaps(self):
        """Return the memory-mapped arrays the feature's workers read, or None."""
        return []

    @classmethod
    def template(cls, **kwargs):
        """Return a constructor of this feature with the given keyword arguments
        fixed."""
        return partial(cls, **kwargs)

    @property
    def a_attr(self):
        """The column of table A the feature reads."""
        return self._a_attr

    @property
    def b_attr(self):
        """The column of table B the feature reads."""
        return self._b_attr

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __call__(self, A : dict, B : pd.DataFrame) -> pd.Series:
        pass

    def batch_call(self, a_recs : pd.DataFrame, b_recs : pd.DataFrame) -> np.ndarray:
        """Return the value for each pair of aligned rows of two record frames."""
        out = np.empty(len(a_recs), dtype=np.float64)
        if len(a_recs) == 0:
            return out
        a_recs = materialize_arrow_columns(a_recs)
        b_recs = materialize_arrow_columns(b_recs)
        b_index = b_recs.index.to_numpy()
        boundaries = np.flatnonzero(b_index[1:] != b_index[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(b_index)]))
        for s, e in zip(starts, ends):
            out[s:e] = np.asarray(self(b_recs.iloc[s], a_recs.iloc[s:e]), dtype=np.float64)
        return out

    def pairs_call(self, a_frame, a_local, b_frame, b_local) -> np.ndarray:
        """Return the value for each pair, given two record frames and the row of each
        pair in them."""
        a = materialize_arrow_columns(_own(a_frame, self.stored_columns('a'))).iloc[a_local]
        b = materialize_arrow_columns(_own(b_frame, self.stored_columns('b'))).iloc[b_local]
        return self.batch_call(a, b)

    @abstractmethod
    def _preprocess(self, data : pd.DataFrame, input_col : str) -> pd.Series:
        pass

    @abstractmethod
    def _preprocess_output_column(self):
        pass

    STORED_KINDS = ('str', 'f64', 'ids', 'vec', 'strset', 'pkl')

    def stored_columns(self, side):
        """Return the columns, with their kinds, that a record store keeps for this
        feature on side `'a'` or `'b'`."""
        attr = self.a_attr if side == 'a' else self.b_attr
        out = [(attr, 'str')]
        name = self.preprocess_output_column(side == 'a')
        if name is not None:
            out.append((name, 'pkl'))
        return out

    def preprocess_output_column(self, for_table_a : bool):
        """Return the name of the preprocessed column for table A or table B."""
        if for_table_a:
            return self._preprocess_output_column(self.a_attr)
        else:
            return self._preprocess_output_column(self.b_attr)

    def preprocess(self, data, is_table_a):
        """Return `data` with the preprocessed column of this feature added."""
        out_col = self.preprocess_output_column(is_table_a)
        if out_col is None:
            return data

        if is_table_a: 
            if self.preprocess_output_column(is_table_a) not in data:
                c = self._preprocess(data, self.a_attr)
                data[c.name] = c
        else:
            if self.preprocess_output_column(False) not in data:
                c = self._preprocess(data, self.b_attr)
                data[c.name] = c

        return data


class ExactMatchFeature(Feature):
    """Case-insensitive exact match of two strings."""
    
    def _preprocess_output_column(self, for_table_a):
        return None

    def _preprocess(self, data, is_table_a):
        return data

    def __call__(self, rec, recs):
        s = rec[self.b_attr]
        strings = recs[self.a_attr]

        if not isinstance(s, str):
            return pd.Series(np.nan, index=strings.index)

        lowered = strings.apply(lambda x : x.lower() if pd.notnull(x) else None)
        out = lowered.eq(s.lower()).astype(np.float64)
        return out.mask(lowered.isna())

    def batch_call(self, a_recs, b_recs):
        """Return the value for each pair of aligned rows of two record frames."""
        a_vals = a_recs[self.a_attr].to_numpy()
        b_vals = b_recs[self.b_attr].to_numpy()
        out = np.full(len(a_vals), np.nan, dtype=np.float64)
        for i in range(len(a_vals)):
            a, b = a_vals[i], b_vals[i]
            if not isinstance(b, str) or not isinstance(a, str):
                continue
            out[i] = 1.0 if a.lower() == b.lower() else 0.0
        return out

    def pairs_call(self, a_frame, a_local, b_frame, b_local):
        """Return the value for each pair, given two record frames and the row of each
        pair in them."""
        a_vals = a_frame[self.a_attr].to_numpy()[a_local]
        b_vals = b_frame[self.b_attr].to_numpy()[b_local]
        out = np.full(len(a_vals), np.nan, dtype=np.float64)
        for i in range(len(a_vals)):
            a, b = a_vals[i], b_vals[i]
            if not isinstance(b, str) or not isinstance(a, str):
                continue
            out[i] = 1.0 if a.lower() == b.lower() else 0.0
        return out

    def __str__(self):
        return f'exact_match({self.a_attr}, {self.b_attr})'


class RelDiffFeature(Feature):
    """Relative difference of two numbers."""
        
    def __init__(self, a_attr, b_attr):
        super().__init__(a_attr, b_attr)
        self._a_float_col = self._preprocess_output_column(a_attr)
        self._b_float_col = self._preprocess_output_column(b_attr)

    def _preprocess_output_column(self, attr):
        return f'float({attr})'

    def stored_columns(self, side):
        """Return the numeric column a record store keeps for this feature on side
        `side`."""
        return [(self._a_float_col if side == 'a' else self._b_float_col, 'f64')]

    def _preprocess(self, data, input_col):
        floats = data[input_col].apply(parse_number)
        floats.name = self._preprocess_output_column(input_col)
        return floats

    def __call__(self, rec, recs):
        f = rec[self._b_float_col]
        floats = recs[self._a_float_col]

        if is_null(f):
            return pd.Series(np.nan, index=floats.index)

        vals = floats.values.astype(np.float32)

        denom = np.maximum(np.abs(vals), np.abs(f))
        rel = np.divide(np.abs(vals - f), denom, out=np.zeros_like(denom), where=denom != 0)
        return pd.Series(rel, index=floats.index).astype(np.float64)

    def batch_call(self, a_recs, b_recs):
        """Return the value for each pair of aligned rows of two record frames."""
        a = pd.to_numeric(a_recs[self._a_float_col], errors='coerce').to_numpy(np.float64)
        b = pd.to_numeric(b_recs[self._b_float_col], errors='coerce').to_numpy(np.float64)
        denom = np.maximum(np.abs(a), np.abs(b))
        out = np.divide(np.abs(a - b), denom, out=np.zeros_like(denom), where=denom != 0)
        return np.where(np.isnan(a) | np.isnan(b), np.nan, out)

    def pairs_call(self, a_frame, a_local, b_frame, b_local):
        """Return the value for each pair, given two record frames and the row of each
        pair in them."""
        a = pd.to_numeric(a_frame[self._a_float_col], errors='coerce').to_numpy(np.float64)[a_local]
        b = pd.to_numeric(b_frame[self._b_float_col], errors='coerce').to_numpy(np.float64)[b_local]
        denom = np.maximum(np.abs(a), np.abs(b))
        out = np.divide(np.abs(a - b), denom, out=np.zeros_like(denom), where=denom != 0)
        return np.where(np.isnan(a) | np.isnan(b), np.nan, out)

    def __str__(self):
        return f'rel_diff({self.a_attr}, {self.b_attr})'

class EditDistanceFeature(Feature):
    """Case-insensitive edit distance of two strings."""
    
    _func = staticmethod(_edit_similarity)

    def _preprocess_output_column(self, for_table_a):
        return None

    def _preprocess(self, data, is_table_a):
        return data

    def __call__(self, rec, recs):
        s = rec[self.b_attr]
        strings = recs[self.a_attr]

        if not isinstance(s, str):
            return pd.Series(np.nan, index=strings.index)

        strings = strings.apply(lambda x : str(x).lower() if pd.notnull(x) else None)
        return strings.apply(lambda x : self._func(s, x) if x is not None else np.nan).astype(np.float64)

    def __str__(self):
        return f'edit_distance({self.a_attr}, {self.b_attr})'

class NeedlemanWunschFeature(Feature):
    """Case-insensitive Needleman-Wunsch score of two strings."""
    
    _func = staticmethod(_needleman_wunsch_score)

    def _preprocess_output_column(self, for_table_a):
        return None

    def _preprocess(self, data, is_table_a):
        return data
    
    def _sim_func(self, x, y):
        div = max(len(x), len(y))
        if div != 0:
            return self._func(x,y) / div
        else:
            return 0.0

    def __call__(self, rec, recs):
        s = rec[self.b_attr]
        strings = recs[self.a_attr]

        if not isinstance(s, str):
            return pd.Series(np.nan, index=strings.index)

        strings = strings.apply(lambda x : str(x).lower() if pd.notnull(x) else None)
        return strings.apply(lambda x : self._sim_func(s, x) if x is not None else np.nan).astype(np.float64)

    def __str__(self):
        return f'needleman_wunch({self.a_attr}, {self.b_attr})'


class SmithWatermanFeature(Feature):
    """Case-insensitive Smith-Waterman score of two strings."""
    
    _func = staticmethod(_smith_waterman_score)

    def _preprocess_output_column(self, for_table_a):
        return None

    def _preprocess(self, data, is_table_a):
        return data

    def _sim_func(self, x, y):
        div = max(len(x), len(y))
        if div != 0:
            return self._func(x,y) / div
        else:
            return 0.0


    def __call__(self, rec, recs):
        s = rec[self.b_attr]
        strings = recs[self.a_attr]

        if not isinstance(s, str):
            return pd.Series(np.nan, index=strings.index)

        strings = strings.apply(lambda x : str(x).lower() if pd.notnull(x) else None)
        return strings.apply(lambda x : self._sim_func(s, x) if x is not None else np.nan).astype(np.float64)

    def __str__(self):
        return f'smith_waterman({self.a_attr}, {self.b_attr})'
