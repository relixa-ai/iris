"""Helpers shared by both engines: logging, bounds on collected rows, token hashing and
the labeled-data checkpoint."""
import pandas as pd
import numpy as np
import numba as nb
from random import randint
import mmh3
import logging
import pyarrow as pa
import pyarrow.parquet as pq
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from math import floor

from . import files
from .frames import is_spark



MAX_DRIVER_ROWS_ENV = 'IRIS_MAX_DRIVER_ROWS'
DEFAULT_MAX_DRIVER_ROWS = 20_000_000


def max_driver_rows():
    """Return the most rows that may be collected into the driving process, from
    `IRIS_MAX_DRIVER_ROWS`."""
    try:
        return int(os.environ.get(MAX_DRIVER_ROWS_ENV, DEFAULT_MAX_DRIVER_ROWS))
    except ValueError:
        return DEFAULT_MAX_DRIVER_ROWS


def driver_room():
    """Return the bytes the driving process may collect, and where the figure comes from:
    its share of the memory budget, or a running Spark driver's heap when smaller."""
    from .. import resources
    from .frames import active_spark_context

    total_gb, _, source = resources.budget()
    room = total_gb * 2**30 * resources.SHARE
    sc = active_spark_context()
    if sc is not None and getattr(sc, '_jvm', None) is not None:
        heap = int(sc._jvm.java.lang.Runtime.getRuntime().maxMemory())
        if heap < room:
            return heap, 'the Spark driver heap, spark.driver.memory'
    return room, source


def check_driver_collect(n_rows, what, remedy, row_bytes=None):
    """Raise an error when collecting `n_rows` would exceed the bound on collected rows,
    or, given the bytes of a row, the memory of the driver."""
    limit = max_driver_rows()
    if n_rows is not None and n_rows > limit:
        raise RuntimeError(
            f'{what} would move {n_rows:,} rows onto the driver, above the limit of '
            f'{limit:,}. {remedy} Raise the limit with {MAX_DRIVER_ROWS_ENV}=<n> only if '
            f'the driver really can hold this frame.'
        )
    if n_rows is None or not row_bytes:
        return
    need = n_rows * row_bytes
    room, source = driver_room()
    if need > room:
        raise RuntimeError(
            f'{what} would move {n_rows:,} rows of about {row_bytes:,.0f} bytes onto the '
            f'driver, about {need / 2**30:.1f} GiB, more than the {room / 2**30:.1f} GiB '
            f'the driver may hold ({source}). {remedy}'
        )


def feature_vector_width(df, col_name='feature_vectors'):
    """Return the width of one feature vector in `df`, or None when it cannot be read
    cheaply."""
    try:
        if isinstance(df, pd.DataFrame):
            if len(df) == 0:
                return None
            first = df[col_name].iloc[0]
            return len(first) if hasattr(first, '__len__') else None
        if is_spark(df):
            from .._spark.utils import vector_width
            return vector_width(df, col_name)
        import pyarrow as pa
        if isinstance(df, pa.Table):
            if df.num_rows == 0:
                return None
            first = df.column(col_name)[0]
            return len(first) if first.is_valid else None
    except (KeyError, IndexError, TypeError):
        return None
    return None


def check_training_data_matches(training_data, expected_width, checkpoint_path,
                                col_name='feature_vectors'):
    """Raise an error when a labeled-data checkpoint has feature vectors of another
    width."""
    if training_data is None or expected_width is None or len(training_data) == 0:
        return
    found = feature_vector_width(training_data, col_name)
    if found is None or found == expected_width:
        return
    raise ValueError(
        f'existing training data at {checkpoint_path!r} has {found}-element feature '
        f'vectors but this run produces {expected_width}. That file is treated as a '
        f'resume checkpoint, so it is almost certainly left over from a different '
        f'dataset or feature set. Delete it, or pass checkpoint_path=<new path> to '
        f'start fresh.'
    )


def type_check(var, var_name, expected):
    """Raise TypeError when `var` is not of the expected type."""
    if not isinstance(var, expected):
        raise TypeError(f'{var_name} must be type {expected} (got {type(var)})')

def type_check_iterable(var, var_name, expected_var_type, expected_element_type):
    """Raise TypeError when `var` or one of its elements is not of the expected type."""
    type_check(var, var_name, expected_var_type)
    for e in var:
        if not isinstance(e, expected_element_type):
            raise TypeError(f'all elements of {var_name} must be type '
                            f'{expected_element_type} (got {type(e)})')

def is_null(o):
    """Return whether `o` is None or a floating-point NaN."""
    r = pd.isnull(o)
    return r if isinstance(r, bool) else False

def get_logger(name):
    """Return the logger for a module, which inherits its level and handlers from the
    program's logging configuration."""
    return logging.getLogger(name)

class SparseVec: 
    """A sparse float32 vector of indexes and values."""
    def __init__(self, size, indexes, values):    
        self._size = size    
        self._indexes = indexes.astype(np.int32)
        self._values = values.astype(np.float32)

    @property
    def indexes(self):
        """The indexes of the non-zero entries."""
        return self._indexes

    @property
    def values(self):
        """The values of the non-zero entries."""
        return self._values
    
    def dot(self, other):    
        """Return the dot product with another sparse vector."""
        return _sparse_dot(self._indexes, self._values, other._indexes, other._values)

@nb.njit('float32(int32[::1], float32[::1],int32[::1], float32[::1])')
def _sparse_dot(l_ind, l_val, r_ind, r_val):
    l = 0
    r = 0
    s = 0.0

    while l < l_ind.size and r < r_ind.size:
        li = l_ind[l]
        ri = r_ind[r]
        if li == ri:
            s += l_val[l] * r_val[r]
            l += 1
            r += 1
        elif li < ri:
            l += 1
        else:
            r += 1

    return s

class PerfectHashFunction:

    """A seeded 64-bit hash function without collisions on a given set of keys."""
    def __init__(self, seed=None):
        self._seed = seed if seed is not None else randint(0, 2**31)
    

    @classmethod
    def create_for_keys(cls, keys):
        """Return a hash function without collisions on `keys`, and the hashes of the
        keys."""
        if len(set(keys)) != len(keys):
            raise ValueError('keys must be unique')
        hashes = {}

        MAX_RETRIES = 10
        for i in range(MAX_RETRIES):
            hashes.clear()
            hash_func = cls()
            hash_vals = map(hash_func.hash, keys)
            for h, k in zip(hash_vals, keys):
                if h in hashes:
                    break
                hashes[h] = k
            if len(hashes) == len(keys):
                break
        else:
            raise RuntimeError(f'max retries ({MAX_RETRIES}) exceeded')

        return hash_func, np.fromiter(hashes.keys(), dtype=np.int64, count=len(hashes))
    
    def hash(self, s):
        """Return the 64-bit hash of `s`, a string hashed as its UTF-8 bytes."""
        if isinstance(s, str):
            s = s.encode('utf-8', 'surrogatepass')
        return mmh3.hash64(s, self._seed)[0]


TOKEN_HASH_SEED = 0x4D617463
TOKEN_HASH = PerfectHashFunction(seed=TOKEN_HASH_SEED)

_TOKEN_CACHE = ContextVar('iris_token_cache', default=None)


@contextmanager
def token_cache(frame):
    """Keep the token hashes computed for the columns of `frame` until the block ends."""
    reset = _TOKEN_CACHE.set((frame, {}))
    try:
        yield
    finally:
        _TOKEN_CACHE.reset(reset)


def token_hashes(data, input_col, tokenizer):
    """Return the sorted distinct token hashes and their counts for each record of a
    column, computed once per column and tokenizer inside `token_cache(data)`."""
    cache = _TOKEN_CACHE.get()
    memo = cache[1] if cache is not None and cache[0] is data else None
    key = tokenizer.out_col_name(input_col)
    hit = None if memo is None else memo.get(key)
    if hit is None:
        hit = data[input_col].apply(lambda s: _hash_tokens(s, tokenizer))
        if memo is not None:
            memo[key] = hit
    return hit


def _hash_tokens(s, tokenizer):
    hashes = getattr(tokenizer, 'token_hashes', None)
    if hashes is not None:
        h = hashes(s, TOKEN_HASH_SEED)
        return None if h is None else np.unique(h, return_counts=True)
    toks = tokenizer.tokenize(s)
    if toks is None:
        return None
    h = np.fromiter((TOKEN_HASH.hash(t) for t in toks), dtype=np.int64, count=len(toks))
    return np.unique(h, return_counts=True)

def check_writable(path):
    """Raise an error now when a local checkpoint at `path` could not be written."""
    if files.scheme(path) is not None:
        return
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        raise FileNotFoundError(f'the directory of the checkpoint {path!r} does not exist')
    if not os.access(directory, os.W_OK) or (os.path.exists(path)
                                             and not os.access(path, os.W_OK)):
        raise PermissionError(f'the checkpoint {path!r} cannot be written; pass a '
                              f'checkpoint_path in a directory this process may write to')


def save_training_data_streaming(new_batch, checkpoint_path, logger=None):
    """Append a batch of labeled pairs to the labeled-data checkpoint, rewriting the
    file atomically."""
    if logger is None:
        logger = get_logger(__name__)
        
    required_columns = ['id1', 'id2', 'feature_vectors', 'label']
    new_batch_clean = new_batch[required_columns].copy().reset_index(drop=True)
    def to_float64_list(x):
        if isinstance(x, np.ndarray):
            return x.astype(np.float64).tolist()
        elif isinstance(x, list):
            return [np.float64(i) for i in x]
        else:
            return [np.float64(i) for i in list(x)]
    
    new_batch_clean['feature_vectors'] = new_batch_clean['feature_vectors'].apply(to_float64_list)
        
    new_table = pa.Table.from_pandas(new_batch_clean, preserve_index=False)
    vectors = new_table.schema.get_field_index('feature_vectors')
    new_table = new_table.set_column(vectors, 'feature_vectors',
                                     new_table.column(vectors).cast(pa.list_(pa.float64())))

    if files.current().exists(checkpoint_path):
        existing_table = _read_parquet(checkpoint_path).select(required_columns)
        existing_table = existing_table.cast(new_table.schema)
        out_table = pa.concat_tables([existing_table, new_table])
        action = 'Appended'
    else:
        out_table = new_table
        action = 'Created'

    _atomic_write_parquet(out_table, checkpoint_path)
    logger.info(f'{action} {len(new_batch)} labeled pairs '
                f'({out_table.num_rows} total) in {checkpoint_path}')


class CheckpointWriter:
    """Labeled pairs appended to the labeled-data checkpoint in order, written once the
    time since the last write is as long as that write took, and on `flush`."""

    def __init__(self, checkpoint_path, logger=None):
        self._path = checkpoint_path
        self._logger = logger
        self._held = []
        self._written = time.monotonic()
        self._took = 0.0

    def add(self, row):
        """Hold a labeled pair, and write the pairs held once the time since the last
        write is as long as that write took."""
        self._held.append(row)
        if time.monotonic() - self._written >= self._took:
            self.flush()

    def flush(self):
        """Write the labeled pairs held."""
        if not self._held:
            return
        start = time.monotonic()
        save_training_data_streaming(pd.DataFrame(self._held), self._path, self._logger)
        self._held = []
        self._written = time.monotonic()
        self._took = self._written - start


def load_training_data_streaming(checkpoint_path, logger=None):
    """Return the labeled pairs in the labeled-data checkpoint, or None when there is no
    file."""
    if logger is None:
        logger = get_logger(__name__)
        
    if not files.current().exists(checkpoint_path):
        return None

    try:
        table = _read_parquet(checkpoint_path)
    except Exception as e:
        raise RuntimeError(
            f'Existing training data at {checkpoint_path} could not be read: {e}'
        ) from e

    training_data = table.to_pandas()

    required_columns = ['id1', 'id2', 'feature_vectors', 'label']
    missing = [c for c in required_columns if c not in training_data.columns]
    if missing:
        raise ValueError(
            f'Training data at {checkpoint_path} is missing required column(s): {missing}')

    training_data = training_data[required_columns].copy()

    for col in ['id1', 'id2']:
        if training_data[col].dtype != 'int64':
            training_data[col] = training_data[col].astype('int64')

    if training_data['label'].dtype != 'float64':
        training_data['label'] = training_data['label'].astype('float64')

    def to_float_list(x):
        if isinstance(x, np.ndarray):
            return x.astype(float).tolist()
        elif isinstance(x, list):
            return [float(i) for i in x]
        else:
            return list(x)
    training_data['feature_vectors'] = training_data['feature_vectors'].apply(to_float_list)

    logger.info(f'Loaded {len(training_data)} labeled pairs from {checkpoint_path}')
    return training_data


def _atomic_write_parquet(table, checkpoint_path):
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    files.current().write_bytes(checkpoint_path, sink.getvalue().to_pybytes())


def _read_parquet(checkpoint_path):
    return pq.read_table(pa.BufferReader(files.current().read_bytes(checkpoint_path)))


def adjust_iterations_for_existing_data(existing_data_size, n_fvs, batch_size, max_iter):
    """Return the iterations of batch active learning left after the labels already
    collected."""
    completed_iterations = floor(existing_data_size / batch_size)
    remaining_iterations = max_iter - completed_iterations    
    return max(0, remaining_iterations)


def adjust_labeled_examples_for_existing_data(existing_data_size, max_labeled):
    """Return the labels continuous active learning may still collect after those
    already collected."""
    if existing_data_size >= max_labeled:
        return 0
    return max_labeled - existing_data_size


def convert_arrays_for_spark(df):
    """Return a pandas DataFrame with its array and list columns converted to forms
    Spark accepts."""
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return df
    
    df = df.copy()

    for col in df.columns:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        sample_value = non_null.iloc[0]

        if hasattr(sample_value, 'tolist') and hasattr(sample_value, 'dtype'):
            df[col] = df[col].apply(
                lambda x: x.tolist() if hasattr(x, 'tolist') else x
            )
        elif isinstance(sample_value, list):
            if len(sample_value) > 0 and hasattr(sample_value[0], 'tolist'):
                df[col] = df[col].apply(
                    lambda x: [item.tolist() if hasattr(item, 'tolist') else item for item in x] if isinstance(x, list) else x
                )

    return df
