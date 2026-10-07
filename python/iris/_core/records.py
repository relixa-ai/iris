"""A table's records, preprocessed by every feature and packed into bytes, for both
engines."""
import numpy as np

from . import recpack

PREPROCESS_ROWS = 2_000


def preprocess(frame, features, for_a, for_b):
    """Return every feature's preprocessing of a frame of text records."""
    if for_a:
        for f in features:
            frame = f.preprocess(frame, True)
    if for_b:
        for f in features:
            frame = f.preprocess(frame, False)
    return frame


def pack_rows(frame, layout, features, for_a, for_b):
    """Yield the records of a frame packed into bytes, a slice of rows at a time."""
    for lo in range(0, len(frame), PREPROCESS_ROWS):
        chunk = frame.iloc[lo:lo + PREPROCESS_ROWS].set_index('_id')
        chunk = preprocess(chunk, features, for_a, for_b)
        data, ends = recpack.pack(layout, chunk)
        yield chunk.index.to_numpy(np.int64), data, ends


def feature_matrix(features, a_frame, a_local, b_frame, b_local, fill_na):
    """Return the feature vector of each pair as a row of a float32 matrix, computed on
    record frames."""
    f_mat = np.stack(
        [np.asarray(f.pairs_call(a_frame, a_local, b_frame, b_local), dtype=np.float64)
         for f in features], axis=-1).astype(np.float32)
    if fill_na is not None:
        np.nan_to_num(f_mat, copy=False, nan=fill_na)
    return f_mat


def vectors(f_mat):
    """Return a feature matrix as an Arrow `list<float>` array with one row per pair."""
    import pyarrow as pa

    n, k = f_mat.shape
    return pa.ListArray.from_arrays(
        pa.array(np.arange(0, n * k + 1, k, dtype=np.int32)),
        pa.array(np.ascontiguousarray(f_mat).ravel(), type=pa.float32()))
