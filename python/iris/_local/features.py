"""Choosing the features on the local engine, from counts made by worker processes."""
from functools import partial

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from .._core import feature_select
from .._core.textform import resolve_timezone, to_text
from . import inputs
from .procpool import run_tasks, workers_for

WORKER_GB = 1.0


def _kind(t):
    if pa.types.is_floating(t):
        return 'float'
    if pa.types.is_integer(t) or pa.types.is_decimal(t):
        return 'number'
    return 'other'


def _union_kind(a_type, b_type):
    kinds = {_kind(a_type)} if b_type is None else {_kind(a_type), _kind(b_type)}
    if 'other' in kinds:
        return 'other'
    return 'float' if 'float' in kinds else 'number'


def _null_counts(obj, cols, names, float_names, rows, nulls):
    for batch in inputs.arrow_batches(obj, cols):
        rows += batch.num_rows
        for col, name in zip(cols, names):
            arr = batch.column(col)
            n = arr.null_count
            if name in float_names and pa.types.is_floating(arr.type):
                n += pc.sum(pc.is_nan(arr)).as_py() or 0
            nulls[name] += n
    return rows


def _count_chunk(task, tz, token_cols, flags):
    chunk, cols, names = task
    table = inputs.read_chunk(chunk, cols)
    counts = {}
    for part in inputs.slices(table):
        frame = pd.DataFrame({name: to_text(part.column(col), tz, col).to_pandas()
                              for col, name in zip(cols, names)})
        feature_select.add_counts(counts, feature_select.chunk_counts(frame, token_cols,
                                                                      flags))
    return counts


def create_features(A, B, a_cols, b_cols, sim_functions=None, tokenizers=None,
                    null_threshold=.5, numeric_threshold=.9, min_doc_freq=1,
                    max_vocab_size=None, timezone=None, workers=None):
    """Return the features for matching the columns `a_cols` of table A with the columns
    `b_cols` of table B."""
    tokenizers = feature_select.default_tokenizers(tokenizers)
    feature_select.check_numeric_threshold(numeric_threshold)
    tz = resolve_timezone(timezone)
    a_cols, b_cols = list(a_cols), list(b_cols)
    b_of = dict(zip(a_cols, b_cols))
    a_schema = inputs.schema(A)
    b_schema = None if B is None else inputs.schema(B)
    kinds = {c: _union_kind(a_schema.field(c).type,
                            None if B is None else b_schema.field(b_of[c]).type)
             for c in a_cols}
    floats = {c for c, k in kinds.items() if k == 'float'}

    nulls = dict.fromkeys(a_cols, 0)
    rows = _null_counts(A, a_cols, a_cols, floats, 0, nulls)
    if B is not None:
        rows = _null_counts(B, b_cols, a_cols, floats, rows, nulls)
    kept = feature_select.kept_columns(a_cols, rows, nulls, null_threshold)
    for c in kept:
        to_text(pa.array([], type=a_schema.field(c).type), tz, c)
        if B is not None:
            to_text(pa.array([], type=b_schema.field(b_of[c]).type), tz, b_of[c])
    typed_numeric = [c for c in kept if kinds[c] != 'other']
    flags = feature_select.numeric_flags(kept, typed_numeric, numeric_threshold)
    token_cols = feature_select.token_columns(kept, tokenizers)

    counts = {}
    if kept and (token_cols or flags):
        with inputs.scratch() as tmp:
            a = inputs.stage(A, tmp, 'a')
            tasks = [(c, kept, kept) for c in inputs.chunks(a)]
            if B is not None:
                b = a if B is A else inputs.stage(B, tmp, 'b')
                tasks += [(c, [b_of[k] for k in kept], kept) for c in inputs.chunks(b)]
            fn = partial(_count_chunk, tz=tz, token_cols=token_cols, flags=flags)
            for part in run_tasks(fn, tasks, workers or workers_for(WORKER_GB)):
                feature_select.add_counts(counts, part)
    numeric = feature_select.numeric_columns(kept, typed_numeric, counts, flags,
                                             numeric_threshold)
    return feature_select.choose(kept, numeric, token_cols, counts, sim_functions,
                                 min_doc_freq, max_vocab_size)
