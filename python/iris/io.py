"""Reading inputs, checking them against the input contract, and writing results."""
from __future__ import annotations

import json
import logging
import os

from ._core import files

log = logging.getLogger(__name__)


_INTEGRAL_TYPES = {'tinyint', 'smallint', 'int', 'bigint'}


def file_format(path):
    """Return `'parquet'`, `'csv'` or `'tsv'` for a file, by its extension; a directory or
    a URI without one is read as Parquet."""
    name = os.fspath(path).rstrip('/')
    low = name.lower()
    if low.endswith('.csv'):
        return 'csv'
    if low.endswith('.tsv'):
        return 'tsv'
    if low.endswith('.parquet') or files.scheme(name) or os.path.isdir(name):
        return 'parquet'
    raise ValueError(f'{name}: tables and pairs are read from a .parquet, .csv or .tsv file, '
                     f'or a directory of Parquet files')


def table_columns(path, names, id_col, columns):
    """Return the attribute columns read from a table, after checking the id column."""
    if id_col not in names:
        raise ValueError(f"{path}: no column '{id_col}' to use as the id; columns: {names}")
    if id_col != '_id' and '_id' in names:
        raise ValueError(f"{path}: has both '_id' and the id column '{id_col}'; read it with "
                         f"id_col='_id', or drop one of them")
    if columns is None:
        return [c for c in names if c != id_col]
    missing = [c for c in columns if c not in names]
    if missing:
        raise ValueError(f'{path}: no column(s) {missing}; columns: {names}')
    return [c for c in columns if c != id_col]


def pair_layout(path, names, is_list, columns, with_label):
    """Return how a file holds its pairs: flat, or grouped with one of the two id columns
    a list, as `(grouped, list_or_a, other_or_b, list_is_a)`."""
    a, b = columns
    if with_label and 'label' not in names:
        raise ValueError(f"{path}: no column 'label' to read with with_label=True; "
                         f'columns: {names}')
    if a in names and b in names:
        if is_list(a) and not is_list(b):
            return True, a, b, True
        if is_list(b) and not is_list(a):
            return True, b, a, False
        return False, a, b, None
    if tuple(columns) == ('id1', 'id2') and 'id2' in names and 'id1_list' in names:
        return True, 'id1_list', 'id2', True
    raise ValueError(f"{path}: has neither the columns {list(columns)} nor the grouped "
                     f"columns 'id2' and 'id1_list'; columns: {names}. Name the file's two id "
                     f"columns with columns=(...)")


def read_parquet(spark, path):
    """Return a Parquet file or directory as a Spark DataFrame, reading dates and
    timestamps as written."""
    return (spark.read.option('datetimeRebaseMode', 'CORRECTED')
                 .option('int96RebaseMode', 'CORRECTED').parquet(path))


def _is_spark(df):
    return hasattr(df, 'toPandas')


def _is_local(df):
    from ._local.inputs import is_local
    return is_local(df)


def _is_frame(df):
    return _is_spark(df) or _is_local(df)


def _columns(df):
    if _is_spark(df):
        return list(df.columns)
    from ._local.inputs import columns
    return columns(df)


def _all_local(*dfs):
    return all(_is_local(df) for df in dfs)


def _as_spark(df, what):
    if _is_spark(df):
        return df
    import pandas as pd
    import pyarrow as pa
    if isinstance(df, pa.Table):
        df = df.to_pandas()
    if isinstance(df, pd.DataFrame):
        from ._spark.io import as_spark
        return as_spark(df)
    if _is_local(df):
        raise TypeError(f'{what} is an Arrow dataset and other inputs are Spark '
                        f'DataFrames: pass all inputs local or all as Spark')
    raise TypeError(f'{what} must be a pandas or Spark DataFrame, got {type(df)}')


def _sample(rows, col='_cid', n=5):
    return [r[col] for r in rows[:n]]


def _listlike(x):
    import numpy as np
    return isinstance(x, (list, tuple, np.ndarray))


def _dangling_error(what, id_col, table_name, ids):
    return ValueError(
        f"{what} '{id_col}' references id(s) not present in {table_name}'s "
        f"'_id' column, e.g. {list(ids)}. Every pair must point at records that "
        f"exist.")


def _misaligned_error(what, col, n_bad, ids):
    return ValueError(
        f"{what} array column '{col}' must have the same length as 'id1_list' in "
        f"every row ({n_bad} row(s) differ; e.g. id2 {list(ids)}). Drop the column "
        f"or re-align it.")


def _shape_error(what, cols):
    return ValueError(
        f"{what} must be flat (columns 'id1', 'id2' -- one row per pair) or grouped "
        f"(columns 'id2', 'id1_list' -- one row per B record); got {sorted(cols)}")


def _warn_self_pairs(what, n, grouped):
    where = ("row(s) contain a self-pair (id2 in its own id1_list)" if grouped
             else 'self-pair(s) (id1 == id2)')
    log.warning(
        f'{what}: {n} {where}. Self-pairs featurize as maximum-similarity and '
        f'poison seeding; clean_candidates(..., drop_self_pairs=True) removes them.')


def _warn_duplicate_ids(what, n):
    log.warning(
        f"{what}: {n} row(s) contain duplicate ids within 'id1_list'; duplicates "
        f"can be labeled twice and leak between training and application sets. "
        f"clean_candidates removes them.")


def _candidate_shape(cols, what):
    cols = set(cols)
    if 'id1' in cols and 'id1_list' not in cols:
        if 'id2' not in cols:
            raise ValueError(f"flat {what} need an 'id2' column; got {sorted(cols)}")
        return 'flat'
    if {'id2', 'id1_list'} <= cols:
        return 'grouped'
    raise _shape_error(what, cols)


def check_tables(table_a, table_b):
    """Check that each table has a unique, non-null integer `_id` column, and return the
    row counts."""
    n_a = _validate_table(table_a, 'table_a')
    n_b = n_a if table_b is table_a else _validate_table(table_b, 'table_b')
    return {'table_a': n_a, 'table_b': n_b}


def _validate_table(df, name):
    if _all_local(df):
        from ._local import validate
        return validate.table(df, name)
    from ._spark.io import validate_table
    return validate_table(_as_spark(df, name), name)


def check_candidates(candidates, table_a, table_b, dedup=None, membership=True,
                        what='candidates'):
    """Check a candidate set, flat or grouped, against the tables and return its number
    of pairs."""
    if dedup is None:
        dedup = table_a is table_b
    if _all_local(candidates, table_a, table_b):
        from ._local import validate
        return validate.candidates(candidates, table_a, table_b, dedup, membership,
                                   what)
    from ._spark.io import validate_candidates
    return validate_candidates(
        _as_spark(candidates, what), _as_spark(table_a, 'table_a'),
        _as_spark(table_b, 'table_b'), dedup, membership, what)


def check_labeled_data(labeled_data, table_a, table_b, label_column='label',
                          dedup=None):
    """Check labeled pairs, flat or grouped, against the tables and return their
    number."""
    if not _is_frame(labeled_data):
        raise TypeError(f'labeled_data must be a pandas or Spark DataFrame, '
                        f'got {type(labeled_data)}')
    if label_column not in _columns(labeled_data):
        raise ValueError(f"labeled_data must have the column '{label_column}'; got "
                         f'{_columns(labeled_data)}')
    return check_candidates(labeled_data, table_a, table_b, dedup=dedup,
                               what='labeled_data')


def check_pairs(pairs, table_a, table_b, what='gold'):
    """Check that every pair of a flat `(id1, id2)` set names a record of each table,
    and return the number of pairs."""
    for c in ('id1', 'id2'):
        if not _is_frame(pairs):
            raise TypeError(f'{what} must be a pandas or Spark DataFrame, '
                            f'got {type(pairs)}')
        if c not in _columns(pairs):
            raise ValueError(f"{what} must have the column '{c}'; got "
                             f'{_columns(pairs)}')
    if _all_local(pairs, table_a, table_b):
        from ._local import validate
        return validate.pairs(pairs, table_a, table_b, what)

    from ._spark.io import check_pairs as check
    return check(_as_spark(pairs, what), _as_spark(table_a, 'table_a'),
                 _as_spark(table_b, 'table_b'), what)


FVS_CONTRACT = 1
FVS_META = '_iris.json'


def write_fvs_meta(path, features, pairs, output_col='feature_vectors'):
    """Write the description of a feature-vector directory: its format, its features in
    order and its pair count."""
    meta = {'contract': FVS_CONTRACT, 'features': [str(f) for f in features],
            'pairs': int(pairs), 'output_col': output_col}
    files.current().write_bytes(files.join(path, FVS_META), json.dumps(meta).encode())


def write_feature_vectors(fvs, path):
    """Write Spark feature vectors to Parquet, replacing any previous set."""
    fvs.write.mode('overwrite').parquet(str(path))


def check_gold_data(gold, table_a, table_b):
    """Check that every gold pair names a record of each table, and return the number of
    pairs."""
    return check_pairs(gold, table_a, table_b, what='gold')
