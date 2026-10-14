"""Reading inputs, checking them against the input contract, and writing results."""
from __future__ import annotations

import json
import logging
import os

from ._core import files

log = logging.getLogger(__name__)


_INTEGRAL_TYPES = {'tinyint', 'smallint', 'int', 'bigint'}


PARQUET_MAGIC = b'PAR1'
PAIR_COLUMNS = (('id1', 'id2'), ('a_id', 'b_id'), ('ltable_id', 'rtable_id'),
                ('left_id', 'right_id'))
PAIR_ID_NAMES = tuple(n for pair in PAIR_COLUMNS for n in pair)


ONE_TABLE_MESSAGE = ('featurize: one table is matched against itself: {n_self:,} pair(s) of a '
                     'record with itself and {repeats:,} repeated or mirrored pair(s) are '
                     'dropped, and each pair is featurized once, as (smaller id, larger id)')


def file_format(path):
    """Return `'parquet'`, `'csv'`, `'tsv'` or `'jsonl'` for a file, by its name checked
    against its contents; a directory or a remote URI is Parquet."""
    name = os.fspath(path).rstrip('/')
    low = name.lower()
    s = files.scheme(name)
    local = files.local_path(name) if s in (None, 'file') else None
    if low.endswith('.csv'):
        fmt = 'csv'
    elif low.endswith('.tsv'):
        fmt = 'tsv'
    elif low.endswith('.jsonl'):
        fmt = 'jsonl'
    elif low.endswith('.parquet') or local is None or os.path.isdir(local):
        fmt = 'parquet'
    else:
        fmt = None
    if local is None or not os.path.isfile(local):
        if fmt is None:
            raise ValueError(f'{name}: tables and pairs are read from a .parquet, .csv, .tsv '
                             f'or .jsonl file, or a directory of Parquet files')
        return fmt
    with open(local, 'rb') as fh:
        head = fh.read(len(PARQUET_MAGIC))
    if head == PARQUET_MAGIC:
        if fmt in ('csv', 'tsv', 'jsonl'):
            raise ValueError(f'{name}: is named .{fmt} but holds Parquet data; rename it to '
                             f'end in .parquet')
        return 'parquet'
    if fmt == 'parquet':
        raise ValueError(f'{name}: is named .parquet but does not hold Parquet data; if it is '
                         f'CSV or TSV text, rename it to end in .csv or .tsv')
    if fmt is None:
        raise ValueError(f'{name}: tables and pairs are read from a .parquet, .csv, .tsv or '
                         f'.jsonl file, or a directory of Parquet files')
    return fmt


def parquet_name(path, what):
    """Return `path` as a string, refusing a name that ends in the extension of a format
    other than Parquet."""
    name = os.fspath(path)
    ext = os.path.splitext(name.rstrip('/'))[1].lower()
    if ext not in ('', '.parquet'):
        raise ValueError(f'{name}: {what} writes Parquet; name it so that it ends in .parquet')
    return name


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
    if with_label and 'label' not in names:
        raise ValueError(f"{path}: no column 'label' to read with with_label=True; "
                         f'columns: {names}')
    default = tuple(columns) == PAIR_COLUMNS[0]
    if default and 'id2' in names and 'id1_list' in names and 'id1' not in names:
        return True, 'id1_list', 'id2', True
    for a, b in (PAIR_COLUMNS if default else (tuple(columns),)):
        if a in names and b in names:
            if is_list(a) and not is_list(b):
                return True, a, b, True
            if is_list(b) and not is_list(a):
                return True, b, a, False
            return False, a, b, None
    if default:
        raise ValueError(f"{path}: has none of the pair columns id1/id2, a_id/b_id, "
                         f"ltable_id/rtable_id or left_id/right_id, nor the grouped columns "
                         f"'id2' and 'id1_list'; columns: {names}. Name the file's two id "
                         f"columns with columns=(...)")
    raise ValueError(f'{path}: has no columns {list(columns)}; columns: {names}')


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
        f'{what}: {n} {where}. featurize drops them when one table is matched against '
        f'itself, and clean_candidates(..., drop_self_pairs=True) removes them from the '
        f'candidate set.')


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
