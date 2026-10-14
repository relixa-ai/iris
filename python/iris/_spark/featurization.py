"""Choosing features and computing feature vectors on Spark."""
from typing import List, Optional, Callable, Any, Union

import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame as SparkDataFrame
from .io import as_spark
from .text import text_columns
from .._core.textform import resolve_timezone
from .._core import feature_select, pool, recpack
from .vocab import build_vocabularies
from . import pairpass
from .._core.utils import check_driver_collect, get_logger
from .utils import check_java_version, repartition_df, ship_arrays

from .._core.feature.catalog import BuildCache


log = get_logger(__name__)


def _null_counts(df):
    float_cols = {f.name for f in df.schema
                  if isinstance(f.dataType, (T.DoubleType, T.FloatType))}
    checks = [F.sum(((df[c].isNull() | F.isnan(df[c])) if c in float_cols
                     else df[c].isNull()).cast('long')).alias(c) for c in df.columns]
    row = df.agg(F.count(F.lit(1)).alias('__rows'), *checks).head()
    return int(row['__rows']), {c: int(row[c] or 0) for c in df.columns}


def _text_counts(df, token_cols, flags):
    names = list(token_cols) + list(flags)
    fields = [f'`{n}#{k}` long' for n in names for k in ('n', 's')]

    def count(chunks):
        for pdf in chunks:
            c = feature_select.chunk_counts(pdf, token_cols, flags)
            yield pd.DataFrame({f'{n}#{k}': [c[n][i]] for n in names
                                for i, k in enumerate(('n', 's'))})

    if not names:
        return {}
    row = df.mapInPandas(count, ', '.join(fields)) \
            .agg(*[F.sum(F.col(f'`{n}#{k}`')).alias(f'{n}#{k}')
                   for n in names for k in ('n', 's')]).head()
    return {n: (int(row[f'{n}#n'] or 0), int(row[f'{n}#s'] or 0)) for n in names}


def create_features(
    A: Union[pd.DataFrame, SparkDataFrame],
    B: Union[pd.DataFrame, SparkDataFrame],
    a_cols: List[str],
    b_cols: List[str],
    sim_functions: Optional[List[Callable[..., Any]]] = None,
    tokenizers: Optional[List[Callable[..., Any]]] = None,
    null_threshold: float = .5,
    numeric_threshold: Optional[float] = .9,
    min_doc_freq: int = 1,
    max_vocab_size: Optional[int] = None,
    timezone: Optional[str] = None,
) -> List[Callable]:
    """Return the features for matching the columns `a_cols` of table A with the columns
    `b_cols` of table B."""
    tokenizers = feature_select.default_tokenizers(tokenizers)
    feature_select.check_numeric_threshold(numeric_threshold)
    A_in = A
    tz = resolve_timezone(timezone)
    if isinstance(A, pd.DataFrame):
        A = as_spark(A)
    if isinstance(B, pd.DataFrame):
        B = A if B is A_in else as_spark(B)
    if not isinstance(A, SparkDataFrame):
        raise TypeError(f'A must be a pandas or Spark DataFrame (got {type(A)})')
    df = A.select(a_cols)
    if B is not None:
        df = df.unionAll(B.select(b_cols))

    union_df = df.persist()
    df = repartition_df(union_df, 5000, [])
    rows, nulls = _null_counts(df)
    cols_to_keep = feature_select.kept_columns(df.columns, rows, nulls, null_threshold)
    df = df.select(*cols_to_keep)

    typed_numeric = [f.name for f in df.schema if isinstance(f.dataType, T.NumericType)]
    flags = feature_select.numeric_flags(df.columns, typed_numeric, numeric_threshold)
    n_rows = union_df.count()
    parts = max(1, min(n_rows, max(n_rows // 5000,
                                   df.sparkSession.sparkContext.defaultParallelism * 4)))
    b_of = dict(zip(a_cols, b_cols))
    df = text_columns(A.select(cols_to_keep).repartition(parts), cols_to_keep, tz)
    if B is not None:
        b_keep = [b_of[c] for c in cols_to_keep]
        df = df.unionAll(text_columns(B.select(b_keep).repartition(parts), b_keep, tz)
                         .toDF(*cols_to_keep))

    token_cols = feature_select.token_columns(df.columns, tokenizers)
    counts = _text_counts(df, token_cols, flags)
    union_df.unpersist()
    numeric_cols = feature_select.numeric_columns(cols_to_keep, typed_numeric, counts,
                                                  flags, numeric_threshold)
    return feature_select.choose(cols_to_keep, numeric_cols, token_cols, counts,
                                 sim_functions, min_doc_freq, max_vocab_size)


def featurize(
    features: List[Callable],
    A,
    B,
    candidates,
    output_col: str = 'feature_vectors',
    fill_na: float = 0.0,
    validate: bool = True,
    timezone: Optional[str] = None,
    out: Optional[str] = None,
) -> Union[pd.DataFrame, SparkDataFrame]:
    """Return the feature vector and score of every candidate pair as a Spark
    DataFrame."""
    tz = resolve_timezone(timezone)
    if validate:
        from ..io import check_candidates
        check_candidates(candidates, A, A if B is None else B)
    return_pandas = False
    spark = SparkSession.builder.getOrCreate()
    check_java_version(spark)
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    return_pandas = isinstance(A, pd.DataFrame) and (B is None or isinstance(B, pd.DataFrame))
    A_in = A
    if isinstance(A, pd.DataFrame):
        A = as_spark(A, spark)
    if isinstance(B, pd.DataFrame):
        B = A if B is A_in else as_spark(B, spark)
    if isinstance(candidates, pd.DataFrame):
        candidates = as_spark(candidates, spark)
    if validate:
        _warn_duplicate_pairs(candidates, one_table=B is None)
    if return_pandas:
        n_pairs = (candidates.count() if is_flat_candidates(candidates)
                   else candidates.select(
                       F.coalesce(F.sum(F.size('id1_list')), F.lit(0)).alias('n')
                   ).head()[0])
        check_driver_collect(
            n_pairs, 'featurize with pandas A and B',
            'Pass A and B as Spark DataFrames to keep the feature vectors distributed.',
        )
    built = _build_records(A, B, features, tz)
    release = [built[0], built[2]]
    fvs = _pair_pass(candidates, built, features, fill_na, output_col, one_table=B is None)
    if out is not None or return_pandas:
        try:
            if out is not None:
                from .._core import files
                from ..io import write_feature_vectors, write_fvs_meta
                from .files import HadoopFiles
                write_feature_vectors(fvs, out)
                fvs = spark.read.parquet(str(out))
                with files.using(HadoopFiles(spark)):
                    write_fvs_meta(out, features, fvs.count(), output_col)
            if return_pandas:
                fvs = fvs.toPandas()
        finally:
            for frame in release:
                frame.unpersist()
    return fvs


def _get_pos_cor_features(features):
    return [int(p) for p in pool.positive_features([str(f) for f in features])]


def _text_tables(A, B, features, tz):
    A = _prepreprocess_table(
        A, _record_columns(A, 'table A', [f.a_attr for f in features]
                           + ([f.b_attr for f in features] if B is None else [])), tz)\
        .persist()
    if B is not None:
        B = _prepreprocess_table(
            B, _record_columns(B, 'table B', [f.b_attr for f in features]), tz).persist()
    return A, B


def _build_vocabularies(A, B, features):
    vocab_feats = [f for f in features if hasattr(f, 'vocab_request')]
    requests = list(dict.fromkeys(f.vocab_request() for f in vocab_feats))
    built = dict(zip(requests, build_vocabularies(A, B, requests)))
    for f in vocab_feats:
        v = built[f.vocab_request()]
        f.set_vocab(v.hashes, v.doc_freqs, v.corpus_size)
    cache = BuildCache()
    for f in features:
        if not hasattr(f, 'vocab_request'):
            f.build(A, B, cache)
    cache.clear()
    arrays = {id(m): m for f in features for m in getattr(f, 'memmaps', list)()}
    ship_arrays(arrays.values())


def _build_records(A, B, features, tz):
    A, B = _text_tables(A, B, features, tz)
    try:
        _build_vocabularies(A, B, features)
        layout_a = recpack.layout(features, True, B is None)
        recs_a = pairpass.build_records(A, layout_a, features, True, B is None)
        recs_a.count()
        if B is None:
            return recs_a, layout_a, recs_a, layout_a
        layout_b = recpack.layout(features, False, True)
        recs_b = pairpass.build_records(B, layout_b, features, False, True)
        recs_b.count()
        return recs_a, layout_a, recs_b, layout_b
    finally:
        A.unpersist()
        if B is not None:
            B.unpersist()


def _pair_pass(candidates, built, features, fill_na=0.0, output_col='feature_vectors',
               one_table=False):
    recs_a, layout_a, recs_b, layout_b = built
    return pairpass.pair_pass(canonical_pairs(candidates, one_table), recs_a, recs_b,
                                    layout_a, layout_b, features, fill_na)\
                         .withColumnRenamed('fv', output_col)


def _record_columns(df, what, wanted):
    wanted = set(wanted)
    missing = sorted(c for c in wanted if c not in df.columns)
    if missing:
        raise ValueError(f'features read column(s) {missing} that {what} does not have '
                         f'(columns: {df.columns})')
    return [c for c in df.columns if c in wanted and c != '_id']


def _prepreprocess_table(df, cols, tz):
    part_size = 5000
    df = repartition_df(df, part_size, '_id').select('_id', *cols)
    return text_columns(df, cols, tz)


def is_flat_candidates(pairs):
    """Return whether a candidate set has one row per pair."""
    names = set(pairs.columns)
    return 'id1' in names and 'id1_list' not in names


def canonical_pairs(candidates, one_table=False):
    """Return a candidate set, flat or grouped, as flat `(id1, id2)` pairs; for one table,
    without self-pairs and with `id1` the smaller id."""
    if is_flat_candidates(candidates):
        pairs = candidates.select(F.col('id1').cast('long').alias('id1'),
                                  F.col('id2').cast('long').alias('id2'))
    else:
        pairs = candidates.select(F.explode('id1_list').alias('id1'), 'id2') \
                          .select(F.col('id1').cast('long').alias('id1'),
                                  F.col('id2').cast('long').alias('id2'))
    if not one_table:
        return pairs
    return pairs.filter(F.col('id1') != F.col('id2')).select(
        F.least('id1', 'id2').alias('id1'), F.greatest('id1', 'id2').alias('id2'))


def _warn_duplicate_pairs(candidates, one_table=False):
    if one_table:
        from ..io import ONE_TABLE_MESSAGE
        n = canonical_pairs(candidates).count()
        pairs = canonical_pairs(candidates, one_table=True)
        row = pairs.agg(F.count(F.lit(1)).alias('k'),
                        F.count_distinct('id1', 'id2').alias('d')).head()
        n_self, repeats = n - int(row['k']), int(row['k']) - int(row['d'])
        if n_self or repeats:
            log.warning(ONE_TABLE_MESSAGE.format(n_self=n_self, repeats=repeats))
        return
    row = canonical_pairs(candidates).agg(
        F.count(F.lit(1)).alias('n'), F.count_distinct('id1', 'id2').alias('d')).head()
    n, d = int(row['n']), int(row['d'])
    if n > d:
        log.warning(f'featurize: {n - d:,} of {n:,} candidate pairs are repeats and are '
                    f'dropped; each distinct pair is featurized once')
