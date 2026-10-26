"""Choosing features and computing feature vectors on Spark."""
from contextlib import nullcontext
from typing import List, Optional, Callable, Any, Union

import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame as SparkDataFrame
from .io import as_spark, check_unique_ids
from .text import text_columns
from .._core.textform import resolve_timezone
from .._core import feature_select, pool, recpack
from .vocab import build_vocabularies
from . import pairpass
from .._core.utils import check_driver_collect, get_logger
from .session import check_retries
from .utils import (bounded_batches, check_java_version, check_row_groups, column,
                    lower_batches, map_in_pandas, parquet_row_bytes, parquet_row_groups,
                    partition_count, repartition_df, row_bytes_column, ship_arrays,
                    table_stats)

from .._core.feature.catalog import BuildCache


log = get_logger(__name__)

PANDAS_ROW_BYTES = 200


def _null_counts(df):
    float_cols = {f.name for f in df.schema
                  if isinstance(f.dataType, (T.DoubleType, T.FloatType))}
    checks = [F.sum(((column(c).isNull() | F.isnan(column(c))) if c in float_cols
                     else column(c).isNull()).cast('long')).alias(c) for c in df.columns]
    row = df.agg(F.count(F.lit(1)).alias('__rows'),
                 F.sum(row_bytes_column(df)).alias('__bytes'), *checks).head()
    rows = int(row['__rows'])
    mean = (row['__bytes'] or 0) / max(rows, 1)
    return rows, {c: int(row[c] or 0) for c in df.columns}, mean


def _text_counts(df, token_cols, flags):
    names = list(token_cols) + list(flags)
    fields = T.StructType([T.StructField(f'{n}#{k}', T.LongType())
                           for n in names for k in ('n', 's')])

    def count(chunks):
        for pdf in chunks:
            c = feature_select.chunk_counts(pdf, token_cols, flags)
            yield pd.DataFrame({f'{n}#{k}': [c[n][i]] for n in names
                                for i, k in enumerate(('n', 's'))})

    if not names:
        return {}
    row = map_in_pandas(df, count, fields) \
            .agg(*[F.sum(column(f'{n}#{k}')).alias(f'{n}#{k}')
                   for n in names for k in ('n', 's')]).head()
    return {n: (int(row[f'{n}#n'] or 0), int(row[f'{n}#s'] or 0)) for n in names}


def _longest(df, cols, names):
    strings = [(c, name) for c, name in zip(cols, names)
               if isinstance(df.schema[c].dataType, T.StringType)]
    if not strings:
        return {}
    row = df.agg(*[F.max(F.length(df[c])).alias(f'_{k}')
                   for k, (c, _) in enumerate(strings)]).head()
    return {name: int(row[f'_{k}'] or 0) for k, (_, name) in enumerate(strings)}


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
    feature_select.check_sim_functions(sim_functions)
    A_in = A
    tz = resolve_timezone(timezone)
    if not isinstance(A, (pd.DataFrame, SparkDataFrame)):
        raise TypeError(f'A must be a pandas or Spark DataFrame (got {type(A)})')
    if B is not None and not isinstance(B, (pd.DataFrame, SparkDataFrame)):
        raise TypeError(f'B must be a pandas or Spark DataFrame (got {type(B)})')
    a_cols, b_cols = list(a_cols), list(b_cols)
    feature_select.check_columns(list(A.columns), a_cols, 'A')
    if B is not None:
        feature_select.check_columns(list(B.columns), b_cols, 'B')
    if isinstance(A, pd.DataFrame):
        A = as_spark(A)
    if isinstance(B, pd.DataFrame):
        B = A if B is A_in else as_spark(B)
    check_retries(A.sparkSession)
    df = A.select(*map(column, a_cols))
    if B is not None:
        df = df.unionAll(B.select(*map(column, b_cols)))

    per_row = _parquet_layout(A, B, a_cols, b_cols, 'create_features')
    with bounded_batches(A.sparkSession, per_row) if per_row else nullcontext():
        rows, nulls, text_bytes = _null_counts(df)
    cols_to_keep = feature_select.kept_columns(df.columns, rows, nulls, null_threshold)
    df = df.select(*map(column, cols_to_keep))

    typed_numeric = [f.name for f in df.schema if isinstance(f.dataType, T.NumericType)]
    flags = feature_select.numeric_flags(df.columns, typed_numeric, numeric_threshold)
    parts = partition_count(rows, rows * text_bytes)
    b_of = dict(zip(a_cols, b_cols))
    df = text_columns(A.select(*map(column, cols_to_keep)).repartition(parts),
                      cols_to_keep, tz)
    if B is not None:
        b_keep = [b_of[c] for c in cols_to_keep]
        df = df.unionAll(text_columns(B.select(*map(column, b_keep)).repartition(parts),
                                      b_keep, tz).toDF(*cols_to_keep))

    token_cols = feature_select.token_columns(df.columns, tokenizers)
    with bounded_batches(df.sparkSession, text_bytes):
        counts = _text_counts(df, token_cols, flags)
    numeric_cols = feature_select.numeric_columns(cols_to_keep, typed_numeric, counts,
                                                  flags, numeric_threshold)
    features = feature_select.choose(cols_to_keep, numeric_cols, token_cols, counts,
                                     sim_functions, min_doc_freq, max_vocab_size)
    aligned = feature_select.alignment_columns(features)
    if aligned:
        with bounded_batches(A.sparkSession, text_bytes):
            longest_a = _longest(A, aligned, aligned)
            longest_b = longest_a if B is None or B is A else _longest(
                B, [b_of[c] for c in aligned], aligned)
        feature_select.warn_long_alignments(features, longest_a, longest_b)
    return features


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
    check_retries(spark)
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
            row_bytes=8 * len(features) + PANDAS_ROW_BYTES,
        )
    _release_kept()
    built = _build_records(A, B, features, tz)
    frames = list({id(f): f for f in (built[0], built[2])}.values())
    kept = False
    try:
        means = pairpass.mean_records(built[0], built[2])
        fvs = _pair_pass(candidates, built, features, fill_na, output_col,
                         one_table=B is None, means=means)
        pass_bytes = pairpass.stream_row_bytes(*means, len(features))
        if out is None and not return_pandas:
            lower_batches(spark, pass_bytes)
            _KEPT.extend(frames)
            kept = True
            return fvs
        with bounded_batches(spark, pass_bytes):
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
        return fvs
    finally:
        if not kept:
            for frame in frames:
                frame.unpersist()


_KEPT = []


def _release_kept():
    while _KEPT:
        frame = _KEPT.pop()
        if frame.sparkSession.sparkContext._jsc is not None:
            frame.unpersist()


def _get_pos_cor_features(features):
    return [int(p) for p in pool.positive_features([str(f) for f in features])]


def _text_tables(A, B, features, tz, stats=(None, None)):
    A = _prepreprocess_table(
        A, _record_columns(A, 'table A', [f.a_attr for f in features]
                           + ([f.b_attr for f in features] if B is None else [])), tz,
        stats[0]).persist(StorageLevel.DISK_ONLY)
    if B is not None:
        B = _prepreprocess_table(
            B, _record_columns(B, 'table B', [f.b_attr for f in features]), tz,
            stats[1]).persist(StorageLevel.DISK_ONLY)
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


def _parquet_layout(A, B, a_cols, b_cols, what):
    groups = parquet_row_groups(A, a_cols)
    if B is not None and B is not A:
        groups += parquet_row_groups(B, b_cols)
    check_row_groups(A.sparkSession, groups, what)
    return parquet_row_bytes(groups)


def _record_stats(A, B, features):
    a_cols = _record_columns(A, 'table A', [f.a_attr for f in features]
                             + ([f.b_attr for f in features] if B is None else []))
    b_cols = None if B is None else _record_columns(B, 'table B',
                                                    [f.b_attr for f in features])
    per_row = _parquet_layout(A, B, a_cols, b_cols, 'featurize')
    with bounded_batches(A.sparkSession, per_row) if per_row else nullcontext():
        stats_a = table_stats(A, a_cols)
        return stats_a, None if B is None else table_stats(B, b_cols)


def _build_records(A, B, features, tz):
    stats = _record_stats(A, B, features)
    with bounded_batches(A.sparkSession, max(s[1] for s in stats if s is not None)):
        A, B = _text_tables(A, B, features, tz, stats)
        built = []
        try:
            check_unique_ids(A, 'table_a')
            if B is not None:
                check_unique_ids(B, 'table_b')
            _build_vocabularies(A, B, features)
            layout_a = recpack.layout(features, True, B is None)
            recs_a = pairpass.build_records(A, layout_a, features, True, B is None)
            built.append(recs_a)
            recs_a.count()
            if B is None:
                return recs_a, layout_a, recs_a, layout_a
            layout_b = recpack.layout(features, False, True)
            recs_b = pairpass.build_records(B, layout_b, features, False, True)
            built.append(recs_b)
            recs_b.count()
            return recs_a, layout_a, recs_b, layout_b
        except BaseException:
            for recs in built:
                recs.unpersist()
            raise
        finally:
            A.unpersist()
            if B is not None:
                B.unpersist()


def _pair_pass(candidates, built, features, fill_na=0.0, output_col='feature_vectors',
               one_table=False, means=(None, None)):
    recs_a, layout_a, recs_b, layout_b = built
    return pairpass.pair_pass(canonical_pairs(candidates, one_table), recs_a, recs_b,
                              layout_a, layout_b, features, fill_na, *means)\
                   .withColumnRenamed('fv', output_col)


def _record_columns(df, what, wanted):
    wanted = set(wanted)
    missing = sorted(c for c in wanted if c not in df.columns)
    if missing:
        raise ValueError(f'features read column(s) {missing} that {what} does not have '
                         f'(columns: {df.columns})')
    return [c for c in df.columns if c in wanted and c != '_id']


def _prepreprocess_table(df, cols, tz, stats=None):
    df = repartition_df(df.select('_id', *map(column, cols)), '_id', stats=stats)
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
