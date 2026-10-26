"""The Spark side of the public functions in `iris.tools`."""
import json
import numbers
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyspark.sql.functions as F
from pyspark import StorageLevel
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, StructField, StructType
from pyspark.sql.window import Window

from .._core import pool
from .._core import seeds as seed_rules
from .._core.utils import (MAX_DRIVER_ROWS_ENV, check_driver_collect,
                           convert_arrays_for_spark, get_logger,
                           max_driver_rows as _max_driver_rows)
from .active_learning.cont_entropy_active_learner import ContinuousEntropyActiveLearner
from .active_learning.ent_active_learner import EntropyActiveLearner
from .featurization import canonical_pairs
from .io import as_spark
from .utils import check_java_version, column, row_bytes

logger = get_logger(__name__)

PANDAS_ROW_BYTES = 200

READ_PIECE_BYTES = 64 * 2**20


def clean_candidates(candidates, drop_self_pairs, canonicalize_mirrors):
    """Return a Spark candidate set as distinct flat pairs, without self-pairs and
    mirrored pairs when requested."""
    out = canonical_pairs(candidates)
    if drop_self_pairs:
        out = out.filter(F.col('id1') != F.col('id2'))
    if canonicalize_mirrors:
        out = out.select(F.least('id1', 'id2').alias('id1'),
                         F.greatest('id1', 'id2').alias('id2'))
    return out.distinct()


def down_sample(fvs, percent, score_column, bucket_size, max_pool_size, seed):
    """Return the labeling pool of Spark feature vectors."""
    total = fvs.count()
    B, n, B_keep = pool.plan(total, percent, bucket_size, max_pool_size)
    if max_pool_size is not None and int(total * percent) > max_pool_size:
        logger.info(f'down_sample: max_pool_size={max_pool_size} caps the pool '
                    f'(percent={percent:g} of {total} would be {int(total * percent)}); '
                    f'the score band stays at the top {percent:g}')
    rank = Window.partitionBy('_b').orderBy(F.col('_s').desc())
    draw = Window.partitionBy('_b', '_band').orderBy('_r', 'id2', 'id1')
    out = (fvs.withColumn('_b', F.pmod(F.xxhash64('id1', 'id2'), F.lit(B)))
              .filter(F.col('_b') < B_keep)
              .withColumn('_s', F.coalesce(column(score_column).cast('double'),
                                           F.lit(float('nan'))))
              .withColumn('_band', F.percent_rank().over(rank) <= F.lit(percent))
              .withColumn('_r', F.xxhash64(F.lit(seed).cast('long'), 'id1', 'id2'))
              .withColumn('_rn', F.row_number().over(draw))
              .filter(F.col('_rn') <= n))
    if max_pool_size is not None and 2 * n * B_keep > max_pool_size:
        out = out.orderBy('_r', 'id2', 'id1').limit(max_pool_size)
    return out.drop('_b', '_s', '_band', '_r', '_rn')


def rows_frame(rows):
    """Return rows held in the driver as a Spark DataFrame."""
    spark = SparkSession.builder.getOrCreate()
    return spark.createDataFrame([
        {k: (v.tolist() if isinstance(v, np.ndarray)
             else int(v) if isinstance(v, np.integer)
             else float(v) if isinstance(v, np.floating) else v)
         for k, v in row.items()} for row in rows])


def _whole_id(value):
    if not isinstance(value, numbers.Number):
        return None
    try:
        whole = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return whole if whole == value and -2**63 <= whole < 2**63 else None


def _gold_ids(lookup):
    if lookup._a is not None:
        return lookup._a, lookup._b
    pairs = [(x, y) for x, y in ((_whole_id(x), _whole_id(y)) for x, y in lookup._pairs)
             if x is not None and y is not None]
    id1 = np.fromiter((x for x, _ in pairs), np.int64, count=len(pairs))
    id2 = np.fromiter((y for _, y in pairs), np.int64, count=len(pairs))
    order = np.lexsort((id2, id1))
    return id1[order], id2[order]


def gold_scores(fvs, labeler, score_column):
    """Return Spark feature vectors with `score_column` set to 1.0 on the labeler's gold
    pairs and 0.0 elsewhere."""
    id1, id2 = _gold_ids(labeler._lookup)
    if not len(id1):
        return fvs.withColumn(score_column, F.lit(0.0))
    first = np.r_[True, (id1[1:] != id1[:-1]) | (id2[1:] != id2[:-1])]
    gold = as_spark(pd.DataFrame({'_gold_id1': id1[first], '_gold_id2': id2[first]}),
                    fvs.sparkSession).withColumn('_gold', F.lit(1.0))
    on = (fvs['id1'] == gold['_gold_id1']) & (fvs['id2'] == gold['_gold_id2'])
    return (fvs.join(gold, on, 'left').drop('_gold_id1', '_gold_id2')
               .withColumn(score_column, F.coalesce(F.col('_gold'), F.lit(0.0)))
               .drop('_gold'))


def seed_rankings(fvs, score_column, nseeds):
    """Return the likely matches and likely non-matches of Spark feature vectors, as row
    dictionaries."""
    score = column(score_column)
    valid = fvs.filter(score.isNotNull() & ~F.isnan(score)).persist()
    try:
        n_valid = valid.count()
        if nseeds > n_valid:
            raise seed_rules.too_few_scores(nseeds, n_valid, fvs.count(), score_column)
        k = seed_rules.scan_size(nseeds, n_valid)
        likely_pos = valid.sort(score.desc(), 'id2', 'id1').limit(k) \
                          .toPandas().to_dict('records')
        likely_neg = valid.sort(score.asc(), 'id2', 'id1').limit(k) \
                          .toPandas().to_dict('records')
    finally:
        valid.unpersist()
    return likely_pos, likely_neg


def _gold_frame(gold, spark):
    if isinstance(gold, SparkDataFrame):
        return gold
    if isinstance(gold, pd.DataFrame):
        return as_spark(gold[['id1', 'id2']], spark)
    if isinstance(gold, (pa.Table, ds.Dataset)):
        table = gold.select(['id1', 'id2']) if isinstance(gold, pa.Table) else \
            gold.to_table(columns=['id1', 'id2'])
        return as_spark(table.to_pandas(), spark)
    raise TypeError(f'gold must be a pandas DataFrame, an Arrow table or dataset, or a '
                    f'Spark DataFrame (got {type(gold)})')


def _not_whole(c):
    return F.isnan(c) | (F.abs(c) == F.lit(float('inf'))) | (c != F.rint(c))


def check_gold_ids(gold):
    """Check that the ids of Spark gold pairs are whole numbers without nulls."""
    fractional = {n for n, t in gold.dtypes if t in ('float', 'double')}
    aggs = [F.sum(F.col(n).isNull().cast('long')).alias(f'{n}_null') for n in ('id1', 'id2')]
    aggs += [F.sum(_not_whole(F.col(n)).cast('long')).alias(f'{n}_bad') for n in fractional]
    row = gold.agg(*aggs).head().asDict()
    for name in ('id1', 'id2'):
        if row[f'{name}_null']:
            raise ValueError(f"gold '{name}' holds {row[f'{name}_null']} null id(s)")
        if row.get(f'{name}_bad'):
            bad = [r[name] for r in gold.filter(_not_whole(F.col(name))).select(name)
                   .limit(5).collect()]
            raise ValueError(f"gold '{name}' must hold whole numbers, got {bad}")


def evaluate_counts(predictions, gold, threshold, dedup):
    """Return the gold count, the predicted count, the true positives and the
    probability histogram of Spark predictions, against gold pairs of either engine."""
    p_match = F.when(F.col('prediction') == 1.0, F.col('confidence')) \
               .otherwise(F.lit(1.0) - F.col('confidence'))
    pos = (F.col('prediction') == 1.0) if threshold == 0.5 else (p_match >= threshold)
    if gold is None:
        n_gold = tp = 0
        hist = None
        n_predicted = predictions.filter(pos).count()
    else:
        if isinstance(gold, SparkDataFrame):
            check_gold_ids(gold)
        gold = _gold_frame(gold, predictions.sparkSession)
        g = gold.select(F.col('id1').cast('long').alias('id1'),
                        F.col('id2').cast('long').alias('id2'))
        if dedup:
            g = g.select(F.least('id1', 'id2').alias('id1'),
                         F.greatest('id1', 'id2').alias('id2'))
        g = g.distinct().persist()
        try:
            n_gold = g.count()
            marked = (predictions
                      .join(g.withColumn('_g', F.lit(1)), ['id1', 'id2'], 'left')
                      .select(pos.alias('_pos'), p_match.alias('_p'),
                              F.coalesce(F.col('_g'), F.lit(0)).alias('_g')))
            counts = marked.agg(
                F.sum(F.when(F.col('_pos'), 1).otherwise(0)).alias('n'),
                F.sum(F.when(F.col('_pos') & (F.col('_g') == 1), 1).otherwise(0))
                 .alias('tp')).head()
            n_predicted, tp = int(counts['n'] or 0), int(counts['tp'] or 0)
            hist = {int(r['b']): (int(r['n']), int(r['pos'])) for r in
                    marked.withColumn('b', F.least(F.floor(F.col('_p') * 100),
                                                   F.lit(100)).cast('int'))
                          .groupBy('b').agg(F.count('*').alias('n'),
                                            F.sum('_g').alias('pos')).collect()}
        finally:
            g.unpersist()
    return n_gold, n_predicted, tp, hist


def label_al(model, mode, labeler, fvs, seeds, checkpoint_path, learner_kwargs):
    """Return the pairs labeled by active learning over a Spark pool."""
    from ..tools import create_seeds

    spark = SparkSession.builder.getOrCreate()
    check_java_version(spark)

    return_pandas = isinstance(fvs, pd.DataFrame)
    if isinstance(fvs, pd.DataFrame):
        fvs = as_spark(fvs, spark)

    fvs = fvs.persist(StorageLevel.MEMORY_AND_DISK)
    try:
        if seeds is None:
            seeds = create_seeds(
                fvs=fvs,
                nseeds=min(10, fvs.count()),
                labeler=labeler,
                score_column='score',
                checkpoint_path=checkpoint_path,
            )
        if isinstance(seeds, SparkDataFrame):
            seeds = seeds.toPandas()
        if mode == "batch":
            learner = EntropyActiveLearner(model, labeler, checkpoint_path=checkpoint_path, **learner_kwargs)
        elif mode == "continuous":
            learner = ContinuousEntropyActiveLearner(model, labeler, checkpoint_path=checkpoint_path, **learner_kwargs)
        else:
            raise ValueError(f"mode must be either 'batch' or 'continuous', not {mode}")
        labeled_data = learner.train(fvs, seeds)
    finally:
        fvs.unpersist()
        labeler.close()

    if return_pandas:
        return labeled_data
    else:
        labeled_data = convert_arrays_for_spark(labeled_data)
        return spark.createDataFrame(labeled_data)


_WIDE_DECIMAL = re.compile(r'Decimal precision (\d+) exceeds max precision')


def _data_file(fs, path):
    if not fs.getFileStatus(path).isDirectory():
        return path
    for status in sorted(fs.listStatus(path), key=lambda s: s.getPath().getName()):
        if status.getPath().getName().startswith(('_', '.')):
            continue
        found = _data_file(fs, status.getPath())
        if found is not None:
            return found
    return None


def _parquet_fields(spark, path):
    jvm = spark._jvm
    parquet = jvm.org.apache.parquet
    conf = spark._jsc.hadoopConfiguration()
    where = jvm.org.apache.hadoop.fs.Path(path)
    first = _data_file(where.getFileSystem(conf), where)
    if first is None:
        return []
    reader = parquet.hadoop.ParquetFileReader.open(
        parquet.hadoop.util.HadoopInputFile.fromPath(first, conf))
    try:
        message = reader.getFooter().getFileMetaData().getSchema()
    finally:
        reader.close()
    convert = jvm.org.apache.spark.sql.execution.datasources.parquet \
        .ParquetToSparkSchemaConverter(spark._jsparkSession.sessionState().conf())
    fields = []
    for field in message.getFields():
        try:
            one = convert.convert(parquet.schema.MessageType(message.getName(), [field]))
        except Exception as e:
            wide = _WIDE_DECIMAL.search(str(e))
            if wide is None:
                raise
            fields.append((field.getName(), None, int(wide.group(1))))
            continue
        fields.append((field.getName(), StructType.fromJson(json.loads(one.json()))[0], None))
    return fields


def _read_parquet(spark, path, keep, advice):
    from ..io import read_parquet
    try:
        return read_parquet(spark, path)
    except Exception as e:
        found = _WIDE_DECIMAL.search(str(e))
        if found is None:
            raise
    try:
        fields = _parquet_fields(spark, path)
    except Exception:
        fields = []
    wide = {name: precision for name, _, precision in fields if precision is not None}
    if not wide:
        raise ValueError(
            f'{path}: holds a decimal of precision {found.group(1)}, and Spark reads '
            f'decimals of at most 38 digits; cast it, for example to a string or a '
            f'float') from None
    needed = {n: p for n, p in wide.items() if keep is None or n in keep}
    if needed:
        listed = ', '.join(f"'{n}' (precision {p})" for n, p in needed.items())
        raise ValueError(
            f'{path}: Spark reads decimals of at most 38 digits, and the decimal '
            f'column(s) {listed} hold more; {advice}') from None
    return read_parquet(spark, path, StructType([f for _, f, _ in fields if f is not None]))


def _read_frame(path, fmt, types, keep=None, advice=None):
    spark = SparkSession.builder.getOrCreate()
    if fmt == 'parquet':
        return _read_parquet(spark, path, keep, advice)
    if fmt == 'jsonl':
        df = spark.read.json(path)
    else:
        df = (spark.read.option('header', True)
              .option('sep', '\t' if fmt == 'tsv' else ',').csv(path))
    for name, typ in types.items():
        if name in df.columns:
            df = df.withColumn(name, column(name).cast(typ))
    return df


def read_table(path, fmt, id_col, columns):
    """Return a table read with Spark, its id column renamed to `_id`."""
    from ..io import table_columns
    df = _read_frame(path, fmt, {id_col: 'long'},
                     None if columns is None else [id_col, *columns],
                     'cast them, for example to a string or a float, or leave them out '
                     'with columns=')
    attrs = table_columns(path, df.columns, id_col, columns)
    return df.select(column(id_col).alias('_id'), *map(column, attrs))


def read_pairs(path, fmt, columns, with_label):
    """Return pairs read with Spark as flat `id1`, `id2` and, with `with_label`, `label`."""
    from ..io import PAIR_ID_NAMES, pair_layout
    df = _read_frame(path, fmt, {c: 'long' for c in PAIR_ID_NAMES + columns} | {'label': 'double'},
                     {*PAIR_ID_NAMES, *columns, 'id1_list', 'label'},
                     'cast them, for example to an integer or a float')
    types = dict(df.dtypes)
    grouped, x, y, list_is_a = pair_layout(
        path, df.columns, lambda c: types.get(c, '').startswith('array'), columns, with_label)
    label = [F.col('label').cast('double').alias('label')] if with_label else []
    if not grouped:
        return df.select(column(x).cast('long').alias('id1'),
                         column(y).cast('long').alias('id2'), *label)
    rows = df.select(column(y).alias('_other'),
                     F.posexplode(column(x)).alias('_pos', '_listed'),
                     *([F.col('label')] if with_label else []))
    listed, other = F.col('_listed').cast('long'), F.col('_other').cast('long')
    ids = [listed.alias('id1'), other.alias('id2')] if list_is_a else [
        other.alias('id1'), listed.alias('id2')]
    if with_label:
        ids.append(F.element_at(F.col('label'), F.col('_pos') + 1).cast('double').alias('label'))
    return rows.select(*ids)


def decided_labels(labeled, label_col):
    """Return the rows labeled match or non-match, the other values of the label column
    apart from unsure, and the numbers of matches and non-matches."""
    label = column(label_col)
    counts = {r['v']: r['n'] for r in labeled.groupBy(label.cast('double').alias('v'))
              .count().withColumnRenamed('count', 'n').collect()}
    other = sorted(v for v in counts if v is not None and v not in (0.0, 1.0, 2.0))
    decided = labeled.filter(label.isin(0.0, 1.0))
    return decided, other, int(counts.get(1.0, 0)), int(counts.get(0.0, 0))


def label_pairs(labeler, pairs):
    """Return Spark pairs with the labels the labeler assigns, streamed to the driver a
    partition at a time."""
    stopped = False
    spark = SparkSession.builder.getOrCreate()
    labeled_pairs = []
    from .._core.labeler import label_of

    cols = pairs.columns
    id1_col, id2_col = ('id1', 'id2') if {'id1', 'id2'} <= set(cols) else cols[:2]
    try:
        for row in pairs.select(id1_col, id2_col).toLocalIterator():
            label = label_of(labeler, row[id1_col], row[id2_col])
            if label == -1.0:
                stopped = True
                break
            labeled_pairs.append({
                id1_col: row[id1_col],
                id2_col: row[id2_col],
                'label': float(label)
            })
        labeler.finish(len(labeled_pairs), stopped)
    finally:
        labeler.close()
    if labeled_pairs:
        labeled_pairs = spark.createDataFrame(labeled_pairs)
    else:
        id1_type = pairs.schema[id1_col].dataType
        id2_type = pairs.schema[id2_col].dataType
        schema = StructType([
            StructField(id1_col, id1_type, True),
            StructField(id2_col, id2_type, True),
            StructField('label', DoubleType(), True)
        ])
        labeled_pairs = spark.createDataFrame([], schema)
    return labeled_pairs


def save_data(dataframe, path):
    """Write a Spark DataFrame to `path` as Parquet."""
    spark = SparkSession.builder.getOrCreate()
    master = spark.sparkContext.master
    path_str = str(path)
    if not master.startswith("local") and "://" not in path_str:
        n_rows = dataframe.count()
        max_rows = _max_driver_rows()
        logger.warning(
            f"save_data: '{path_str}' has no filesystem scheme and the Spark "
            f"master is '{master}', so the DataFrame ({n_rows} rows) must be collected "
            f"to the driver to be written as a single parquet file. This needs enough "
            f"driver memory to hold the whole frame. Use a distributed path "
            f"(hdfs://, s3a://, gs://, ...) to write it in parallel instead."
        )
        if n_rows > max_rows:
            raise RuntimeError(
                f"save_data would collect {n_rows} rows to the driver, above the "
                f"{max_rows}-row limit. Write to a distributed path (hdfs://, s3a://, ...) "
                f"so the write stays parallel, or raise the limit with "
                f"{MAX_DRIVER_ROWS_ENV}=<n> if the driver really can hold this frame."
            )
        check_driver_collect(
            n_rows, f"save_data of '{path_str}'",
            'Write to a distributed path (hdfs://, s3a://, ...) so the write stays '
            'parallel.', row_bytes=row_bytes(dataframe) + PANDAS_ROW_BYTES)
        pdf = dataframe.toPandas()
        logger.info(f"Saving pandas DataFrame with shape {pdf.shape} to {path_str}")
        pdf.to_parquet(path_str)
        logger.info(f"Successfully saved pandas DataFrame to {path_str}")
        return
    logger.info(f"Saving Spark DataFrame to {path}")
    dataframe.write.mode('overwrite').parquet(str(path))
    logger.info(f"Successfully saved Spark DataFrame to {path}")


def save_matches(predictions, path, threshold):
    """Write the predicted matches at `threshold` as `id1`, `id2` and `probability`, and
    return their number."""
    p_match = F.when(F.col('prediction') == 1.0, F.col('confidence')) \
               .otherwise(F.lit(1.0) - F.col('confidence'))
    pos = (F.col('prediction') == 1.0) if threshold == 0.5 else (p_match >= threshold)
    matches = predictions.filter(pos).select(
        F.col('id1').cast('long').alias('id1'), F.col('id2').cast('long').alias('id2'),
        p_match.cast('float').alias('probability'))
    n = matches.count()
    save_data(matches, path)
    return n


def load_data(path):
    """Read a Parquet file as a Spark DataFrame."""
    spark = SparkSession.builder.getOrCreate()
    master = spark.sparkContext.master
    path_str = str(path)
    if not master.startswith("local") and "://" not in path_str:
        logger.warning(
            f"load_data: '{path_str}' has no filesystem scheme and the Spark "
            f"master is '{master}', so the file is read into driver memory before "
            f"being distributed. Use a distributed path (hdfs://, s3a://, ...) to read "
            f"it in parallel instead."
        )
        return _read_on_driver(spark, path_str)
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    dataframe = spark.read.parquet(str(path))
    logger.info(f"Successfully loaded Spark DataFrame from {path}")
    return dataframe


def _read_on_driver(spark, path):
    data = ds.dataset(path, format='parquet')
    rows = nbytes = 0
    for frag in data.get_fragments():
        meta = frag.metadata
        for k in range(meta.num_row_groups):
            rows += meta.row_group(k).num_rows
            nbytes += meta.row_group(k).total_byte_size
    check_driver_collect(
        rows, f"load_data of '{path}'",
        'Read it from a distributed path (hdfs://, s3a://, ...) so that the executors read '
        'it in parallel.', row_bytes=nbytes / max(rows, 1) + PANDAS_ROW_BYTES)
    pieces, held, out = [], 0, None
    for batch in data.to_batches():
        pieces.append(batch)
        held += batch.nbytes
        if held >= READ_PIECE_BYTES:
            out = _append(out, as_spark(pa.Table.from_batches(pieces).to_pandas(), spark))
            pieces, held = [], 0
    if pieces or out is None:
        table = pa.Table.from_batches(pieces, schema=data.schema)
        out = _append(out, as_spark(table.to_pandas(), spark))
    logger.info(f"Successfully loaded {rows} rows from {path} through the driver")
    return out


def _append(out, piece):
    return piece if out is None else out.unionByName(piece)
