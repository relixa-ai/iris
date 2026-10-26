"""Helpers shared by the modules of the Spark engine."""
import logging
import math
import os
import shutil
from contextlib import contextmanager

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark import SparkContext, StorageLevel

BATCH_KEY = 'spark.sql.execution.arrow.maxRecordsPerBatch'

CACHE_BATCH_KEY = 'spark.sql.inMemoryColumnarStorage.batchSize'

READER_BATCH_KEY = 'spark.sql.parquet.columnarReaderBatchSize'

BATCH_BYTES = 1 << 20

BATCH_ROWS = 10_000

BATCH_KEYS = {BATCH_KEY: BATCH_ROWS, CACHE_BATCH_KEY: BATCH_ROWS, READER_BATCH_KEY: 4096}

RESERVED_HEAP_BYTES = 300 * 2**20

READ_FACTOR = 2

VALUE_BYTES = 24

PARTITION_BYTES = 64 * 2**20

MAX_PARTITIONS = 20_000

_STRING_TYPES = (T.StringType, T.BinaryType)


def _value_bytes(field):
    c = column(field.name)
    typ = field.dataType
    if isinstance(typ, _STRING_TYPES):
        return F.coalesce(F.octet_length(c), F.lit(0)) + 4
    if isinstance(typ, T.ArrayType):
        if isinstance(typ.elementType, _STRING_TYPES):
            inner = F.coalesce(F.octet_length(F.concat_ws('', c)), F.lit(0))
            return inner + F.greatest(F.size(c), F.lit(0)) * 4 + 4
        return F.greatest(F.size(c), F.lit(0)) * 8 + 4
    return F.lit(VALUE_BYTES)


def row_bytes_column(df, cols=None):
    """Return a Spark column of the bytes of each row over `cols`, as Arrow and its text
    form hold it."""
    wanted = set(df.columns if cols is None else cols)
    parts = [_value_bytes(f) for f in df.schema.fields if f.name in wanted]
    total = F.lit(0)
    for p in parts:
        total = total + p
    return total.cast('long')


def row_bytes(df, cols=None):
    """Return the mean bytes of a row of a Spark DataFrame over `cols`, measured in one
    pass over its rows."""
    mean = df.agg(F.avg(row_bytes_column(df, cols))).head()[0]
    return float(mean or 0.0)


def vector_row_bytes(width):
    """Return the bytes of a row of feature vectors of the given width, with its ids and
    score."""
    return 8 * int(width) + 32


def batch_rows(per_row, limit=None, default=BATCH_ROWS):
    """Return the rows of a batch of about `BATCH_BYTES` at `per_row` bytes a row, at
    most `limit` rows, or `default` when no limit is set."""
    rows = max(1, int(BATCH_BYTES // max(int(per_row), 1)))
    limit = default if limit is None else int(limit)
    return min(rows, limit) if limit > 0 else rows


_SCOPES = []


@contextmanager
def bounded_batches(spark, per_row):
    """Bound the Arrow, cached column and Parquet reader batches of the Spark stages in a
    block to about `BATCH_BYTES` at `per_row` bytes a row, restoring the settings after."""
    before = {key: spark.conf.get(key, None) for key in BATCH_KEYS}
    outer = _SCOPES[-1] if _SCOPES else {}
    scope = {}
    for key, default in BATCH_KEYS.items():
        set_by, limit = outer.get(key, (None, None))
        limit = limit if outer and set_by == before[key] else before[key]
        scope[key] = (str(batch_rows(per_row, limit, default)), limit)
        spark.conf.set(key, scope[key][0])
    _SCOPES.append(scope)
    try:
        yield
    finally:
        _SCOPES.pop()
        for key in BATCH_KEYS:
            if before[key] is None:
                spark.conf.unset(key)
            else:
                spark.conf.set(key, before[key])


def lower_batches(spark, per_row):
    """Lower the session's Arrow batches to about `BATCH_BYTES` at `per_row` bytes a row,
    for a lazy result whose stages run when the caller evaluates it."""
    before = spark.conf.get(BATCH_KEY, None)
    current = BATCH_ROWS if before is None else int(before)
    rows = batch_rows(per_row, before)
    if current <= 0 or rows < current:
        spark.conf.set(BATCH_KEY, str(rows))
    return rows


def column(name):
    """Return the Spark column of the given name, quoted so that a dot or a backtick in
    it is part of the name."""
    return F.col('`' + name.replace('`', '``') + '`')


def _plain(names):
    return not any('.' in n or '`' in n for n in names)


def _slots(names):
    return [f'_{i}' for i in range(len(names))]


def map_in_arrow(df, fn, schema):
    """Return `df.mapInArrow(fn, schema)` for input columns of any name, which pyspark
    would otherwise look up unquoted."""
    names = df.columns
    if _plain(names):
        return df.mapInArrow(fn, schema)

    def run(batches):
        import pyarrow as pa

        def named():
            for b in batches:
                yield pa.RecordBatch.from_arrays(b.columns, schema=pa.schema(
                    [f.with_name(n) for f, n in zip(b.schema, names)]))
        return fn(named())

    return df.toDF(*_slots(names)).mapInArrow(run, schema)


def map_in_pandas(df, fn, schema):
    """Return `df.mapInPandas(fn, schema)` for input columns of any name, which pyspark
    would otherwise look up unquoted."""
    names = df.columns
    if _plain(names):
        return df.mapInPandas(fn, schema)

    def run(frames):
        def named():
            for pdf in frames:
                pdf.columns = names
                yield pdf
        return fn(named())

    return df.toDF(*_slots(names)).mapInPandas(run, schema)


def parquet_row_groups(df, cols):
    """Return the row groups of the Parquet files a Spark DataFrame reads, as `(file,
    rows, packed bytes, plain bytes)` of the columns `cols`, from the files' footers."""
    from py4j.protocol import Py4JJavaError

    jvm = df.sparkSession._jvm
    conf = df.sparkSession._jsc.hadoopConfiguration()
    wanted = set(cols)
    try:
        names = df.inputFiles()
    except Exception:
        return []
    out = []
    for name in names:
        try:
            reader = jvm.org.apache.parquet.hadoop.ParquetFileReader.open(
                jvm.org.apache.parquet.hadoop.util.HadoopInputFile.fromPath(
                    jvm.org.apache.hadoop.fs.Path(name), conf))
        except Py4JJavaError:
            continue
        try:
            for block in reader.getFooter().getBlocks():
                packed = plain = 0
                for chunk in block.getColumns():
                    if chunk.getPath().toArray()[0] in wanted:
                        packed += chunk.getTotalSize()
                        plain += chunk.getTotalUncompressedSize()
                out.append((name, int(block.getRowCount()), int(packed), int(plain)))
        finally:
            reader.close()
    return out


def task_heap(spark):
    """Return the heap of a JVM that runs tasks, the tasks it runs at once, and the
    setting that sizes the heap."""
    sc = spark.sparkContext
    if sc.master.startswith('local'):
        heap = int(sc._jvm.java.lang.Runtime.getRuntime().maxMemory())
        return heap, sc.defaultParallelism, 'spark.driver.memory'
    conf = sc._jsc.sc().conf()
    heap = int(conf.getSizeAsBytes('spark.executor.memory', '1g'))
    tasks = conf.getInt('spark.executor.cores', 1) // conf.getInt('spark.task.cpus', 1)
    return heap, max(1, tasks), 'spark.executor.memory'


def check_row_groups(spark, groups, what):
    """Raise an error when the largest Parquet row groups, which Spark reads whole, one
    per task, would not fit the heap of the JVM that reads them."""
    if not groups:
        return
    heap, tasks, setting = task_heap(spark)
    largest = sorted(groups, key=lambda g: g[2], reverse=True)[:tasks]
    need = READ_FACTOR * sum(g[2] for g in largest)
    room = heap - RESERVED_HEAP_BYTES
    if need <= room:
        return
    name, rows, packed, _ = largest[0]
    raise RuntimeError(
        f'{what} reads {name}, which holds a row group of {rows:,} rows and '
        f'{packed / 2**30:.2f} GiB. Spark reads a row group whole, one per task, in '
        f'buffers it cannot spill, and needs about {READ_FACTOR} times its bytes: '
        f'{len(largest)} task(s) reading the largest row groups at once need about '
        f'{need / 2**30:.1f} GiB of the {room / 2**30:.1f} GiB that {setting} leaves '
        f'after the 300 MiB Spark reserves. Rewrite the file with smaller row groups, '
        f'for example pyarrow.parquet.write_table(table, path, row_group_size=10_000), '
        f'or raise {setting}.')


def parquet_row_bytes(groups):
    """Return the mean bytes of a row of the columns read, from the row groups of
    `parquet_row_groups`, or None without any."""
    rows = sum(g[1] for g in groups)
    return sum(g[3] for g in groups) / rows if rows else None


def vector_width(df, col_name='feature_vectors'):
    """Return the width of one feature vector in a Spark DataFrame, or None when it
    cannot be read cheaply."""
    if not isinstance(df.schema[col_name].dataType, T.ArrayType):
        return None
    row = df.select(F.size(col_name).alias('n')).head(1)
    return row[0]['n'] if row else None


def replace_or_append_fields(schema, new_fields):
    """Return a schema with each new field replacing the field of the same name, or
    appended."""
    fields = list(schema.fields)
    index_by_name = {f.name: i for i, f in enumerate(fields)}
    for nf in new_fields:
        if nf.name in index_by_name:
            fields[index_by_name[nf.name]] = nf
        else:
            index_by_name[nf.name] = len(fields)
            fields.append(nf)
    return T.StructType(fields)

@contextmanager
def persisted(df, storage_level=StorageLevel.MEMORY_AND_DISK):
    """Persist a Spark DataFrame for the duration of a block."""
    if df is not None:
        df = df.persist(storage_level)
    try:
        yield df
    finally:
        if df is not None:
            df.unpersist()

def is_persisted(df):
    """Return whether a Spark DataFrame is persisted."""
    sl = df.storageLevel
    return sl.useMemory or sl.useDisk

MAX_SUPPORTED_JAVA_VERSION = 17
_java_version_checked = False


def check_java_version(spark=None):
    """Warn when Spark runs on a Java release it does not support."""
    global _java_version_checked
    if _java_version_checked:
        return
    _java_version_checked = True

    try:
        if spark is None:
            sc = SparkContext._active_spark_context
            if sc is None:
                return
            jvm = sc._jvm
        else:
            jvm = spark.sparkContext._jvm
        raw = jvm.System.getProperty('java.version')
        if not raw:
            return
        parts = raw.split('.')
        major = int(parts[1]) if parts[0] == '1' and len(parts) > 1 else int(parts[0].split('-')[0])
    except Exception:
        return

    if major > MAX_SUPPORTED_JAVA_VERSION:
        logging.getLogger(__name__).warning(
            f'Iris is running on Java {major}, but PySpark 3.5.x supports Java 8, 11 '
            f'and 17 only. On newer releases Arrow cannot access sun.misc.Unsafe, so Spark '
            f'falls back to slow non-Arrow conversion and later fails with errors that look '
            f'unrelated (for example "CANNOT_MERGE_TYPE" while converting a pandas table). '
            f'Install a Java {MAX_SUPPORTED_JAVA_VERSION} JDK and point JAVA_HOME at it.'
        )


def table_stats(df, cols=None):
    """Return the rows of a Spark DataFrame and the mean bytes of a row over `cols`,
    measured in one pass."""
    row = df.agg(F.count(F.lit(1)).alias('n'),
                 F.avg(row_bytes_column(df, cols)).alias('b')).head()
    return int(row['n']), float(row['b'] or 0.0)


def partition_count(n_rows, nbytes, part_bytes=PARTITION_BYTES):
    """Return the partitions for rows of the given bytes in all: about `part_bytes` each,
    at least four per core and at most `MAX_PARTITIONS`, never more than the rows."""
    parallelism = SparkContext.getOrCreate().defaultParallelism
    n = max(4 * parallelism, math.ceil(nbytes / part_bytes))
    return int(max(1, min(MAX_PARTITIONS, n, n_rows)))


def repartition_df(df, by=None, part_bytes=PARTITION_BYTES, stats=None):
    """Return the Spark DataFrame repartitioned into partitions of about `part_bytes`
    bytes, from its rows and their measured bytes, or the `stats` given."""
    if part_bytes <= 0:
        raise ValueError(f'part_bytes must be > 0 (got {part_bytes})')
    cnt, mean = table_stats(df) if stats is None else stats
    if cnt == 0:
        return df
    n = partition_count(cnt, cnt * mean, part_bytes)
    if by is not None:
        return df.repartition(n, by)
    return df.repartition(n)


def shipping_dir(sc):
    """Return the directory of the files shipped to the Spark context's workers, inside the
    driver's directory of added files, which Spark removes when the context stops."""
    directory = os.path.join(sc._jvm.org.apache.spark.SparkFiles.getRootDirectory(),
                             'iris-shipped')
    os.makedirs(directory, exist_ok=True)
    return directory


def ship_arrays(arrays):
    """Add memory-mapped arrays to the Spark job's files, for workers to open locally,
    from copies that live as long as the Spark context."""
    sc = SparkContext.getOrCreate()
    directory = shipping_dir(sc)
    for arr in arrays:
        target = os.path.join(directory, arr.path.name)
        if not os.path.exists(target):
            try:
                os.link(arr.path, target)
            except OSError:
                shutil.copyfile(arr.path, target)
            sc.addFile(target)
        arr.locate_with(spark_file)


def spark_file(name):
    """Return the local path of a file added to the Spark job."""
    from pyspark import SparkFiles

    return SparkFiles.get(name)
