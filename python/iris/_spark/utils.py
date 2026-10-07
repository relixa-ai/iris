"""Helpers shared by the modules of the Spark engine."""
import logging
from contextlib import contextmanager

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark import SparkContext, StorageLevel


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


def repartition_df(df, part_size, by=None):
    """Return the Spark DataFrame repartitioned into partitions of about `part_size`
    rows."""
    if part_size <= 0:
        raise ValueError(f'part_size must be > 0 (got {part_size})')
    cnt = df.count()
    if cnt == 0:
        return df
    n = max(cnt // part_size, SparkContext.getOrCreate().defaultParallelism * 4)
    n = min(n, cnt)
    if by is not None:
        return df.repartition(n, by)
    else:
        return df.repartition(n)


def ship_arrays(arrays):
    """Add memory-mapped arrays to the Spark job's files, for workers to open
    locally."""
    sc = SparkContext.getOrCreate()
    for arr in arrays:
        sc.addFile(str(arr.path))
        arr.locate_with(spark_file)


def spark_file(name):
    """Return the local path of a file added to the Spark job."""
    from pyspark import SparkFiles

    return SparkFiles.get(name)
