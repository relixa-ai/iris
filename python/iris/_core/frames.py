"""The record frame features read, telling Spark DataFrames apart without importing
Spark, and telling Arrow datasets that carry a filter."""
import sys


def is_spark(df):
    """Return whether `df` is a Spark DataFrame, without importing pyspark."""
    sql = sys.modules.get('pyspark.sql')
    return sql is not None and isinstance(df, sql.DataFrame)


def has_filter(dataset):
    """Return whether an Arrow dataset carries a row filter, which reading its files one
    by one would not apply."""
    options = getattr(dataset, '_scan_options', None)
    if isinstance(options, dict):
        return options.get('filter') is not None
    try:
        next(iter(dataset.get_fragments()), None)
    except ValueError:
        return True
    return False


def active_spark_context():
    """Return the running SparkContext, or None, without importing pyspark or starting
    one."""
    pyspark = sys.modules.get('pyspark')
    return None if pyspark is None else pyspark.SparkContext._active_spark_context


class AttrsByRef(dict):
    """A dictionary that a deep copy shares by reference, for the attributes of a record
    frame."""

    def __deepcopy__(self, memo):
        return self
