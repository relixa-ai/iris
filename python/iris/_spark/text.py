"""Spark columns in the text form, formatted in the Python workers."""
from typing import Iterator

import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T

from .._core.textform import unsupported

_TEXT_TYPES = (T.BooleanType, T.ByteType, T.ShortType, T.IntegerType, T.LongType,
               T.FloatType, T.DoubleType, T.DecimalType, T.DateType, T.TimestampType,
               T.TimestampNTZType, T.StringType, T.NullType)


def check_text_types(df, cols):
    """Raise an error for the first column whose Spark type has no text form."""
    for c in cols:
        typ = df.schema[c].dataType
        if not isinstance(typ, _TEXT_TYPES):
            raise unsupported(c, typ.simpleString())


def text_columns(df, cols, tz, suffix=None):
    """Return the Spark DataFrame with the columns `cols` in the text form, replaced or
    added beside them."""
    cols = list(dict.fromkeys(cols))
    if not cols:
        return df
    check_text_types(df, cols)
    targets = set(cols)
    fields = [T.StructField(f.name, T.StringType(), True)
              if f.name in targets and suffix is None else f
              for f in df.schema.fields]
    if suffix is not None:
        fields += [T.StructField(c + suffix, T.StringType(), True) for c in cols]
    schema = T.StructType(fields)
    names = [f.name for f in fields]

    def fmt(batches):
        import pyarrow as pa

        from iris._core.textform import to_text

        for batch in batches:
            arrays = [to_text(arr, tz, name) if name in targets and suffix is None else arr
                      for name, arr in zip(batch.schema.names, batch.columns)]
            if suffix is not None:
                arrays += [to_text(batch.column(c), tz, c) for c in cols]
            yield pa.RecordBatch.from_arrays(arrays, names=names)

    return df.mapInArrow(fmt, schema)


def tokens_column(tokenizer, input_col):
    """Return a Spark column of the tokens of `input_col`, computed in the Python
    workers."""
    @F.pandas_udf(T.ArrayType(T.StringType()))
    def t(itr: Iterator[pd.Series]) -> Iterator[pd.Series]:
        for s in itr:
            yield s.apply(tokenizer.tokenize)

    return t(input_col)
