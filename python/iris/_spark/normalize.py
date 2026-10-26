"""The normalization parsers on Spark tables, run in the Python workers."""
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .._core.textform import resolve_timezone
from ..normalize import duration_seconds, integer_number, year_number
from .text import text_columns
from .utils import column


def add_normalized_columns(df, dur_cols, year_cols, concat_cols, int_cols, timezone):
    """Return the Spark table with parsed numeric columns and a concatenated text column
    added."""
    tz = resolve_timezone(timezone)

    for group, cols in (('normalize-durations', dur_cols), ('normalize-years', year_cols),
                        ('normalize-integers', int_cols), ('concat-cols', concat_cols)):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f'[normalize] WARNING: --{group} column(s) {missing} not in table '
                  f'(columns: {[c for c in df.columns if c != "_id"]}); skipping them',
                  flush=True)

    sources = [c for c in dict.fromkeys([*dur_cols, *year_cols, *int_cols, *concat_cols])
               if c in df.columns]
    suffix = '__iris_text'
    other = [c for c in sources if not isinstance(df.schema[c].dataType, T.StringType)]
    df = text_columns(df, other, tz, suffix=suffix)
    text = {c: column(c + suffix if c in other else c) for c in sources}

    dur_udf = F.pandas_udf(duration_seconds, 'double')
    year_udf = F.pandas_udf(year_number, 'double')
    int_udf = F.pandas_udf(integer_number, 'double')
    for c in dur_cols:
        if c in df.columns:
            df = df.withColumn(f'{c}_seconds', dur_udf(text[c]))
    for c in year_cols:
        if c in df.columns:
            df = df.withColumn(f'{c}_numeric', year_udf(text[c]))
    for c in int_cols:
        if c in df.columns:
            df = df.withColumn(f'{c}_numeric', int_udf(text[c]))
    present = [c for c in concat_cols if c in df.columns]
    if present:
        df = df.withColumn('combined_text', F.concat_ws(
            ' ', *[F.coalesce(text[c], F.lit('')) for c in present]))
    return df.drop(*[c + suffix for c in other])


def parse_rates(df, dur_cols, year_cols, sample_rows):
    """Return the fraction of values each parser parsed, over a sample of the Spark
    table."""
    aggs = [F.mean(column(f'{c}_seconds').isNotNull().cast('double')).alias(c)
            for c in dur_cols if f'{c}_seconds' in df.columns]
    aggs += [F.mean(column(f'{c}_numeric').isNotNull().cast('double')).alias(f'{c}_yr')
             for c in year_cols if f'{c}_numeric' in df.columns]
    if not aggs:
        return {}
    row = df.limit(sample_rows).select(*aggs).toPandas()
    return {c: float(row[c].iloc[0]) for c in row.columns}
