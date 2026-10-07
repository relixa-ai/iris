"""Parsers that add numeric columns for durations, years and integers written as
text."""
from __future__ import annotations

import numpy as np
import pandas as pd


def duration_seconds(raw: pd.Series) -> pd.Series:
    """Return the durations in a series of strings as seconds, NaN where a value does
    not parse."""
    s = raw.astype('string').str.strip().str.lower()
    sec = pd.Series(np.nan, index=s.index, dtype=float)

    ex = s.str.extract(r'^(\d+)m\s*(\d+(?:\.\d+)?)sec$')
    m = ex[0].notna()
    sec[m] = ex[0][m].astype(float) * 60 + ex[1][m].astype(float)

    ex = s.str.extract(r'^(?:(\d+):)?(\d{1,2}):(\d{2})$')
    m = ex[2].notna() & sec.isna()
    sec[m] = (ex[0][m].fillna('0').astype(float) * 3600
              + ex[1][m].astype(float) * 60 + ex[2][m].astype(float))

    ex = s.str.extract(r'^(\d+)\.(\d+)$')
    m = ex[0].notna() & sec.isna()
    if m.any():
        mins = ex[0][m].astype(float)
        frac = ex[1][m]
        two = frac.str.len() == 2
        secs = pd.Series(np.nan, index=frac.index, dtype=float)
        secs[two] = frac[two].astype(float)
        secs[~two] = ('0.' + frac[~two]).astype(float) * 60.0
        overflow = secs >= 60
        secs[overflow] = ('0.' + frac[overflow]).astype(float) * 60.0
        sec[m] = mins * 60 + secs

    ex = s.str.extract(r'^(\d+)$')[0]
    m = ex.notna() & sec.isna()
    if m.any():
        iv = ex[m].astype(float)
        sec[m] = np.where((iv >= 10) & (iv <= 5400), iv,
                          np.where(iv > 5400, iv / 1000.0, iv))
    return sec


def year_number(raw: pd.Series) -> pd.Series:
    """Return the years in a series of strings as numbers, NaN where a value does not
    parse."""
    s = raw.astype('string').str.strip().str.lstrip("'")
    ex = s.str.extract(r'^(\d{2}|\d{4})$')[0]
    y = pd.Series(np.nan, index=s.index, dtype=float)
    m = ex.notna()
    if m.any():
        v = ex[m].astype(float)
        y[m] = np.where(v < 100, np.where(v <= 30, v + 2000, v + 1900), v)
    return y


def integer_number(raw: pd.Series) -> pd.Series:
    """Return the numbers in a series of digit strings, NaN where a value does not
    parse."""
    s = raw.astype('string').str.strip()
    ex = s.str.extract(r'^(\d+(?:\.\d+)?)$')[0]
    y = pd.Series(np.nan, index=s.index, dtype=float)
    m = ex.notna()
    if m.any():
        y[m] = ex[m].astype(float)
    return y


def add_normalized_columns(df, dur_cols=(), year_cols=(), concat_cols=(), int_cols=(),
                           timezone=None):
    """Return the table with parsed numeric columns and a concatenated text column
    added."""
    from ._local.inputs import is_local
    if is_local(df):
        from ._local.normalize import add_normalized_columns as add
    else:
        from ._spark.normalize import add_normalized_columns as add

    return add(df, dur_cols, year_cols, concat_cols, int_cols, timezone)


def parse_rates(df, dur_cols=(), year_cols=(), sample_rows=100_000):
    """Return the fraction of values each parser parsed, over a sample of the table."""
    from ._local.inputs import is_local
    if is_local(df):
        from ._local.normalize import parse_rates as rates
    else:
        from ._spark.normalize import parse_rates as rates

    return rates(df, dur_cols, year_cols, sample_rows)
