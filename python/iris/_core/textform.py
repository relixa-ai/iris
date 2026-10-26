"""The text form of a value: how both engines turn a column into the strings features
read."""
from __future__ import annotations

import datetime as _dt
import os
from fractions import Fraction

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

SUPPORTED = ('bool, integers, floats, decimals, dates, timestamps and strings')

_EPOCH = _dt.datetime(1970, 1, 1)
_EPOCH_UTC = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)
_PER_SECOND = {'s': 1, 'ms': 1_000, 'us': 1_000_000, 'ns': 1_000_000_000}


def check_timezone(name):
    """Return `name` when it is an IANA time zone this machine knows, and raise
    ValueError otherwise."""
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    if not isinstance(name, str) or not name:
        raise ValueError(f'timezone must be an IANA zone name such as "UTC" or '
                         f'"Europe/Paris", got {name!r}')
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError(f'unknown timezone {name!r}: use an IANA zone name such as '
                         f'"UTC" or "Europe/Paris"') from None
    return name


def machine_timezone():
    """Return this machine's IANA time zone, found as the JVM finds its default."""
    candidates = []
    if os.environ.get('TZ'):
        candidates.append(os.environ['TZ'].lstrip(':'))
    try:
        link = os.path.realpath('/etc/localtime')
        if 'zoneinfo/' in link:
            candidates.append(link.split('zoneinfo/', 1)[1])
    except OSError:
        pass
    try:
        with open('/etc/timezone') as fh:
            candidates.append(fh.read().strip())
    except OSError:
        pass
    for name in candidates:
        try:
            return check_timezone(name)
        except ValueError:
            continue
    return 'UTC'


def resolve_timezone(name=None):
    """Return the time zone of a run: `name` checked, or this machine's zone when it is
    None."""
    return machine_timezone() if name is None else check_timezone(name)


def _shortest(x, dtype):
    if dtype == np.float64:
        text = repr(abs(float(x)))
    else:
        text = np.format_float_scientific(np.float32(abs(x)), unique=True, trim='0')
    mantissa, _, exp = text.partition('e')
    whole, _, frac = mantissa.partition('.')
    digits = whole + frac
    point = len(whole) + (int(exp) if exp else 0)
    stripped = digits.lstrip('0')
    point -= len(digits) - len(stripped)
    return stripped.rstrip('0') or '0', point


def _rounds_to(value, x, dtype):
    x = dtype(x)
    lo = Fraction(float(np.nextafter(x, dtype(-np.inf))))
    hi = Fraction(float(np.nextafter(x, dtype(np.inf))))
    fx = Fraction(float(x))
    lo_mid, hi_mid = (lo + fx) / 2, (fx + hi) / 2
    even = (int(np.frombuffer(x.tobytes(), dtype=np.uint64 if dtype == np.float64
                              else np.uint32)[0]) & 1) == 0
    if lo_mid < value < hi_mid:
        return True
    return even and value in (lo_mid, hi_mid)


def _two_digit(digits, point, x, dtype):
    x = dtype(abs(x))
    if float(np.spacing(x)) < 0.1 * 10.0 ** (point - 2):
        return digits, point
    exact = Fraction(float(x))
    best = None
    for c in range(10, 100):
        value = Fraction(c) * Fraction(10) ** (point - 2)
        if not _rounds_to(value, x, dtype):
            continue
        key = (abs(value - exact), c % 2)
        if best is None or key < best[0]:
            best = (key, c)
    if best is None:
        return digits, point
    out = str(best[1]).rstrip('0')
    return out, point


def _java_layout(negative, digits, point):
    sign = '-' if negative else ''
    if 0 < point < 8:
        whole = digits[:point].ljust(point, '0')
        return f'{sign}{whole}.{digits[point:] or "0"}'
    if -3 < point <= 0:
        return f'{sign}0.{"0" * -point}{digits}'
    return f'{sign}{digits[0]}.{digits[1:] or "0"}E{point - 1}'


def format_float(x, dtype=np.float64):
    """Return a float as text with the shortest digits that read back to it at the given
    precision."""
    x = dtype(x)
    if np.isnan(x):
        return 'NaN'
    if np.isinf(x):
        return 'Infinity' if x > 0 else '-Infinity'
    negative = bool(np.signbit(x))
    if x == 0:
        return '-0.0' if negative else '0.0'
    digits, point = _shortest(x, dtype)
    if len(digits) == 1:
        digits, point = _two_digit(digits, point, x, dtype)
    return _java_layout(negative, digits, point)


def _floats(arr):
    dtype = np.float32 if pa.types.is_float32(arr.type) else np.float64
    values = arr.to_numpy(zero_copy_only=False)
    if arr.null_count:
        values = np.where(arr.is_valid().to_numpy(zero_copy_only=False), values, 0)
    values = np.ascontiguousarray(values, dtype=dtype)
    bits = values.view(np.uint64 if dtype == np.float64 else np.uint32)
    uniq, inverse = np.unique(bits, return_inverse=True)
    texts = np.array([format_float(v, dtype) for v in uniq.view(dtype)], dtype=object)
    return pa.array(texts[inverse], type=pa.string(), mask=_null_mask(arr))


def _null_mask(arr):
    if not arr.null_count:
        return None
    return arr.is_null().to_numpy(zero_copy_only=False)


def _decimals(arr):
    return pa.array([None if v is None else format(v, 'f') for v in arr.to_pylist()],
                    type=pa.string())


def _out_of_range(column, what, value):
    return ValueError(f'column {column!r} holds {what} {value}, outside the years 1 to '
                      f'9999 that Iris can show as text')


def _dates(arr, column):
    if pa.types.is_date64(arr.type):
        ms = arr.cast(pa.int64()).to_numpy(zero_copy_only=False)
        known = ~_null_mask(arr) if arr.null_count else np.ones(len(arr), dtype=bool)
        if np.any(ms[known] % 86_400_000):
            raise ValueError(f'column {column!r} is date64 with values that are not whole '
                             f'days; store it as a timestamp, or as date32')
        arr = pc.cast(arr, pa.date32())
    days = arr.cast(pa.int32()).to_numpy(zero_copy_only=False)
    valid = ~_null_mask(arr) if arr.null_count else np.ones(len(arr), dtype=bool)
    lo, hi = (_dt.date(1, 1, 1) - _dt.date(1970, 1, 1)).days, \
        (_dt.date(9999, 12, 31) - _dt.date(1970, 1, 1)).days
    bad = valid & ((days < lo) | (days > hi))
    if bad.any():
        raise _out_of_range(column, 'the date with day number', int(days[bad][0]))
    return pc.cast(arr, pa.string())


def _timestamp_text(d):
    text = (f'{d.year:04d}-{d.month:02d}-{d.day:02d} '
            f'{d.hour:02d}:{d.minute:02d}:{d.second:02d}')
    if d.microsecond:
        text += '.' + f'{d.microsecond:06d}'.rstrip('0')
    return text


def _timestamps(arr, tz, column):
    from zoneinfo import ZoneInfo

    unit = arr.type.unit
    raw = arr.cast(pa.int64()).to_numpy(zero_copy_only=False)
    valid = ~_null_mask(arr) if arr.null_count else np.ones(len(arr), dtype=bool)
    raw = np.where(valid, raw, 0)
    factor = 1_000_000 // _PER_SECOND[unit] if unit != 'ns' else 1
    if factor > 1 and np.any(np.abs(raw) > np.iinfo(np.int64).max // factor):
        bad = raw[np.abs(raw) > np.iinfo(np.int64).max // factor][0]
        raise _out_of_range(column, 'a timestamp of', f'{int(bad)} {unit} from 1970')
    micros = raw * factor if unit != 'ns' else raw // 1_000
    zone = ZoneInfo(tz) if arr.type.tz is not None else None
    uniq, inverse = np.unique(micros, return_inverse=True)
    texts = []
    for us in uniq.tolist():
        try:
            if zone is None:
                d = _EPOCH + _dt.timedelta(microseconds=us)
            else:
                d = (_EPOCH_UTC + _dt.timedelta(microseconds=us)).astimezone(zone)
        except OverflowError:
            raise _out_of_range(column, 'a timestamp of', f'{us} us from 1970') from None
        texts.append(_timestamp_text(d))
    texts = np.array(texts, dtype=object)[inverse] if texts else np.array([], dtype=object)
    valid_rows = valid if arr.null_count else None
    if valid_rows is not None:
        return pa.array(texts, type=pa.string(), mask=~valid_rows)
    return pa.array(texts, type=pa.string())


def unsupported(column, typ):
    """Return the error raised for a column whose type has no text form."""
    return TypeError(
        f'column {column!r} is {typ}, which has no text form in Iris (supported: '
        f'{SUPPORTED}). Leave it out of the feature columns, or convert it first.')


def to_text(arr, tz, column=None):
    """Return an Arrow array as a string array of the same length, in the text form."""
    if isinstance(arr, pa.ChunkedArray):
        return pa.chunked_array([to_text(c, tz, column) for c in arr.chunks],
                                type=pa.string())
    t = arr.type
    if pa.types.is_dictionary(t):
        return to_text(arr.dictionary_decode(), tz, column)
    if pa.types.is_null(t):
        return pa.nulls(len(arr), type=pa.string())
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        try:
            arr.validate(full=True)
        except pa.ArrowInvalid as e:
            raise ValueError(f'column {column!r} holds text that is not valid UTF-8 ({e}); '
                             f'decode it, or replace the bad bytes, before Iris reads it') \
                from None
        return arr if pa.types.is_string(t) else arr.cast(pa.string())
    if pa.types.is_boolean(t) or pa.types.is_integer(t):
        return pc.cast(arr, pa.string())
    if pa.types.is_float32(t) or pa.types.is_float64(t):
        return _floats(arr)
    if pa.types.is_decimal(t):
        return _decimals(arr)
    if pa.types.is_date(t):
        return _dates(arr, column)
    if pa.types.is_timestamp(t):
        return _timestamps(arr, tz, column)
    raise unsupported(column, t)
