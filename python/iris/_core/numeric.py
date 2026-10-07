"""Deciding whether a text value is a number."""
from __future__ import annotations

import math
import re

_WS = r'[ \t\n\r\f\v]*'
_DECIMAL = r'(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?'
_HEX = r'0[xX](?:[0-9a-fA-F]+\.?[0-9a-fA-F]*|\.[0-9a-fA-F]+)(?:[pP][+-]?[0-9]+)?'
_INF = r'[iI][nN][fF](?:[iI][nN][iI][tT][yY])?'

NUMBER = re.compile(rf'{_WS}(?P<sign>[+-]?)(?:(?P<hex>{_HEX})|(?P<dec>{_DECIMAL})|(?P<inf>{_INF})){_WS}')


def parse_number(value):
    """Return `value` as a float, or None when it is missing or not a number."""
    if value is None:
        return None
    if isinstance(value, str):
        m = NUMBER.fullmatch(value)
        if m is None:
            return None
        sign = -1.0 if m.group('sign') == '-' else 1.0
        if m.group('hex') is not None:
            return sign * float.fromhex(m.group('hex'))
        if m.group('inf') is not None:
            return sign * math.inf
        return sign * float(m.group('dec'))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def is_number(value):
    """Return whether a text value parses as a number."""
    return isinstance(value, str) and NUMBER.fullmatch(value) is not None


def numeric_rate(values):
    """Return the share of non-missing values that parse as numbers, and the number of
    non-missing values."""
    parsed = present = 0
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        present += 1
        if isinstance(v, str):
            parsed += NUMBER.fullmatch(v) is not None
        else:
            parsed += parse_number(v) is not None
    return (parsed / present if present else 0.0), present
