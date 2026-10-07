"""The encoding of one preprocessed record as one block of bytes, the same in both
engines."""
from __future__ import annotations

import pickle

import numba as nb
import numpy as np
import pandas as pd
import pyarrow as pa

from .frames import AttrsByRef

KINDS = ('str', 'f64', 'ids', 'vec', 'strset', 'pkl')
VEC_IDX_SUFFIX = '#idx'
VEC_VAL_SUFFIX = '#val'


def layout(features, for_a, for_b):
    """Return the columns, with their kinds, that a record holds for the given features
    and sides."""
    sides = (['a'] if for_a else []) + (['b'] if for_b else [])
    out = {}
    for f in features:
        for side in sides:
            for name, kind in f.stored_columns(side):
                if kind not in KINDS:
                    raise ValueError(f'{f} declares column {name!r} of unknown kind '
                                     f'{kind!r}; kinds are {KINDS}')
                if out.setdefault(name, kind) != kind:
                    raise ValueError(f'column {name!r} is stored as {out[name]!r} for '
                                     f'one feature and as {kind!r} for {f}')
    return list(out.items())


def _uvarint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _missing(v):
    return v is None or (isinstance(v, float) and v != v)


def _section(kind, v):
    if kind == 'str':
        return None if _missing(v) else str(v).encode('utf-8')
    if kind == 'f64':
        return None if _missing(v) else np.float64(v).tobytes()
    if kind == 'ids':
        return None if v is None else np.ascontiguousarray(v, dtype='<i8').tobytes()
    if kind == 'vec':
        if v is None:
            return None
        return (np.ascontiguousarray(v.indexes, dtype='<i4').tobytes()
                + np.ascontiguousarray(v.values, dtype='<f4').tobytes())
    if kind == 'strset':
        if v is None:
            return None
        toks = sorted({str(t).encode('utf-8') for t in v})
        return (_uvarint(len(toks)) + b''.join(_uvarint(len(t)) for t in toks)
                + b''.join(toks))
    if kind == 'pkl':
        return None if v is None else pickle.dumps(v, protocol=5)
    raise ValueError(f'unknown stored kind {kind!r}; kinds are {KINDS}')


def pack(layout, frame):
    """Return every row of a frame of preprocessed values packed against `layout`."""
    cols = [(frame[name].tolist(), kind) for name, kind in layout]
    blobs = []
    for i in range(len(frame)):
        dirs, payload = [], []
        for values, kind in cols:
            sec = _section(kind, values[i])
            if sec is None:
                dirs.append(b'\x00')
            else:
                dirs.append(_uvarint(len(sec) + 1))
                payload.append(sec)
        blobs.append(b''.join(dirs) + b''.join(payload))
    ends = np.cumsum(np.fromiter((len(b) for b in blobs), dtype=np.int64, count=len(blobs)))
    data = np.frombuffer(b''.join(blobs), dtype=np.uint8).copy()
    return data, ends


_U8 = nb.types.Array(nb.types.uint8, 1, 'C')
_U8_RO = nb.types.Array(nb.types.uint8, 1, 'C', readonly=True)
_I64 = nb.types.Array(nb.types.int64, 1, 'C')
_I64_2D = nb.types.Array(nb.types.int64, 2, 'C')

@nb.njit([nb.types.Tuple((_I64_2D, _I64_2D))(u8, _I64, _I64, nb.types.int64)
          for u8 in (_U8, _U8_RO)])
def scan_dirs(data, starts, ends, n_cols):
    """Return the start and length of every column section of every record, with length
    -1 for null."""
    n = len(starts)
    off = np.empty((n, n_cols), dtype=np.int64)
    ln = np.empty((n, n_cols), dtype=np.int64)
    for r in range(n):
        p = starts[r]
        lens = np.empty(n_cols, dtype=np.int64)
        for c in range(n_cols):
            v = 0
            shift = 0
            while True:
                b = data[p]
                p += 1
                v |= (np.int64(b) & 0x7F) << shift
                if b < 0x80:
                    break
                shift += 7
            lens[c] = v - 1
        q = p
        for c in range(n_cols):
            off[r, c] = q
            ln[r, c] = lens[c]
            if lens[c] > 0:
                q += lens[c]
        if q != ends[r]:
            raise ValueError('record is corrupt: its sections do not end where it does')
    return off, ln


@nb.njit([nb.types.Tuple((_U8, _I64))(u8, _I64, _I64) for u8 in (_U8, _U8_RO)])
def copy_col(data, off, ln):
    """Return the sections of one column copied end to end, with their byte offsets."""
    n = len(off)
    offsets = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        offsets[i + 1] = offsets[i] + max(ln[i], 0)
    out = np.empty(offsets[n], dtype=np.uint8)
    for i in range(n):
        if ln[i] > 0:
            out[offsets[i]:offsets[i + 1]] = data[off[i]:off[i] + ln[i]]
    return out, offsets


@nb.njit([nb.types.Tuple((_U8, _I64, _I64))(u8, _I64, _I64) for u8 in (_U8, _U8_RO)])
def _strsets(data, off, ln):
    n = len(off)
    n_tok = np.zeros(n + 1, dtype=np.int64)
    total = 0
    for i in range(n):
        if ln[i] <= 0:
            n_tok[i + 1] = n_tok[i]
            continue
        p = off[i]
        k = 0
        shift = 0
        while True:
            b = data[p]
            p += 1
            k |= (np.int64(b) & 0x7F) << shift
            if b < 0x80:
                break
            shift += 7
        n_tok[i + 1] = n_tok[i] + k
        for _ in range(k):
            m = 0
            shift = 0
            while True:
                b = data[p]
                p += 1
                m |= (np.int64(b) & 0x7F) << shift
                if b < 0x80:
                    break
                shift += 7
            total += m
    out = np.empty(total, dtype=np.uint8)
    tok_off = np.zeros(n_tok[n] + 1, dtype=np.int64)
    t = 0
    for i in range(n):
        if ln[i] <= 0:
            continue
        p = off[i]
        k = 0
        shift = 0
        while True:
            b = data[p]
            p += 1
            k |= (np.int64(b) & 0x7F) << shift
            if b < 0x80:
                break
            shift += 7
        lens = np.empty(k, dtype=np.int64)
        for j in range(k):
            m = 0
            shift = 0
            while True:
                b = data[p]
                p += 1
                m |= (np.int64(b) & 0x7F) << shift
                if b < 0x80:
                    break
                shift += 7
            lens[j] = m
        for j in range(k):
            start = tok_off[t]
            out[start:start + lens[j]] = data[p:p + lens[j]]
            p += lens[j]
            tok_off[t + 1] = start + lens[j]
            t += 1
    return out, tok_off, n_tok


@nb.njit('Tuple((uint8[::1], uint8[::1]))(uint8[::1], int64[::1])')
def _split_vec(raw, offsets):
    half = len(raw) // 2
    idx = np.empty(half, dtype=np.uint8)
    val = np.empty(half, dtype=np.uint8)
    for i in range(len(offsets) - 1):
        a, b = offsets[i], offsets[i + 1]
        h = (b - a) // 2
        idx[a // 2:a // 2 + h] = raw[a:a + h]
        val[a // 2:a // 2 + h] = raw[a + h:b]
    return idx, val


def _null_mask(ln):
    return pa.array(ln < 0)


def _list_array(values, elem_offsets, ln, typ):
    return pa.ListArray.from_arrays(pa.array(elem_offsets.astype(np.int32)),
                                    pa.array(values, type=typ), mask=_null_mask(ln))


def decode_frame(layout, data, starts, ends):
    """Return the packed records `data[starts[i]:ends[i]]` decoded into a frame, in the
    order given."""
    data = np.ascontiguousarray(data, dtype=np.uint8)
    starts = np.ascontiguousarray(starts, dtype=np.int64)
    ends = np.ascontiguousarray(ends, dtype=np.int64)
    n = len(starts)
    off, ln = scan_dirs(data, starts, ends, len(layout))
    scalar, arrow = {}, {}
    for c, (name, kind) in enumerate(layout):
        o, l = np.ascontiguousarray(off[:, c]), np.ascontiguousarray(ln[:, c])
        if kind == 'strset':
            toks, tok_off, n_tok = _strsets(data, o, l)
            strings = pa.StringArray.from_buffers(
                len(tok_off) - 1, pa.py_buffer(tok_off.astype(np.int32)), pa.py_buffer(toks))
            arrow[name] = pa.ListArray.from_arrays(pa.array(n_tok.astype(np.int32)),
                                                   strings, mask=_null_mask(l))
            continue
        raw, offsets = copy_col(data, o, l)
        if kind == 'str':
            validity = pa.array(l >= 0).buffers()[1] if (l < 0).any() else None
            arr = pa.StringArray.from_buffers(n, pa.py_buffer(offsets.astype(np.int32)),
                                              pa.py_buffer(raw), validity)
            scalar[name] = arr.to_numpy(zero_copy_only=False)
        elif kind == 'f64':
            vals = np.full(n, np.nan)
            present = l >= 0
            vals[present] = raw.view('<f8')
            scalar[name] = vals
        elif kind == 'ids':
            arrow[name] = _list_array(raw.view('<i8'), offsets // 8, l, pa.int64())
        elif kind == 'vec':
            idx, val = _split_vec(raw, offsets)
            elem = offsets // 8
            arrow[name + VEC_IDX_SUFFIX] = _list_array(idx.view('<i4'), elem, l, pa.int32())
            arrow[name + VEC_VAL_SUFFIX] = _list_array(val.view('<f4'), elem, l, pa.float32())
        elif kind == 'pkl':
            scalar[name] = np.array(
                [None if l[i] < 0 else pickle.loads(raw[offsets[i]:offsets[i + 1]].tobytes())
                 for i in range(n)], dtype=object)
        else:
            raise ValueError(f'unknown stored kind {kind!r}; kinds are {KINDS}')
    df = pd.DataFrame(scalar, index=np.arange(n)) if scalar else pd.DataFrame(index=np.arange(n))
    df.attrs['_mf_arrow'] = AttrsByRef(arrow)
    return df
