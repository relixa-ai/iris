"""The 64-bit token hash, MurmurHash3 x64 128 of a token's UTF-8 bytes as `mmh3.hash64`
gives its first half, computed for every q-gram of a text without making the tokens."""
import numba as nb
import numpy as np

_C1 = np.uint64(0x87C37B91114253D5)
_C2 = np.uint64(0x4CF5AD432745937F)
_M1 = np.uint64(0xFF51AFD7ED558CCD)
_M2 = np.uint64(0xC4CEB9FE1A85EC53)
_FIVE = np.uint64(5)
_N1 = np.uint64(0x52DCE729)
_N2 = np.uint64(0x38495AB5)


@nb.njit(inline='always')
def _rotl(x, r):
    return (x << np.uint64(r)) | (x >> np.uint64(64 - r))


@nb.njit(inline='always')
def _fmix(k):
    k ^= k >> np.uint64(33)
    k *= _M1
    k ^= k >> np.uint64(33)
    k *= _M2
    k ^= k >> np.uint64(33)
    return k


@nb.njit(inline='always')
def _byte(data, i):
    return np.uint64(data[i])


@nb.njit(cache=False)
def _murmur(data, start, end, seed):
    n = end - start
    h1 = np.uint64(seed)
    h2 = np.uint64(seed)
    nblocks = n // 16
    for b in range(nblocks):
        p = start + 16 * b
        k1 = np.uint64(0)
        k2 = np.uint64(0)
        for j in range(8):
            k1 |= _byte(data, p + j) << np.uint64(8 * j)
            k2 |= _byte(data, p + 8 + j) << np.uint64(8 * j)
        k1 *= _C1
        k1 = _rotl(k1, 31)
        k1 *= _C2
        h1 ^= k1
        h1 = _rotl(h1, 27)
        h1 += h2
        h1 = h1 * _FIVE + _N1
        k2 *= _C2
        k2 = _rotl(k2, 33)
        k2 *= _C1
        h2 ^= k2
        h2 = _rotl(h2, 31)
        h2 += h1
        h2 = h2 * _FIVE + _N2
    tail = start + 16 * nblocks
    rest = n & 15
    k1 = np.uint64(0)
    k2 = np.uint64(0)
    if rest > 8:
        for j in range(rest - 8):
            k2 |= _byte(data, tail + 8 + j) << np.uint64(8 * j)
        k2 *= _C2
        k2 = _rotl(k2, 33)
        k2 *= _C1
        h2 ^= k2
    if rest > 0:
        for j in range(min(rest, 8)):
            k1 |= _byte(data, tail + j) << np.uint64(8 * j)
        k1 *= _C1
        k1 = _rotl(k1, 31)
        k1 *= _C2
        h1 ^= k1
    h1 ^= np.uint64(n)
    h2 ^= np.uint64(n)
    h1 += h2
    h2 += h1
    h1 = _fmix(h1)
    h2 = _fmix(h2)
    h1 += h2
    return h1


@nb.njit(cache=False)
def _qgrams(data, q, seed):
    starts = np.empty(len(data) + 1, np.int64)
    n = 0
    for i in range(len(data)):
        if (data[i] & 0xC0) != 0x80:
            starts[n] = i
            n += 1
    starts[n] = len(data)
    count = n - q + 1
    out = np.empty(max(count, 0), np.uint64)
    for i in range(count):
        out[i] = _murmur(data, starts[i], starts[i + q], seed)
    return out


def qgram_hashes(text, q, seed):
    """Return the hash of every substring of `q` characters of `text`, in order, as
    int64: the hashes of `[text[i:i + q] for i in ...]`."""
    data = np.frombuffer(text.encode('utf-8', 'surrogatepass'), dtype=np.uint8)
    return _qgrams(data, q, np.uint64(seed)).view(np.int64)
