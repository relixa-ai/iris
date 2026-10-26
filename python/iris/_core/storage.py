"""Scratch files on disk: memory-mapped arrays and the checks on the space they need."""
import ctypes
import mmap
import os
import re
import shutil
import socket
import tempfile
import time
import weakref
from pathlib import Path
from tempfile import mkstemp

import numpy as np

from .utils import get_logger

log = get_logger(__name__)

SPILL_DIR_ENV = 'IRIS_SPILL_DIR'


def unknown_id_error(bad_ids, side=None):
    """Return the error raised for candidate ids that name no record."""
    bad = np.asarray(bad_ids)
    sample = ', '.join(str(int(x)) for x in bad[:5])
    suffix = ', ...' if len(bad) > 5 else ''
    table = {'a': 'table_a', 'b': 'table_b'}.get(side, 'the tables')
    return ValueError(
        f'unknown id(s) [{sample}{suffix}] ({len(bad)} total) not in {table}: the '
        f'pairs point at records that do not exist. Run '
        f'iris.check_candidates(candidates, A, B) before featurizing to '
        f'catch this up front.')


_HOST = re.sub(r'[^A-Za-z0-9]', '', socket.gethostname())[:32] or 'host'

_SWEPT = []

SWEEP_AGE_SECONDS = 3600


def scratch_prefix(kind):
    """Return the prefix of a scratch directory or file of this process: its kind, host
    and process id, by which a later run finds what a killed run left."""
    return f'iris-{kind}-{_HOST}-{os.getpid()}-'


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def sweep(directory=None):
    """Remove the scratch directories and files that runs on this host left when they
    were killed: this user's, unchanged for an hour, whose process has ended."""
    directory = directory or spill_dir() or tempfile.gettempdir()
    if directory in _SWEPT:
        return 0
    _SWEPT.append(directory)
    pattern = re.compile(rf'^iris-[a-z]+-{_HOST}-(\d+)-')
    removed = 0
    try:
        names = os.listdir(directory)
    except OSError:
        return 0
    uid = os.getuid() if hasattr(os, 'getuid') else None
    for name in names:
        found = pattern.match(name)
        if not found or int(found.group(1)) == os.getpid():
            continue
        path = os.path.join(directory, name)
        try:
            st = os.stat(path)
        except OSError:
            continue
        if (uid is not None and st.st_uid != uid) or \
                time.time() - st.st_mtime < SWEEP_AGE_SECONDS or _alive(int(found.group(1))):
            continue
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            _unlink_quietly(path)
        removed += 1
    if removed:
        log.info(f'removed {removed} scratch director{"y" if removed == 1 else "ies"} or '
                 f'file{"" if removed == 1 else "s"} that killed runs left in {directory}')
    return removed


def spill_dir():
    """Return the directory for scratch files: `IRIS_SPILL_DIR`, else the first
    `spark.local.dir`, else None."""
    env_dir = os.environ.get(SPILL_DIR_ENV)
    if env_dir:
        os.makedirs(env_dir, exist_ok=True)
        return env_dir

    try:
        from .frames import active_spark_context
        sc = active_spark_context()
        if sc is not None:
            local_dir = sc.getConf().get('spark.local.dir', None)
            if local_dir:
                first = local_dir.split(',')[0].strip()
                if first and os.path.isdir(first):
                    return first
    except Exception:
        pass

    return None


def _mkstemp(suffix):
    fd, path = mkstemp(prefix=scratch_prefix('mmap'), suffix=suffix, dir=spill_dir())
    os.close(fd)
    return Path(path)


def require_free_space(nbytes, path=None):
    """Raise an error when the scratch directory cannot hold `nbytes`."""
    target = str(path) if path is not None else (spill_dir() or tempfile.gettempdir())
    try:
        free = shutil.disk_usage(target).free
    except OSError:
        return
    if free < nbytes:
        raise RuntimeError(
            f'not enough free disk space for Iris scratch files in {target!r}: '
            f'need {nbytes / 2**30:.2f} GiB, {free / 2**30:.2f} GiB available. '
            f'These files are memory-mapped, so the space is needed regardless of '
            f'available RAM. Set {SPILL_DIR_ENV} to a directory on a larger volume, or '
            f'free space on this one.'
        )


def _unlink_quietly(path):
    try:
        os.unlink(path)
    except OSError:
        pass


_libc = None


def _c_library():
    global _libc
    if _libc is None:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                              ctypes.c_int, ctypes.c_long)
        libc.munmap.restype = ctypes.c_int
        libc.munmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        _libc = libc
    return _libc


def map_read_only(path, dtype, shape):
    """Return a read-only array over the file at `path`, mapped and the file closed again,
    so that the arrays a process maps hold none of its open files."""
    dtype = np.dtype(dtype)
    shape = tuple(int(x) for x in np.atleast_1d(shape))
    n = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if os.name != 'posix' or n == 0:
        return np.memmap(path, mode='r', shape=shape, dtype=dtype)
    libc = _c_library()
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        if size < n:
            raise ValueError(f'{path} holds {size:,} bytes, fewer than the {n:,} of its array')
        addr = libc.mmap(None, n, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
    finally:
        os.close(fd)
    if addr is None or addr == ctypes.c_void_p(-1).value:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), str(path))
    buf = (ctypes.c_char * n).from_address(addr)
    weakref.finalize(buf, libc.munmap, addr, n)
    arr = np.frombuffer(buf, dtype=dtype).reshape(shape)
    arr.flags.writeable = False
    return arr


class MemmapArray:

    """A NumPy array in a memory-mapped file that worker processes open by name."""
    def __init__(self, arr):
        self._dtype = arr.dtype
        self._shape = arr.shape
        self._local_mmap_file = _mkstemp(suffix='.mmap_arr')
        require_free_space(arr.nbytes, self._local_mmap_file.parent)
        mmap = np.memmap(self._local_mmap_file, shape=self._shape, dtype=self._dtype, mode='w+')
        mmap[:] = arr[:]
        mmap.flush()
        del mmap
        self._mmap_arr = None
        self._locate = None
        self._finalizer = weakref.finalize(self, _unlink_quietly, str(self._local_mmap_file))

    @classmethod
    def from_file(cls, path, shape, dtype):
        """Return a `MemmapArray` over an existing file, without copying it."""
        obj = cls.__new__(cls)
        obj._dtype = np.dtype(dtype)
        obj._shape = tuple(shape)
        obj._local_mmap_file = Path(path)
        obj._mmap_arr = None
        obj._locate = None
        obj._finalizer = weakref.finalize(obj, _unlink_quietly, str(path))
        return obj

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_mmap_arr'] = None
        state['_finalizer'] = None
        return state

    @property
    def values(self):
        """The opened array, or None before `init`."""
        return self._mmap_arr

    @property
    def path(self):
        """The file the array was built in."""
        return self._local_mmap_file

    @property
    def shape(self):
        """The shape of the array."""
        return self._shape
    
    def __len__(self):
        return self._shape[0]

    def locate_with(self, locate):
        """Open the array at the path that `locate` returns for its file name, where the
        file was shipped."""
        self._locate = locate

    def init(self):
        """Open the memory-mapped file."""
        if self._mmap_arr is None:
            f = self._local_mmap_file
            if getattr(self, '_locate', None) is not None:
                f = self._locate(f.name)
                if not os.path.exists(f):
                    raise RuntimeError(f'cannot find the memory-mapped file {f}')
            self._mmap_arr = map_read_only(f, self._dtype, self._shape)
    
    def delete(self):
        """Close the array and delete its file."""
        if self._local_mmap_file.exists():
            self._local_mmap_file.unlink()
        if getattr(self, '_finalizer', None) is not None:
            self._finalizer.detach()


class MemmapArrayBuilder:
    """A writer that builds a `MemmapArray` chunk by chunk, without holding the whole
    array in memory or its file open between writes."""

    def __init__(self, n, dtype):
        self._dtype = np.dtype(dtype)
        self._shape = (int(n),)
        self._path = _mkstemp(suffix='.mmap_arr')
        require_free_space(self._shape[0] * self._dtype.itemsize, self._path.parent)
        os.truncate(self._path, self._shape[0] * self._dtype.itemsize)
        self._pos = 0

    def write(self, chunk):
        """Append `chunk` to the array."""
        chunk = np.ascontiguousarray(chunk, dtype=self._dtype)
        end = self._pos + len(chunk)
        if end > self._shape[0]:
            raise ValueError(
                f'MemmapArrayBuilder overflow: allocated {self._shape[0]} elements, '
                f'received at least {end}'
            )
        with open(self._path, 'r+b') as fh:
            fh.seek(self._pos * self._dtype.itemsize)
            fh.write(memoryview(chunk).cast('B'))
        self._pos = end

    def finish(self):
        """Return the finished `MemmapArray`."""
        if self._pos != self._shape[0]:
            raise ValueError(
                f'MemmapArrayBuilder underflow: allocated {self._shape[0]} elements, '
                f'received {self._pos}'
            )
        return MemmapArray.from_file(self._path, self._shape, self._dtype)


def memmap_map(source, func, dtype, chunk_size=1 << 20):
    """Return a new `MemmapArray` with an elementwise function applied to a memory-
    mapped array, chunk by chunk."""
    n = len(source)
    builder = MemmapArrayBuilder(n, dtype)
    for start in range(0, n, chunk_size):
        builder.write(func(np.array(source[start:start + chunk_size])))
    return builder.finish()
