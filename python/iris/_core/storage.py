"""Scratch files on disk: memory-mapped arrays and the checks on the space they need."""
import os
import shutil
import tempfile
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
    fd, path = mkstemp(suffix=suffix, dir=spill_dir())
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
            self._mmap_arr = np.memmap(f, mode='r', shape=self._shape, dtype=self._dtype)
    
    def delete(self):
        """Close the array and delete its file."""
        if self._local_mmap_file.exists():
            self._local_mmap_file.unlink()
        if getattr(self, '_finalizer', None) is not None:
            self._finalizer.detach()


class MemmapArrayBuilder:
    """A writer that builds a `MemmapArray` chunk by chunk, without holding the whole
    array in memory."""

    def __init__(self, n, dtype):
        self._dtype = np.dtype(dtype)
        self._shape = (int(n),)
        self._path = _mkstemp(suffix='.mmap_arr')
        require_free_space(self._shape[0] * self._dtype.itemsize, self._path.parent)
        self._mmap = np.memmap(self._path, shape=self._shape, dtype=self._dtype, mode='w+')
        self._pos = 0

    def write(self, chunk):
        """Append `chunk` to the array."""
        chunk = np.asarray(chunk, dtype=self._dtype)
        end = self._pos + len(chunk)
        if end > self._shape[0]:
            raise ValueError(
                f'MemmapArrayBuilder overflow: allocated {self._shape[0]} elements, '
                f'received at least {end}'
            )
        self._mmap[self._pos:end] = chunk
        self._pos = end

    def finish(self):
        """Return the finished `MemmapArray`."""
        if self._pos != self._shape[0]:
            raise ValueError(
                f'MemmapArrayBuilder underflow: allocated {self._shape[0]} elements, '
                f'received {self._pos}'
            )
        self._mmap.flush()
        del self._mmap
        return MemmapArray.from_file(self._path, self._shape, self._dtype)


def memmap_map(source, func, dtype, chunk_size=1 << 20):
    """Return a new `MemmapArray` with an elementwise function applied to a memory-
    mapped array, chunk by chunk."""
    n = len(source)
    builder = MemmapArrayBuilder(n, dtype)
    for start in range(0, n, chunk_size):
        builder.write(func(np.array(source[start:start + chunk_size])))
    return builder.finish()
