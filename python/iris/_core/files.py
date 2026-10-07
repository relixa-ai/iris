"""The filesystem through which a run reads and writes its own files, by path or URI."""
import contextlib
import contextvars
import os
import posixpath
import re

_SCHEME = re.compile(r'^([a-zA-Z][a-zA-Z0-9+.-]*)://')
_active = contextvars.ContextVar('iris_files', default=None)


def scheme(name):
    """Return the scheme of a URI, such as `'s3a'`, or None for a plain path."""
    m = _SCHEME.match(str(name))
    return m.group(1).lower() if m else None


def join(base, *names):
    """Return a base path or URI joined with further names."""
    return posixpath.join(str(base), *names)


def basename(name):
    """Return the last component of a path or URI."""
    return posixpath.basename(str(name).rstrip('/'))


def is_remote(name):
    """Return whether `name` is a URI for storage other than this machine's files."""
    return scheme(name) not in (None, 'file')


def local_path(name):
    """Return `name` as a path of this machine, refusing a URI for other storage."""
    name = str(name)
    s = scheme(name)
    if s is None:
        return name
    if s == 'file':
        return name[len('file://'):]
    raise ValueError(
        f'{name}: the local engine reads and writes paths of this machine, and a {s}:// '
        f'URI names storage that Spark reaches. Pass Spark DataFrames to run on Spark.')


class LocalFiles:
    """This machine's filesystem."""

    _path = staticmethod(local_path)

    def exists(self, name):
        """Return whether `name` exists."""
        return os.path.exists(self._path(name))

    def makedirs(self, name):
        """Create the directory `name` and its parents."""
        os.makedirs(self._path(name), exist_ok=True)

    def delete(self, name):
        """Delete the file or directory `name`."""
        import shutil

        path = self._path(name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)

    def read_bytes(self, name):
        """Return the contents of the file `name`."""
        with open(self._path(name), 'rb') as fh:
            return fh.read()

    def write_bytes(self, name, data):
        """Write `data` to the file `name` atomically."""
        path = self._path(name)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = os.path.join(os.path.dirname(os.path.abspath(path)),
                           '.' + os.path.basename(path) + '.tmp')
        with open(tmp, 'wb') as fh:
            fh.write(data)
        os.replace(tmp, path)


def current():
    """Return the filesystem active in this context, by default this machine's."""
    return _active.get() or LocalFiles()


@contextlib.contextmanager
def using(files):
    """Make `files` the active filesystem for the duration of a block."""
    token = _active.set(files)
    try:
        yield files
    finally:
        _active.reset(token)
