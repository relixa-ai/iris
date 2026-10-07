"""Spark's Hadoop filesystem, for the files the driver reads and writes itself."""

OBJECT_STORES = {'s3a', 's3', 's3n', 'gs', 'wasb', 'wasbs', 'abfs', 'abfss', 'oss'}


class HadoopFiles:
    """The filesystem of a Spark session, reached through Hadoop's `FileSystem`, with
    atomic writes."""

    def __init__(self, spark):
        self._jvm = spark._jvm
        self._conf = spark._jsc.hadoopConfiguration()

    def _fs(self, name):
        path = self._jvm.org.apache.hadoop.fs.Path(str(name))
        fs = path.getFileSystem(self._conf)
        fs.setWriteChecksum(False)
        return fs, path

    def _pending(self, path):
        return self._jvm.org.apache.hadoop.fs.Path(path.getParent(),
                                                   '.' + path.getName() + '.tmp')

    def exists(self, name):
        """Return whether `name` exists."""
        fs, path = self._fs(name)
        return bool(fs.exists(path) or (path.getParent() is not None
                                        and fs.exists(self._pending(path))))

    def makedirs(self, name):
        """Create the directory `name` and its parents."""
        fs, path = self._fs(name)
        fs.mkdirs(path)

    def delete(self, name):
        """Delete the file or directory `name`."""
        fs, path = self._fs(name)
        fs.delete(path, True)
        if path.getParent() is not None:
            fs.delete(self._pending(path), False)

    def read_bytes(self, name):
        """Return the contents of the file `name`."""
        fs, path = self._fs(name)
        if not fs.exists(path) and path.getParent() is not None \
                and fs.exists(self._pending(path)):
            path = self._pending(path)
        stream = fs.open(path)
        try:
            out = self._jvm.java.io.ByteArrayOutputStream()
            self._jvm.org.apache.hadoop.io.IOUtils.copyBytes(stream, out, 1 << 16, False)
            return bytes(out.toByteArray())
        finally:
            stream.close()

    def write_bytes(self, name, data):
        """Write `data` to the file `name` atomically."""
        fs, path = self._fs(name)
        if fs.getScheme() in OBJECT_STORES or path.getParent() is None:
            _write(fs, path, data)
            return
        tmp = self._pending(path)
        _write(fs, tmp, data)
        fs.delete(path, False)
        if not fs.rename(tmp, path):
            raise OSError(f'could not move {tmp.toString()} to {path.toString()}')


def active():
    """Return the filesystem of the active SparkSession, or None when no session is
    active."""
    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession()
    return HadoopFiles(spark) if spark is not None else None


def _write(fs, path, data):
    out = fs.create(path, True)
    try:
        out.write(bytearray(data))
    finally:
        out.close()
