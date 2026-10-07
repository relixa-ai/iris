"""The `IRIS_*` environment settings: reading a `.env` file and setting values from
Python."""
import os
import sys


def load_env(path='.env', override=False):
    """Copy the assignments in a `.env` file into the environment and return those
    applied."""
    applied = {}
    try:
        with open(path) as fh:
            lines = fh.readlines()
    except OSError:
        return applied
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, val = line.partition('=')
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = val
            applied[key] = val
    return applied


def pin_worker_python():
    """Set Spark's Python workers to this interpreter unless `PYSPARK_PYTHON` is already
    set."""
    if sys.executable and not os.environ.get('PYSPARK_PYTHON'):
        os.environ['PYSPARK_PYTHON'] = sys.executable
    chosen = os.environ.get('PYSPARK_PYTHON')
    sc = _active_spark_context()
    if chosen and sc is not None and getattr(sc, 'pythonExec', None) == 'python3':
        sc.pythonExec = chosen
    return chosen


def _active_spark_context():
    from ._core.frames import active_spark_context
    return active_spark_context()


def configure(**settings):
    """Set `IRIS_*` environment settings from keyword arguments, and unset those given
    as None."""
    for key, val in settings.items():
        name = key if key.startswith('IRIS_') else 'IRIS_' + key.upper()
        if val is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(val)
