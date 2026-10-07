"""The worker processes of the local engine."""
import contextlib
import multiprocessing as mp
import multiprocessing.spawn
import os
import sys
import threading
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool

try:
    import cloudpickle
except ImportError:
    from joblib.externals import cloudpickle

from .._core.utils import get_logger

log = get_logger(__name__)

PRELOAD = ['numpy', 'pandas', 'pyarrow', 'numba', 'iris._core.recpack',
           'iris._core.feature']

_THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')

_fn = None

WORKERS = None

WORKERS_ENV = 'IRIS_WORKERS'


def workers_for(per_worker_gb):
    """Return the number of workers a stage runs: `WORKERS`, then `IRIS_WORKERS`, then as
    many as fit the memory each worker is planned at."""
    if WORKERS is not None:
        return int(WORKERS)
    env = os.environ.get(WORKERS_ENV, '').strip()
    if env:
        try:
            n = int(env)
        except ValueError:
            n = 0
        if n < 1:
            raise ValueError(f'{WORKERS_ENV} must be a positive integer, got {env!r}')
        return n
    from .. import resources
    return resources.local_workers(per_worker_gb)


@contextlib.contextmanager
def using_workers(n):
    """Fix the number of workers for the duration of a block."""
    global WORKERS
    before, WORKERS = WORKERS, n
    try:
        yield
    finally:
        WORKERS = before

_starting = threading.local()
_preparation_data = multiprocessing.spawn.get_preparation_data


def _without_main(name, *args, **kwargs):
    data = _preparation_data(name, *args, **kwargs)
    if getattr(_starting, 'pool', False):
        data.pop('init_main_from_path', None)
        data.pop('init_main_from_name', None)
    return data


if getattr(multiprocessing.spawn.get_preparation_data, '__name__', '') != '_without_main':
    multiprocessing.spawn.get_preparation_data = _without_main


@contextlib.contextmanager
def _starting_workers():
    _starting.pool = True
    try:
        yield
    finally:
        _starting.pool = False


def _context():
    if sys.platform == 'win32':
        return mp.get_context('spawn')
    ctx = mp.get_context('forkserver')
    ctx.set_forkserver_preload(PRELOAD)
    return ctx


def _init(payload):
    global _fn
    for var in _THREAD_VARS:
        os.environ[var] = '1'
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    _fn = cloudpickle.loads(payload)


def _call(task):
    return _fn(task)


def run_tasks(fn, tasks, workers, restarts=2):
    """Return `fn` applied to every task, computed by worker processes, in task
    order."""
    tasks = list(tasks)
    results = [None] * len(tasks)
    pending = list(range(len(tasks)))
    payload = cloudpickle.dumps(fn)
    broken = 0
    while pending:
        pool = ProcessPoolExecutor(max_workers=max(1, min(workers, len(pending))),
                                   mp_context=_context(), initializer=_init,
                                   initargs=(payload,))
        try:
            with _starting_workers():
                futures = {pool.submit(_call, tasks[i]): i for i in pending}
            wait(futures, return_when=FIRST_EXCEPTION)
            finished = {f for f in futures if f.done() and not f.cancelled()}
            errors = [f.exception() for f in finished if f.exception() is not None]
            for f in finished:
                if f.exception() is None:
                    results[futures[f]] = f.result()
            pending = [i for f, i in futures.items()
                       if f not in finished or f.exception() is not None]
            real = [e for e in errors if not isinstance(e, BrokenProcessPool)]
            if real:
                raise real[0]
            if errors:
                broken += 1
                if broken > restarts:
                    raise RuntimeError(
                        f'worker processes died {broken} times; {len(pending)} of '
                        f'{len(tasks)} task(s) unfinished. A worker killed by the '
                        f'operating system usually ran out of memory: run with fewer '
                        f'workers.') from errors[0]
                log.warning(f'a worker process died; restarting the pool and rerunning '
                            f'{len(pending)} unfinished task(s) '
                            f'({broken} of {restarts} restarts)')
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    return results


@contextlib.contextmanager
def atomic_output(path):
    """Yield a temporary path beside `path`, moved into place when the block
    succeeds."""
    path = os.fspath(path)
    head, name = os.path.split(path)
    tmp = os.path.join(head, f'.{name}.tmp')
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
