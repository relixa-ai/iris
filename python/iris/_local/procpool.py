"""The worker processes of the local engine, started as the memory they are measured to
use allows."""
import collections
import contextlib
import multiprocessing as mp
import multiprocessing.connection
import multiprocessing.spawn
import os
import sys
import threading
import time
import traceback

try:
    import cloudpickle
except ImportError:
    from joblib.externals import cloudpickle

from .. import resources
from .._core.utils import get_logger

log = get_logger(__name__)

PRELOAD = ['numpy', 'pandas', 'pyarrow', 'numba', 'iris._core.recpack',
           'iris._core.feature']

_THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')

WORKERS = None

WORKERS_ENV = 'IRIS_WORKERS'

AVAILABLE_SHARE = 0.9

MARGIN = 1.25

DEFAULT_TASK_BYTES = 2**30

SAMPLE_SECONDS = 0.01

MONITOR_SECONDS = 0.1

LAST = {}


def workers_for(*_):
    """Return the most workers a stage runs: `WORKERS`, then `IRIS_WORKERS`, then the
    cores this process may use."""
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
    return resources.budget()[1]


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


WORKER_ENV = {'MallocLargeCache': '0'} if sys.platform == 'darwin' else {}


@contextlib.contextmanager
def _starting_workers():
    _starting.pool = True
    saved = {k: os.environ.get(k) for k in (*WORKER_ENV, *_THREAD_VARS)}
    os.environ.update({k: v for k, v in WORKER_ENV.items() if saved[k] is None})
    os.environ.update(dict.fromkeys(_THREAD_VARS, '1'))
    try:
        yield
    finally:
        _starting.pool = False
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _context():
    if sys.platform == 'win32':
        return mp.get_context('spawn')
    ctx = mp.get_context('forkserver')
    ctx.set_forkserver_preload(PRELOAD)
    return ctx


class _Meter:

    def __init__(self):
        self.pid = os.getpid()
        self.base = resources.private_bytes(self.pid)
        self.recorded = resources.peak_private_bytes(self.pid)
        self.peak = self.base
        self._stop = threading.Event()
        self._thread = None
        if self.base is not None:
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()

    def _sample(self):
        while not self._stop.wait(SAMPLE_SECONDS):
            now = resources.private_bytes(self.pid)
            if now is not None and now > self.peak:
                self.peak = now

    def stop(self):
        """Return the private memory before the task and the most during it."""
        if self._thread is None:
            return None, None
        self._stop.set()
        self._thread.join()
        now = resources.private_bytes(self.pid) or 0
        peak = max(self.peak, now)
        recorded = resources.peak_private_bytes(self.pid)
        if recorded is not None and self.recorded is not None and recorded > self.recorded:
            peak = max(peak, recorded)
        return self.base, peak


def _watch():
    parent = mp.parent_process()
    if parent is not None:
        parent.join()
        os._exit(1)


def _serve(conn, payload):
    threading.Thread(target=_watch, daemon=True).start()
    if sys.platform.startswith('linux'):
        try:
            with open('/proc/self/oom_score_adj', 'w') as fh:
                fh.write('1000')
        except OSError:
            pass
    fn = cloudpickle.loads(payload)
    conn.send(('ready', None, resources.private_bytes(os.getpid()), None))
    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            return
        if message is None:
            return
        i, task = message
        meter = _Meter()
        try:
            reply = ('done', i, fn(task))
        except MemoryError as e:
            reply = ('memory', i, str(e) or None)
        except BaseException as e:
            reply = ('error', i, (e, traceback.format_exc()))
        measured = meter.stop()
        try:
            conn.send(reply + (measured,))
        except Exception as e:
            conn.send(('error', i, (RuntimeError(f'task {i} gave a result that cannot be '
                                                 f'returned: {e}'), traceback.format_exc()),
                       measured))


class _RemoteTraceback(Exception):
    def __init__(self, tb):
        self.tb = tb

    def __str__(self):
        return self.tb


class _Worker:
    def __init__(self, ctx, payload):
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_serve, args=(child, payload), daemon=True)
        with _starting_workers():
            self.proc.start()
        child.close()
        self.ready = False
        self.task = None
        self.base = None
        self.private = 0
        self.at_start = 0
        self.started = 0.0

    def assign(self, i, task):
        self.task = i
        self.at_start = self.private
        self.started = time.monotonic()
        self.conn.send((i, task))

    def stop(self):
        try:
            self.conn.send(None)
        except (OSError, ValueError):
            pass

    def end(self):
        if self.proc.is_alive():
            self.proc.terminate()
        self.proc.join(5)
        self.conn.close()


class Plan:
    """How many tasks fit at once: the limit, what each task is measured to use, and
    what the processes of the run hold now."""

    def __init__(self, task_bytes=None, limit=None):
        if limit is None:
            total_gb, _, self.source = resources.budget()
            limit = total_gb * 2**30 * resources.SHARE
            here = resources.private_bytes(os.getpid())
            available = resources.available_gb()
            if here is not None and available is not None:
                limit = min(limit, here + available * 2**30 * AVAILABLE_SHARE)
        else:
            self.source = 'given'
        self.limit = limit
        self.prior = resources.PRIOR * task_bytes if task_bytes else DEFAULT_TASK_BYTES
        self.task = None
        self.worker = None
        self.main = 0

    def estimate(self):
        """Return the memory planned for one more task."""
        return self.prior if self.task is None else self.task * MARGIN

    def record(self, base, peak):
        """Take in one task's memory: its worker's before it, and the most during it."""
        if base is None or peak is None:
            return
        grew = max(0, peak - base)
        self.task = grew if self.task is None else max(self.task, grew)

    def in_use(self, workers):
        """Return the memory the run holds now, with what its running and starting
        workers may still take."""
        start = self.worker if self.worker is not None else resources.WORKER_BYTES_GUESS
        held = self.main + sum(w.private for w in workers if w.ready)
        growth = sum(max(0, self.estimate() - (w.private - w.at_start))
                     for w in workers if w.ready and w.task is not None)
        starting = sum(start + self.estimate() for w in workers if not w.ready)
        return held + growth + starting

    def fits_task(self, workers):
        """Return whether one more task fits on a worker that is already running."""
        return self.in_use(workers) + self.estimate() <= self.limit

    def fits_worker(self, workers):
        """Return whether one more worker, with one task, fits."""
        start = self.worker if self.worker is not None else resources.WORKER_BYTES_GUESS
        return self.in_use(workers) + start + self.estimate() <= self.limit

    def spare(self, workers):
        """Return the idle worker holding most, to stop so a waiting task fits, or None; the
        last worker only when it holds more than a fresh worker and a task."""
        idle = [w for w in workers if w.ready and w.task is None]
        if not idle or all(w.task is None for w in workers) or self.fits_task(workers):
            return None
        worst = max(idle, key=lambda w: w.private)
        if len(idle) > 1 or (self.worker is not None
                             and worst.private > self.worker + self.estimate()):
            return worst
        return None


def _measure(plan, workers):
    plan.main = resources.private_bytes(os.getpid()) or 0
    for w in workers:
        found = resources.private_bytes(w.proc.pid)
        if found is not None:
            w.private = found
    return plan.main + sum(w.private for w in workers)


def _raise(error, tb):
    raise error from _RemoteTraceback(f'\n"""\n{tb}"""')


def memory_error(error, text):
    """Return a MemoryError whose message adds `text` to that of `error`, for a task to say
    what it held when memory ran out."""
    return MemoryError(f'{error}; {text}' if str(error) else text)


def _gave_up(i, times, kind, plan, n_left, n, error=None):
    seen = '' if plan is None or plan.task is None else (
        f', after tasks of up to {plan.task / 2**20:,.0f} MiB within {plan.limit / 2**30:.1f} '
        f'GiB')
    what = (f'worker processes died {times} times running task {i} alone' if kind == 'died'
            else f'task {i} ran out of memory {times} times running alone')
    said = f' The task failed with: {error}.' if error else ''
    return RuntimeError(
        f'{what}; {n_left} of {n} task(s) unfinished{seen}.{said} A worker killed by the '
        f'operating system usually ran out of memory: give Iris more memory (close other '
        f'programs, or raise {resources.MEMORY_ENV} or the limit of its container or job), '
        f'or split the input when a single record is very large.')


def _never_started(times, code):
    return RuntimeError(
        f'worker processes ended {times} times before they could take a task (exit code '
        f'{code}). A worker that ends before its first task failed while starting, for '
        f'example importing a module or unpickling the function it runs, and its error '
        f'is printed above on standard error. Set {WORKERS_ENV}=1 to see the error of one '
        f'worker alone, and check that the Python the workers run can import Iris and the '
        f'modules of your features, model and labeler.')


def run_tasks(fn, tasks, workers=None, restarts=2, task_bytes=None, done=None):
    """Return `fn` applied to every task by worker processes, in task order, starting a task
    only when its measured memory is free and calling `done(i, result)` as task `i` ends."""
    tasks = list(tasks)
    results = [None] * len(tasks)
    if not tasks:
        return results
    cap = max(1, min(workers or workers_for(), len(tasks)))
    measured = resources.private_bytes(os.getpid()) is not None
    plan = Plan(task_bytes) if measured else None
    payload = cloudpickle.dumps(fn)
    ctx = _context()
    pending = collections.deque(range(len(tasks)))
    alone = collections.Counter()
    errors = {}
    pool, left, unstarted, most, held, eased, freed, last = [], len(tasks), 0, 0, 0, 0, 0, 0.0

    def retire(w):
        w.stop()
        w.end()
        pool.remove(w)

    def failed(w, i, kind):
        nonlocal cap
        others = sum(x.task is not None for x in pool if x is not w)
        pending.appendleft(i)
        if others:
            cap = max(1, min(cap, others + 1) // 2)
        else:
            alone[i] += 1
            if alone[i] > restarts:
                raise _gave_up(i, alone[i], kind, plan, len(pending), len(tasks),
                               errors.get(i))
        log.warning(f'a worker process {"died" if kind == "died" else "ran out of memory"} '
                    f'running task {i}; rerunning it with up to {cap} task(s) at once')

    try:
        while left:
            if measured and time.monotonic() - last >= MONITOR_SECONDS:
                now = _measure(plan, pool)
                held = max(held, now)
                last = time.monotonic()
                running = [w for w in pool if w.task is not None]
                if now > plan.limit and len(running) > 1:
                    idle = [w for w in pool if w.ready and w.task is None]
                    if idle:
                        retire(max(idle, key=lambda w: w.private))
                    else:
                        newest = max(running, key=lambda w: w.started)
                        plan.record(newest.at_start, newest.private)
                        pending.appendleft(newest.task)
                        retire(newest)
                        cap = max(1, len(running) - 1)
                        eased += 1
                        log.warning(f'the run held {now / 2**30:.1f} GiB, over its '
                                    f'{plan.limit / 2**30:.1f} GiB: stopped the newest task '
                                    f'to rerun it with up to {cap} task(s) at once')
            busy = sum(w.task is not None for w in pool)
            for w in pool:
                if not pending or busy >= cap:
                    break
                if w.ready and w.task is None and (
                        busy == 0 or not measured or plan.fits_task(pool)):
                    i = pending.popleft()
                    w.assign(i, tasks[i])
                    busy += 1
            starting = sum(not w.ready for w in pool)
            idle = [w for w in pool if w.ready and w.task is None]
            while (pending and len(pool) < cap and len(idle) + starting < len(pending)
                   and (busy + starting == 0 or not measured or plan.fits_worker(pool))):
                pool.append(_Worker(ctx, payload))
                starting += 1
            if idle and len(pool) > cap:
                retire(max(idle, key=lambda w: w.private))
            elif pending and measured:
                spare = plan.spare(pool)
                if spare is not None:
                    retire(spare)
                    freed += 1
            most = max(most, busy)
            waits = [w.conn for w in pool] + [w.proc.sentinel for w in pool]
            for ready in mp.connection.wait(waits, timeout=MONITOR_SECONDS):
                w = next((x for x in pool if x.conn is ready or x.proc.sentinel == ready),
                         None)
                if w is None:
                    continue
                if ready is w.conn:
                    try:
                        kind, i, value, usage = w.conn.recv()
                    except (EOFError, OSError):
                        continue
                    if kind == 'ready':
                        w.ready, w.base, w.private = True, value, value or 0
                        if measured and value is not None:
                            plan.worker = value if plan.worker is None else max(plan.worker,
                                                                                value)
                        continue
                    if measured and usage is not None:
                        plan.record(*usage)
                    if kind == 'done':
                        w.task = None
                        results[i] = value
                        left -= 1
                        if done is not None:
                            done(i, value)
                    elif kind == 'error':
                        w.task = None
                        _raise(*value)
                    else:
                        if value:
                            errors[i] = value
                        failed(w, i, 'memory')
                        w.task = None
                elif not w.proc.is_alive():
                    i = w.task
                    pool.remove(w)
                    w.end()
                    if not w.ready:
                        unstarted += 1
                        if unstarted > restarts:
                            raise _never_started(unstarted, w.proc.exitcode)
                    elif i is not None:
                        w.task = None
                        failed(w, i, 'died')
    finally:
        for w in pool:
            w.stop()
        for w in pool:
            w.end()
        LAST.clear()
        LAST.update(most=most, held=held, limit=plan.limit if measured else None,
                    task=plan.task if measured else None, eased=eased, freed=freed)
    if measured:
        log.info(f'{len(tasks)} task(s) on up to {most} worker(s), the largest using '
                 f'{(plan.task or 0) / 2**20:,.0f} MiB, within {plan.limit / 2**30:.1f} GiB '
                 f'({plan.source})')
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
