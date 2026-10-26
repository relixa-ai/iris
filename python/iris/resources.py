"""Sizing a Spark session and the local engine to the memory and cores this process may
use, and measuring the memory of a process."""
from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass

_CGROUP_ROOT = '/sys/fs/cgroup'
_SELF_CGROUP = '/proc/self/cgroup'
_MEMINFO = '/proc/meminfo'

_UNLIMITED = 2**62

MEMORY_ENV = 'IRIS_MEMORY_GB'

SHARE = 0.85

PRIOR = 10

WORKER_BYTES_GUESS = 2**28

PIECE_FLOOR = 4 * 2**20

ROWS_PER_SHUFFLE_TASK = 500_000

TASK_ATTEMPTS = 4


def _read(path):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def _own_cgroups():
    out = {}
    for line in (_read(_SELF_CGROUP) or '').splitlines():
        parts = line.split(':', 2)
        if len(parts) == 3:
            for controller in (parts[1].split(',') if parts[1] else ['']):
                out[controller] = parts[2]
    return out


def _dirs(base, path):
    out = []
    path = (path or '/').strip('/')
    while path:
        d = os.path.join(base, path)
        if os.path.isdir(d):
            out.append(d)
        path = os.path.dirname(path)
    return out + [base]


def _v1_base(controller):
    try:
        names = os.listdir(_CGROUP_ROOT)
    except OSError:
        return None
    for name in sorted(names):
        if controller in name.split(','):
            return os.path.join(_CGROUP_ROOT, name)
    return None


def _number(raw):
    try:
        value = int(raw.split()[0])
    except (AttributeError, ValueError, IndexError):
        return None
    return value if 0 < value < _UNLIMITED else None


def _cgroup_memory():
    own = _own_cgroups()
    found = []
    if '' in own:
        for d in _dirs(_CGROUP_ROOT, own['']):
            for name in ('memory.max', 'memory.high'):
                value = _number(_read(os.path.join(d, name)))
                if value:
                    found.append((value, d))
    base = _v1_base('memory') if 'memory' in own else None
    if base:
        for d in _dirs(base, own['memory']):
            value = _number(_read(os.path.join(d, 'memory.limit_in_bytes')))
            if value:
                found.append((value, d))
    return min(found) if found else (None, None)


def _cgroup_cpus():
    own = _own_cgroups()
    found = []
    if '' in own:
        for d in _dirs(_CGROUP_ROOT, own['']):
            parts = (_read(os.path.join(d, 'cpu.max')) or '').split()
            quota, period = (_number(parts[0]) if parts else None,
                             _number(parts[1]) if len(parts) > 1 else None)
            if quota and period:
                found.append(quota / period)
    base = _v1_base('cpu') if 'cpu' in own else None
    if base:
        for d in _dirs(base, own['cpu']):
            quota = _number(_read(os.path.join(d, 'cpu.cfs_quota_us')))
            period = _number(_read(os.path.join(d, 'cpu.cfs_period_us')))
            if quota and period:
                found.append(quota / period)
    return min(found) if found else None


def _affinity():
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


class _WinMemory(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                ('total_phys', ctypes.c_ulonglong), ('avail_phys', ctypes.c_ulonglong),
                ('total_page', ctypes.c_ulonglong), ('avail_page', ctypes.c_ulonglong),
                ('total_virtual', ctypes.c_ulonglong), ('avail_virtual', ctypes.c_ulonglong),
                ('avail_extended', ctypes.c_ulonglong)]


def _physical():
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (AttributeError, ValueError, OSError):
        status = _WinMemory()
        status.length = ctypes.sizeof(_WinMemory)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return int(status.total_phys)


def _setting():
    raw = os.environ.get(MEMORY_ENV, '').strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        value = 0.0
    if not value > 0:
        raise ValueError(f'{MEMORY_ENV} must be a positive number of GiB, got {raw!r}')
    return value * 2**30


def budget():
    """Return the memory in GiB and the cores this process may use, and where the
    figures came from."""
    phys = _physical()
    ncpu = os.cpu_count() or 1
    limit, where = _cgroup_memory()
    quota = _cgroup_cpus()
    affinity = _affinity()
    setting = _setting()

    total = min(x for x in (phys, limit, setting) if x)
    cores = max(1, min(x for x in (ncpu, affinity, int(quota) if quota else None) if x))
    machine = f'machine reports {phys / 2**30:.1f} GiB / {ncpu} cores'
    if setting and setting <= total:
        src = f'setting {MEMORY_ENV} ({total / 2**30:.1f} GiB / {cores} cores; {machine})'
    elif limit or quota or (affinity and affinity < ncpu):
        src = (f'cgroup{" " + where if where else ""} ({total / 2**30:.1f} GiB / '
               f'{cores} cores; {machine})')
    else:
        src = 'machine (no cgroup limit set)'
    return total / 2**30, cores, src


def piece_bytes(ceiling):
    """Return the size in bytes of a piece of work on this machine: what one task per core
    leaves room for, between PIECE_FLOOR and `ceiling`."""
    total_gb, cores, _ = budget()
    room = (total_gb * 2**30 * SHARE / cores - WORKER_BYTES_GUESS) / PRIOR
    return int(min(ceiling, max(PIECE_FLOOR, room)))


def _meminfo(name):
    for line in (_read(_MEMINFO) or '').splitlines():
        if line.startswith(name + ':'):
            return int(line.split()[1]) * 1024
    return None


def _stat(d, name):
    for line in (_read(os.path.join(d, 'memory.stat')) or '').splitlines():
        key, _, value = line.partition(' ')
        if key == name:
            return int(value)
    return None


def _sysctl(name):
    value = ctypes.c_uint64(0)
    size = ctypes.c_size_t(ctypes.sizeof(value))
    libc = ctypes.CDLL(None)
    if libc.sysctlbyname(name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0):
        return None
    return value.value


def available_gb():
    """Return the operating system's estimate of the memory available to this process in
    GiB, or None where it gives none."""
    if sys.platform.startswith('linux'):
        found = [x for x in (_meminfo('MemAvailable'),) if x]
        limit, d = _cgroup_memory()
        if limit and d:
            used = _number(_read(os.path.join(d, 'memory.current'))) or \
                _number(_read(os.path.join(d, 'memory.usage_in_bytes')))
            cache = _stat(d, 'inactive_file') or _stat(d, 'total_inactive_file') or 0
            if used:
                found.append(max(0, limit - used + cache))
        return min(found) / 2**30 if found else None
    if sys.platform == 'darwin':
        level = _sysctl('kern.memorystatus_level')
        return None if level is None else _physical() * level / 100 / 2**30
    return None


class _Rusage(ctypes.Structure):
    _fields_ = [('uuid', ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        'user_time', 'system_time', 'pkg_idle_wkups', 'interrupt_wkups', 'pageins',
        'wired_size', 'resident_size', 'phys_footprint', 'proc_start_abstime',
        'proc_exit_abstime', 'child_user_time', 'child_system_time',
        'child_pkg_idle_wkups', 'child_interrupt_wkups', 'child_pageins',
        'child_elapsed_abstime', 'diskio_bytesread', 'diskio_byteswritten',
        'cpu_time_qos_default', 'cpu_time_qos_maintenance', 'cpu_time_qos_background',
        'cpu_time_qos_utility', 'cpu_time_qos_legacy', 'cpu_time_qos_user_initiated',
        'cpu_time_qos_user_interactive', 'billed_system_time', 'serviced_system_time',
        'logical_writes', 'lifetime_max_phys_footprint', 'instructions', 'cycles',
        'billed_energy', 'serviced_energy', 'interval_max_phys_footprint', 'runnable_time')]


_libproc = None


def _rusage(pid):
    global _libproc
    if _libproc is None:
        _libproc = ctypes.CDLL('/usr/lib/libproc.dylib')
    info = _Rusage()
    if _libproc.proc_pid_rusage(int(pid), 4, ctypes.byref(info)) != 0:
        return None
    return info


def private_bytes(pid):
    """Return the memory that belongs to process `pid` alone, in bytes, or None where
    this platform does not report it."""
    if sys.platform.startswith('linux'):
        text = _read(f'/proc/{int(pid)}/status')
        if text is None:
            return None
        fields = dict(line.split(':', 1) for line in text.splitlines() if ':' in line)
        kib = sum(int(fields[k].split()[0]) for k in ('RssAnon', 'VmSwap') if k in fields)
        return kib * 1024 if 'RssAnon' in fields else None
    if sys.platform == 'darwin':
        info = _rusage(pid)
        return None if info is None else int(info.phys_footprint)
    return None


def peak_private_bytes(pid):
    """Return the most memory process `pid` has held alone, as the operating system
    records it, or None where it records none."""
    if sys.platform == 'darwin':
        info = _rusage(pid)
        return None if info is None else int(info.lifetime_max_phys_footprint)
    return None


WORKER_BYTES = int(0.5 * 2**30)

JVM_RSS_FACTOR = 1.35

HEAP_SHARE = 0.35

MAX_HEAP_GB = 24.0

MIN_HEAP_GB = 0.5


def local_plan(total_gb, cores):
    """Return the Spark workers, heap, per-worker memory and reserve that fit a memory
    budget: the heap scaled to the budget, with room left for at least one worker."""
    per_worker_gb = WORKER_BYTES / 2**30
    room = total_gb * SHARE
    heap_gb = min(MAX_HEAP_GB, total_gb * HEAP_SHARE,
                  (room - per_worker_gb) / JVM_RSS_FACTOR)
    heap_gb = max(MIN_HEAP_GB, heap_gb)
    jvm_gb = heap_gb * JVM_RSS_FACTOR
    workers = int(max(1, min(cores, (room - jvm_gb) // per_worker_gb)))
    return workers, heap_gb, per_worker_gb, total_gb - jvm_gb - workers * per_worker_gb


def local_workers(per_worker_gb, total_gb=None, cores=None):
    """Return how many worker processes of the local engine fit the budget at the given
    memory each."""
    if total_gb is None or cores is None:
        found_gb, found_cores, _ = budget()
        total_gb = found_gb if total_gb is None else total_gb
        cores = found_cores if cores is None else cores
    return int(max(1, min(cores, (total_gb * 0.85) // per_worker_gb)))


def shuffle_partitions(n_candidates):
    """Return the number of shuffle partitions for a candidate set of the given size."""
    return int(min(20_000, max(200, n_candidates // ROWS_PER_SHUFFLE_TASK)))


@dataclass
class SparkPlan:
    """The Spark master, driver memory and worker count planned for a session."""
    master: str
    driver_memory: str
    workers: int
    heap_gb: float
    per_worker_gb: float
    reserve_gb: float
    total_gb: float
    cores: int
    source: str

    def describe(self):
        """Return two lines describing the budget and what the plan commits against
        it."""
        return (
            f'[plan] budget from {self.source}\n'
            f'[plan] {self.total_gb:.1f} GiB / {self.cores} cores -> {self.master}, '
            f'heap {self.driver_memory}; {self.workers} x {self.per_worker_gb:.2f} GiB '
            f'workers + {self.heap_gb * JVM_RSS_FACTOR:.1f} GiB JVM = '
            f'{self.workers * self.per_worker_gb + self.heap_gb * JVM_RSS_FACTOR:.1f} GiB '
            f'committed, {self.reserve_gb:.1f} GiB left over')


def plan(master=None, driver_memory=None):
    """Return the `SparkPlan` for this machine, with the master and driver memory given
    taking precedence."""
    total_gb, cores, source = budget()
    workers, heap_gb, per_worker_gb, reserve_gb = local_plan(total_gb, cores)
    if reserve_gb < 0 and driver_memory is None:
        import logging
        logging.getLogger(__name__).warning(
            f'a Spark session needs at least {total_gb - reserve_gb:.1f} GiB, a '
            f'{MIN_HEAP_GB:.1f} GiB heap and one {per_worker_gb:.1f} GiB worker, more than '
            f'the {total_gb:.1f} GiB budget ({source}). Raise {MEMORY_ENV}, or use the '
            f'local engine.')
    return SparkPlan(
        master=master or f'local[{workers}, {TASK_ATTEMPTS}]',
        driver_memory=driver_memory or f'{int(heap_gb * 1024)}m',
        workers=workers, heap_gb=heap_gb, per_worker_gb=per_worker_gb,
        reserve_gb=reserve_gb, total_gb=total_gb, cores=cores, source=source)


def build_session(plan_, app_name='iris', shuffle_partitions=2000):
    """Start, or attach to, the `SparkSession` a `SparkPlan` describes."""
    from ._spark.session import build_session as build

    return build(plan_, app_name, shuffle_partitions)
