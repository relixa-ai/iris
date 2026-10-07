"""Sizing a Spark session and the local engine to the memory and cores this process may
use."""
from __future__ import annotations

import os
from dataclasses import dataclass

_CGROUP_MEM_PATHS = ('/sys/fs/cgroup/memory.max',
                     '/sys/fs/cgroup/memory/memory.limit_in_bytes')
_CGROUP_CPU_PATHS = ('/sys/fs/cgroup/cpu.max',)
_CGROUP_CPU_V1_PATHS = ('/sys/fs/cgroup/cpu/cpu.cfs_quota_us',
                        '/sys/fs/cgroup/cpu/cpu.cfs_period_us')

ROWS_PER_SHUFFLE_TASK = 500_000


def _cgroup_value(paths, parse):
    for path in paths:
        try:
            with open(path) as fh:
                raw = fh.read().strip()
        except OSError:
            continue
        try:
            value = parse(raw)
        except (ValueError, IndexError, ZeroDivisionError):
            continue
        if value is not None:
            return value
    return None


def budget():
    """Return the memory in GiB and the cores this process may use, and where the
    figures came from."""
    phys_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30
    ncpu = os.cpu_count() or 1

    limit_gb = _cgroup_value(
        _CGROUP_MEM_PATHS,
        lambda raw: None if raw == 'max' or int(raw) >= 2**62 else int(raw) / 2**30)
    quota = _cgroup_value(
        _CGROUP_CPU_PATHS,
        lambda raw: None if raw.split()[0] == 'max'
        else int(raw.split()[0]) / int(raw.split()[1]))
    if quota is None:
        q = _cgroup_value(_CGROUP_CPU_V1_PATHS[:1],
                          lambda raw: None if int(raw) <= 0 else int(raw))
        period = _cgroup_value(_CGROUP_CPU_V1_PATHS[1:],
                               lambda raw: int(raw) or None)
        quota = q / period if q and period else None

    total_gb = min(phys_gb, limit_gb) if limit_gb else phys_gb
    cores = max(1, min(ncpu, int(quota))) if quota else ncpu
    src = (f'cgroup ({total_gb:.1f} GiB / {cores} cores; machine reports '
           f'{phys_gb:.1f} GiB / {ncpu})' if (limit_gb or quota)
           else 'machine (no cgroup limit set)')
    return total_gb, cores, src


WORKER_BYTES = int(0.5 * 2**30)

JVM_RSS_FACTOR = 1.35


def local_plan(total_gb, cores):
    """Return the Spark workers, heap, per-worker memory and reserve that fit a memory
    budget."""
    heap_gb = max(1.0, min(24.0, total_gb * 0.35))
    jvm_gb = heap_gb * JVM_RSS_FACTOR
    per_worker_gb = WORKER_BYTES / 2**30
    workers = int(max(1, min(cores, (total_gb * 0.85 - jvm_gb) // per_worker_gb)))
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
            f'workers + {self.heap_gb * 1.35:.1f} GiB JVM = '
            f'{self.workers * self.per_worker_gb + self.heap_gb * 1.35:.1f} GiB '
            f'committed, {self.reserve_gb:.1f} GiB left over')


def plan(master=None, driver_memory=None):
    """Return the `SparkPlan` for this machine, with the master and driver memory given
    taking precedence."""
    total_gb, cores, source = budget()
    workers, heap_gb, per_worker_gb, reserve_gb = local_plan(total_gb, cores)
    return SparkPlan(
        master=master or f'local[{workers}]',
        driver_memory=driver_memory or f'{heap_gb:.0f}g',
        workers=workers, heap_gb=heap_gb, per_worker_gb=per_worker_gb,
        reserve_gb=reserve_gb, total_gb=total_gb, cores=cores, source=source)


def build_session(plan_, app_name='iris', shuffle_partitions=2000):
    """Start, or attach to, the `SparkSession` a `SparkPlan` describes."""
    from ._spark.session import build_session as build

    return build(plan_, app_name, shuffle_partitions)
