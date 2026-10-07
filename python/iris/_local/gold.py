"""Gold pairs made distinct, oriented and sorted on scratch disk, and searched there."""
import math
import os
import shutil
from functools import partial

import numpy as np
import pyarrow.parquet as pq

from .._core import metrics
from . import candidates as cand_mod
from . import inputs
from .procpool import run_tasks, workers_for

SAMPLE = 1 << 20


class GoldSet:
    """Gold pairs sorted by `(id1, id2)` in two memory-mapped arrays."""

    def __init__(self, directory, n, dedup):
        self.directory = directory
        self.dedup = dedup
        if n:
            self.id1 = np.load(os.path.join(directory, 'id1.npy'), mmap_mode='c')
            self.id2 = np.load(os.path.join(directory, 'id2.npy'), mmap_mode='c')
        else:
            self.id1 = self.id2 = np.empty(0, np.int64)

    def __len__(self):
        return len(self.id1)

    def __getitem__(self, name):
        return {'id1': self.id1, 'id2': self.id2}[name]

    def contains(self, id1, id2):
        """Return whether each pair `(id1[i], id2[i])` is a gold pair."""
        return metrics.is_member_sorted(id1, id2, self.id1, self.id2)


def _boundaries(gold, n, parts, dedup):
    if parts <= 1:
        return np.empty(0, np.int64)
    stride = max(1, n // SAMPLE)
    sample, seen = [], 0
    for b in inputs.arrow_batches(gold, ['id1', 'id2']):
        first = (-seen) % stride
        id1 = b.column('id1').to_numpy()[first::stride]
        if dedup:
            id1 = np.minimum(id1, b.column('id2').to_numpy()[first::stride])
        sample.append(id1.astype(np.int64, copy=False))
        seen += b.num_rows
    sample = np.sort(np.concatenate(sample))
    return np.unique(sample[(np.arange(1, parts) * len(sample)) // parts])


def build(gold, dedup, directory, workers=None):
    """Return the gold pairs as a `GoldSet` written to `directory` by worker
    processes."""
    workers = workers or workers_for(cand_mod.WORKER_GB)
    os.makedirs(directory, exist_ok=True)
    n = cand_mod.count(gold)
    inputs.require_disk(48 * n, directory, 'the gold pairs')
    parts = max(1, math.ceil(n / cand_mod.PAIRS_PER_PART))
    boundaries = _boundaries(gold, n, parts, dedup)
    routed = os.path.join(directory, 'routed')
    sorted_dir = os.path.join(directory, 'sorted')
    os.makedirs(routed)
    os.makedirs(sorted_dir)
    with inputs.scratch() as tmp:
        src = inputs.stage(gold, tmp, 'gold')
        run_tasks(partial(cand_mod._route, flat=True, boundaries=boundaries,
                          out_dir=routed, orient=dedup, by='id1'),
                  list(enumerate(inputs.chunks(src))), workers)
    tasks = [(k, os.path.join(routed, name))
             for k, name in enumerate(sorted(os.listdir(routed)))]
    out = run_tasks(partial(cand_mod._sort, out_dir=sorted_dir, by='id1'), tasks, workers)
    shutil.rmtree(routed, ignore_errors=True)
    total = sum(n_out for _, _, n_out in out)
    if total:
        arrays = [np.lib.format.open_memmap(os.path.join(directory, f'{c}.npy'), 'w+',
                                            np.int64, (total,)) for c in ('id1', 'id2')]
        at = 0
        for path, _, n_out in out:
            part = pq.read_table(path)
            for arr, c in zip(arrays, ('id1', 'id2')):
                arr[at:at + n_out] = part.column(c).to_numpy()
            at += n_out
            os.remove(path)
        for arr in arrays:
            arr.flush()
        del arrays
    shutil.rmtree(sorted_dir, ignore_errors=True)
    return GoldSet(directory, total, dedup)
