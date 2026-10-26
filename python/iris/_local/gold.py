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


def build(gold, dedup, directory, workers=None):
    """Return the gold pairs as a `GoldSet` written to `directory` by worker
    processes."""
    workers = workers or workers_for()
    os.makedirs(directory, exist_ok=True)
    n = cand_mod.count(gold)
    parts = max(1, math.ceil(n / cand_mod.PAIRS_PER_PART))
    routed = os.path.join(directory, 'routed')
    sorted_dir = os.path.join(directory, 'sorted')
    os.makedirs(routed)
    os.makedirs(sorted_dir)
    with inputs.scratch() as tmp:
        src = inputs.stage(gold, tmp, 'gold', workers=workers)
        inputs.require_disk(16 * n, routed, 'the routed gold pairs', kept=0)
        chunks = inputs.chunks(src, workers=workers)
        cuts = cand_mod.boundaries(chunks, n, parts, workers, orient=dedup, by='id1')
        kept = run_tasks(partial(cand_mod._route, flat=True, boundaries=cuts,
                                 out_dir=routed, orient=dedup, by='id1'),
                         list(enumerate(chunks)), workers, task_bytes=inputs.chunk_bytes())
    _, out = cand_mod._sorted(routed, sorted_dir, sum(kept), workers, by='id1')
    shutil.rmtree(routed, ignore_errors=True)
    total = sum(n_out for _, _, n_out in out)
    if total:
        inputs.require_disk(16 * total, directory, 'the gold pairs')
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
