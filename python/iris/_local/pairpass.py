"""The pair pass of the local engine: the feature vector and score of every candidate
pair."""
import os
from functools import partial

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .. import resources
from .._core import pool, records
from . import disk
from .procpool import atomic_output, memory_error, run_tasks

UNIT_BYTES = 128 * 2**20

MAX_UNIT = 200_000


def _first(ids):
    first = np.zeros(len(ids), bool)
    first[np.unique(ids, return_index=True)[1]] = True
    return first


def units(id1, id2, size_a, size_b, n_features, unit_bytes, max_unit):
    """Return the start and end of each unit of the pairs: at most `max_unit` pairs whose
    distinct records and feature values come to at most `unit_bytes`, or a single pair."""
    per_pair = 4 * n_features + 32
    out, lo, n, window = [], 0, len(id1), 4096
    while lo < n:
        hi = min(n, lo + min(window, max_unit))
        cost = np.cumsum(np.where(_first(id1[lo:hi]), size_a[lo:hi], 0)
                         + np.where(_first(id2[lo:hi]), size_b[lo:hi], 0) + per_pair)
        k = max(1, int(np.searchsorted(cost, unit_bytes, side='right')))
        if lo + k == hi < n and window < max_unit:
            window *= 4
            continue
        out.append((lo, lo + k))
        lo += k
        window = max(4096, 2 * k)
    return out


def schema(output_col):
    """Return the Arrow schema of the feature vectors."""
    return pa.schema([('id1', pa.int64()), ('id2', pa.int64()),
                      (output_col, pa.list_(pa.float32())), ('score', pa.float64())])


def _score_part(task, store_a, store_b, features, fill_na, positive, output_col, out_dir,
                unit_bytes, max_unit):
    k, pairs_path = task
    table = pq.read_table(pairs_path)
    id1 = table.column('id1').to_numpy()
    id2 = table.column('id2').to_numpy()
    cuts = units(id1, id2, store_a.sizes(id1, 'a'), store_b.sizes(id2, 'b'), len(features),
                 unit_bytes, max_unit)
    path = os.path.join(out_dir, f'part-{k:05d}.parquet')
    lo, hi = 0, len(id1)
    try:
        with atomic_output(path) as tmp:
            with pq.ParquetWriter(tmp, schema(output_col)) as writer:
                for lo, hi in cuts:
                    p1, p2 = id1[lo:hi], id2[lo:hi]
                    u1, a_local = np.unique(p1, return_inverse=True)
                    u2, b_local = np.unique(p2, return_inverse=True)
                    a_frame = store_a.frame(u1, 'a')
                    b_frame = store_b.frame(u2, 'b')
                    f_mat = records.feature_matrix(features, a_frame, a_local, b_frame,
                                                   b_local, fill_na)
                    writer.write_batch(pa.RecordBatch.from_arrays(
                        [pa.array(p1), pa.array(p2), records.vectors(f_mat),
                         pa.array(pool.score(f_mat, positive))],
                        schema=schema(output_col)))
    except MemoryError as e:
        raise memory_error(e, _largest(id1[lo:hi], id2[lo:hi], store_a, store_b)) from e
    return len(id1)


def _largest(id1, id2, store_a, store_b):
    if not len(id1):
        return 'it held no pairs'
    size_a, size_b = store_a.sizes(id1, 'a'), store_b.sizes(id2, 'b')
    a, b = int(np.argmax(size_a)), int(np.argmax(size_b))
    return (f'the largest records of the {len(id1):,} pair(s) it was scoring are id1 '
            f'{int(id1[a])}, {size_a[a] / 2**20:,.1f} MiB packed, and id2 {int(id2[b])}, '
            f'{size_b[b] / 2**20:,.1f} MiB packed')


def _meter(meter, out_dir):
    finished = {}

    def done(k, n):
        finished[f'part-{k:05d}.parquet'] = n
        meter.wrote(sum(finished.values()), *disk.written_in(out_dir, finished))
    return done


def run(parts, store_a, store_b, features, fill_na, output_col, out_dir, workers=None):
    """Write the feature vector and score of every pair of the parts to `out_dir`, and
    return the number of pairs."""
    os.makedirs(out_dir, exist_ok=True)
    positive = pool.positive_features([str(f) for f in features])
    unit_bytes = resources.piece_bytes(UNIT_BYTES)
    fn = partial(_score_part, store_a=store_a, store_b=store_b, features=features,
                 fill_na=fill_na, positive=positive, output_col=output_col,
                 out_dir=out_dir, unit_bytes=unit_bytes, max_unit=MAX_UNIT)
    tasks = [(k, path) for k, (path, _) in enumerate(parts)]
    total = sum(n for _, n in parts)
    meter = disk.Meter('the feature vectors', out_dir, total)
    n = sum(disk.measured(run_tasks, fn, tasks, workers, meter,
                          total * (4 * len(features) + 24), _meter(meter, out_dir),
                          task_bytes=unit_bytes))
    if not tasks:
        with atomic_output(os.path.join(out_dir, 'part-00000.parquet')) as tmp:
            pq.write_table(schema(output_col).empty_table(), tmp)
    from ..io import write_fvs_meta
    write_fvs_meta(out_dir, features, n, output_col)
    open(os.path.join(out_dir, '_SUCCESS'), 'w').close()
    return n
