"""The pair pass of the local engine: the feature vector and score of every candidate
pair."""
import os
from functools import partial

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .._core import pool, records
from .procpool import atomic_output, run_tasks, workers_for

UNIT_BYTES = 256 * 2**20

MIN_UNIT, MAX_UNIT = 8_000, 200_000

WORKER_GB = 1.5

def unit_pairs(mean_a, mean_b):
    """Return the number of pairs in a unit, from the mean record sizes of the two
    stores."""
    per_pair = max(1.0, float(mean_a) + float(mean_b))
    return int(min(max(UNIT_BYTES // per_pair, MIN_UNIT), MAX_UNIT))


def schema(output_col):
    """Return the Arrow schema of the feature vectors."""
    return pa.schema([('id1', pa.int64()), ('id2', pa.int64()),
                      (output_col, pa.list_(pa.float32())), ('score', pa.float64())])


def _score_part(task, store_a, store_b, features, fill_na, positive, unit, output_col,
                out_dir):
    k, pairs_path = task
    table = pq.read_table(pairs_path)
    id1 = table.column('id1').to_numpy()
    id2 = table.column('id2').to_numpy()
    path = os.path.join(out_dir, f'part-{k:05d}.parquet')
    with atomic_output(path) as tmp:
        with pq.ParquetWriter(tmp, schema(output_col)) as writer:
            for lo in range(0, len(id1), unit):
                p1, p2 = id1[lo:lo + unit], id2[lo:lo + unit]
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
    return len(id1)


def run(parts, store_a, store_b, features, fill_na, output_col, out_dir, workers=None):
    """Write the feature vector and score of every pair of the parts to `out_dir`, and
    return the number of pairs."""
    os.makedirs(out_dir, exist_ok=True)
    positive = pool.positive_features([str(f) for f in features])
    mean = lambda st: (sum(s['bytes'] for s in st.segments) / st.rows) if st.rows else 1.0
    unit = unit_pairs(mean(store_a), mean(store_b))
    fn = partial(_score_part, store_a=store_a, store_b=store_b, features=features,
                 fill_na=fill_na, positive=positive, unit=unit, output_col=output_col,
                 out_dir=out_dir)
    tasks = [(k, path) for k, (path, _) in enumerate(parts)]
    n = sum(run_tasks(fn, tasks, workers or workers_for(WORKER_GB)))
    if not tasks:
        with atomic_output(os.path.join(out_dir, 'part-00000.parquet')) as tmp:
            pq.write_table(schema(output_col).empty_table(), tmp)
    from ..io import write_fvs_meta
    write_fvs_meta(out_dir, features, n, output_col)
    open(os.path.join(out_dir, '_SUCCESS'), 'w').close()
    return n
