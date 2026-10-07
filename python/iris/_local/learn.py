"""Seeds and active learning on the local engine, over a pool held in memory or memory-
mapped."""
import os
import tempfile

import numba as nb
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds

from .._core import select
from .._core.learn import BatchLearner, ContinuousLearner

SCORE_ROWS = 1 << 20


@nb.njit(cache=False)
def _positions(sorted2, sorted1, q2, q1):
    out = np.full(len(q2), -1, np.int64)
    for i in range(len(q2)):
        lo, hi = 0, len(sorted2)
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted2[mid] < q2[i] or (sorted2[mid] == q2[i] and sorted1[mid] < q1[i]):
                lo = mid + 1
            else:
                hi = mid
        if lo < len(sorted2) and sorted2[lo] == q2[i] and sorted1[lo] == q1[i]:
            out[i] = lo
    return out


def vector_matrix(column):
    """Return an Arrow column of equal-length vectors as a two-dimensional float32
    array."""
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    n = len(column)
    if n == 0:
        return np.empty((0, 0), np.float32)
    values = column.flatten().to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float32).reshape(n, -1)


class _Vectors:

    def __init__(self, column, nan_fill):
        chunks = column.chunks if isinstance(column, pa.ChunkedArray) else [column]
        self._parts = [vector_matrix(c) for c in chunks if len(c)]
        self.width = self._parts[0].shape[1] if self._parts else 0
        self._starts = np.cumsum([0] + [len(p) for p in self._parts])
        self._nan_fill = nan_fill

    def take(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        out = np.empty((len(idx), self.width), dtype=np.float32)
        part = np.searchsorted(self._starts, idx, side='right') - 1
        for k in np.unique(part).tolist():
            m = part == k
            out[m] = self._parts[k][idx[m] - self._starts[k]]
        if self._nan_fill is not None:
            np.copyto(out, np.float32(self._nan_fill), where=np.isnan(out))
        return out


def _as_table(fvs):
    if isinstance(fvs, pa.Table):
        return fvs
    if not hasattr(fvs, 'columns'):
        return fvs.to_table()
    frame = fvs.copy()
    frame['feature_vectors'] = [np.asarray(v, dtype=np.float32) for v in frame['feature_vectors']]
    table = pa.Table.from_pandas(frame, preserve_index=False)
    return table.set_column(table.schema.get_field_index('feature_vectors'),
                            'feature_vectors',
                            table.column('feature_vectors').cast(pa.list_(pa.float32())))


class LocalPool:
    """The labeling pool as arrays, for the active-learning loops."""

    def __init__(self, fvs, model):
        if type(model).__module__ == 'iris._spark.ml_model':
            raise TypeError('SparkMLModel trains and predicts on Spark: pass the pool as '
                            'a Spark DataFrame, or use an SKLearnModel on local data')
        table = _as_table(fvs)
        id1 = table.column('id1').to_numpy().astype(np.int64, copy=False)
        id2 = table.column('id2').to_numpy().astype(np.int64, copy=False)
        if len(id1) > 1 and not np.all((id2[1:] > id2[:-1]) |
                                       ((id2[1:] == id2[:-1]) & (id1[1:] > id1[:-1]))):
            order = np.lexsort((id1, id2))
            table = table.take(pa.array(order))
            id1, id2 = id1[order], id2[order]
        self._table = table
        self._id1, self._id2 = id1, id2
        self._X = _Vectors(table.column('feature_vectors'), model.nan_fill)

    def count(self):
        """Return the number of pairs in the pool."""
        return len(self._id1)

    def vector_width(self):
        """Return the width of the feature vectors."""
        return self._X.width if self.count() else None

    def _rows(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        frame = self._table.drop(['feature_vectors']).take(pa.array(idx)).to_pandas()
        names = self._table.column_names
        frame['feature_vectors'] = [row.tolist() for row in self._X.take(idx)]
        return frame[names].reset_index(drop=True)

    def _remaining(self, *excluded):
        mask = np.ones(self.count(), dtype=bool)
        for pairs in excluded:
            if isinstance(pairs, (set, frozenset, list)):
                if not pairs:
                    continue
                arr = np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2)
                q1, q2 = arr[:, 0], arr[:, 1]
            else:
                q1 = pairs['id1'].to_numpy(np.int64)
                q2 = pairs['id2'].to_numpy(np.int64)
            pos = _positions(self._id2, self._id1, np.ascontiguousarray(q2),
                             np.ascontiguousarray(q1))
            mask[pos[pos >= 0]] = False
        return np.flatnonzero(mask)

    def _entropy(self, model, rows):
        out = np.empty(len(rows), dtype=np.float64)
        for lo in range(0, len(rows), SCORE_ROWS):
            part = rows[lo:lo + SCORE_ROWS]
            out[lo:lo + len(part)] = model.entropy_of(self._X.take(part))
        return out

    def batch(self, model, labeled, skipped, n, selection, i):
        """Return the next batch of pairs to label."""
        rem = self._remaining(labeled, skipped)
        ent = None if selection == 'random' else self._entropy(model, rem)
        idx = select.choose(self._id1[rem], self._id2[rem], ent, n, selection, i)
        return self._rows(rem[idx])

    def top_entropy(self, model, seen, k, i, window):
        """Return the `k` unlabeled pairs of highest entropy."""
        rem = self._remaining(seen)
        if window is not None and len(rem):
            fraction = min(1.0, window / max(len(rem), 1))
            if fraction < 1.0:
                rem = rem[np.random.default_rng(i).random(len(rem)) < fraction]
        ent = self._entropy(model, rem)
        idx = select.choose(self._id1[rem], self._id2[rem], ent, k, 'entropy', i)
        out = self._rows(rem[idx])
        out['entropy'] = ent[idx]
        return out

    def everything(self):
        """Return every pair of the pool as pandas."""
        return self._rows(np.arange(self.count()))

    def train(self, model, frame):
        """Train the model on the labeled pairs in `frame`."""
        model.train(frame, 'feature_vectors', 'label')

    def close(self):
        """Release the pool."""
        pass


class LocalBatchLearner(BatchLearner):
    """Batch-mode active learning over a local pool."""

    def _open_pool(self, fvs):
        return LocalPool(fvs, self._model)


class LocalContinuousLearner(ContinuousLearner):
    """Continuous-mode active learning over a local pool."""

    def _open_pool(self, fvs):
        return LocalPool(fvs, self._model)


def seed_rows(fvs, score_column, nseeds):
    """Return the likely matches and likely non-matches of the feature vectors, as row
    dictionaries."""
    from .._core import seeds as seed_rules

    table = fvs if isinstance(fvs, pa.Table) else fvs.to_table(
        columns=['id1', 'id2', score_column])
    id1 = table.column('id1').to_numpy()
    id2 = table.column('id2').to_numpy()
    score = table.column(score_column).to_numpy(zero_copy_only=False).astype(np.float64)
    n_valid = int((~np.isnan(score)).sum())
    if nseeds > n_valid:
        raise ValueError("number of seeds would exceed the size of the fvs DataFrame")
    pos_idx, neg_idx = seed_rules.rankings(id1, id2, score, nseeds)

    def rows(idx):
        frame = fvs.take(pa.array(np.asarray(idx, dtype=np.int64))).to_pandas()
        return frame.to_dict('records')

    return rows(pos_idx), rows(neg_idx)


def gold_scores(fvs, labeler, score_column):
    """Return the feature vectors with `score_column` set to 1.0 on gold pairs and 0.0
    elsewhere."""
    table = fvs if isinstance(fvs, pa.Table) else fvs.to_table()
    is_gold = labeler._lookup.contains_many(table.column('id1').to_numpy(),
                                            table.column('id2').to_numpy())
    return table.append_column(score_column, pa.array(np.where(is_gold, 1.0, 0.0)))


def pool_table(fvs):
    """Return a pool given as an Arrow dataset as a memory-mapped table on scratch disk,
    and any other pool as it is."""
    if not isinstance(fvs, ds.Dataset):
        return fvs
    from .._core.pool import DEFAULT_MAX_POOL_SIZE
    from .._core.storage import spill_dir
    from . import inputs
    from .featurize import _remove_at_exit

    n = fvs.count_rows()
    if n > DEFAULT_MAX_POOL_SIZE:
        raise ValueError(f'the pool has {n:,} pairs, more than the {DEFAULT_MAX_POOL_SIZE:,} '
                         f'that active learning scores in each round; draw a pool with '
                         f'down_sample')
    directory = tempfile.mkdtemp(prefix='iris-pool-', dir=spill_dir())
    _remove_at_exit(directory)
    inputs.require_disk(4 * sum(os.path.getsize(f) for f in fvs.files), directory, 'the pool')
    path = os.path.join(directory, 'pool.arrow')
    scanner = fvs.scanner(batch_size=1 << 16, use_threads=False, **inputs._scan_options(fvs))
    with pa.OSFile(path, 'wb') as sink, pa.ipc.new_file(sink, scanner.projected_schema) as out:
        for batch in scanner.to_batches():
            out.write_batch(batch)
    return pa.ipc.open_file(pa.memory_map(path)).read_all()


def label_al(model, mode, labeler, fvs, seeds, checkpoint_path, learner_kwargs):
    """Return the pairs labeled by active learning over a local pool, as pandas."""
    from ..tools import create_seeds

    fvs = pool_table(fvs)
    n = fvs.num_rows if isinstance(fvs, pa.Table) else len(fvs)
    try:
        if seeds is None:
            seeds = create_seeds(fvs=fvs, nseeds=min(10, n), labeler=labeler,
                                 score_column='score', checkpoint_path=checkpoint_path)
        if mode == 'batch':
            learner = LocalBatchLearner(model, labeler, checkpoint_path=checkpoint_path,
                                        **learner_kwargs)
        elif mode == 'continuous':
            learner = LocalContinuousLearner(model, labeler,
                                             checkpoint_path=checkpoint_path,
                                             **learner_kwargs)
        else:
            raise ValueError(f"mode must be either 'batch' or 'continuous', not {mode}")
        return learner.train(fvs, seeds)
    finally:
        labeler.close()
