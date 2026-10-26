"""Seeds and active learning on the local engine, over a pool held in memory or memory-
mapped."""
import os
import tempfile

import numba as nb
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds

from .. import resources
from .._core import select
from .._core.learn import BatchLearner, ContinuousLearner
from . import disk, inputs
from .pool import in_pair_order, write_file_in_order

SCORE_BYTES = 64 * 2**20

SCAN_ROW_BYTES = 64


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
    if column.null_count:
        raise ValueError(f'{column.null_count} feature vector(s) are missing; every row '
                         f'needs one')
    lengths = np.diff(np.asarray(column.offsets))
    if len(lengths) and lengths.min() != lengths.max():
        raise ValueError(f'feature vectors differ in length: {lengths.min()} to '
                         f'{lengths.max()} values; vectors from one featurize all have '
                         f'as many values as its features')
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


def _ids(table):
    return (table.column('id1').to_numpy().astype(np.int64, copy=False),
            table.column('id2').to_numpy().astype(np.int64, copy=False))


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


def _queries(excluded):
    q1, q2 = [], []
    for pairs in excluded:
        if isinstance(pairs, (set, frozenset, list)):
            if not pairs:
                continue
            arr = np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2)
            q1.append(arr[:, 0])
            q2.append(arr[:, 1])
        else:
            q1.append(pairs['id1'].to_numpy(np.int64))
            q2.append(pairs['id2'].to_numpy(np.int64))
    if not q1:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return (np.ascontiguousarray(np.concatenate(q1)),
            np.ascontiguousarray(np.concatenate(q2)))


class _Best:

    def __init__(self, n, selection, i):
        self.n, self.selection, self.i = n, selection, i
        self.rows = self.id1 = self.id2 = np.empty(0, np.int64)
        self.ent = None if selection == 'random' else np.empty(0, np.float64)

    def add(self, rows, id1, id2, ent):
        rows = np.concatenate([self.rows, rows])
        id1 = np.concatenate([self.id1, id1])
        id2 = np.concatenate([self.id2, id2])
        if self.ent is not None:
            ent = np.concatenate([self.ent, ent])
        keep = np.sort(select.choose(id1, id2, ent, self.n, self.selection, self.i))
        self.rows, self.id1, self.id2 = rows[keep], id1[keep], id2[keep]
        if self.ent is not None:
            self.ent = ent[keep]

    def order(self):
        return select.choose(self.id1, self.id2, self.ent, self.n, self.selection, self.i)


class LocalPool:
    """The labeling pool as a table in pair order, scanned a piece at a time by the
    active-learning loops."""

    def __init__(self, fvs, model):
        if type(model).__module__ == 'iris._spark.ml_model':
            raise TypeError('SparkMLModel trains and predicts on Spark: pass the pool as '
                            'a Spark DataFrame, or use an SKLearnModel on local data')
        table = _as_table(fvs)
        self._X = _Vectors(table.column('feature_vectors'), model.nan_fill)
        self._step = max(1, resources.piece_bytes(SCORE_BYTES)
                         // (4 * self._X.width + SCAN_ROW_BYTES))
        if not in_pair_order(table, self._step):
            table = _ordered_copy(table)
            self._X = _Vectors(table.column('feature_vectors'), model.nan_fill)
        self._table = table

    def count(self):
        """Return the number of pairs in the pool."""
        return self._table.num_rows

    def vector_width(self):
        """Return the width of the feature vectors."""
        return self._X.width if self.count() else None

    def _rows(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        frame = inputs.take(self._table.drop(['feature_vectors']), idx).to_pandas()
        names = self._table.column_names
        frame['feature_vectors'] = [row.tolist() for row in self._X.take(idx)]
        return frame[names].reset_index(drop=True)

    def _pieces(self, q1, q2):
        found = np.zeros(len(q1), bool)
        for lo in range(0, self.count(), self._step):
            id1, id2 = _ids(self._table.slice(lo, self._step))
            keep = np.ones(len(id1), bool)
            todo = np.flatnonzero(~found)
            if len(todo):
                pos = _positions(id2, id1, q2[todo], q1[todo])
                hit = pos >= 0
                found[todo[hit]] = True
                keep[pos[hit]] = False
            yield lo, id1, id2, keep

    def _remaining(self, *excluded, window=None, i=None):
        q1, q2 = _queries(excluded)
        rng = fraction = None
        if window is not None:
            left = sum(int(keep.sum()) for *_, keep in self._pieces(q1, q2))
            fraction = min(1.0, window / max(left, 1))
            if left and fraction < 1.0:
                rng = np.random.default_rng(i)
        for lo, id1, id2, keep in self._pieces(q1, q2):
            at = np.flatnonzero(keep)
            if rng is not None:
                at = at[rng.random(len(at)) < fraction]
            yield lo + at, id1[at], id2[at]

    def _entropy(self, model, rows):
        out = np.empty(len(rows), dtype=np.float64)
        if len(rows):
            out[:] = model.entropy_of(self._X.take(rows))
        return out

    def batch(self, model, labeled, skipped, n, selection, i):
        """Return the next batch of pairs to label."""
        if selection == 'hybrid':
            first, second = _Best((n + 1) // 2, 'entropy', i), _Best(n, 'random', i)
        else:
            first, second = _Best(n, selection, i), None
        for rows, id1, id2 in self._remaining(labeled, skipped):
            first.add(rows, id1, id2,
                      None if selection == 'random' else self._entropy(model, rows))
            if second is not None:
                second.add(rows, id1, id2, None)
        rows = first.rows[first.order()]
        if second is not None:
            other = ~np.isin(second.rows, rows)
            more = _Best(n - len(rows), 'random', i)
            more.add(second.rows[other], second.id1[other], second.id2[other], None)
            rows = np.concatenate([rows, more.rows[more.order()]])
        return self._rows(rows)

    def top_entropy(self, model, seen, k, i, window):
        """Return the `k` unlabeled pairs of highest entropy."""
        best = _Best(k, 'entropy', i)
        for rows, id1, id2 in self._remaining(seen, window=window, i=i):
            best.add(rows, id1, id2, self._entropy(model, rows))
        idx = best.order()
        out = self._rows(best.rows[idx])
        out['entropy'] = best.ent[idx]
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


def _top(keys, rows, k):
    order = np.lexsort(keys)[:k]
    return [x[order] for x in rows]


def _streamed_rankings(fvs, score_column, nseeds):
    from .._core import seeds as seed_rules

    empty = np.empty(0, np.int64)
    pos = [np.empty(0), empty, empty, empty]
    neg = [np.empty(0), empty, empty, empty]
    n_valid = at = 0
    for batch in inputs.arrow_batches(fvs, ['id1', 'id2', score_column]):
        score = batch.column(score_column).to_numpy(zero_copy_only=False).astype(np.float64)
        valid = np.flatnonzero(~np.isnan(score))
        n_valid += len(valid)
        k = seed_rules.scan_size(nseeds, max(n_valid, nseeds))
        rows = [score[valid], batch.column('id2').to_numpy()[valid].astype(np.int64),
                batch.column('id1').to_numpy()[valid].astype(np.int64), valid + at]
        at += batch.num_rows
        both = [np.concatenate([p, r]) for p, r in zip(pos, rows)]
        pos = _top((both[2], both[1], -both[0]), both, k)
        both = [np.concatenate([p, r]) for p, r in zip(neg, rows)]
        neg = _top((both[2], both[1], both[0]), both, k)
    if nseeds > n_valid:
        raise seed_rules.too_few_scores(nseeds, n_valid, at, score_column)
    k = seed_rules.scan_size(nseeds, n_valid)
    return pos[3][:k], neg[3][:k]


def seed_rows(fvs, score_column, nseeds):
    """Return the likely matches and likely non-matches of the feature vectors, as row
    dictionaries."""
    pos_idx, neg_idx = _streamed_rankings(fvs, score_column, nseeds)

    def rows(idx):
        frame = inputs.take(fvs, idx).to_pandas()
        return frame.to_dict('records')

    return rows(pos_idx), rows(neg_idx)


def any_score(fvs, score_column):
    """Return whether any feature vector has a score that is not missing or NaN."""
    for batch in inputs.arrow_batches(fvs, [score_column]):
        score = batch.column(0).to_numpy(zero_copy_only=False).astype(np.float64)
        if not np.isnan(score).all():
            return True
    return False


def _with_column(table, name, values):
    at = table.schema.get_field_index(name)
    if at < 0:
        return table.append_column(name, values)
    return table.set_column(at, name, values)


@disk.disk_step('create_seeds')
def gold_scores(fvs, labeler, score_column):
    """Return the feature vectors with `score_column` set to 1.0 on gold pairs and 0.0
    elsewhere."""
    if isinstance(fvs, pa.Table):
        is_gold = labeler._lookup.contains_many(fvs.column('id1').to_numpy(),
                                                fvs.column('id2').to_numpy())
        return _with_column(fvs, score_column, pa.array(np.where(is_gold, 1.0, 0.0)))
    import pyarrow.parquet as pq

    from .._core.storage import scratch_prefix, spill_dir
    from .featurize import _remove_at_exit

    directory = tempfile.mkdtemp(prefix=scratch_prefix('scored'), dir=spill_dir())
    _remove_at_exit(directory)
    disk.output(directory)
    path = os.path.join(directory, 'part-00000.parquet')
    meter = disk.Meter('the feature vectors with their gold scores', directory,
                       fvs.count_rows())
    writer, done = None, 0
    try:
        for batch in inputs.arrow_batches(fvs, fvs.schema.names):
            is_gold = labeler._lookup.contains_many(batch.column('id1').to_numpy(),
                                                    batch.column('id2').to_numpy())
            scored = _with_column(pa.Table.from_batches([batch]), score_column,
                                  pa.array(np.where(is_gold, 1.0, 0.0)))
            if writer is None:
                writer = pq.ParquetWriter(path, scored.schema)
            writer.write_table(scored)
            done += scored.num_rows
            meter.wrote(done, os.path.getsize(path))
    finally:
        if writer is not None:
            writer.close()
    meter.finish()
    if writer is None:
        return gold_scores(fvs.to_table(), labeler, score_column)
    return inputs.dataset(directory)


@disk.disk_step('label_al')
def pool_table(fvs):
    """Return a pool given as an Arrow dataset as a memory-mapped table on scratch disk,
    and any other pool as it is."""
    if not isinstance(fvs, ds.Dataset):
        return fvs
    from .._core.pool import DEFAULT_MAX_POOL_SIZE
    from .._core.storage import scratch_prefix, spill_dir
    from .featurize import _remove_at_exit

    n = fvs.count_rows()
    if n > DEFAULT_MAX_POOL_SIZE:
        raise ValueError(f'the pool has {n:,} pairs, more than the {DEFAULT_MAX_POOL_SIZE:,} '
                         f'that active learning scores in each round; draw a pool with '
                         f'down_sample')
    directory = tempfile.mkdtemp(prefix=scratch_prefix('pool'), dir=spill_dir())
    _remove_at_exit(directory)
    disk.output(directory)
    path = os.path.join(directory, 'pool.arrow')
    meter = disk.Meter('the pool', directory, n)
    scanner = fvs.scanner(batch_size=1 << 16, use_threads=False, **inputs._scan_options(fvs))
    with pa.OSFile(path, 'wb') as sink, pa.ipc.new_file(sink, scanner.projected_schema) as out:
        done = 0
        for batch in scanner.to_batches():
            out.write_batch(batch)
            done += batch.num_rows
            meter.wrote(done, sink.tell())
    meter.finish()
    table = pa.ipc.open_file(pa.memory_map(path)).read_all()
    if in_pair_order(table, inputs.take_rows(16)):
        return table
    del table
    return _mapped_in_order(path, directory)


def _mapped_in_order(path, directory):
    ordered = os.path.join(directory, 'ordered.arrow')
    inputs.require_disk(os.path.getsize(path), directory, 'the pool in order')
    write_file_in_order(path, ordered)
    os.remove(path)
    return pa.ipc.open_file(pa.memory_map(ordered)).read_all()


def _ordered_copy(table):
    from .._core.storage import scratch_prefix, spill_dir
    from .featurize import _remove_at_exit

    directory = tempfile.mkdtemp(prefix=scratch_prefix('pool'), dir=spill_dir())
    _remove_at_exit(directory)
    disk.output(directory)
    inputs.require_disk(table.nbytes, directory, 'the pool')
    path = os.path.join(directory, 'pool.arrow')
    step = inputs.take_rows(table.nbytes / max(table.num_rows, 1))
    with pa.OSFile(path, 'wb') as sink, pa.ipc.new_file(sink, table.schema) as out:
        for lo in range(0, table.num_rows, step):
            out.write_table(table.slice(lo, step))
    return _mapped_in_order(path, directory)


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
