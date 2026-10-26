"""Applying a matcher and counting its results against gold pairs, on the local
engine."""
import contextlib
import os
import tempfile
from functools import partial

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .._core import metrics
from .._core.storage import scratch_prefix, spill_dir
from . import disk, inputs
from .learn import vector_matrix
from .procpool import atomic_output, run_tasks, workers_for


MATCH_SCHEMA = pa.schema([pa.field('id1', pa.int64()), pa.field('id2', pa.int64()),
                          pa.field('probability', pa.float32())])

PREDICT_ROWS = 1 << 18


def _predict_chunk(task, model, feature_col, prediction_col, confidence_col, out_dir):
    k, chunk = task
    table = inputs.read_chunk(chunk, None)
    X = vector_matrix(table.column(feature_col))
    preds, confs = [], []
    for lo in range(0, table.num_rows, PREDICT_ROWS):
        out = model.predict_of(X[lo:lo + PREDICT_ROWS], confidence_col is not None)
        if confidence_col is not None:
            preds.append(out[0])
            confs.append(out[1])
        else:
            preds.append(out)
    cat = lambda parts: np.concatenate(parts) if parts else np.empty(0, np.float64)
    table = table.append_column(prediction_col, pa.array(cat(preds), pa.float64()))
    if confidence_col is not None:
        table = table.append_column(confidence_col, pa.array(cat(confs), pa.float64()))
    with atomic_output(os.path.join(out_dir, f'part-{k:05d}.parquet')) as tmp:
        pq.write_table(table, tmp)
    return table.num_rows


def _check_width(model, fvs, feature_col):
    expected = getattr(getattr(model, '_trained_model', None), 'n_features_in_', None)
    if expected is None:
        return
    for batch in inputs.arrow_batches(fvs, [feature_col]):
        if batch.num_rows:
            first = batch.column(0)[0]
            if first.is_valid and len(first) != int(expected):
                raise ValueError(f'the matcher was trained on feature vectors of '
                                 f'{int(expected)} values, and these have {len(first)}; '
                                 f'apply it to vectors of the same features')
            return


@disk.disk_step('apply_matcher')
def apply_matcher(model, fvs, feature_col, prediction_col, confidence_col=None,
                  out=None, workers=None):
    """Return the feature vectors with predictions, as a dataset of Parquet parts
    written by worker processes."""
    if type(model).__module__ == 'iris._spark.ml_model':
        raise TypeError('SparkMLModel predicts on Spark: pass the feature vectors as a '
                        'Spark DataFrame, or use an SKLearnModel on local data')
    _check_width(model, fvs, feature_col)
    if out is None:
        from .featurize import _remove_at_exit
        out = tempfile.mkdtemp(prefix=scratch_prefix('preds'), dir=spill_dir())
        _remove_at_exit(out)
        disk.output(out)
    os.makedirs(out, exist_ok=True)
    workers = workers or workers_for()
    with inputs.scratch() as tmp:
        src = inputs.stage(fvs, tmp, 'fvs', workers=workers)
        fn = partial(_predict_chunk, model=model, feature_col=feature_col,
                     prediction_col=prediction_col, confidence_col=confidence_col,
                     out_dir=out)
        chunks = inputs.chunks(src, workers=workers)
        rows = sum(c.rows for c in chunks)
        meter = disk.Meter('the predictions', out, rows)
        disk.measured(run_tasks, fn, list(enumerate(chunks)), workers, meter,
                      inputs.column_bytes(src) + 16 * rows, _meter(meter, out),
                      task_bytes=inputs.chunk_bytes())
    open(os.path.join(out, '_SUCCESS'), 'w').close()
    return inputs.dataset(out)


def _meter(meter, out):
    finished = {}

    def done(k, rows):
        finished[f'part-{k:05d}.parquet'] = rows
        meter.wrote(sum(finished.values()), *disk.written_in(out, finished))
    return done


def _gold_ids(gold):
    if hasattr(gold, 'id1') and hasattr(gold, 'contains'):
        return gold.id1, gold.id2
    if hasattr(gold, 'columns') and not isinstance(gold, pa.Table):
        return gold['id1'].to_numpy(), gold['id2'].to_numpy()
    table = gold if isinstance(gold, pa.Table) else gold.to_table(columns=['id1', 'id2'])
    return table.column('id1').to_numpy(), table.column('id2').to_numpy()


def save_matches(predictions, path, threshold):
    """Write the predicted matches at `threshold` to one Parquet file as `id1`, `id2` and
    `probability`, and return their number."""
    partial = f'{path}.partial-{os.getpid()}'
    written = 0
    writer = pq.ParquetWriter(partial, MATCH_SCHEMA, compression='zstd')
    try:
        for batch in inputs.arrow_batches(predictions, ['id1', 'id2', 'prediction', 'confidence']):
            pred = batch.column('prediction').to_numpy(zero_copy_only=False)
            conf = batch.column('confidence').to_numpy(zero_copy_only=False)
            keep = np.flatnonzero(metrics.positives(pred, conf, threshold))
            if not len(keep):
                continue
            prob = metrics.p_match(pred[keep], conf[keep]).astype(np.float32)
            writer.write_batch(pa.RecordBatch.from_arrays(
                [pa.array(batch.column('id1').to_numpy(zero_copy_only=False)[keep], pa.int64()),
                 pa.array(batch.column('id2').to_numpy(zero_copy_only=False)[keep], pa.int64()),
                 pa.array(prob, pa.float32())], schema=MATCH_SCHEMA))
            written += len(keep)
        writer.close()
        os.replace(partial, path)
    except BaseException:
        writer.close()
        if os.path.exists(partial):
            os.remove(partial)
        raise
    return written


def _not_whole(values, k=5):
    bad = []
    for v in values:
        try:
            np.asarray([v], dtype=np.int64)
        except (TypeError, ValueError, OverflowError):
            bad.append(v)
            if len(bad) == k:
                break
    return bad or f'values of the types {sorted({type(v).__name__ for v in values})}'


def _check_gold_column(name, column):
    if column.null_count:
        raise ValueError(f"gold '{name}' holds {column.null_count} null id(s)")
    if pa.types.is_integer(column.type):
        return
    values = column.to_numpy(zero_copy_only=False)
    if pa.types.is_floating(column.type):
        bad = ~np.isfinite(values) | (values != np.floor(values))
        if bad.any():
            raise ValueError(f"gold '{name}' must hold whole numbers, got "
                             f"{values[bad][:5].tolist()}")
        return
    try:
        np.asarray(values, dtype=np.int64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"gold '{name}' must hold whole numbers, got "
                         f"{_not_whole(values)}") from None


def check_gold_ids(gold):
    """Check that the gold pairs' ids are whole numbers without nulls."""
    for name in ('id1', 'id2'):
        try:
            for batch in inputs.arrow_batches(gold, [name]):
                _check_gold_column(name, batch.column(0))
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            if not isinstance(gold, pd.DataFrame):
                raise
            raise ValueError(f"gold '{name}' must hold whole numbers, got "
                             f"{_not_whole(gold[name])}") from None


@contextlib.contextmanager
def gold_arrays(gold, dedup):
    """Yield the distinct gold pairs sorted by `(id1, id2)`, oriented `id1 <= id2` when
    `dedup`, built on scratch disk by workers when the gold set is large."""
    from . import candidates as cand_mod
    from . import gold as gold_mod

    if isinstance(gold, gold_mod.GoldSet):
        yield (gold.id1, gold.id2) if gold.dedup or not dedup else \
            metrics.canonical_gold(gold.id1, gold.id2, dedup)
        return
    check_gold_ids(gold)
    if cand_mod.count(gold) <= cand_mod.PAIRS_PER_PART:
        yield metrics.canonical_gold(*_gold_ids(gold), dedup)
        return
    with inputs.scratch() as tmp:
        with disk.disk_step('evaluate'):
            built = gold_mod.build(gold, dedup, os.path.join(tmp, 'gold'))
        yield built.id1, built.id2


def evaluate_counts(predictions, gold, threshold, dedup):
    """Return the gold count, the predicted count, the true positives and the
    probability histogram of the predictions."""
    with contextlib.ExitStack() as stack:
        g1 = g2 = None
        if gold is not None:
            g1, g2 = stack.enter_context(gold_arrays(gold, dedup))
        return _counts(predictions, g1, g2, gold is not None, threshold)


def _counts(predictions, g1, g2, with_gold, threshold):
    gold = True if with_gold else None
    n_predicted = tp = 0
    rows = np.zeros(101, np.int64)
    gold_rows = np.zeros(101, np.int64)
    cols = ['id1', 'id2', 'prediction', 'confidence']
    for batch in inputs.arrow_batches(predictions, cols):
        pred = batch.column('prediction').to_numpy(zero_copy_only=False)
        conf = batch.column('confidence').to_numpy(zero_copy_only=False)
        pos = metrics.positives(pred, conf, threshold)
        n_predicted += int(pos.sum())
        if gold is not None:
            is_gold = metrics.is_member_sorted(batch.column('id1').to_numpy(),
                                               batch.column('id2').to_numpy(), g1, g2)
            tp += int((pos & is_gold).sum())
            b = metrics.bucket(metrics.p_match(pred, conf))
            rows += np.bincount(b, minlength=101)
            gold_rows += np.bincount(b[is_gold], minlength=101)
    if gold is None:
        return 0, n_predicted, 0, None
    if tp > len(g1):
        raise ValueError(f'the predictions list some pairs more than once: {tp:,} true '
                         f'positives against {len(g1):,} gold pairs; evaluate counts each '
                         f'pair once, as apply_matcher gives them')
    hist = {int(i): (int(rows[i]), int(gold_rows[i])) for i in np.flatnonzero(rows)}
    return len(g1), n_predicted, tp, hist
