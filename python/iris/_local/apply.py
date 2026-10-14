"""Applying a matcher and counting its results against gold pairs, on the local
engine."""
import os
import tempfile
from functools import partial

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .._core import metrics
from .._core.storage import spill_dir
from . import inputs
from .learn import vector_matrix
from .procpool import atomic_output, run_tasks, workers_for

WORKER_GB = 1.0

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


def apply_matcher(model, fvs, feature_col, prediction_col, confidence_col=None,
                  out=None, workers=None):
    """Return the feature vectors with predictions, as a dataset of Parquet parts
    written by worker processes."""
    if type(model).__module__ == 'iris._spark.ml_model':
        raise TypeError('SparkMLModel predicts on Spark: pass the feature vectors as a '
                        'Spark DataFrame, or use an SKLearnModel on local data')
    if out is None:
        from .featurize import _remove_at_exit
        out = tempfile.mkdtemp(prefix='iris-preds-', dir=spill_dir())
        _remove_at_exit(out)
    os.makedirs(out, exist_ok=True)
    with inputs.scratch() as tmp:
        src = inputs.stage(fvs, tmp, 'fvs')
        fn = partial(_predict_chunk, model=model, feature_col=feature_col,
                     prediction_col=prediction_col, confidence_col=confidence_col,
                     out_dir=out)
        run_tasks(fn, list(enumerate(inputs.chunks(src))),
                  workers or workers_for(WORKER_GB))
    open(os.path.join(out, '_SUCCESS'), 'w').close()
    return inputs.dataset(out)


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


def evaluate_counts(predictions, gold, threshold, dedup):
    """Return the gold count, the predicted count, the true positives and the
    probability histogram of the predictions."""
    from .gold import GoldSet

    if isinstance(gold, GoldSet) and (gold.dedup or not dedup):
        g1, g2 = gold.id1, gold.id2
    elif gold is not None:
        g1, g2 = metrics.sorted_gold(*metrics.canonical_gold(*_gold_ids(gold), dedup))
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
    hist = {int(i): (int(rows[i]), int(gold_rows[i])) for i in np.flatnonzero(rows)}
    return len(g1), n_predicted, tp, hist
