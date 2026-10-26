"""Batch-mode active learning over a Spark DataFrame of feature vectors."""

import numpy as np
import pandas as pd
import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark import StorageLevel
from pyspark.sql import SparkSession

from ..._core.learn import SELECTION_STRATEGIES, BatchLearner
from ..._core.ml_model import normalize_fv_cells
from ..._core.utils import feature_vector_width, get_logger, type_check
from ..ml_model import SparkMLModel

log = get_logger(__name__)

POOL_PART_ROWS = 25_000


def _seeded(seed):
    return F.xxhash64(F.lit(seed).cast('long'), 'id1', 'id2')


def entropy_order(seed):
    """Return the sort keys for selection by entropy: highest entropy first, ties by the
    seeded pair hash."""
    return [F.desc('entropy'), _seeded(seed), F.asc('id2'), F.asc('id1')]


def random_order(seed):
    """Return the sort keys for a random draw: the seeded pair hash."""
    return [_seeded(seed), F.asc('id2'), F.asc('id1')]


def select_batch(model, cand_fvs, batch_size, selection, seed):
    """Return the `batch_size` rows to label next, in labeling order."""
    if selection not in SELECTION_STRATEGIES:
        raise ValueError(f'selection must be one of {SELECTION_STRATEGIES}, '
                         f'got {selection!r}')

    def _by_entropy(df, n):
        return model.entropy(df, 'feature_vectors', 'entropy')\
                    .sort(*entropy_order(seed))\
                    .limit(n)\
                    .drop('entropy')

    def _at_random(df, n, iter_seed):
        if n <= 0:
            return df.limit(0)
        return df.sort(*random_order(iter_seed)).limit(n)

    if selection == 'entropy':
        return _by_entropy(cand_fvs, batch_size)
    if selection == 'random':
        return _at_random(cand_fvs, batch_size, seed)

    n_unc = (batch_size + 1) // 2
    unc = _by_entropy(cand_fvs, n_unc)
    rnd = _at_random(cand_fvs.join(unc.select('id1', 'id2'), ['id1', 'id2'], 'left_anti'),
                     batch_size - n_unc, seed)
    return unc.unionByName(rnd)
    
    
def train_in_order(model, spark, frame, schema=None):
    """Train `model` on a labeled pandas frame, its rows in labeling order."""
    if isinstance(model, SparkMLModel):
        frame = spark.createDataFrame(frame, schema=schema)
    model.train(frame, 'feature_vectors', 'label')


def driver_schema(spark, driver_df):
    """Return the Spark schema of the labeled frame held in the driver."""
    inferred = spark.createDataFrame(driver_df.head(1)).schema
    return T.StructType([
        T.StructField(f.name, T.ArrayType(T.DoubleType()), f.nullable)
        if f.name == 'feature_vectors' else f
        for f in inferred.fields
    ])


class SparkPool:
    """The labeling pool as a Spark DataFrame, for the active-learning loops."""

    def __init__(self, fvs, model):
        type_check(fvs, 'fvs', pyspark.sql.DataFrame)
        self._spark = SparkSession.builder.getOrCreate()
        n = fvs.count()
        if n:
            parts = max(n // POOL_PART_ROWS, 4 * self._spark.sparkContext.defaultParallelism)
            fvs = fvs.repartition(min(parts, n), 'id1', 'id2')
        self._fvs = model.prep_fvs(fvs).persist(StorageLevel.MEMORY_AND_DISK)
        self._n = None

    def count(self):
        """Return the number of pairs in the pool."""
        if self._n is None:
            self._n = self._fvs.count()
        return self._n

    def vector_width(self):
        """Return the width of the feature vectors."""
        return feature_vector_width(self._fvs)

    def _pairs(self, frame):
        return self._spark.createDataFrame(pd.DataFrame({
            'id1': frame['id1'].to_numpy(np.int64),
            'id2': frame['id2'].to_numpy(np.int64)}))

    def batch(self, model, labeled, skipped, n, selection, i):
        """Return the next batch of pairs to label."""
        cand_fvs = self._fvs.join(self._pairs(labeled), on=['id1', 'id2'], how='left_anti')
        if skipped:
            cand_fvs = cand_fvs.join(
                self._spark.createDataFrame(sorted(skipped), 'id1 long, id2 long'),
                ['id1', 'id2'], 'left_anti')
        out = select_batch(model, cand_fvs, n, selection, seed=42 + i).toPandas()
        normalize_fv_cells(out, 'feature_vectors')
        return out

    def top_entropy(self, model, seen, k, i, window):
        """Return the `k` unlabeled pairs of highest entropy."""
        cand_fvs = self._fvs.join(self._pairs(seen), on=['id1', 'id2'], how='left_anti')
        if window is not None:
            remaining = max(self.count() - len(seen), 1)
            fraction = min(1.0, window / remaining)
            if fraction < 1.0:
                cand_fvs = cand_fvs.sample(withReplacement=False, fraction=fraction, seed=i)
        out = (model.entropy(cand_fvs, 'feature_vectors', 'entropy')
                    .sort(*entropy_order(42 + i))
                    .limit(k)
                    .toPandas())
        normalize_fv_cells(out, 'feature_vectors')
        return out

    def everything(self):
        """Return every pair of the pool as pandas."""
        out = self._fvs.orderBy('id2', 'id1').toPandas()
        normalize_fv_cells(out, 'feature_vectors')
        return out

    def train(self, model, frame):
        """Train the model on the labeled pairs in `frame`."""
        schema = driver_schema(self._spark, frame) if isinstance(model, SparkMLModel) else None
        train_in_order(model, self._spark, frame, schema)

    def close(self):
        """Release the pool."""
        self._fvs.unpersist()


class EntropyActiveLearner(BatchLearner):
    """Batch-mode active learning over a Spark DataFrame of feature vectors."""

    def _open_pool(self, fvs):
        return SparkPool(fvs, self._model)
