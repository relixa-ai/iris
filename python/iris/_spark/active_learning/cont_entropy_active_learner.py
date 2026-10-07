"""Continuous-mode active learning over a Spark DataFrame of feature vectors."""
from ..._core.learn import ContinuousLearner
from .ent_active_learner import SparkPool


class ContinuousEntropyActiveLearner(ContinuousLearner):
    """Continuous-mode active learning over a Spark DataFrame of feature vectors."""

    def _open_pool(self, fvs):
        return SparkPool(fvs, self._model)
