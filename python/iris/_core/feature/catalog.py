"""The built-in tokenizers and similarity functions, and the cache through which
features share vocabulary builds."""
from threading import Lock

from ..tokenizer import (
    AlphaNumericTokenizer,
    NumericTokenizer,
    QGramTokenizer,
    StrippedQGramTokenizer,
    StrippedWhiteSpaceTokenizer,
)
from .token_feature import CosineFeature, JaccardFeature, OverlapCoeffFeature
from .vector_feature import SIFFeature, TFIDFFeature

TOKENIZERS = [
        StrippedWhiteSpaceTokenizer(),
        NumericTokenizer(),
        QGramTokenizer(3),
]

EXTRA_TOKENIZERS = [
        AlphaNumericTokenizer(),
        QGramTokenizer(5),
        StrippedQGramTokenizer(3),
        StrippedQGramTokenizer(5),
]

SIM_FUNCTIONS = [
    TFIDFFeature,
    JaccardFeature, 
    SIFFeature,
    OverlapCoeffFeature, 
    CosineFeature,
]


class BuildCache:
    """A cache through which features that need the same vocabulary share one build."""
    def __init__(self):
        self._cache = []
        self._lock = Lock()

    def add_or_get(self, builder):
        """Return the cached builder equal to `builder`, adding `builder` when there is
        none."""
        with self._lock:
            try:
                builder = self._cache[self._cache.index(builder)]
            except ValueError:
                self._cache.append(builder)

        return builder

    def clear(self):
        """Remove every cached builder."""
        with self._lock:
            self._cache.clear()


def get_base_sim_functions():
    """Return the similarity functions `create_features` uses by default."""
    return SIM_FUNCTIONS


def get_base_tokenizers():
    """Return the tokenizers `create_features` uses by default."""
    return TOKENIZERS


def get_extra_tokenizers():
    """Return the additional tokenizers `create_features` can use."""
    return EXTRA_TOKENIZERS
