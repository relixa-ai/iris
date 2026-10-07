"""Iris: machine-learning matching for entity matching, on one machine or a Spark
cluster."""

from .config import load_env, configure, pin_worker_python

load_env()
pin_worker_python()

from .tools import (
    create_features,
    get_base_sim_functions,
    get_base_tokenizers,
    get_extra_tokenizers,
    featurize,
    down_sample,
    create_seeds,
    evaluate,
    train_matcher,
    apply_matcher,
    label_al,
    label_pairs,
    save_features,
    load_features,
    save_data,
    load_data,
    read_table,
    read_pairs,
    clean_candidates,
)

from .candidates import (
    candidates_from_flat,
    candidates_from_result,
    write_flat_candidates,
)

from ._core.labeler import (
    Labeler,
    CustomLabeler,
    CLILabeler,
    GoldLabeler,
    DelayedGoldLabeler,
    WebUILabeler,
)

from ._core.tokenizer.tokenizer import Tokenizer
from ._core.tokenizer.vectorizer import Vectorizer
from ._core.feature.feature import Feature
from ._core.ml_model import MLModel, SKLearnModel

from . import io
from .io import (
    check_tables,
    check_candidates,
    check_labeled_data,
    check_pairs,
    check_gold_data,
)

__version__ = '0.1.0'

__all__ = [

    'configure', 'load_env',

    'create_features', 'featurize', 'down_sample', 'create_seeds', 'label_al',
    'label_pairs', 'train_matcher', 'apply_matcher', 'evaluate', 'clean_candidates',
    'get_base_sim_functions', 'get_base_tokenizers', 'get_extra_tokenizers',
    'save_features', 'load_features', 'save_data', 'load_data', 'read_table', 'read_pairs',

    'candidates_from_flat', 'candidates_from_result', 'write_flat_candidates',

    'Labeler', 'CustomLabeler', 'CLILabeler', 'GoldLabeler', 'DelayedGoldLabeler',
    'WebUILabeler',

    'Tokenizer', 'Vectorizer', 'Feature', 'MLModel', 'SKLearnModel', 'SparkMLModel',

    'check_tables', 'check_candidates', 'check_labeled_data',
    'check_pairs', 'check_gold_data',
    'io',
]


def __getattr__(name):
    if name == 'SparkMLModel':
        from ._spark.ml_model import SparkMLModel
        return SparkMLModel
    import difflib
    match = difflib.get_close_matches(name, __all__, n=1)
    hint = f' -- did you mean {match[0]!r}?' if match else ''
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}{hint}')
