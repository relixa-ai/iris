"""The public functions of Iris, each running on the local engine or on Spark according
to its inputs."""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Callable, Any, Union, Literal
import pandas as pd
import numpy as np
import pickle
import os
import sys

from ._core import files, metrics, pool
from ._core import seeds as seed_rules
from ._core.frames import is_spark
from ._local.inputs import is_arrow
from ._core.ml_model import MLModel
from ._core.utils import get_logger, save_training_data_streaming, load_training_data_streaming
from ._core.labeler import Labeler, GoldLabeler
from ._core.feature.catalog import (
    get_base_sim_functions,
    get_base_tokenizers,
    get_extra_tokenizers,
)
from ._core.feature.feature import NeedlemanWunschFeature
from ._core.feature.token_feature import (JaccardFeature, OverlapCoeffFeature,
                                          CosineFeature)
from ._core.feature.vector_feature import TFIDFFeature, SIFFeature
from ._core.tokenizer.tokenizer import (AlphaNumericTokenizer, NumericTokenizer,
                                        QGramTokenizer, ShingleTokenizer,
                                        StrippedQGramTokenizer,
                                        StrippedWhiteSpaceTokenizer,
                                        WhiteSpaceTokenizer)

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame

logger = get_logger(__name__)

_FIXED_TOKENIZERS = {
    StrippedWhiteSpaceTokenizer.NAME: StrippedWhiteSpaceTokenizer,
    WhiteSpaceTokenizer.NAME: WhiteSpaceTokenizer,
    NumericTokenizer.NAME: NumericTokenizer,
    AlphaNumericTokenizer.NAME: AlphaNumericTokenizer,
}

SIM_FUNCTION_NAMES = {
    'tf_idf': TFIDFFeature,
    'sif': SIFFeature,
    'jaccard': JaccardFeature,
    'overlap_coeff': OverlapCoeffFeature,
    'cosine': CosineFeature,
    'needleman_wunch': NeedlemanWunschFeature,
}


def resolve_tokenizers(spec):
    """Return the tokenizers named by a preset ('base', 'extra' or 'all') or by a list
    of names."""
    import re

    if spec is None or spec == 'base':
        return get_base_tokenizers()
    if spec == 'extra':
        return get_extra_tokenizers()
    if spec == 'all':
        return get_base_tokenizers() + get_extra_tokenizers()
    if isinstance(spec, str):
        raise ValueError(f"unknown tokenizer preset {spec!r}: use 'base', 'extra', "
                         f"'all', or a list of tokenizer names")
    out = []
    for name in spec:
        if name in _FIXED_TOKENIZERS:
            out.append(_FIXED_TOKENIZERS[name]())
            continue
        m = re.fullmatch(r'(stripped_)?(\d+)gram_tokens', name)
        if m:
            cls = StrippedQGramTokenizer if m.group(1) else QGramTokenizer
            out.append(cls(int(m.group(2))))
            continue
        m = re.fullmatch(r'(\d+)shingle_tokens', name)
        if m:
            out.append(ShingleTokenizer(int(m.group(1))))
            continue
        raise ValueError(
            f'unknown tokenizer name {name!r}; valid: '
            f'{sorted(_FIXED_TOKENIZERS)} plus "{{n}}gram_tokens", '
            f'"stripped_{{n}}gram_tokens", "{{n}}shingle_tokens"')
    return out


def resolve_sim_functions(spec):
    """Return the similarity functions named in a list, and the per-column string
    features it names."""
    if spec is None:
        return None, []
    classes, extras = [], []
    for name in spec:
        if name not in SIM_FUNCTION_NAMES:
            raise ValueError(f'unknown similarity function {name!r}; valid: '
                             f'{sorted(SIM_FUNCTION_NAMES)}')
        cls = SIM_FUNCTION_NAMES[name]
        (extras if cls is NeedlemanWunschFeature else classes).append(cls)
    return classes, extras

from ._core.utils import check_training_data_matches, feature_vector_width


__all__ = [
    'create_features',
    'get_base_sim_functions',
    'get_base_tokenizers',
    'get_extra_tokenizers',
    'featurize',
    'down_sample',
    'create_seeds',
    'train_matcher',
    'apply_matcher',
    'label_al',
    'label_pairs',
    'save_features',
    'load_features',
    'save_data',
    'load_data',
    'clean_candidates'
]


DEFAULT_MAX_POOL_SIZE = pool.DEFAULT_MAX_POOL_SIZE


def create_features(
    A: Union[pd.DataFrame, SparkDataFrame],
    B: Union[pd.DataFrame, SparkDataFrame],
    a_cols: List[str],
    b_cols: List[str],
    sim_functions: Optional[List[Callable[..., Any]]] = None,
    tokenizers: Optional[List[Callable[..., Any]]] = None,
    null_threshold: float = .5,
    numeric_threshold: Optional[float] = .9,
    min_doc_freq: int = 1,
    max_vocab_size: Optional[int] = None,
    timezone: Optional[str] = None,
) -> List[Callable]:
    """Return the features for matching the columns `a_cols` of table A with the columns
    `b_cols` of table B."""
    if is_spark(A) or is_spark(B):
        from ._spark.featurization import create_features as create
    else:
        from ._local.features import create_features as create

    return create(A, B, a_cols, b_cols, sim_functions, tokenizers, null_threshold,
                  numeric_threshold, min_doc_freq, max_vocab_size, timezone)


def featurize(
    features: List[Callable],
    A,
    B,
    candidates,
    output_col: str = 'feature_vectors',
    fill_na: float = 0.0,
    validate: bool = True,
    timezone: Optional[str] = None,
    out: Optional[str] = None,
) -> Union[pd.DataFrame, SparkDataFrame]:
    """Return the feature vector and score of every candidate pair, written to `out` when
    it is given. Passing the same table as `A` and `B` deduplicates that table."""
    if B is A:
        B = None
    if is_spark(A) or is_spark(B) or is_spark(candidates):
        from ._spark.featurization import featurize as run
    else:
        from ._local.featurize import featurize as run

    return run(features, A, B, candidates, output_col, fill_na, validate, timezone, out)


def _pair_arrays(candidates):
    if 'id1' in candidates.columns and 'id1_list' not in candidates.columns:
        return (candidates['id1'].to_numpy(np.int64), candidates['id2'].to_numpy(np.int64))
    lists = [np.asarray([] if x is None else x, dtype=np.int64)
             for x in candidates['id1_list'].to_list()]
    sizes = np.fromiter((len(x) for x in lists), dtype=np.int64, count=len(lists))
    id1 = np.concatenate(lists) if lists else np.empty(0, np.int64)
    return id1, np.repeat(candidates['id2'].to_numpy(np.int64), sizes)


def clean_candidates(
    candidates: Union[pd.DataFrame, SparkDataFrame],
    drop_self_pairs: bool = False,
    canonicalize_mirrors: bool = False,
) -> Union[pd.DataFrame, SparkDataFrame]:
    """Return the candidate set as distinct flat `(id1, id2)` pairs, without self-pairs
    and mirrored pairs when requested."""
    logger = get_logger(__name__)

    if isinstance(candidates, pd.DataFrame):
        id1, id2 = _pair_arrays(candidates)
        before = len(id1)
        if drop_self_pairs:
            keep = id1 != id2
            id1, id2 = id1[keep], id2[keep]
        if canonicalize_mirrors:
            id1, id2 = np.minimum(id1, id2), np.maximum(id1, id2)
        pairs = (np.unique(np.stack([id1, id2], axis=1), axis=0) if len(id1)
                 else np.empty((0, 2), dtype=np.int64))
        logger.info(f'clean_candidates: {before - len(pairs)} of {before} pair(s) removed')
        return pd.DataFrame({'id1': pairs[:, 0], 'id2': pairs[:, 1]})

    if is_spark(candidates):
        from ._spark import steps
        return steps.clean_candidates(candidates, drop_self_pairs, canonicalize_mirrors)

    if is_arrow(candidates):
        from ._local.candidates import clean
        return clean(candidates, drop_self_pairs, canonicalize_mirrors)

    raise TypeError(
        f'candidates must be a pandas DataFrame, an Arrow dataset or a Spark DataFrame '
        f'(got {type(candidates)})'
    )


def down_sample(
    fvs: Union[pd.DataFrame, SparkDataFrame],
    percent: float,
    score_column: str = 'score',
    bucket_size: int = 1_000,
    max_pool_size: Optional[int] = DEFAULT_MAX_POOL_SIZE,
    seed: Optional[int] = 42,
) -> Union[pd.DataFrame, SparkDataFrame]:
    """Return the labeling pool, a sample of the feature vectors that covers the range
    of scores."""
    if bucket_size < 1000:
        raise ValueError('bucket_size must be >= 1000')
    if percent <= 0 or percent > 1.0:
        raise ValueError('percent must be in the range (0.0, 1.0]')
    if max_pool_size is not None and max_pool_size <= 0:
        raise ValueError(f'max_pool_size must be > 0 (got {max_pool_size})')
    if seed is None:
        seed = pool.draw_seed(None)
        logger.info(f'down_sample: drew seed {seed}; pass seed={seed} to draw this pool again')

    if isinstance(fvs, pd.DataFrame):
        idx = pool.select(fvs['id1'], fvs['id2'], fvs[score_column], percent,
                          bucket_size, max_pool_size, seed)
        return fvs.iloc[idx].reset_index(drop=True)

    if is_arrow(fvs):
        from ._local.pool import down_sample as draw
        return draw(fvs, percent, score_column, bucket_size, max_pool_size, seed)

    if is_spark(fvs):
        from ._spark import steps
        return steps.down_sample(fvs, percent, score_column, bucket_size, max_pool_size,
                                 seed)

    raise TypeError(f'fvs must be a pandas DataFrame, an Arrow dataset or a Spark DataFrame '
                    f'(got {type(fvs)})')


def _files_beside(arg):
    import functools
    import inspect

    def wrap(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def inner(*args, **kwargs):
            frame = sig.bind_partial(*args, **kwargs).arguments.get(arg)
            if not is_spark(frame):
                return fn(*args, **kwargs)
            from ._spark.files import HadoopFiles
            with files.using(HadoopFiles(frame.sparkSession)):
                return fn(*args, **kwargs)
        return inner
    return wrap


@_files_beside('fvs')
def create_seeds(
    fvs: Union[pd.DataFrame, SparkDataFrame],
    nseeds: int,
    labeler: Labeler,
    score_column: str = 'score',
    checkpoint_path: str = 'active-matcher-training-data.parquet'
) -> Union[pd.DataFrame, SparkDataFrame]:
    """Return labeled pairs drawn from likely matches and likely non-matches, to start
    active learning."""
    if nseeds <= 0:
        raise ValueError("no seeds would be created")


    arrow = is_arrow(fvs)
    return_pandas = isinstance(fvs, pd.DataFrame) or arrow
    if not return_pandas and not is_spark(fvs):
        raise TypeError(f'fvs must be a pandas DataFrame, an Arrow dataset or a Spark '
                        f'DataFrame (got {type(fvs)})')

    def as_output(rows):
        if return_pandas:
            return pd.DataFrame(rows)
        from ._spark import steps
        return steps.rows_frame(rows)

    existing_training_data = load_training_data_streaming(checkpoint_path, logger)
    check_training_data_matches(existing_training_data, feature_vector_width(fvs),
                                checkpoint_path)

    labeled = set()
    pos = neg = 0
    seeds = []
    if existing_training_data is not None:
        logger.warning(
            f'Resuming from {len(existing_training_data)} labeled examples already in '
            f'{checkpoint_path!r} -- no new seeds will be created from this fvs. '
            f'Delete that file or pass checkpoint_path=<new path> to start fresh.')
        if len(existing_training_data) >= nseeds:
            logger.info(f'Already have {len(existing_training_data)} labeled examples, '
                        f'returning existing data')
            return as_output(existing_training_data.head(nseeds).to_dict('records'))
        seeds = existing_training_data.to_dict('records')
        pos = int(existing_training_data['label'].sum())
        neg = len(existing_training_data) - pos
        labeled = set(zip(existing_training_data['id1'].tolist(),
                          existing_training_data['id2'].tolist()))

    if isinstance(labeler, GoldLabeler):
        if arrow:
            if score_column not in fvs.schema.names:
                from ._local.learn import gold_scores
                fvs = gold_scores(fvs, labeler, score_column)
        elif return_pandas and (score_column not in fvs.columns or fvs[score_column].isna().all()):
            gold_pairs = labeler._gold
            is_gold_mask = [tuple(x) in gold_pairs for x in fvs[['id1', 'id2']].to_numpy()]
            fvs = fvs.assign(**{score_column: np.where(is_gold_mask, 1.0, 0.0)})
        elif not return_pandas and score_column not in fvs.columns:
            from ._spark import steps
            fvs = steps.gold_scores(fvs, labeler._gold, score_column)

    if arrow:
        from ._local.learn import seed_rows
        likely_pos, likely_neg = seed_rows(fvs, score_column, nseeds)
    elif return_pandas:
        n_valid = int(fvs[score_column].notna().sum())
        if nseeds > n_valid:
            raise ValueError("number of seeds would exceed the size of the fvs DataFrame")
        pos_idx, neg_idx = seed_rules.rankings(fvs['id1'], fvs['id2'], fvs[score_column],
                                               nseeds)
        likely_pos = fvs.iloc[pos_idx].to_dict('records')
        likely_neg = fvs.iloc[neg_idx].to_dict('records')
    else:
        from ._spark import steps
        likely_pos, likely_neg = steps.seed_rankings(fvs, score_column, nseeds)

    def checkpoint(row, label):
        save_training_data_streaming(pd.DataFrame([row]), checkpoint_path, logger)

    new, pos, neg = seed_rules.label_seeds(likely_pos, likely_neg, labeler, nseeds,
                                           labeled=labeled, pos=pos, neg=neg,
                                           on_label=checkpoint)
    seeds = seeds + new
    if not seeds:
        raise RuntimeError("No seeds were labeled before stopping")
    logger.info(f'seeds: {pos} positive, {neg} negative')
    return as_output(seeds)


def evaluate(
    predictions: Union[pd.DataFrame, SparkDataFrame],
    gold: Optional[Union[pd.DataFrame, SparkDataFrame]],
    threshold: float = 0.5,
    dedup: bool = False,
) -> dict:
    """Return precision, recall, F1 and a threshold sweep of the predictions against the
    gold pairs."""
    if isinstance(predictions, pd.DataFrame):
        pred = predictions['prediction'].to_numpy(np.float64)
        conf = predictions['confidence'].to_numpy(np.float64)
        pos = metrics.positives(pred, conf, threshold)
        n_predicted = int(pos.sum())
        if gold is None:
            n_gold = tp = 0
            hist = None
        else:
            g1, g2 = metrics.canonical_gold(gold['id1'], gold['id2'], dedup)
            n_gold = len(g1)
            is_gold = metrics.is_member(predictions['id1'], predictions['id2'], g1, g2)
            tp = int((pos & is_gold).sum())
            hist = metrics.histogram(metrics.p_match(pred, conf), is_gold)
    elif is_arrow(predictions):
        from ._local.apply import evaluate_counts
        n_gold, n_predicted, tp, hist = evaluate_counts(predictions, gold, threshold,
                                                        dedup)
    elif is_spark(predictions):
        from ._spark import steps
        n_gold, n_predicted, tp, hist = steps.evaluate_counts(predictions, gold, threshold,
                                                              dedup)
    else:
        raise TypeError(f'predictions must be a pandas DataFrame, an Arrow dataset or a '
                        f'Spark DataFrame (got {type(predictions)})')
    precision, recall, f1 = metrics.prf(tp, n_predicted, n_gold)
    curve, best = metrics.sweep(hist, n_gold) if hist is not None else ([], None)
    return {'n_gold': n_gold, 'n_predicted': n_predicted, 'tp': tp,
            'fp': n_predicted - tp, 'fn': n_gold - tp,
            'precision': precision, 'recall': recall, 'f1': f1,
            'threshold_curve': curve,
            'best_threshold': float(best['threshold']) if best else None,
            'best_f1': float(best['f1']) if best else None}


def train_matcher(
    model: MLModel,
    labeled_data: Union[pd.DataFrame, SparkDataFrame],
    feature_col: str = "feature_vectors",
    label_col: str = "label",
) -> MLModel:
    """Return the model trained on the labeled feature vectors, leaving out the pairs
    labeled unsure."""
    if isinstance(labeled_data, pd.DataFrame):
        labels = labeled_data[label_col]
        other = sorted(set(labels.dropna().unique().tolist()) - {0.0, 1.0, 2.0})
        decided = labeled_data[labels.isin([0.0, 1.0])]
        n_match = int((decided[label_col] == 1.0).sum())
        n_non = len(decided) - n_match
    elif is_spark(labeled_data):
        from ._spark import steps
        decided, other, n_match, n_non = steps.decided_labels(labeled_data, label_col)
    else:
        raise TypeError(f'labeled_data must be a pandas DataFrame or Spark DataFrame '
                        f'(got {type(labeled_data)})')
    if other:
        raise ValueError(f'{label_col} holds {other}; a label is 1.0 (match), 0.0 '
                         f'(non-match) or 2.0 (unsure)')
    if not n_match or not n_non:
        raise ValueError(f'training needs matches and non-matches, but the labels hold '
                         f'{n_match} match(es) and {n_non} non-match(es)')
    return model.train(decided, feature_col, label_col)


def apply_matcher(
    model: MLModel,
    df: Union[pd.DataFrame, SparkDataFrame],
    feature_col: str,
    prediction_col: str,
    confidence_col: Optional[str] = None,
) -> Union[pd.DataFrame, SparkDataFrame]:
    """Return the model's prediction for every row, and its confidence when
    `confidence_col` is given."""
    if is_arrow(df):
        from ._local.apply import apply_matcher as apply
        return apply(model, df, feature_col, prediction_col, confidence_col)
    if confidence_col is not None:
        return model.predict_with_confidence(df, feature_col, prediction_col, confidence_col)
    else:
        return model.predict(df, feature_col, prediction_col)


@_files_beside('fvs')
def label_al(
    model: MLModel,
    mode: Literal["batch", "continuous"],
    labeler: Labeler,
    fvs: Union[pd.DataFrame, SparkDataFrame],
    seeds: Optional[Union[pd.DataFrame, SparkDataFrame]] = None,
    checkpoint_path: str = 'active-matcher-training-data.parquet',
    **learner_kwargs
) -> Union[pd.DataFrame, SparkDataFrame]:
    """Return labeled pairs collected from the labeler by active learning, in batch or
    continuous mode."""

    if is_spark(fvs):
        from ._spark import steps
        return steps.label_al(model, mode, labeler, fvs, seeds, checkpoint_path,
                              learner_kwargs)
    from ._local import learn
    return learn.label_al(model, mode, labeler, fvs, seeds, checkpoint_path,
                          learner_kwargs)


def label_pairs(
        labeler: Labeler,
        pairs: Union[pd.DataFrame, SparkDataFrame]
) -> Union[pd.DataFrame, SparkDataFrame]:
    """Return the given pairs with the labels the labeler assigns, without active
    learning."""
    stopped = False
    if is_arrow(pairs):
        pairs = pairs.to_pandas() if hasattr(pairs, 'to_pandas') else pairs.to_table().to_pandas()
    if isinstance(pairs, pd.DataFrame):
        labeled_pairs = []
        id1_col, id2_col = pairs.columns[0], pairs.columns[1]
        try:
            for _, row in pairs.iterrows():
                label = labeler(row[id1_col], row[id2_col])
                if label == -1.0:
                    stopped = True
                    break
                labeled_pairs.append({
                    id1_col: row[id1_col],
                    id2_col: row[id2_col],
                    'label': label
                })
            labeler.finish(len(labeled_pairs), stopped)
        finally:
            labeler.close()
        labeled_pairs = pd.DataFrame(labeled_pairs)
    elif is_spark(pairs):
        from ._spark import steps
        labeled_pairs = steps.label_pairs(labeler, pairs)
    else:
        raise TypeError(f'pairs must be a pandas DataFrame, an Arrow dataset or a Spark '
                        f'DataFrame (got {type(pairs)})')

    return labeled_pairs

def _files_for(path):
    if files.scheme(path) is not None and 'pyspark' in sys.modules:
        from ._spark.files import active
        found = active()
        if found is not None:
            return found
    return files.LocalFiles()


def save_features(features, path):
    """Write a list of features to `path`, a path or a URI of storage Spark reaches."""
    path = os.fspath(path)
    logger.info(f"Saving {len(features)} features to {path}")
    _files_for(path).write_bytes(path, pickle.dumps(features))
    logger.info(f"Successfully saved features to {path}")


def load_features(path):
    """Read a list of features written by `save_features`."""
    path = os.fspath(path)
    logger.info(f"Loading features from {path}")
    features = pickle.loads(_files_for(path).read_bytes(path))
    logger.info(f"Successfully loaded {len(features)} features from {path}")
    return features


def save_data(dataframe, path):
    """Write a pandas or Spark DataFrame to `path` as Parquet."""
    from .io import parquet_name
    path = parquet_name(path, 'save_data')
    if isinstance(dataframe, pd.DataFrame):
        logger.info(f"Saving pandas DataFrame with shape {dataframe.shape} to {path}")
        local = files.local_path(path)
        partial = f"{local}.partial-{os.getpid()}"
        try:
            dataframe.to_parquet(partial)
            os.replace(partial, local)
        finally:
            if os.path.exists(partial):
                os.remove(partial)
        logger.info(f"Successfully saved pandas DataFrame to {path}")
    elif is_spark(dataframe):
        from ._spark import steps
        steps.save_data(dataframe, path)
    else:
        raise TypeError(f"Unsupported DataFrame type: {type(dataframe)}. "
                       f"Expected pandas.DataFrame or pyspark.sql.DataFrame")


def save_predictions(predictions, path, threshold: float = 0.5) -> int:
    """Write the predicted matches, the pairs whose probability of a match is at or above
    `threshold`, to Parquet as `id1`, `id2` and `probability`, and return their number."""
    from .io import parquet_name, _columns
    path = parquet_name(path, 'save_predictions')
    missing = [c for c in ('id1', 'id2', 'prediction', 'confidence') if c not in _columns(predictions)]
    if missing:
        raise ValueError(f'save_predictions needs the columns {missing}; pass the result of '
                         f"apply_matcher(..., 'prediction', 'confidence')")
    if is_spark(predictions):
        from ._spark import steps
        return steps.save_matches(predictions, path, threshold)
    if isinstance(predictions, pd.DataFrame) or is_arrow(predictions):
        from ._local.apply import save_matches
        return save_matches(predictions, files.local_path(path), threshold)
    raise TypeError(f'predictions must be a pandas DataFrame, an Arrow dataset or a Spark '
                    f'DataFrame (got {type(predictions)})')


def load_data(path, df_type):
    """Read a Parquet file or directory as pandas, a Spark DataFrame or an Arrow dataset,
    according to `df_type`: 'pandas', 'sparkdf' or 'arrow'."""
    path = os.fspath(path)
    logger.info(f"Loading DataFrame from {path} as {df_type}")
    if df_type.lower() == 'pandas':
        dataframe = pd.read_parquet(files.local_path(path))
        logger.info(f"Successfully loaded pandas DataFrame with shape {dataframe.shape} from {path}")
        return dataframe
    elif df_type.lower() == 'sparkdf':
        from ._spark import steps
        return steps.load_data(path)
    elif df_type.lower() == 'arrow':
        from ._local.inputs import dataset
        return dataset(files.local_path(path))
    else:
        raise ValueError(f"Unsupported DataFrame type: {df_type}. "
                        f"Expected 'pandas', 'sparkdf' or 'arrow'")



def _reader_type(df_type):
    kind = str(df_type).lower()
    if kind not in ('pandas', 'arrow', 'sparkdf'):
        raise ValueError(f"df_type is 'pandas', 'arrow' or 'sparkdf', got {df_type!r}")
    return kind


def read_table(
    path,
    id_col: str = '_id',
    columns: Optional[List[str]] = None,
    df_type: str = 'pandas',
):
    """Read a table of records from a Parquet, CSV, TSV or JSON Lines file, with its id
    column named `_id`."""
    from .io import file_format, table_columns

    kind = _reader_type(df_type)
    fmt = file_format(path)
    path = os.fspath(path)
    if kind == 'sparkdf':
        from ._spark import steps
        return steps.read_table(path, fmt, id_col, columns)
    import pyarrow as pa
    import pyarrow.dataset as ds

    from ._local import readers
    path = files.local_path(path)
    src = readers.source(path, fmt, {id_col: pa.int64()})
    attrs = table_columns(path, src.schema.names, id_col, columns)
    projection = {'_id': ds.field(id_col), **{c: ds.field(c) for c in attrs}}
    as_is = fmt == 'parquet' and id_col == '_id' and columns is None
    return readers.output(path, src, projection, kind, as_is)


def read_pairs(
    path,
    columns: tuple = ('id1', 'id2'),
    with_label: bool = False,
    df_type: str = 'pandas',
):
    """Read a set of pairs from a Parquet, CSV, TSV or JSON Lines file as flat `id1`, `id2`
    rows, with `label` when `with_label` is True."""
    from .io import PAIR_ID_NAMES, file_format, pair_layout

    kind = _reader_type(df_type)
    fmt = file_format(path)
    path = os.fspath(path)
    columns = tuple(columns)
    if kind == 'sparkdf':
        from ._spark import steps
        return steps.read_pairs(path, fmt, columns, with_label)
    import pyarrow as pa
    import pyarrow.dataset as ds

    from ._local import readers
    path = files.local_path(path)
    src = readers.source(path, fmt, {**{c: pa.int64() for c in PAIR_ID_NAMES + columns},
                                     'label': pa.float64()})
    schema = src.schema
    grouped, x, y, list_is_a = pair_layout(
        path, schema.names,
        lambda c: pa.types.is_list(schema.field(c).type) or pa.types.is_large_list(schema.field(c).type),
        columns, with_label)
    label = {'label': ds.field('label')} if with_label else {}
    if grouped:
        projection = {x: ds.field(x), y: ds.field(y), **label}
        return readers.output(path, src, projection, kind, False,
                              readers.flatten(x, y, list_is_a, with_label))
    projection = {'id1': ds.field(x), 'id2': ds.field(y),
                  **({'label': ds.field('label').cast(pa.float64())} if with_label else {})}
    wanted = {'id1', 'id2'} | ({'label'} if with_label else set())
    as_is = fmt == 'parquet' and (x, y) == ('id1', 'id2') and set(schema.names) == wanted
    return readers.output(path, src, projection, kind, as_is)