"""Feature vectors on the local engine: vocabularies, record stores and the pair
pass."""
import atexit
import os
import shutil
import tempfile

import pandas as pd

from .._core.feature.catalog import BuildCache
from .._core.storage import spill_dir
from .._core.textform import resolve_timezone
from .._core.utils import check_driver_collect, get_logger
from . import candidates as cand_mod
from .._core import files
from . import inputs, pairpass, store
from .vocab import build_vocabularies

log = get_logger(__name__)

_RESULTS = []


def _remove_at_exit(path):
    if not _RESULTS:
        atexit.register(lambda: [shutil.rmtree(p, ignore_errors=True) for p in _RESULTS])
    _RESULTS.append(path)


def build_vocabularies_for(A, B, features, tz, workers=None):
    """Build every TF/IDF and SIF vocabulary the features need, once per distinct
    request."""
    vocab_feats = [f for f in features if hasattr(f, 'vocab_request')]
    requests = list(dict.fromkeys(f.vocab_request() for f in vocab_feats))
    built = dict(zip(requests, build_vocabularies(A, B, requests, tz, workers)))
    for f in vocab_feats:
        v = built[f.vocab_request()]
        f.set_vocab(v.hashes, v.doc_freqs, v.corpus_size)
    cache = BuildCache()
    for f in features:
        if not hasattr(f, 'vocab_request'):
            f.build(A, B, cache)
    cache.clear()


def featurize(features, A, B, candidates, output_col='feature_vectors', fill_na=0.0,
              validate=True, timezone=None, out=None, workers=None):
    """Return the feature vector and score of every candidate pair, as pandas or as a
    dataset of Parquet parts."""
    tz = resolve_timezone(timezone)
    if validate:
        from ..io import check_candidates
        check_candidates(candidates, A, A if B is None else B)
    return_pandas = isinstance(A, pd.DataFrame) and (B is None or isinstance(B, pd.DataFrame))
    n_listed = cand_mod.count(candidates)
    if return_pandas:
        check_driver_collect(
            n_listed, 'featurize with pandas A and B',
            'Pass A and B as Arrow datasets (parquet) to keep the feature vectors on '
            'disk.')
    if out is None:
        out = tempfile.mkdtemp(prefix='iris-fvs-', dir=spill_dir())
        if not return_pandas:
            _remove_at_exit(out)
        own_out = True
    else:
        out = files.local_path(os.fspath(out))
        shutil.rmtree(out, ignore_errors=True)
        own_out = False
    try:
        with inputs.scratch() as tmp:
            build_vocabularies_for(A, B, features, tz, workers)
            store_a = store.build(A, features, True, B is None, tz,
                                  os.path.join(tmp, 'store_a'), 'table_a', workers)
            store_b = store_a if B is None else store.build(
                B, features, False, True, tz, os.path.join(tmp, 'store_b'), 'table_b',
                workers)
            _check_scratch(n_listed, len(features), out)
            b_ids = _store_ids(store_b)
            pairs_dir = os.path.join(tmp, 'pairs')
            os.makedirs(pairs_dir)
            parts, repeats = cand_mod.canonical(candidates, b_ids, pairs_dir, workers)
            del b_ids
            if validate and repeats:
                log.warning(f'featurize: {repeats:,} of {n_listed:,} candidate pairs are '
                            f'repeats and are dropped; each distinct pair is featurized '
                            f'once')
            pairpass.run(parts, store_a, store_b, features, fill_na, output_col, out,
                         workers)
        if return_pandas:
            return inputs.dataset(out).to_table().to_pandas()
        return inputs.dataset(out)
    finally:
        if return_pandas and own_out:
            shutil.rmtree(out, ignore_errors=True)


def _store_ids(st):
    import numpy as np

    parts = [np.fromfile(store._segment_path(st.directory, s['k'], 'ids'), np.int64)
             for s in st.segments]
    return np.concatenate(parts) if parts else np.empty(0, np.int64)


def _check_scratch(n_pairs, n_features, out):
    inputs.require_disk(int(n_pairs * (4 * n_features + 24)), out, 'the feature vectors',
                        'Write them to a larger volume (out=...), or free space.')
