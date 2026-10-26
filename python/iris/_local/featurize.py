"""Feature vectors on the local engine: vocabularies, record stores and the pair
pass."""
import atexit
import os
import shutil
import tempfile
from collections import Counter

import pandas as pd

from .._core.feature.catalog import BuildCache
from .._core.storage import SPILL_DIR_ENV, scratch_prefix, spill_dir
from .._core.textform import resolve_timezone
from .._core.utils import check_driver_collect, get_logger
from . import candidates as cand_mod
from .._core import files
from . import disk, inputs, pairpass, store
from .vocab import build_vocabularies

log = get_logger(__name__)

_RESULTS = []

PANDAS_ROW_BYTES = 200

STORE_A, STORE_B = 'the record store of table_a', 'the record store of table_b'

PAIRS = 'the candidate pairs'

REMEDY = (f'Set {SPILL_DIR_ENV} to a directory on a larger volume, write the feature '
          f'vectors to one (out=...), or free space.')


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


@disk.disk_step('featurize', REMEDY)
def featurize(features, A, B, candidates, output_col='feature_vectors', fill_na=0.0,
              validate=True, timezone=None, out=None, workers=None):
    """Return the feature vector and score of every candidate pair, as pandas or as a
    dataset of Parquet parts."""
    from ..io import ONE_TABLE_MESSAGE

    tz = resolve_timezone(timezone)
    _check_names(A, features, 'table_a', True, B is None)
    if B is not None:
        _check_names(B, features, 'table_b', False, True)
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
        _check_room(n_listed, len(features))
    if out is None:
        out = tempfile.mkdtemp(prefix=scratch_prefix('fvs'), dir=spill_dir())
        if not return_pandas:
            _remove_at_exit(out)
        disk.output(out)
        own_out = True
    else:
        out = files.local_path(os.fspath(out))
        shutil.rmtree(out, ignore_errors=True)
        own_out = False
    step = disk.current()
    _plan(step, B is None, n_listed)
    try:
        with inputs.scratch() as tmp:
            build_vocabularies_for(A, B, features, tz, workers)
            for name, what, table, for_a, for_b in _stores(A, B):
                step.phase(name, kept=[(store.estimate(table, features, for_a, for_b, tz,
                                                       what), tmp)])
            step.begin(STORE_A)
            store_a = store.build(A, features, True, B is None, tz,
                                  os.path.join(tmp, 'store_a'), 'table_a', workers)
            if B is not None:
                step.begin(STORE_B)
            store_b = store_a if B is None else store.build(
                B, features, False, True, tz, os.path.join(tmp, 'store_b'), 'table_b',
                workers)
            step.begin(PAIRS)
            pairs_dir = os.path.join(tmp, 'pairs')
            os.makedirs(pairs_dir)
            parts, repeats, n_self = cand_mod.canonical(candidates, pairs_dir, workers,
                                                        one_table=B is None)
            if validate and B is None and (repeats or n_self):
                log.warning(ONE_TABLE_MESSAGE.format(n_self=n_self, repeats=repeats))
            elif validate and repeats:
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


def _check_names(table, features, what, for_a, for_b):
    names = inputs.columns(table)
    read = {'_id'}
    if for_a:
        read |= {f.a_attr for f in features}
    if for_b:
        read |= {f.b_attr for f in features}
    if isinstance(table, pd.DataFrame):
        read = set(names)
    repeated = sorted((c for c, n in Counter(names).items() if n > 1 and c in read), key=str)
    if repeated:
        raise ValueError(f'{what} has columns of the same name: {repeated}; give each column '
                         f'its own name')


def _check_room(n_pairs, n_features):
    from .. import resources

    need = n_pairs * (8 * n_features + PANDAS_ROW_BYTES)
    total_gb, _, source = resources.budget()
    room = total_gb * 2**30 * resources.SHARE
    if need > room:
        raise RuntimeError(
            f'featurize with pandas A and B would bring {n_pairs:,} feature vectors of '
            f'{n_features} values into this process, about {need / 2**30:.1f} GiB, more '
            f'than the {room / 2**30:.1f} GiB Iris plans to use ({source}). Pass A and B '
            f'as Arrow datasets (parquet) to keep the feature vectors on disk.')


def _stores(A, B):
    if B is None:
        return [(STORE_A, 'table_a', A, True, True)]
    return [(STORE_A, 'table_a', A, True, False), (STORE_B, 'table_b', B, False, True)]


def _plan(step, one_table, n_pairs):
    for name in [STORE_A] if one_table else [STORE_A, STORE_B]:
        step.phase(name)
    step.phase(PAIRS, scratch=[(16 * n_pairs, disk.scratch_root())])
    step.begin()
