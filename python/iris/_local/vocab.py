"""Building the TF/IDF and SIF vocabularies on the local engine, with worker
processes."""
import os
from functools import partial

import numpy as np
import pandas as pd

from .._core import vocab
from .._core.storage import MemmapArray
from .._core.textform import to_text
from .._core.utils import get_logger, token_hashes
from .._core.vocab import Vocabulary
from . import inputs
from .procpool import atomic_output, run_tasks, workers_for

log = get_logger(__name__)

WORKER_GB = 1.0


def _count_chunk(task, tz, jobs, out_dir):
    chunk, side, k = task
    cols = list(dict.fromkeys(col for _, col, _ in jobs[side]))
    table = inputs.read_chunk(chunk, cols)
    empty = np.empty(0, np.int64)
    arrays = {key: empty for i, _, _ in jobs[side] for key in (f'h{i}', f'n{i}')}
    for part in inputs.slices(table):
        frame = pd.DataFrame({c: to_text(part.column(c), tz, c).to_pandas() for c in cols})
        for i, col, tokenizer in jobs[side]:
            rows = token_hashes(frame, col, tokenizer)
            h, n = vocab.chunk_counts(None if r is None else r[0] for r in rows)
            arrays[f'h{i}'], arrays[f'n{i}'] = vocab.merge_counts(
                arrays[f'h{i}'], arrays[f'n{i}'], h, n)
    path = os.path.join(out_dir, f'{side}-{k}.npz')
    with atomic_output(path) as tmp, open(tmp, 'wb') as fh:
        np.savez(fh, **arrays)
    return path


def build_vocabularies(A, B, requests, tz, workers=None):
    """Return one vocabulary per request, in order, built from tables A and B."""
    requests = list(requests)
    if not requests:
        return []
    with inputs.scratch() as tmp:
        a = inputs.stage(A, tmp, 'a')
        b = None if B is None else (a if B is A else inputs.stage(B, tmp, 'b'))
        corpus_size = a.count_rows() + (b.count_rows() if b is not None else 0)
        jobs = {'a': [(i, r.a_attr, r.tokenizer) for i, r in enumerate(requests)],
                'b': [(i, r.b_attr, r.tokenizer) for i, r in enumerate(requests)]}
        tasks = [(c, 'a', k) for k, c in enumerate(inputs.chunks(a))]
        if b is not None:
            tasks += [(c, 'b', k) for k, c in enumerate(inputs.chunks(b))]
        out_dir = os.path.join(tmp, 'counts')
        os.makedirs(out_dir)
        fn = partial(_count_chunk, tz=tz, jobs=jobs, out_dir=out_dir)
        paths = run_tasks(fn, tasks, workers or workers_for(WORKER_GB))

        vocabularies = []
        for i, r in enumerate(requests):
            h = n = np.empty(0, np.int64)
            for path in paths:
                with np.load(path) as counts:
                    h, n = vocab.merge_counts(h, n, counts[f'h{i}'], counts[f'n{i}'])
            h, n = vocab.select(h, n, r.min_doc_freq, r.max_vocab_size)
            log.info(f'vocabulary for {r.describe()}: {len(h)} tokens')
            if not len(h):
                raise vocab.empty_error(r)
            vocabularies.append(Vocabulary(MemmapArray(h), MemmapArray(n), corpus_size))
    return vocabularies
