"""Building the TF/IDF and SIF vocabularies on the local engine, with worker
processes."""
import os
from functools import partial

import numpy as np
import pandas as pd

from .._core import vocab
from .._core.storage import MemmapArrayBuilder
from .._core.textform import to_text
from .._core.utils import get_logger, token_hashes
from .._core.vocab import Vocabulary
from . import inputs
from .procpool import atomic_output, run_tasks, workers_for

log = get_logger(__name__)



def _count_chunk(task, tz, jobs, out_dir):
    chunk, side, k = task
    cols = list(dict.fromkeys(col for _, col, _ in jobs[side]))
    empty = np.empty(0, np.int64)
    arrays = {key: empty for i, _, _ in jobs[side] for key in (f'h{i}', f'n{i}')}
    for part in inputs.read_slices(chunk, cols):
        frame = pd.DataFrame({c: to_text(part.column(c), tz, c).to_pandas() for c in cols})
        for i, col, tokenizer in jobs[side]:
            rows = token_hashes(frame, col, tokenizer)
            h, n = vocab.chunk_counts(None if r is None else r[0] for r in rows)
            arrays[f'h{i}'], arrays[f'n{i}'] = vocab.merge_counts(
                arrays[f'h{i}'], arrays[f'n{i}'], h, n)
    ids = sorted(i for i, _, _ in jobs[side])
    path = os.path.join(out_dir, f'{side}-{k}')
    inputs.require_disk(sum(16 * len(arrays[f'h{i}']) + 8 for i in ids) + 8, out_dir,
                        'the token counts', kept=0)
    for ext, parts in (('h', [arrays[f'h{i}'] for i in ids]),
                       ('n', [arrays[f'n{i}'] for i in ids]),
                       ('off', [np.cumsum([0] + [len(arrays[f'h{i}']) for i in ids])])):
        with atomic_output(f'{path}.{ext}.npy') as tmp, open(tmp, 'wb') as fh:
            np.save(fh, np.concatenate(parts).astype(np.int64, copy=False))
    return path


def _bounds(s, shards):
    lo = None if s == 0 else np.int64(-2**63 + (2**64 * s) // shards)
    hi = None if s == shards - 1 else np.int64(-2**63 + (2**64 * (s + 1)) // shards)
    return lo, hi


def _merge_shard(task, paths, out_dir, min_doc_freqs):
    i, s, shards = task
    min_doc_freq = min_doc_freqs[i]
    lo, hi = _bounds(s, shards)
    hs, ns = [], []
    for path in paths:
        off = np.load(f'{path}.off.npy')
        h = np.load(f'{path}.h.npy', mmap_mode='r')[off[i]:off[i + 1]]
        a = 0 if lo is None else int(np.searchsorted(h, lo))
        b = len(h) if hi is None else int(np.searchsorted(h, hi))
        if b > a:
            hs.append(np.array(h[a:b]))
            ns.append(np.array(np.load(f'{path}.n.npy', mmap_mode='r')[off[i] + a:off[i] + b]))
    h = np.concatenate(hs) if hs else np.empty(0, np.int64)
    n = np.concatenate(ns) if ns else np.empty(0, np.int64)
    order = np.argsort(h, kind='stable')
    h, n = h[order], n[order]
    first = np.flatnonzero(np.r_[True, h[1:] != h[:-1]]) if len(h) else np.empty(0, np.int64)
    h, n = h[first], (np.add.reduceat(n, first) if len(first) else n)
    keep = n >= min_doc_freq
    h, n = h[keep], n[keep]
    out = os.path.join(out_dir, f'{i}-{s:05d}')
    inputs.require_disk(h.nbytes + n.nbytes, out_dir, 'the merged token counts', kept=0)
    for ext, arr in (('h', h), ('n', n)):
        with atomic_output(f'{out}.{ext}.npy') as tmp, open(tmp, 'wb') as fh:
            np.save(fh, arr)
    values, counts = np.unique(n, return_counts=True)
    return out, len(h), dict(zip(values.tolist(), counts.tolist()))


def _assemble(shards, r, corpus_size):
    found = sum(n for _, n, _ in shards)
    count, quota = (None, None)
    if r.max_vocab_size is not None and found > r.max_vocab_size:
        count, quota = vocab.cutoff([hist for _, _, hist in shards], r.max_vocab_size)
    size = found if count is None else r.max_vocab_size
    log.info(f'vocabulary for {r.describe()}: {size} tokens')
    if not size:
        raise vocab.empty_error(r)
    hashes, freqs = MemmapArrayBuilder(size, np.int64), MemmapArrayBuilder(size, np.int64)
    for s, (path, _, _) in enumerate(shards):
        h, n = np.load(f'{path}.h.npy'), np.load(f'{path}.n.npy')
        if count is not None:
            ties = np.flatnonzero(n == count)[:quota[s]]
            keep = n > count
            keep[ties] = True
            h, n = h[keep], n[keep]
        hashes.write(h)
        freqs.write(n)
    return Vocabulary(hashes.finish(), freqs.finish(), corpus_size)


def build_vocabularies(A, B, requests, tz, workers=None):
    """Return one vocabulary per request, in order, built from tables A and B."""
    requests = list(requests)
    if not requests:
        return []
    workers = workers or workers_for()
    jobs = {'a': [(i, r.a_attr, r.tokenizer) for i, r in enumerate(requests)],
            'b': [(i, r.b_attr, r.tokenizer) for i, r in enumerate(requests)]}
    cols = {side: list(dict.fromkeys(col for _, col, _ in jobs[side])) for side in jobs}
    with inputs.scratch() as tmp:
        a = inputs.stage(A, tmp, 'a', cols['a'] if B is not A else None, workers)
        b = None if B is None else (a if B is A else
                                    inputs.stage(B, tmp, 'b', cols['b'], workers))
        corpus_size = a.count_rows() + (b.count_rows() if b is not None else 0)
        tasks = [(c, 'a', k) for k, c in enumerate(inputs.chunks(a, workers=workers))]
        if b is not None:
            tasks += [(c, 'b', k) for k, c in enumerate(inputs.chunks(b, workers=workers))]
        out_dir = os.path.join(tmp, 'counts')
        os.makedirs(out_dir)
        fn = partial(_count_chunk, tz=tz, jobs=jobs, out_dir=out_dir)
        piece = inputs.chunk_bytes()
        paths = run_tasks(fn, tasks, workers, task_bytes=piece)

        merged = os.path.join(tmp, 'merged')
        os.makedirs(merged)
        entries = [sum(int(np.diff(np.load(f'{p}.off.npy')[i:i + 2])[0]) for p in paths)
                   for i in range(len(requests))]
        shards = [max(1, -(-e * 16 // piece), min(2 * workers, e // 65_536)) for e in entries]
        merge = [(i, s, shards[i]) for i in range(len(requests)) for s in range(shards[i])]
        fn = partial(_merge_shard, paths=paths, out_dir=merged,
                     min_doc_freqs=[r.min_doc_freq for r in requests])
        done = run_tasks(fn, merge, workers,
                         task_bytes=max(1, max(e * 16 // n for e, n in zip(entries, shards))))
        vocabularies, at = [], 0
        for i, r in enumerate(requests):
            vocabularies.append(_assemble(done[at:at + shards[i]], r, corpus_size))
            at += shards[i]
    return vocabularies
