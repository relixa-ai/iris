"""Conversions between the flat and the grouped forms of a candidate set."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


def _score_column(columns, score_col):
    if score_col in columns:
        return score_col
    if score_col == "blocking_score" and "score" in columns:
        return "score"
    return None


def candidates_from_flat(
    pairs: pd.DataFrame,
    a_col: str = "id1",
    b_col: str = "id2",
    score_col: str = "blocking_score",
    sort_by_score: bool = True,
) -> pd.DataFrame:
    """Return a flat pair table grouped into one row per record of table B, with the list
    `id1_list` and, when the pairs have a blocking score, the list `blocking_scores`."""
    missing = {a_col, b_col} - set(pairs.columns)
    if missing:
        raise ValueError(
            f"candidates_from_flat: input is missing column(s) {sorted(missing)}; "
            f"got {list(pairs.columns)}")
    score = _score_column(pairs.columns, score_col)
    lists = ["id1_list"] + (["blocking_scores"] if score else [])

    if len(pairs) == 0:
        return pd.DataFrame({"id2": [], **{c: [] for c in lists}})

    a = pairs[a_col].to_numpy()
    b = pairs[b_col].to_numpy()
    scores = pairs[score].to_numpy(np.float64) if score else None
    if pd.isna(a).any() or pd.isna(b).any():
        raise ValueError(f"candidates_from_flat: '{a_col}' and '{b_col}' must not hold "
                         f"null ids")
    order = (np.lexsort((-scores, b)) if sort_by_score and score
             else np.arange(len(b)))
    return _grouped(a[order], b[order], None if scores is None else scores[order], lists)


def _grouped(a, b, scores, lists):
    codes, keys = pd.factorize(b, sort=False)
    order = np.argsort(codes, kind='stable')
    cuts = np.cumsum(np.bincount(codes, minlength=len(keys)))[:-1]
    out = {'id2': [int(x) for x in keys],
           'id1_list': [part.tolist() for part in
                        np.split(np.asarray(a[order], dtype=np.int64), cuts)]}
    if scores is not None:
        out[lists[1]] = [part.tolist() for part in
                         np.split(np.asarray(scores[order], dtype=np.float64), cuts)]
    return pd.DataFrame(out).reset_index(drop=True)


def candidates_from_result(result, sort_by_score: bool = True) -> pd.DataFrame:
    """Return the pairs of a blocker result grouped into one row per record of table B,
    without building the flat table."""
    parts = [(np.asarray(a, np.int64), np.asarray(b, np.int64), np.asarray(s, np.float64))
             for a, b, s in result.iter_pairs()]
    if not parts or not sum(len(b) for _, b, _ in parts):
        return pd.DataFrame({"id2": [], "id1_list": [], "blocking_scores": []})
    a = np.concatenate([p[0] for p in parts])
    b = np.concatenate([p[1] for p in parts])
    scores = np.concatenate([p[2] for p in parts])
    del parts
    if sort_by_score:
        codes, _ = pd.factorize(b, sort=False)
        order = np.lexsort((-scores, codes))
        a, b, scores = a[order], b[order], scores[order]
    return _grouped(a, b, scores, ['id1_list', 'blocking_scores'])


def write_flat_candidates(result, path, chunk_rows: int = 8_000_000) -> int:
    """Write the pairs of a blocker result to a Parquet file as flat `(id1, id2,
    blocking_score)` rows and return the number written."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .io import parquet_name

    name = parquet_name(path, "write_flat_candidates")
    schema = pa.schema([
        pa.field('id1', pa.int64()),
        pa.field('id2', pa.int64()),
        pa.field('blocking_score', pa.float32()),
    ])
    partial = f"{name}.partial-{os.getpid()}"
    written = 0
    writer = pq.ParquetWriter(partial, schema, compression='zstd')
    try:
        for a_ids, b_ids, scores in result.iter_pairs(chunk_rows=chunk_rows):
            batch = pa.RecordBatch.from_arrays(
                [pa.array(a_ids, type=pa.int64()),
                 pa.array(b_ids, type=pa.int64()),
                 pa.array(scores, type=pa.float32())],
                schema=schema)
            writer.write_batch(batch)
            written += batch.num_rows
        writer.close()
        os.replace(partial, name)
    except BaseException:
        writer.close()
        if os.path.exists(partial):
            os.remove(partial)
        raise
    return written
