"""Conversions between the flat and the grouped forms of a candidate set."""
from __future__ import annotations

import os

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

    df = pairs[[b_col, a_col] + ([score] if score else [])]
    if sort_by_score and score:
        df = df.sort_values([b_col, score], ascending=[True, False], kind="stable")

    grouped = df.groupby(b_col, sort=False)
    id1_list = grouped[a_col].apply(lambda s: [int(x) for x in s])
    out = {"id2": [int(x) for x in id1_list.index], "id1_list": id1_list.to_numpy()}
    if score:
        out["blocking_scores"] = grouped[score].apply(lambda s: [float(x) for x in s]).to_numpy()
    return pd.DataFrame(out).reset_index(drop=True)


def candidates_from_result(result, sort_by_score: bool = True) -> pd.DataFrame:
    """Return the pairs of a blocker result grouped into one row per record of table B,
    without building the flat table."""
    groups: dict[int, tuple[list, list]] = {}
    for a_ids, b_ids, scores in result.iter_pairs():
        a_list = a_ids.tolist()
        b_list = b_ids.tolist()
        s_list = scores.tolist()
        for i in range(len(b_list)):
            bucket = groups.get(b_list[i])
            if bucket is None:
                bucket = ([], [])
                groups[b_list[i]] = bucket
            bucket[0].append(int(a_list[i]))
            bucket[1].append(float(s_list[i]))

    if not groups:
        return pd.DataFrame({"id2": [], "id1_list": [], "blocking_scores": []})

    id2, id1_list, scores = [], [], []
    for b_id, (a_list, s_list) in groups.items():
        if sort_by_score and len(s_list) > 1:
            order = sorted(range(len(s_list)), key=lambda i: s_list[i], reverse=True)
            a_list = [a_list[i] for i in order]
            s_list = [s_list[i] for i in order]
        id2.append(int(b_id))
        id1_list.append(a_list)
        scores.append(s_list)

    return pd.DataFrame({"id2": id2, "id1_list": id1_list, "blocking_scores": scores})


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
