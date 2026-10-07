"""Conversions between the flat and the grouped forms of a candidate set."""
from __future__ import annotations

import pandas as pd


def candidates_from_flat(
    pairs: pd.DataFrame,
    a_col: str = "id1",
    b_col: str = "id2",
    score_col: str = "score",
    sort_by_score: bool = True,
) -> pd.DataFrame:
    """Return a flat pair table grouped into one row per record of table B, with the
    lists `id1_list` and `scores`."""
    required = {a_col, b_col, score_col}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(
            f"candidates_from_flat: input is missing column(s) {sorted(missing)}; "
            f"got {list(pairs.columns)}")

    if len(pairs) == 0:
        return pd.DataFrame({"id2": [], "id1_list": [], "scores": []})

    df = pairs[[b_col, a_col, score_col]]
    if sort_by_score:
        df = df.sort_values([b_col, score_col], ascending=[True, False], kind="stable")

    grouped = df.groupby(b_col, sort=False)
    id1_list = grouped[a_col].apply(lambda s: [int(x) for x in s])
    scores = grouped[score_col].apply(lambda s: [float(x) for x in s])

    out = pd.DataFrame({
        "id2": [int(x) for x in id1_list.index],
        "id1_list": id1_list.to_numpy(),
        "scores": scores.to_numpy(),
    })
    return out.reset_index(drop=True)


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
        return pd.DataFrame({"id2": [], "id1_list": [], "scores": []})

    id2, id1_list, scores = [], [], []
    for b_id, (a_list, s_list) in groups.items():
        if sort_by_score and len(s_list) > 1:
            order = sorted(range(len(s_list)), key=lambda i: s_list[i], reverse=True)
            a_list = [a_list[i] for i in order]
            s_list = [s_list[i] for i in order]
        id2.append(int(b_id))
        id1_list.append(a_list)
        scores.append(s_list)

    return pd.DataFrame({"id2": id2, "id1_list": id1_list, "scores": scores})


def write_flat_candidates(result, path, chunk_rows: int = 8_000_000) -> int:
    """Write the pairs of a blocker result to a Parquet file as flat `(id1, id2, score)`
    rows and return the number written."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        pa.field('id1', pa.int64()),
        pa.field('id2', pa.int64()),
        pa.field('score', pa.float32()),
    ])
    written = 0
    writer = None
    try:
        for a_ids, b_ids, scores in result.iter_pairs(chunk_rows=chunk_rows):
            batch = pa.RecordBatch.from_arrays(
                [pa.array(a_ids, type=pa.int64()),
                 pa.array(b_ids, type=pa.int64()),
                 pa.array(scores, type=pa.float32())],
                schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(str(path), schema, compression='zstd')
            writer.write_batch(batch)
            written += batch.num_rows
    finally:
        if writer is not None:
            writer.close()
    return written
