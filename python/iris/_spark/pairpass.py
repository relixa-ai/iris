"""The pair pass of the Spark engine: a two-hop join that brings each pair to its two
records."""
import math

import numpy as np
import pyarrow as pa
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark import StorageLevel

from .._core import pool, records, recpack
from .utils import BATCH_BYTES, MAX_PARTITIONS, PARTITION_BYTES, map_in_arrow

OUT_SCHEMA = T.StructType([T.StructField('id1', T.LongType()),
                           T.StructField('id2', T.LongType()),
                           T.StructField('fv', T.ArrayType(T.FloatType())),
                           T.StructField('score', T.DoubleType())])

SCRATCH_MARGIN = 1.5

STREAM_KEY_BYTES = 17

OUT_KEY_BYTES = 24

INT32_LIMIT = 2**31 - 1


def pieces(ends, limit):
    """Return the start and end of each run of packed records, in order, whose bytes come
    to at most `limit`, or a single record."""
    bounds = np.r_[0, np.asarray(ends, dtype=np.int64)]
    out, lo, n = [], 0, len(bounds) - 1
    while lo < n:
        hi = max(lo + 1, int(np.searchsorted(bounds, bounds[lo] + limit, side='right')) - 1)
        if bounds[hi] - bounds[lo] > INT32_LIMIT:
            raise ValueError(
                f'a packed record of {bounds[hi] - bounds[lo]:,} bytes passes the 2 GiB '
                f'that one Spark value holds; leave its columns out of the features or '
                f'shorten their values')
        out.append((lo, hi))
        lo = hi
    return out


def build_records(text, layout, features, for_a, for_b):
    """Return every row of a text table preprocessed and packed, as `(_id, rec)`
    rows."""
    limit = BATCH_BYTES

    def pack(batches):
        for batch in batches:
            for ids, data, ends in records.pack_rows(batch.to_pandas(), layout, features,
                                                     for_a, for_b):
                bounds = np.r_[0, ends]
                for lo, hi in pieces(ends, limit):
                    offsets = (bounds[lo:hi + 1] - bounds[lo]).astype(np.int32)
                    recs = pa.BinaryArray.from_buffers(
                        pa.binary(), hi - lo,
                        [None, pa.py_buffer(offsets),
                         pa.py_buffer(data[bounds[lo]:bounds[hi]])])
                    yield pa.RecordBatch.from_arrays([pa.array(ids[lo:hi]), recs],
                                                     names=['_id', 'rec'])

    return map_in_arrow(text, pack, '_id long, rec binary').persist(StorageLevel.DISK_ONLY)


def _slices(batches, limit):
    import pyarrow.compute as pc

    for batch in batches:
        sizes = pc.binary_length(batch.column('rec')).fill_null(0)
        ends = np.cumsum(sizes.to_numpy(zero_copy_only=False), dtype=np.int64)
        if not len(ends) or ends[-1] <= limit:
            yield batch
            continue
        for lo, hi in pieces(ends, limit):
            yield batch.slice(lo, hi - lo)


def _binary(arr):
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    n = len(arr)
    _, offs_buf, data_buf = arr.buffers()
    width = np.int64 if pa.types.is_large_binary(arr.type) else np.int32
    offs = np.frombuffer(offs_buf, dtype=width, count=arr.offset + n + 1)[arr.offset:]
    data = (np.frombuffer(data_buf, dtype=np.uint8) if data_buf is not None
            else np.empty(0, dtype=np.uint8))
    valid = (~arr.is_null().to_numpy(zero_copy_only=False) if arr.null_count
             else np.ones(n, dtype=bool))
    return data, offs[:-1].astype(np.int64), offs[1:].astype(np.int64), valid


def _merge_and_score(batches, layout_a, layout_b, features, fill_na, positive,
                     limit=BATCH_BYTES):
    from .._core.storage import unknown_id_error

    cur_id2, cur_blob = None, None
    last_pair = None
    for batch in _slices(batches, limit):
        id2 = batch.column('id2').to_numpy(zero_copy_only=False).astype(np.int64)
        tag = batch.column('tag').to_numpy(zero_copy_only=False)
        id1 = batch.column('id1').fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
        data, starts, ends, valid = _binary(batch.column('rec'))
        is_b = tag == 0
        b_rows = np.flatnonzero(is_b)
        p = np.flatnonzero(~is_b)

        if len(p):
            if not valid[p].all():
                raise unknown_id_error(np.unique(id1[p][~valid[p]]), 'a')
            keep = np.ones(len(p), dtype=bool)
            keep[1:] = (id2[p][1:] != id2[p][:-1]) | (id1[p][1:] != id1[p][:-1])
            if last_pair is not None:
                keep[0] = (int(id1[p[0]]), int(id2[p[0]])) != last_pair
            p = p[keep]
        if len(p):
            b_id2 = id2[b_rows]
            pos = np.minimum(np.searchsorted(b_id2, id2[p]), max(len(b_rows) - 1, 0))
            here = (np.zeros(len(p), dtype=bool) if not len(b_rows)
                    else b_id2[pos] == id2[p])
            carried = ~here & (id2[p] == cur_id2) if cur_id2 is not None \
                else np.zeros(len(p), dtype=bool)
            missing = ~(here | carried)
            if missing.any():
                raise unknown_id_error(np.unique(id2[p][missing]), 'b')

            used_carry = bool(carried.any())
            if used_carry:
                b_data = np.concatenate([np.frombuffer(cur_blob, dtype=np.uint8), data])
                shift = len(cur_blob)
                b_starts = np.r_[0, starts[b_rows] + shift]
                b_ends = np.r_[len(cur_blob), ends[b_rows] + shift]
                b_local = np.where(here, pos + 1, 0)
            else:
                b_data, b_starts, b_ends = data, starts[b_rows], ends[b_rows]
                b_local = pos
            b_frame = recpack.decode_frame(layout_b, b_data, b_starts, b_ends)

            uniq, first, a_local = np.unique(id1[p], return_index=True, return_inverse=True)
            rows = p[first]
            a_frame = recpack.decode_frame(layout_a, data, starts[rows], ends[rows])

            f_mat = records.feature_matrix(features, a_frame, a_local, b_frame, b_local,
                                           fill_na)
            fv = records.vectors(f_mat)
            last_pair = (int(id1[p[-1]]), int(id2[p[-1]]))
            yield pa.RecordBatch.from_arrays(
                [pa.array(id1[p]), pa.array(id2[p]), fv,
                 pa.array(pool.score(f_mat, positive))],
                names=['id1', 'id2', 'fv', 'score'])

        if len(b_rows) and id2[b_rows[-1]] == id2[-1]:
            last = b_rows[-1]
            cur_id2, cur_blob = int(id2[last]), data[starts[last]:ends[last]].tobytes()


def _local_dirs(spark):
    import os

    raw = (os.environ.get('SPARK_LOCAL_DIRS')
           or spark.sparkContext.getConf().get('spark.local.dir', None)
           or spark.sparkContext._jvm.java.lang.System.getProperty('java.io.tmpdir'))
    return [d.strip() for d in str(raw).split(',') if d.strip()]


def check_scratch(spark, stream_bytes):
    """Raise an error when a local session's scratch directories cannot hold the join's
    shuffle."""
    import os
    import shutil

    from .._core.utils import get_logger

    log = get_logger(__name__)
    need = stream_bytes * SCRATCH_MARGIN
    if not spark.sparkContext.master.startswith('local'):
        log.info(f'pair pass: about {stream_bytes / 2**30:.1f} GiB of records cross the '
                 f'shuffle; executors need about {need / 2**30:.1f} GiB of scratch disk '
                 f'between them')
        return
    free, volumes = 0, set()
    for d in _local_dirs(spark):
        try:
            dev = os.stat(d).st_dev
            if dev not in volumes:
                volumes.add(dev)
                free += shutil.disk_usage(d).free
        except OSError:
            return
    where = ', '.join(_local_dirs(spark))
    if free < stream_bytes:
        raise RuntimeError(
            f'not enough scratch disk for the pair pass: about {stream_bytes / 2**30:.1f} '
            f'GiB of records cross the shuffle (up to {need / 2**30:.1f} GiB on disk), and '
            f'{where} has {free / 2**30:.1f} GiB free. Point SPARK_LOCAL_DIRS (or '
            f'spark.local.dir) at a larger volume, or free space.')
    if free < need:
        log.warning(f'pair pass: {where} has {free / 2**30:.1f} GiB free; the shuffle may '
                    f'need up to {need / 2**30:.1f} GiB')


def mean_records(recs_a, recs_b):
    """Return the mean bytes of a packed record of each side."""
    mean_a = float(recs_a.agg(F.avg(F.length('rec'))).head()[0] or 0.0)
    if recs_b is recs_a:
        return mean_a, mean_a
    return mean_a, float(recs_b.agg(F.avg(F.length('rec'))).head()[0] or 0.0)


def stream_row_bytes(mean_a, mean_b, n_features):
    """Return the bytes of one row of the join's stream: a record and its keys, and the
    feature vector the pass gives it."""
    return max(mean_a, mean_b) + STREAM_KEY_BYTES + 4 * n_features + OUT_KEY_BYTES


def pair_pass(pairs, recs_a, recs_b, layout_a, layout_b, features, fill_na=0.0,
              mean_a=None, mean_b=None):
    """Return the feature vector and score of every distinct pair, as a Spark
    DataFrame."""
    spark = pairs.sparkSession
    positive = pool.positive_features([str(f) for f in features])
    a = recs_a.select(F.col('_id').alias('id1'), 'rec')
    hop1 = pairs.select('id1', 'id2').join(a, 'id1', 'left')
    stream = hop1.select('id2', F.lit(1).cast('byte').alias('tag'), 'id1', 'rec').unionByName(
        recs_b.select(F.col('_id').alias('id2'), F.lit(0).cast('byte').alias('tag'),
                      F.lit(None).cast('long').alias('id1'), 'rec'))
    if mean_a is None or mean_b is None:
        mean_a, mean_b = mean_records(recs_a, recs_b)
    n_pairs = pairs.count()
    n_b = recs_b.count()
    stream_bytes = n_pairs * mean_a + n_b * mean_b
    check_scratch(spark, stream_bytes)
    parallelism = spark.sparkContext.defaultParallelism
    parts = int(min(MAX_PARTITIONS,
                    max(parallelism, math.ceil(stream_bytes / PARTITION_BYTES))))

    limit = BATCH_BYTES

    def run(batches):
        return _merge_and_score(batches, layout_a, layout_b, features, fill_na, positive,
                                limit)

    return (stream.repartition(parts, 'id2')
                  .sortWithinPartitions('id2', 'tag', 'id1')
                  .mapInArrow(run, OUT_SCHEMA))
