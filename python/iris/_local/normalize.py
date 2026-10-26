"""The normalization parsers on the local engine, applied by worker processes a chunk at
a time."""
import os
import tempfile
from functools import partial

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .._core.storage import scratch_prefix, spill_dir
from .._core.textform import to_text
from . import inputs
from .procpool import atomic_output, run_tasks, workers_for



def _text(column, tz, name):
    if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
        return column.to_pandas()
    return to_text(column, tz, name).to_pandas()


def _normalized(table, dur_cols, year_cols, concat_cols, int_cols, tz):
    from ..normalize import duration_seconds, integer_number, year_number

    names = table.column_names
    sources = [c for c in dict.fromkeys([*dur_cols, *year_cols, *int_cols, *concat_cols])
               if c in names]
    text = {c: _text(table.column(c), tz, c) for c in sources}
    as_double = lambda s: pa.array(s.to_numpy(dtype='float64', na_value=float('nan')),
                                   type=pa.float64(), from_pandas=True)
    for c in dur_cols:
        if c in names:
            table = table.append_column(f'{c}_seconds', as_double(duration_seconds(text[c])))
    for c in year_cols:
        if c in names:
            table = table.append_column(f'{c}_numeric', as_double(year_number(text[c])))
    for c in int_cols:
        if c in names:
            table = table.append_column(f'{c}_numeric', as_double(integer_number(text[c])))
    present = [c for c in concat_cols if c in names]
    if present:
        joined = [' '.join('' if v is None else v for v in row)
                  for row in zip(*(text[c].tolist() for c in present))]
        table = table.append_column('combined_text', pa.array(joined, pa.string()))
    return table


def _normalize_chunk(task, dur_cols, year_cols, concat_cols, int_cols, tz, out_dir):
    k, chunk = task
    rows, writer = 0, None
    with atomic_output(os.path.join(out_dir, f'part-{k:05d}.parquet')) as tmp:
        try:
            parts = inputs.read_slices(chunk._replace(schema=None), None)
            for part in parts:
                table = _normalized(part, dur_cols, year_cols, concat_cols, int_cols, tz)
                if writer is None:
                    writer = pq.ParquetWriter(tmp, table.schema)
                writer.write_table(table)
                rows += table.num_rows
            if writer is None:
                with chunk.filesystem.open_input_file(chunk.path) as fh:
                    empty = pq.ParquetFile(fh).schema_arrow.empty_table()
                pq.write_table(_normalized(empty, dur_cols, year_cols, concat_cols,
                                           int_cols, tz), tmp)
        finally:
            if writer is not None:
                writer.close()
    return rows


def add_normalized_columns(table, dur_cols=(), year_cols=(), concat_cols=(), int_cols=(),
                           timezone=None, directory=None, workers=None):
    """Return the table with parsed numeric columns and a concatenated text column
    added, as an Arrow dataset."""
    from .._core.textform import resolve_timezone
    from .featurize import _remove_at_exit

    tz = resolve_timezone(timezone)
    names = inputs.columns(table)
    for group, cols in (('normalize-durations', dur_cols), ('normalize-years', year_cols),
                        ('normalize-integers', int_cols), ('concat-cols', concat_cols)):
        missing = [c for c in cols if c not in names]
        if missing:
            print(f'[normalize] WARNING: --{group} column(s) {missing} not in table '
                  f'(columns: {[c for c in names if c != "_id"]}); skipping them',
                  flush=True)
    if directory is None:
        directory = tempfile.mkdtemp(prefix=scratch_prefix('normalized'), dir=spill_dir())
        _remove_at_exit(directory)
    os.makedirs(directory, exist_ok=True)
    with inputs.scratch() as tmp:
        workers = workers or workers_for()
        src = inputs.stage(table, tmp, 'in', workers=workers)
        fn = partial(_normalize_chunk, dur_cols=tuple(dur_cols), year_cols=tuple(year_cols),
                     concat_cols=tuple(concat_cols), int_cols=tuple(int_cols), tz=tz,
                     out_dir=directory)
        run_tasks(fn, list(enumerate(inputs.chunks(src, workers=workers))), workers,
                  task_bytes=inputs.chunk_bytes())
    return inputs.dataset(directory)


def parse_rates(table, dur_cols=(), year_cols=(), sample_rows=100_000):
    """Return the fraction of values each parser parsed, over the first `sample_rows`
    rows."""
    names = inputs.columns(table)
    cols = ([f'{c}_seconds' for c in dur_cols if f'{c}_seconds' in names]
            + [f'{c}_numeric' for c in year_cols if f'{c}_numeric' in names])
    if not cols:
        return {}
    parts, held = [], 0
    for b in inputs.batches(table, cols):
        parts.append(b)
        held += len(b)
        if held >= sample_rows:
            break
    head = pd.concat(parts).head(sample_rows)
    out = {}
    for c in dur_cols:
        if f'{c}_seconds' in names:
            out[c] = float(head[f'{c}_seconds'].notna().mean())
    for c in year_cols:
        if f'{c}_numeric' in names:
            out[f'{c}_yr'] = float(head[f'{c}_numeric'].notna().mean())
    return out
