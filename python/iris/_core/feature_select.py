"""The choice of features `create_features` makes, from counts over the rows of tables A
and B."""
from .feature.catalog import SIM_FUNCTIONS, TOKENIZERS
from .feature.feature import (EditDistanceFeature, ExactMatchFeature, RelDiffFeature,
                              SmithWatermanFeature)
from .feature.token_feature import MongeElkanFeature
from .feature.vector_feature import TFIDFFeature
from .numeric import is_number
from .tokenizer import AlphaNumericTokenizer
from .utils import get_logger

log = get_logger(__name__)


def check_numeric_threshold(numeric_threshold):
    """Check that `numeric_threshold` is None or a rate between 0 and 1."""
    if numeric_threshold is not None:
        if isinstance(numeric_threshold, (bool, str)) or not isinstance(numeric_threshold, (int, float)):
            raise TypeError(f'numeric_threshold is a rate in [0, 1] or None, got '
                            f'{numeric_threshold!r}')
        if not 0.0 <= numeric_threshold <= 1.0:
            raise ValueError(f'numeric_threshold is a rate in [0, 1] or None, got '
                             f'{numeric_threshold}')


def kept_columns(columns, rows, nulls, null_threshold):
    """Return the columns whose share of null rows is below `null_threshold`, in
    order."""
    if rows == 0:
        return []
    return [c for c in columns if nulls[c] / rows < null_threshold]


def token_columns(columns, tokenizers):
    """Return each token column name with the tokenizer and the column it comes from."""
    out = {}
    for t in tokenizers:
        for c in columns:
            out[t.out_col_name(c)] = (t, c)
    return out


def numeric_flags(columns, typed_numeric, numeric_threshold):
    """Return the name of each counted parse-rate flag with its text column."""
    if numeric_threshold is None:
        return {}
    return {f'_numeric({c})': c for c in columns if c not in typed_numeric}


def chunk_counts(frame, token_cols, flags):
    """Return the token and parse counts over one chunk of text rows."""
    counts = {}
    for name, (tokenizer, col) in token_cols.items():
        n_rows = n_tokens = 0
        for v in frame[col]:
            toks = tokenizer.tokenize(v)
            if toks is not None:
                n_rows += 1
                n_tokens += len(toks)
        counts[name] = (n_rows, n_tokens)
    for flag, col in flags.items():
        n_values = n_numbers = 0
        for v in frame[col]:
            if v is not None:
                n_values += 1
                n_numbers += bool(is_number(v))
        counts[flag] = (n_values, n_numbers)
    return counts


def add_counts(total, counts):
    """Add one chunk's counts to `total` in place and return `total`."""
    for k, (x, y) in counts.items():
        tx, ty = total.get(k, (0, 0))
        total[k] = (tx + x, ty + y)
    return total


def _mean(pair):
    n, s = pair
    return s / n if n else float('nan')


def numeric_columns(kept, typed_numeric, counts, flags, threshold):
    """Return the kept columns that are numeric, in column order."""
    rates = {}
    for flag, src in flags.items():
        rate = _mean(counts.get(flag, (0, 0)))
        rates[src] = 0.0 if rate != rate else rate
    typed = set(typed_numeric)
    numeric = [c for c in kept
               if c in typed or (threshold is not None and c in rates and rates[c] >= threshold)]
    if rates and threshold is not None:
        found = [f'{c} ({rates[c]:.1%})' for c in kept if c in rates and c in numeric]
        near = [f'{c} ({rates[c]:.1%})' for c in kept
                if c in rates and c not in numeric and rates[c] >= 0.5]
        if found:
            log.info(f'numeric by content (at least {threshold:.0%} of values parse as '
                     f'numbers): ' + ', '.join(found))
        if near:
            log.info('mostly numbers but under the threshold, so treated as text: '
                     + ', '.join(near) + '. Lower numeric_threshold to include them.')
    return numeric


def choose(kept, numeric, token_cols, counts, sim_functions=None,
           min_doc_freq=1, max_vocab_size=None):
    """Return the features for the kept columns, chosen from the counts."""
    if sim_functions is None:
        sim_functions = SIM_FUNCTIONS
    features = [ExactMatchFeature(c, c) for c in kept]
    features += [RelDiffFeature(c, c) for c in numeric]
    for token_col_name, (tokenizer, column_name) in token_cols.items():
        avg_count = _mean(counts.get(token_col_name, (0, 0)))
        if avg_count >= 3:
            for f in sim_functions:
                if issubclass(f, TFIDFFeature):
                    features.append(f(column_name, column_name, tokenizer=tokenizer,
                                      min_doc_freq=min_doc_freq,
                                      max_vocab_size=max_vocab_size))
                else:
                    features.append(f(column_name, column_name, tokenizer=tokenizer))
        if str(tokenizer) == AlphaNumericTokenizer.NAME:
            if avg_count <= 10:
                features.append(MongeElkanFeature(column_name, column_name, tokenizer=tokenizer))
                features.append(EditDistanceFeature(column_name, column_name))
                features.append(SmithWatermanFeature(column_name, column_name))
    return features


def default_tokenizers(tokenizers):
    """Return `tokenizers`, or the base tokenizers when it is None."""
    return TOKENIZERS if tokenizers is None else tokenizers
