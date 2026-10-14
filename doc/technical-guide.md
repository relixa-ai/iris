## Iris technical guide

This guide describes every public function of Iris, in the order a run calls them, and the file formats, the data shapes, the labelers, the settings and the options for large runs. The [tutorial](tutorial.md) shows the functions in use, and [How Iris works](how-it-works.md) explains the concepts behind them.

Contents:

- [Conventions and data shapes](#conventions-and-data-shapes)
- [Reading and writing files](#reading-and-writing-files)
- [Checking the inputs](#checking-the-inputs)
- [Features and feature vectors](#features-and-feature-vectors)
- [The pool and active learning](#the-pool-and-active-learning)
- [Labelers](#labelers)
- [Matchers](#matchers)
- [Spark sessions and storage](#spark-sessions-and-storage)
- [Options for large runs](#options-for-large-runs)
- [Settings and the command line](#settings-and-the-command-line)
- [Extending Iris](#extending-iris)

## Conventions and data shapes

Every function is imported from the package `iris`, as in `iris.featurize`. Each function runs on the *local engine*, which uses Python processes on one machine, when it is given pandas DataFrames or Arrow tables and datasets, and on the *Spark engine*, as Spark jobs, when it is given Spark DataFrames. [How Iris works](how-it-works.md#the-two-engines) describes both. The inputs of one call are all of one kind, and a function returns the kind it was given, except where an entry says otherwise. The entries below say "Runs on either engine." when this is so.

The data of a run has the following shapes.

| Data | Columns | Notes |
| --- | --- | --- |
| A table | `_id`, and the columns to compare | `_id` is an integer column of unique, non-null, non-negative values. |
| A flat candidate set | `id1`, `id2` | `id1` is an `_id` of table A, and `id2` an `_id` of table B. A blocker's score, in `blocking_score` or in `score` as earlier blockers name it, and any other column are ignored. |
| A grouped candidate set | `id2`, `id1_list` | One row per record of B, with the list of the records of A paired with it. A list column of the same length, such as `blocking_scores`, is ignored. |
| Labeled pairs | `id1`, `id2`, `label` | `label` is 1.0 for a match, 0.0 for a non-match, and 2.0 for a pair labeled unsure. The grouped form has `label` as a list aligned with `id1_list`. |
| A gold set | `id1`, `id2` | The pairs known to be matches. |
| Feature vectors | `id1`, `id2`, `feature_vectors`, `score` | `feature_vectors` holds one float32 array per pair, with one value per feature, in the order of the list of features. |
| Predictions | the columns of the feature vectors, `prediction`, `confidence` | `prediction` is 1.0 or 0.0, and `confidence` is the probability of the predicted class. |
| Predicted matches | `id1`, `id2`, `probability` | The file that `save_predictions` writes, with only the pairs predicted to match and the probability of a match of each. |

In a deduplication, table A and table B are the same table, passed as the same object. A file whose columns have other names is read with `read_table` and `read_pairs`, which give it these names.

### Tables keyed by strings

Iris needs integer ids. A table keyed by strings, such as product codes, is given an integer `_id` once, before blocking, and the gold set is mapped to the same ids. The blocker, Iris and the evaluation then use the same ids. In a deduplication, `table_b` is `table_a`.

```python
import pandas as pd

table_a = pd.read_csv('table_a.csv')
table_b = pd.read_csv('table_b.csv')
gold = pd.read_csv('gold.csv')
table_a.insert(0, '_id', range(len(table_a)))
table_b.insert(0, '_id', range(len(table_b)))
id_a = dict(zip(table_a['code'], table_a['_id']))
id_b = dict(zip(table_b['code'], table_b['_id']))
gold = pd.DataFrame({'id1': gold['code_a'].map(id_a), 'id2': gold['code_b'].map(id_b)})
table_a.to_parquet('table_a.parquet', index=False)
table_b.to_parquet('table_b.parquet', index=False)
gold.to_parquet('gold.parquet', index=False)
```

## Reading and writing files

### File formats

`read_table` and `read_pairs` read the tables and the sets of pairs from Parquet files, directories of Parquet files, CSV files, TSV files and JSON Lines files, with any names for their id columns, and give them the column names of Iris. A file is recognised as Parquet by its contents. A file named `.parquet` that holds CSV text, or a CSV or TSV file that holds Parquet data, is refused with a message, and a file without an extension is read when it holds Parquet data. The feature vectors and the predicted matches are written as Parquet, and a function that writes Parquet refuses a path whose name ends in another extension. A directory written by Spark or by a function of Iris holds a file `_SUCCESS` when it was written to completion. The directory of feature vectors written by `featurize` also holds a file `_iris.json`, which lists the features that made them.

A Parquet file is read in *row groups*. The local engine reads a table a row group at a time, and a labeler that looks up a record reads the row group that holds it. Tables written with row groups of about 65,536 rows keep both the memory of each read and the time of each lookup small, as `pyarrow.parquet.write_table(table, path, row_group_size=65536)` does.

### read_table()

```python
def read_table(
    path,                            # a .parquet, .csv, .tsv or .jsonl file, or a directory of Parquet files
    id_col: str = '_id',             # the integer id column of the file
    columns: List[str] = None,       # the attribute columns to read; all by default
    df_type: str = 'pandas',         # 'pandas', 'arrow' or 'sparkdf'
)
```

Reads a table of records, with its id column named `_id`. Runs on either engine.

**Parameters:**

- `path`: the file, read by its extension. With `df_type='sparkdf'` it may be a URI, such as `s3a://bucket/data/table_a.csv`.
- `id_col`: the name of the id column in the file. It is renamed to `_id`.
- `columns`: the attribute columns to read, in addition to the id. `None` reads all of them.
- `df_type`: `'pandas'` reads the table into memory. `'arrow'` returns an Arrow dataset that the local engine reads a row group at a time. `'sparkdf'` reads it with the active Spark session.

This function:

1. Reads a Parquet file with the types it has. Reads a CSV or TSV file, whose first line names the columns, with the id as a 64-bit integer and every other column as text, an empty cell being a missing value. Reads a JSON Lines file, one JSON object per line, with the id as a 64-bit integer and every other key as JSON gives it.
2. Checks that the id column exists, and that the file has no other column named `_id`.
3. Renames the id column to `_id` and keeps the columns asked for.
4. For `'arrow'`, opens a Parquet file that needs no renaming as it is, without reading it. Any other file is written once as Parquet to the scratch directory, a batch at a time, and opened from there.

**Returns:** a `pandas.DataFrame`, a `pyarrow.dataset.Dataset` or a `pyspark.sql.DataFrame` with the column `_id` and the attribute columns.

**Raises:** `ValueError` when the id column or a column of `columns` is missing, when the file has both `_id` and another id column, when the extension is not one of the four or does not match the contents of the file, or when `df_type` is unknown.

**Example:**

```python
table_a = iris.read_table('data/abt_buy/table_a.csv', id_col='product_id', df_type='arrow')
```

### read_pairs()

```python
def read_pairs(
    path,                            # a .parquet, .csv, .tsv or .jsonl file, or a directory of Parquet files
    columns=('id1', 'id2'),          # the file's id columns of table A and of table B
    with_label: bool = False,        # also read the column label
    df_type: str = 'pandas',         # 'pandas', 'arrow' or 'sparkdf'
)
```

Reads a set of pairs, which can be a candidate set, a gold set or labeled pairs, as flat `id1`, `id2` rows. Runs on either engine.

**Parameters:**

- `columns`: the names of the two id columns in the file, the column of table A's ids first. With the default, a file that has no columns `id1` and `id2` is read from the first of the pairs `a_id`/`b_id`, `ltable_id`/`rtable_id` and `left_id`/`right_id` that it has. Other names are given here, as in `columns=('product_a', 'product_b')`.
- `with_label`: also reads the column `label`, as 1.0, 0.0 or 2.0.
- `df_type`: as in `read_table`.

This function:

1. Reads the file as `read_table` does, with the two id columns as 64-bit integers and `label` as a float.
2. Finds the form of the pairs. When one of the two named columns is a list, the file is grouped, and each row is expanded into one pair per listed id, with the list `label` expanded alongside it. With the default names, a file with `id2` and `id1_list`, the grouped form that earlier blockers write, and that blockers write when asked for it, is read the same way.
3. Returns the two id columns as `id1` and `id2`, and `label` with `with_label`. Other columns, such as a blocker's `blocking_score`, are not read.
4. For `'arrow'`, opens as it is a Parquet file whose only columns are `id1`, `id2` and, with `with_label`, `label`. Any other file is written once as Parquet to the scratch directory and opened from there.

**Returns:** a `pandas.DataFrame`, a `pyarrow.dataset.Dataset` or a `pyspark.sql.DataFrame` with the columns `id1`, `id2` and, with `with_label`, `label`.

**Raises:** `ValueError` when the file has none of the pairs of id columns looked for, nor the grouped form, when `with_label` is given and the file has no `label`, when the extension is not one of the four or does not match the contents of the file, or when `df_type` is unknown.

**Example:**

```python
candidates = iris.read_pairs('out/sparkly.csv', df_type='arrow')
pairs = iris.read_pairs('out/blocker.parquet', columns=('product_a', 'product_b'))
labeled = iris.read_pairs('data/dblp_acm/labeled_pairs.parquet', with_label=True)
```

### load_data()

```python
def load_data(
    path,              # a Parquet file or directory, or a URI on Spark
    df_type,           # 'pandas', 'arrow' or 'sparkdf'
)
```

Reads a Parquet file or directory as a pandas DataFrame, an Arrow dataset or a Spark DataFrame, with its columns as they are. `read_table` and `read_pairs` read the inputs of a run, and `load_data` reads anything else written as Parquet, such as the predicted matches.

**Parameters:**

- `path`: a path, or a URI such as `s3a://bucket/path` when `df_type` is `'sparkdf'`.
- `df_type`: `'pandas'` reads the whole file into memory. `'arrow'` opens it as an Arrow dataset and reads nothing yet. The local engine then reads it a row group at a time. `'sparkdf'` reads it with the active Spark session.

**Returns:** a `pandas.DataFrame`, a `pyarrow.dataset.Dataset` or a `pyspark.sql.DataFrame`.

**Raises:** `ValueError` when `df_type` is none of the three, or when a URI such as `s3a://`, which only Spark can read, is given with `'pandas'` or `'arrow'`.

**Example:**

```python
matches = iris.load_data('examples/out/passive/matches.parquet', 'pandas')
```

### save_data()

```python
def save_data(
    dataframe,         # a pandas or Spark DataFrame
    path,              # the file or directory to write
)
```

Writes a pandas or Spark DataFrame to Parquet, under exactly the name given. A pandas DataFrame is written to a temporary name and renamed to `path` when the file is complete.

**Parameters:**

- `dataframe`: the frame to write.
- `path`: a path, or for a Spark DataFrame a URI of storage that every machine of the cluster can read and write. On a cluster, a Spark DataFrame written to a path without a scheme such as `s3a://` is collected to the *driver*, the Python process that runs the script, and written there as one file, up to the limit `IRIS_MAX_DRIVER_ROWS`.

**Raises:** `ValueError` when the name of `path` ends in an extension other than `.parquet`. `TypeError` for another kind of frame. `RuntimeError` when a Spark DataFrame would collect more rows to the driver than `IRIS_MAX_DRIVER_ROWS`.

**Example:**

```python
iris.save_data(labels, 'out/labels.parquet')
```

### save_predictions()

```python
def save_predictions(
    predictions,                 # the result of apply_matcher, with prediction and confidence
    path,                        # the Parquet file, or for Spark the directory, to write
    threshold: float = 0.5,      # the probability of a match at or above which a pair is written
) -> int
```

Writes the predicted matches to Parquet, as the columns `id1`, `id2` and `probability`. Runs on either engine.

**Parameters:**

- `predictions`: the result of `apply_matcher` with a `confidence_col`, as pandas, an Arrow dataset or a Spark DataFrame.
- `path`: a path whose name ends in `.parquet` or has no extension. A pandas or Arrow result is written as one file, and a Spark result as `save_data` writes it.
- `threshold`: a pair is written when its probability of being a match is at or above it. At 0.5 the pairs written are those whose `prediction` is 1.0.

This function:

1. Computes the probability of a match of each pair from its `prediction` and `confidence`.
2. Keeps the pairs at or above `threshold`.
3. Writes `id1`, `id2` and `probability` (float32) to a temporary name, and renames it to `path` when the file is complete.

**Returns:** the number of matches written, as an `int`.

**Raises:** `ValueError` when `predictions` has no `confidence` column, or when the name of `path` ends in an extension other than `.parquet`. `TypeError` for another kind of input.

**Example:**

```python
predictions = iris.apply_matcher(matcher, fvs, 'feature_vectors', 'prediction', 'confidence')
n = iris.save_predictions(predictions, 'out/matches.parquet')
```

### Converting the output of a blocker

A blocker's output in a file is read with `read_pairs`. The functions below work on a blocker's output in memory.

```python
def candidates_from_flat(
    pairs: pd.DataFrame,          # one row per pair
    a_col: str = 'id1',           # the column of table A's ids
    b_col: str = 'id2',           # the column of table B's ids
    score_col: str = 'blocking_score',  # the blocker's score, when the pairs have one
    sort_by_score: bool = True,   # order each list by descending score
) -> pd.DataFrame

def candidates_from_result(result, sort_by_score: bool = True) -> pd.DataFrame

def write_flat_candidates(result, path, chunk_rows: int = 8_000_000) -> int
```

`candidates_from_flat` groups a flat pandas pair table into the grouped form, with the columns `id2`, `id1_list` and, when the pairs have a score in `score_col` or in `score`, `blocking_scores`. `candidates_from_result` does the same for the result object of a blocker that provides `iter_pairs()`, such as Sparkly's, without building the flat table. `write_flat_candidates` writes the pairs of such a result to a Parquet file as flat `(id1, id2, blocking_score)` rows, `chunk_rows` at a time, and returns the number of pairs written. It writes to a temporary name and renames it to `path` when the file is complete. Each function runs on the local engine.

**Raises:** `ValueError` from `candidates_from_flat` when one of the two id columns is missing, and from `write_flat_candidates` when the name of `path` ends in an extension other than `.parquet`.

## Checking the inputs

The `check_*` functions check the inputs before any work starts. `featurize` runs the same checks on its inputs unless it is given `validate=False`. Each function raises an error that names the invalid ids, and the messages are listed in [Troubleshooting](troubleshooting.md#checking-the-inputs).

### check_tables()

```python
def check_tables(table_a, table_b)
```

Checks that each table has an `_id` column of unique, non-null, non-negative integers. Runs on either engine.

**Returns:** `{'table_a': int, 'table_b': int}`, the number of rows of each table.

**Raises:** `ValueError` when a table is empty, has no `_id` column, or has null, non-integer, negative or repeated ids.

### check_candidates()

```python
def check_candidates(
    candidates,              # flat or grouped
    table_a,
    table_b,
    dedup=None,              # None: a deduplication when table_a is table_b
    membership=True,         # check that every id names a record
    what='candidates',       # the name used in messages
)
```

Checks the shape of a candidate set, its ids, and that every id names a record of its table. Runs on either engine.

**Parameters:**

- `dedup`: whether the run is a deduplication. `None` takes it to be one when `table_a` and `table_b` are the same object. In a deduplication, pairs of a record with itself are reported with a warning.
- `membership`: when `False`, the ids are not looked up in the tables, which saves a pass over a large candidate set.

This function:

1. Checks that the candidate set is flat or grouped, and not empty.
2. Checks that the ids are integers and not null, and in the grouped form that each record of B has one row and that any list column is aligned with `id1_list`.
3. With `membership`, checks that every `id1` is an `_id` of table A and every `id2` an `_id` of table B.
4. Warns about pairs of a record with itself in a deduplication, and about ids repeated within an `id1_list`.

**Returns:** the number of pairs, as an `int`.

**Raises:** `ValueError` when a check fails. `TypeError` when the inputs mix Arrow datasets and Spark DataFrames.

### check_gold_data(), check_labeled_data() and check_pairs()

```python
def check_gold_data(gold, table_a, table_b)
def check_labeled_data(labeled_data, table_a, table_b, label_column='label', dedup=None)
def check_pairs(pairs, table_a, table_b, what='gold')
```

`check_gold_data` and `check_pairs` check that every pair of a flat set names a record of each table. `check_labeled_data` checks labeled pairs, flat or grouped, as `check_candidates` does, and that the column `label_column` exists. Each returns the number of pairs and runs on either engine.

**Raises:** `ValueError` when a column is missing or an id names no record. `TypeError` for an input that is not a DataFrame.

### clean_candidates()

```python
def clean_candidates(
    candidates,                         # flat or grouped
    drop_self_pairs: bool = False,      # remove pairs with id1 == id2
    canonicalize_mirrors: bool = False, # orient every pair so that id1 < id2
)
```

Returns the candidate set as distinct flat `(id1, id2)` pairs. Runs on either engine.

**Parameters:**

- `drop_self_pairs`: removes the pairs of a record with itself. Use it only in a deduplication. In a match of two tables, `id1 == id2` names two different records.
- `canonicalize_mirrors`: orients every pair so that `id1` is the smaller id, so that a pair the blocker produced in both orders appears once. Use it only in a deduplication.

This function:

1. Turns a grouped candidate set into flat pairs, and drops every column other than `id1` and `id2`.
2. Removes the pairs of a record with itself, and orients the pairs, when asked.
3. Removes repeated pairs.

**Returns:** a frame of the kind it was given with the columns `id1` and `id2`, both int64. A pandas result is sorted by `id1` and `id2`. On the local engine, an Arrow input returns an Arrow dataset written to scratch files.

**Raises:** `TypeError` for another kind of input.

**Example:**

```python
candidates = iris.clean_candidates(candidates, drop_self_pairs=True, canonicalize_mirrors=True)
```

## Features and feature vectors

### create_features()

```python
def create_features(
    A,                                  # table A
    B,                                  # table B, or A again in a deduplication
    a_cols: List[str],                  # the columns of A to compare
    b_cols: List[str],                  # the columns of B, in the same order
    sim_functions=None,                 # None: the five base similarity functions
    tokenizers=None,                    # None: the three base tokenizers
    null_threshold: float = 0.5,        # drop a column at least this share empty
    numeric_threshold: float = 0.9,     # share of values that parse as numbers
    min_doc_freq: int = 1,              # TF/IDF: the fewest records a token must be in
    max_vocab_size: int = None,         # TF/IDF: keep at most this many tokens
    timezone: str = None,               # the time zone for date and time columns
) -> List[Feature]
```

Selects the features for the given columns from the contents of the tables. Runs on either engine.

**Parameters:**

- `a_cols` and `b_cols`: the columns compared, pair by pair. The tutorial passes the same list for both.
- `sim_functions` and `tokenizers`: the similarity functions and the tokenizers to choose from. `get_base_sim_functions()` and `get_base_tokenizers()` return the defaults, and `get_extra_tokenizers()` four more.
- `null_threshold`: a column empty in this share or more of the rows of A and B, counted together, is not used.
- `numeric_threshold`: a column of text whose values parse as numbers in at least this share of rows is treated as numeric. `None` treats as numeric only the columns whose type is numeric.
- `min_doc_freq` and `max_vocab_size`: bound the vocabulary of the TF/IDF features.
- `timezone`: an IANA time-zone name, such as `"UTC"`, in which date and time values are written as text before they are compared. `None` uses the time zone of the machine that runs the call. Pass the same zone to `create_features` and `featurize`, such as `"UTC"`, so that a run gives the same result on machines in different time zones.

This function:

1. Counts the empty values of each column and drops the columns that reach `null_threshold`.
2. Detects the numeric columns, by type or by `numeric_threshold`.
3. Counts the tokens of each column under each tokenizer.
4. Returns an exact-match feature for every column kept, a relative-difference feature for every numeric column, and for every tokenizer under which a column averages at least three tokens, one feature per similarity function. With the alphanumeric tokenizer, a column that averages at most ten tokens also gets the Monge-Elkan, edit-distance and Smith-Waterman features.

**Returns:** a list of features. `str(f)` names a feature, such as `jaccard(3gram_tokens(title), 3gram_tokens(title))`, and `f.a_attr` and `f.b_attr` name its columns.

**Raises:** `ValueError` for a `numeric_threshold` outside 0 to 1, or an unknown time zone. `TypeError` for a column whose type has no text form.

**Example:**

```python
columns = ['title', 'authors', 'venue', 'year']
features = iris.create_features(table_a, table_b, columns, columns)
```

### Features and tokenizers

Each feature compares a column of A with the column of B at the same position in `b_cols`.

| Feature | Name | Value |
| --- | --- | --- |
| Exact match | `exact_match(c, c)` | 1.0 when the two values are equal, ignoring case, else 0.0. |
| Relative difference | `rel_diff(c, c)` | `|x − y| / max(|x|, |y|)` for two numbers. |
| TF/IDF | `tf_idf_<tokenizer>(c, c)` | The cosine of the two token vectors, each token weighted by its rarity in the tables. |
| SIF | `sif_<tokenizer>(c, c)` | The cosine of the two token vectors under smooth inverse-frequency weights. |
| Jaccard | `jaccard(<tokenizer>(c), <tokenizer>(c))` | The size of the intersection of the two token sets over the size of their union. |
| Overlap coefficient | `overlap_coeff(...)` | The size of the intersection over the size of the smaller set. |
| Cosine | `cosine(...)` | The size of the intersection over the geometric mean of the two sizes. |
| Monge-Elkan | `monge_elkan_jw(...)` | The mean best Jaro-Winkler similarity of each token of one value to the tokens of the other. |
| Edit distance | `edit_distance(c, c)` | One minus the Levenshtein distance of the two values divided by the length of the longer value. |
| Smith-Waterman | `smith_waterman(c, c)` | The local alignment score of the two values, ignoring case, divided by the length of the longer value. |

| Tokenizer | Name | Tokens | Set |
| --- | --- | --- | --- |
| Words | `stripped_whitespace_tokens` | The words, split on white space, with punctuation stripped. | base |
| Numbers | `num_tokens` | The numbers in the text. | base |
| 3-grams | `3gram_tokens` | The 3-character fragments. | base |
| Alphanumeric words | `alnum_tokens` | The runs of letters and digits. | extra |
| 5-grams | `5gram_tokens` | The 5-character fragments. | extra |
| Stripped 3-grams and 5-grams | `stripped_3gram_tokens`, `stripped_5gram_tokens` | The fragments of the text with spaces and punctuation removed. | extra |

When either record's value of a column is empty, the features of that column are missing for the pair, and `featurize` sets them to `fill_na`, 0.0 by default. Every feature except the relative difference counts toward the score that ranks the pairs.

### Editing a set of features

A set of features is a Python list, and can be filtered, reordered or extended before `featurize` is called, as Part 2 of the tutorial shows. A matcher must be applied to feature vectors of the set it was trained on, in the same order.

### featurize()

```python
def featurize(
    features: List[Feature],         # from create_features
    A,                               # table A
    B,                               # table B, or A again in a deduplication
    candidates,                      # flat or grouped
    output_col: str = 'feature_vectors',
    fill_na: float = 0.0,            # the value of a missing feature
    validate: bool = True,           # check the inputs first
    timezone: str = None,            # as in create_features
    out: str = None,                 # a directory to write the feature vectors to
)
```

Computes the feature vector and the score of every candidate pair. Passing the same table object as `A` and `B` makes the run a deduplication of that table, in which `featurize` drops the pairs of a record with itself and featurizes each pair once, as (smaller id, larger id). Runs on either engine.

**Parameters:**

- `features`: the features, in the order of the vector.
- `candidates`: the candidate set. A pair that appears more than once is featurized once, with a warning.
- `fill_na`: the value written for a feature whose column is empty in either record of the pair.
- `validate`: runs `check_candidates` on the inputs first.
- `out`: a directory, or on Spark a URI, to write the feature vectors to as Parquet. An existing directory at `out` is replaced. Without `out`, the local engine writes them to scratch files, and the Spark engine returns a DataFrame that is computed again each time it is used.

On the local engine, this function:

1. Checks the inputs, and the free disk space for the feature vectors.
2. Builds the vocabulary of each TF/IDF and SIF feature from the records of A and B. In a deduplication each record is counted once.
3. Packs the records of A and B into a record store of scratch files, with each value preprocessed for the features, such as split into its tokens. In a deduplication there is one store.
4. Sorts the candidate pairs into parts, dropping repeated pairs. In a deduplication it first drops the pairs of a record with itself and orients each pair as (smaller id, larger id).
5. Featurizes the parts with worker processes, each reading the records of its pairs from the store, and writes the feature vectors.

On the Spark engine, this function cleans the pairs of a deduplication in the same way, joins the candidate pairs with the records of A and then with the records of B, and featurizes them in Spark's tasks.

**Returns:** for pandas tables, a `pandas.DataFrame` of the feature vectors. For Arrow tables on the local engine, a `pyarrow.dataset.Dataset` over the files written. For Spark DataFrames, a `pyspark.sql.DataFrame`. The columns are `id1`, `id2`, `output_col` and `score`.

**Raises:** the errors of `check_candidates`. `RuntimeError` when the disk cannot hold the feature vectors, or when pandas tables would bring more than `IRIS_MAX_DRIVER_ROWS` feature vectors into memory.

**Example:**

```python
fvs = iris.featurize(features, table_a, table_b, candidates, out='out/fvs.parquet')
```

### save_features() and load_features()

```python
def save_features(features, path)
def load_features(path)
```

`save_features` writes a list of features to a file, and `load_features` reads it back. A path is a file of this machine. A URI, such as `s3a://bucket/run/features.pkl`, is written and read through the active Spark session.

**Raises:** `FileNotFoundError` from `load_features` when the file does not exist.

## The pool and active learning

### down_sample()

```python
def down_sample(
    fvs,                              # the feature vectors, with a score
    percent: float,                   # the share of fvs to draw, in (0, 1]
    score_column: str = 'score',      # the column that ranks the pairs
    bucket_size: int = 1000,          # pairs per bucket, at least 1,000
    max_pool_size: int = 10_000_000,  # the largest pool; None for no limit
    seed: int = 42,                   # None draws a new seed
)
```

Draws the pool for active learning, a stratified sample of the feature vectors. Runs on either engine.

**Parameters:**

- `percent`: the share of the feature vectors to draw, such as 0.3. The pool holds `percent` of them, and at most `max_pool_size`. Since each bucket gives as many pairs from its high-scoring band as from the rest, `percent=1.0` draws about half of the feature vectors. To run active learning over all of them, skip `down_sample` and pass the feature vectors to `create_seeds` and `label_al` as the pool.
- `bucket_size`: the pairs are divided into buckets of about this many pairs by a hash of the pair, and each bucket contributes to the pool.
- `seed`: the same seed draws the same pool from the same feature vectors. With `None`, a new seed is drawn and logged.

This function:

1. Assigns every pair to a bucket by a hash of the pair.
2. Within each bucket, marks the pairs whose score is in the top `percent` of the bucket as the high-scoring band.
3. From each bucket, draws as many pairs from the high-scoring band as from the rest, chosen by a hash of the pair and the seed.
4. Keeps at most `max_pool_size` pairs.

**Returns:** for pandas, a `pandas.DataFrame`. For an Arrow dataset, a `pyarrow.Table` memory-mapped from scratch files and sorted by `id2` and `id1`. For Spark, a `pyspark.sql.DataFrame`. The rows are those of the feature vectors.

**Raises:** `ValueError` when `percent` is outside (0, 1], `bucket_size` is below 1,000, or `max_pool_size` is not positive.

**Example:**

```python
pool = iris.down_sample(fvs, percent=0.3, seed=42)
```

### create_seeds()

```python
def create_seeds(
    fvs,                         # the pool
    nseeds: int,                 # the number of labels to collect
    labeler: Labeler,
    score_column: str = 'score',
    checkpoint_path: str = 'active-matcher-training-data.parquet',
)
```

Collects the first labels of active learning, from both ends of the score ranking of the pool. Runs on either engine.

**Parameters:**

- `nseeds`: the number of labels to collect. More are collected when the first `nseeds` do not include a match and a non-match.
- `labeler`: the labeler that labels each pair.
- `checkpoint_path`: the file that every label is written to as it is collected, by default in the current directory, or a URI on Spark.

This function:

1. When the checkpoint file exists, returns up to `nseeds` of the labels in it instead, with a warning.
2. Ranks the pool by score, and examines up to 50 × `nseeds` pairs from each end.
3. Asks the labeler about pairs from the high end and the low end in turn, taking the next pair from the end of the class that has fewer labels so far, until it has `nseeds` labels with both classes.
4. Sets aside a pair labeled unsure, and stops at a stop label.

**Returns:** the labeled pairs, with the columns of the pool and `label`, as a frame of the kind of the pool.

**Raises:** `ValueError` when `nseeds` is not positive. `RuntimeError` when the labeler stops before any pair is labeled.

**Example:**

```python
seeds = iris.create_seeds(pool, 50, iris.GoldLabeler(gold), checkpoint_path='out/labeled.parquet')
```

### label_al()

```python
def label_al(
    model: MLModel,                  # the matcher retrained in each round
    mode: str,                       # 'batch' or 'continuous'
    labeler: Labeler,
    fvs,                             # the pool
    seeds=None,                      # the labels from create_seeds
    checkpoint_path: str = 'active-matcher-training-data.parquet',
    **learner_kwargs,                # the options of the mode
)
```

Collects labels by active learning over the pool. Runs on either engine.

**Parameters:**

- `fvs`: the pool, from `down_sample`, or the feature vectors themselves, as they were returned by `featurize`, when the candidate set has at most 10,000,000 pairs. Feature vectors on disk are copied once to a memory-mapped file in the scratch directory.
- `mode`: `'batch'` or `'continuous'`, as described in [How Iris works](how-it-works.md#active-learning).
- `seeds`: the first labels. They must include a match and a non-match.
- `checkpoint_path`: as in `create_seeds`. When the file holds labels beyond the seeds, labeling resumes from them, with a warning.

The options of batch mode are the following:

- `batch_size` (10): the number of pairs labeled in each round.
- `max_iter` (50): the number of rounds.
- `selection` (`'entropy'`): `'entropy'` selects the pairs whose predictions are least certain, `'random'` selects them at random, and `'hybrid'` takes half of each batch by uncertainty and the rest at random.

The options of continuous mode are the following:

- `queue_size` (50): the number of selected pairs kept ready for the labeler.
- `max_labeled` (550): the number of labels after which labeling ends.
- `on_demand_stop` (`True`): whether the labeler's stop label ends the labeling.

This function, in batch mode:

1. Trains the matcher on the labels collected so far.
2. Scores every unlabeled pair of the pool and selects `batch_size` of them by `selection`. Pairs of equal uncertainty are ordered by a hash of the pair.
3. Asks the labeler about each, writes each label to the checkpoint file, and sets aside the pairs labeled unsure.
4. Repeats for `max_iter` rounds, or until the labeler stops.

**Returns:** the labeled pairs, seeds included, with the columns `id1`, `id2`, `feature_vectors`, `label` and `labeled_in_iteration`. On the Spark engine the result is a pandas DataFrame, since the labels are collected on the driver.

**Raises:** `ValueError` for an unknown `mode` or `selection`, or a non-positive `batch_size` or `max_iter`. `RuntimeError` when the seeds do not hold both classes. `ValueError` when the checkpoint file holds feature vectors of another length, which happens when it is left from another run.

**Example:**

```python
model = iris.SKLearnModel(XGBClassifier, nan_fill=0.0)
labeled = iris.label_al(model, 'batch', labeler, pool, seeds, batch_size=10, max_iter=50)
```

### label_pairs()

```python
def label_pairs(labeler: Labeler, pairs)
```

Asks the labeler about each pair of a given set, in order, without active learning. Runs on either engine.

**Parameters:**

- `pairs`: a frame whose first two columns are the ids of table A and table B. An Arrow table or dataset is read into pandas, since the pairs a person labels are few.

**Returns:** the pairs with the column `label`. A pair labeled unsure has the label 2.0. A stop label ends the labeling, and the result holds the labels collected before it.

**Raises:** `TypeError` for another kind of input.

### Checkpoints and resuming

`create_seeds` and `label_al` write each label to `checkpoint_path` as it is collected. When the file exists at the start of a call, labeling resumes from the labels in it: `create_seeds` returns them as its seeds, and `label_al` continues from them. A session that was interrupted, by the labeler stopping, by an error, or by the machine shutting down, therefore continues where it stopped when it is run again with the same file. The default file, `active-matcher-training-data.parquet`, is in the current directory, so two runs in the same directory share it unless one of them names another file. A file whose feature vectors have another length than the run's is refused, since it comes from another dataset or another set of features.

## Labelers

### The contract

A labeler is called with the ids of a pair, `labeler(id1, id2)`, and returns one of the following values.

| Value | Meaning |
| --- | --- |
| 1.0 | A match. |
| 0.0 | A non-match. |
| 2.0 | Unsure. The pair is set aside and not asked again. |
| −1.0 | Stop. Labeling ends with the labels collected so far. |

A labeler that shows records, namely `CLILabeler`, `WebUILabeler` and `CustomLabeler`, looks each record up by its id in the tables it was given. It does not load the tables into memory beyond what they already occupy. For an Arrow dataset, it reads the `_id` column once and then the row group that holds each record. For a Spark DataFrame of at most 100,000 rows, it collects the table to the driver once. For a larger one, it looks each record up with a small Spark job.

### GoldLabeler and DelayedGoldLabeler

```python
GoldLabeler(gold, symmetric: bool = False)
DelayedGoldLabeler(gold, delay_secs, symmetric: bool = False)
```

`GoldLabeler` labels a pair 1.0 when it is in the gold set and 0.0 otherwise. With `symmetric=True`, the pair (y, x) is labeled as (x, y), which a deduplication needs when the gold set lists each pair in one order. `DelayedGoldLabeler` waits `delay_secs` seconds before each label, to simulate a person's pace.

**Raises:** `TypeError` when the ids of the gold set and of the pairs asked about have different types, since every pair would be labeled 0.0.

### CLILabeler

```python
CLILabeler(a_df, b_df, id_col: str = '_id')
```

Shows each pair at the terminal, as a table of the two records side by side, and reads the label typed: `y` for a match, `n` for a non-match, `u` for unsure, `h` to select the columns shown, and `s` to stop, after a confirmation. Part 3 of the tutorial shows its screen.

### WebUILabeler

```python
WebUILabeler(a_df, b_df, id_col: str = '_id', flask_port: int = 5005,
             streamlit_port: int = 8501, flask_host: str = '127.0.0.1')
```

Shows each pair on a page in a browser, with buttons for match, non-match, unsure and stop. It starts a Flask server on `flask_port`, through which Iris sends the pairs to the page and receives the labels. It also starts a Streamlit page on `streamlit_port`, and Streamlit prints the address of the page. It needs the browser labeling support of Iris, installed with `pip install 'iris[web]'`. Open the address in a browser on the same machine, or from another machine forward the port over SSH, for example with `ssh -L 8501:localhost:8501 machine`, and open `http://localhost:8501`.

**Raises:** `ImportError` when the browser labeling support is not installed. `RuntimeError` when a port is in use or the page does not start.

### CustomLabeler

```python
class MyLabeler(iris.CustomLabeler):
    def label_pair(self, row1, row2):
        ...
```

A subclass of `CustomLabeler` labels with its own code. `label_pair` receives the record of A and the record of B as dictionaries of their columns, and returns one of the values of the contract. A rule that decides some pairs can pass the others to another labeler, as Part 5 of the tutorial shows.

## Matchers

### SKLearnModel and SparkMLModel

```python
SKLearnModel(model, nan_fill=None, use_floats=True, **model_args)
SparkMLModel(model, nan_fill=0.0, **model_args)
```

`SKLearnModel` wraps a classifier class with the interface of scikit-learn, such as `xgboost.XGBClassifier` or `sklearn.ensemble.RandomForestClassifier`, and the arguments it is created with. It trains in the Python process that runs the script, which on Spark is the driver, and predicts on either engine. `SparkMLModel` wraps a classifier of Spark ML, such as `pyspark.ml.classification.RandomForestClassifier`, and runs on the Spark engine only.

**Parameters:**

- `model`: the classifier class, not an instance.
- `nan_fill`: the value that replaces a missing value in a feature vector before training and prediction. `None` leaves missing values in place, which XGBoost accepts and a random forest of scikit-learn does not.
- `use_floats`: builds the feature matrix in float32, which halves its memory.
- `model_args`: the arguments of the classifier, such as `max_depth=6, seed=42`.

### train_matcher()

```python
def train_matcher(model: MLModel, labeled_data, feature_col='feature_vectors', label_col='label')
```

Trains the matcher on labeled feature vectors. Runs on either engine.

This function:

1. Leaves out the pairs labeled unsure (2.0).
2. Checks that the labels are 1.0 or 0.0, and that both occur.
3. Trains the classifier on the feature vectors and the labels.

**Returns:** the trained model.

**Raises:** `ValueError` when the label column holds other values, or only one class.

### apply_matcher()

```python
def apply_matcher(model: MLModel, df, feature_col: str, prediction_col: str, confidence_col: str = None)
```

Applies a trained matcher to every feature vector. Runs on either engine.

**Parameters:**

- `df`: the feature vectors, as `featurize` returned them.
- `prediction_col` and `confidence_col`: the names of the two columns added, conventionally `'prediction'` and `'confidence'`. Without `confidence_col`, only the prediction is added.

**Returns:** the rows of `df` with the prediction, 1.0 or 0.0, and its confidence. For an Arrow dataset, worker processes write the predictions to scratch files, which are returned as an Arrow dataset.

**Raises:** `RuntimeError` when the model has not been trained. `TypeError` when a `SparkMLModel` is applied on the local engine.

### evaluate()

```python
def evaluate(predictions, gold, threshold: float = 0.5, dedup: bool = False) -> dict
```

Compares the predictions with a gold set. Runs on either engine.

**Parameters:**

- `threshold`: a pair is a predicted match when its probability of being a match is at or above it. At 0.5 it agrees with the column `prediction`.
- `dedup`: counts (x, y) and (y, x) as one pair, by orienting the gold pairs so that `id1` is the smaller id. The predictions must be oriented the same way, as `featurize` orients the pairs of a deduplication.

**Returns:** a dictionary with the following keys.

| Key | Value |
| --- | --- |
| `n_gold`, `n_predicted` | The numbers of gold pairs and of predicted matches. |
| `tp`, `fp`, `fn` | The true positives, false positives and false negatives. |
| `precision`, `recall`, `f1` | At `threshold`. Recall is measured against the whole gold set. |
| `threshold_curve` | One entry per threshold from 1.0 down to 0.0 in steps of 0.05, with `threshold`, `predicted`, `precision`, `recall` and `f1`. |
| `best_threshold`, `best_f1` | The point of the curve with the best F1. |

**Example:**

```python
metrics = iris.evaluate(predictions, gold, dedup=True)
```

## Spark sessions and storage

A function runs on the Spark engine when it is given Spark DataFrames, with the session that created them. On one machine, `iris.resources` starts a session sized to the machine:

```python
from iris.resources import build_session, plan

spark = build_session(plan())
```

`plan(master=None, driver_memory=None)` reads the memory and the cores of the machine, or of its container, and plans Spark's local mode, `local[n]` with `n` the number of parallel tasks the memory and the cores allow, and the driver's memory. `build_session(plan_, app_name='iris', shuffle_partitions=2000)` starts the session with Arrow enabled. On a cluster, the session comes from `spark-submit`, and the script takes it with `SparkSession.builder.getOrCreate()`, as Part 8 of the tutorial shows.

On Spark, the paths given to `read_table`, `read_pairs`, `featurize`, `save_data`, `load_data`, `checkpoint_path` and `save_features` may be URIs of storage that every machine of the cluster can read and write, such as `s3a://` or `hdfs://`. The files that Iris writes itself, namely the checkpoint, the description of the feature vectors and the saved features, are written through the Spark session, with the Hadoop library for the storage of the URI, so the machines of a cluster need no shared file system. [How to install Iris on a Spark cluster](installation/install-spark-cluster.md) describes the Hadoop library for each kind of storage.

## Options for large runs

The functions are the same at every size. What changes is how the data is read and where the results go.

### On one machine

| Need | Option |
| --- | --- |
| Keep the tables and the candidate set on disk | Read them with `read_table` and `read_pairs` and `df_type='arrow'`, instead of pandas. |
| Keep the feature vectors on disk | Pass `out` to `featurize`. With pandas tables, the feature vectors are brought into memory, up to `IRIS_MAX_DRIVER_ROWS` pairs. |
| Put the scratch files on a larger disk | Set `IRIS_SPILL_DIR`, or call `iris.configure(spill_dir=...)`. By default they go to the system's temporary directory. |
| Use fewer worker processes, for example when workers run out of memory | Set `IRIS_WORKERS`, or call `iris.configure(workers=...)`. By default it is up to one per core, and fewer when the memory is short, as `iris info` shows. |
| Keep each labeler lookup small | Write the tables with row groups of about 65,536 rows. |

The local engine checks the free disk space before it writes, and stops with an error that names the space it needs. The memory of a run is the pool, the labels, and one part of pairs per worker process, not the candidate set or its feature vectors.

### On a Spark cluster

| Need | Option |
| --- | --- |
| Room for the shuffle of `featurize` | Scratch disk on every worker, `SPARK_LOCAL_DIRS` or `spark.local.dir`. `featurize` logs the size of the shuffle and stops with an error when a worker's scratch disk is too small. |
| Memory for each task | `spark.executor.memory`. Each task holds a batch of records. When the records are large, `featurize` lowers `spark.sql.execution.arrow.maxRecordsPerBatch`, the number of rows Spark passes to Python at a time, in a session that Iris started with `build_session`. In a session from `spark-submit`, it logs a warning that names the value to set. |
| Results that stay distributed | Write to URIs with `out` and `save_data`. A step that would collect more rows to the driver than `IRIS_MAX_DRIVER_ROWS` stops with an error. |
| Feature vectors computed once | Pass `out` to `featurize`. Without it, the feature vectors are computed again by each step that uses them. |
| A quiet terminal | `--conf spark.log.level=WARN` on the `spark-submit` command line. |

## Settings and the command line

### Settings

| Setting | Default | What it does |
| --- | --- | --- |
| `IRIS_WORKERS` | up to one per core | The number of worker processes of the local engine. |
| `IRIS_SPILL_DIR` | the system's temporary directory | Where the local engine writes its scratch files. |
| `IRIS_MAX_DRIVER_ROWS` | 20,000,000 | The most rows a step brings into the memory of the driver, or of this process with pandas. |

A setting is an environment variable. `iris.configure(workers=4)` sets `IRIS_WORKERS` from Python, and `iris.load_env(path='.env')` reads the settings from a file of `NAME=value` lines. The file `.env.example` lists them.

### iris info and iris smoke

`iris info` prints the versions of Iris and Python, the memory and the cores Iris sees, the number of worker processes and the scratch directory of the local engine, and whether PySpark and Java are present. `iris smoke` runs a short test on the local engine, which featurizes 64 pairs of two small built-in tables and trains and applies a matcher. `iris smoke --engine spark` runs it on Spark. The installer runs `iris smoke`.

## Extending Iris

The base classes are public, for code of your own.

| Class | Subclass it to |
| --- | --- |
| `Labeler` | Label pairs from a source of your own, by implementing `__call__(id1, id2)`. `CustomLabeler` is the simpler choice when the records are needed. |
| `MLModel` | Use a classifier that is neither of scikit-learn's interface nor of Spark ML. |
| `Feature` | Compare two columns in a new way. |
| `Tokenizer` | Break text into tokens in a new way, for use in `create_features`. |
| `Vectorizer` | Weigh tokens in a new way, for features such as TF/IDF. |

A subclass of `Feature`, `Tokenizer` or `Vectorizer` is used by the record store and the vocabulary, and must implement the methods of its base class that they call. The classes of `iris._core.feature` and `iris._core.tokenizer` are the examples to follow.
