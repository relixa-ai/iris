## Troubleshooting

This page lists the error messages and warnings Iris raises, what each means, and what to do. The messages are quoted as the code produces them, with the variable parts in `<angle brackets>`. Installation problems are covered in the [installation guides](installation/README.md), each of which ends with its own troubleshooting table.

### Reading the files

These are raised by `read_table` and `read_pairs`.

| What you see | What it means | What to do |
|---|---|---|
| `<path>: no column '<id_col>' to use as the id; columns: [...]` | The table's id column has another name. | Pass its name, as in `read_table(path, id_col='product_id')`. |
| `<path>: no column(s) [...]; columns: [...]` | `columns` names attribute columns the table does not have. | Use the names listed. |
| `<path>: has both '_id' and the id column '<id_col>'; read it with id_col='_id', or drop one of them` | The file has a column `_id` in addition to the id column named by `id_col`. | Read it with `id_col='_id'`, or remove one of the two columns. |
| `<path>: has none of the pair columns id1/id2, a_id/b_id, ltable_id/rtable_id or left_id/right_id, nor the grouped columns 'id2' and 'id1_list'; columns: [...]. Name the file's two id columns with columns=(...)` | The pairs' id columns have names that are not recognised. | Pass them, table A's first, as in `read_pairs(path, columns=('product_a', 'product_b'))`. |
| `<path>: has no columns [...]; columns: [...]` | The columns given with `columns` are not in the file. | Use the names listed. |
| `<path>: is named .parquet but does not hold Parquet data; if it is CSV or TSV text, rename it to end in .csv or .tsv` | The file holds another format than its name says, often CSV written by a blocker under a `.parquet` name. | Rename the file as the message says. |
| `<path>: is named .csv but holds Parquet data; rename it to end in .parquet` | The file holds Parquet data under a `.csv` or `.tsv` name. | Rename it to end in `.parquet`. |
| `<path>: no column 'label' to read with with_label=True; columns: [...]` | The file has no label column. | Leave out `with_label`, or name the labels `label`. |
| `<path>: tables and pairs are read from a .parquet, .csv, .tsv or .jsonl file, or a directory of Parquet files` | The file has another extension. | Convert it to one of these formats. |
| `<path>: the file has no header row` | A CSV or TSV file whose first line does not name the columns. | Add a first line with the column names. |
| `df_type is 'pandas', 'arrow' or 'sparkdf', got '<value>'` | An unknown kind of result was asked for. | Use one of the three. |

### Checking the inputs

These are raised by the `check_*` functions, and by `featurize` before any work.

| What you see | What it means | What to do |
|---|---|---|
| `<table> must have an '_id' column; got [...]` | The table's id column is not named `_id`. | Read the table with `read_table(path, id_col=...)`, which renames it. |
| `<table> '_id' has <n> null value(s); every record needs an id` | Some rows have no id. | Fill them in. Every record needs a distinct integer id. |
| `<table> '_id' must be an integer column (ids are int64 join keys), got <type>; a table keyed by strings is given integer ids as 'Tables keyed by strings' in the technical guide describes` | The ids are text or floating-point numbers. | Give the table integer ids as [Tables keyed by strings](technical-guide.md#tables-keyed-by-strings) describes, and the candidate set and the gold set the same ids. |
| `<table> '_id' must be non-negative; <n> negative id(s), e.g. ids [...]` | Some ids are below zero. | Renumber the ids from zero, and the ids of the candidate set and the gold set with them. |
| `<table> '_id' must be unique; <n> duplicate row(s), e.g. ids [...]` | Two rows share an id. | Give every record a distinct id. |
| `<table> is empty` | The table has no rows. | Check the path and the filter that produced the table. |
| `candidates must be flat (columns 'id1', 'id2' -- one row per pair) or grouped (columns 'id2', 'id1_list' -- one row per B record); got [...]` | The candidate set has neither accepted shape. | Read the blocker's output with `read_pairs(path, columns=(...))`, which names its id columns `id1` and `id2`. |
| `candidates set is empty: there is nothing to match` | The candidate set has no pairs. | Check the blocker's output. |
| `candidates contain null ids (<n> in id1, <n> in id2)` | Some pairs lack an id. | Drop those rows. |
| `candidates 'id1' references id(s) not present in table_a's '_id' column, e.g. [...]. Every pair must point at records that exist.` | The candidate set names records that the table does not have, usually because the table was filtered or the candidate set belongs to other tables. | Use the tables the blocker ran on. |
| `candidates 'id2' must be unique (one row per B record); <n> duplicate row(s)` | A grouped candidate set lists a record of B on more than one row. | Merge those rows, or convert the candidate set to the flat form. |
| `candidates: <n> self-pair(s) (id1 == id2). featurize drops them when one table is matched against itself, and clean_candidates(..., drop_self_pairs=True) removes them from the candidate set.` | A warning in a deduplication. A record is paired with itself. | Nothing for `featurize`. To remove them from the candidate set itself, use `clean_candidates` as the message says. |
| `candidates: <n> row(s) contain duplicate ids within 'id1_list'; duplicates can be labeled twice and leak between training and application sets. clean_candidates removes them.` | A warning. A grouped row lists the same record twice. | Clean the candidate set with `clean_candidates`. |
| `labeled_data must have the column '<label>'; got [...]` | The labeled pairs have no label column of that name. | Name the column `label`, or pass its name. |
| `<input> is an Arrow dataset and other inputs are Spark DataFrames: pass all inputs local or all as Spark` | Some inputs of one call are Arrow datasets and others are Spark DataFrames. | Read every input of the run with the same `df_type`. |

### Features and feature vectors

| What you see | What it means | What to do |
|---|---|---|
| `column '<column>' is <type>, which has no text form in Iris (supported: <types>). Leave it out of the feature columns, or convert it first.` | A column has a type that Iris cannot compare, such as a nested list. | Leave the column out of the columns given to `create_features`, or convert it to text. |
| `column '<column>' holds <what> <value>, outside the years 1 to 9999 that Iris can show as text` | A date or time column holds a value outside the supported range. | Fix or remove the value. |
| `features read column(s) [...] that <table> does not have (columns: [...])` | The features were created for other tables, or loaded from a file saved for other data. | Create the features from the tables being featurized. |
| `the vocabulary for '<column>'/'<column>' with tokenizer <tokenizer> is empty after filtering (min_doc_freq=<n>, max_vocab_size=<n>). Lower min_doc_freq or raise max_vocab_size.` | Every token of the column was filtered out, or the column has no values in either table, as when features saved for other data are used on tables where that column is empty. | If the column is empty, leave it out of the features. Otherwise lower `min_doc_freq` or raise `max_vocab_size` in `create_features`. |
| `numeric_threshold is a rate in [0, 1] or None, got <value>` | `numeric_threshold` was given as a percentage or as text. | Pass a fraction such as 0.9, or None to detect numeric columns by type only. |
| `unknown timezone '<name>': use an IANA zone name such as "UTC" or "Europe/Paris"` | The `timezone` argument names no time zone. | Use a name from the IANA time-zone database. |
| `featurize: <n> of <n> candidate pairs are repeats and are dropped; each distinct pair is featurized once` | A warning. The candidate set lists some pairs more than once. | Nothing. To remove the repeats from the candidate set itself, use `clean_candidates`. |
| `featurize: one table is matched against itself: <n> pair(s) of a record with itself and <n> repeated or mirrored pair(s) are dropped, and each pair is featurized once, as (smaller id, larger id)` | A warning in a deduplication. The candidate set holds pairs of a record with itself, or the same pair in both orders. | Nothing. To clean the candidate set itself, use `clean_candidates(candidates, drop_self_pairs=True, canonicalize_mirrors=True)`. |

### Disk, memory and workers on one machine

| What you see | What it means | What to do |
|---|---|---|
| `not enough free disk space for Iris scratch files in '<directory>': need <x> GiB, <y> GiB available. These files are memory-mapped, so the space is needed regardless of available RAM. Set IRIS_SPILL_DIR to a directory on a larger volume, or free space on this one.` | The local engine checked the disk before writing its scratch files, and the space is short. | Set `IRIS_SPILL_DIR` as the message says. |
| `not enough free disk for <what> in '<directory>': need about <x> GiB, <y> GiB free. Set IRIS_SPILL_DIR to a directory on a larger volume, or free space.` | The same check, for the record store, the candidate pairs or the feature vectors. | Set `IRIS_SPILL_DIR`, or write the output with `out` to a larger volume. |
| `a worker process died; restarting the pool and rerunning <n> unfinished task(s) (<i> of <n> restarts)` | A warning. A worker process ended without finishing its task, and the work is being repeated. | Nothing, unless it repeats. Then see the next row. |
| `worker processes died <n> times; <n> of <n> task(s) unfinished. A worker killed by the operating system usually ran out of memory: run with fewer workers.` | Workers kept ending, most often because the operating system stopped them for using too much memory. | Set `IRIS_WORKERS` to a smaller number, for example half of what `iris info` reports. |
| `IRIS_WORKERS must be a positive integer, got '<value>'` | The setting holds something other than a positive whole number. | Set it to a number such as 4, or unset it. |
| `<path>: the local engine reads and writes paths of this machine, and a <scheme>:// URI names storage that Spark reaches. Pass Spark DataFrames to run on Spark.` | A URI such as `s3a://` was given to a function running on the local engine. | Use a local path, or read the data with Spark and pass Spark DataFrames. |

### The pool and labeling

| What you see | What it means | What to do |
|---|---|---|
| `percent must be in the range (0.0, 1.0]` | `down_sample` was given a percentage, such as 30, instead of a fraction. | Pass a fraction such as 0.3. |
| `ran out of candidates with <n> positive / <n> negative seeds; active learning needs at least one of each` | A warning from `create_seeds`. Neither end of the score ranking held a pair of the missing class. | See the next row. |
| `active learning needs both matches and non-matches to start, but the seeds are <n> positive / <n> negative. create_seeds ranks candidates by the score column and labels from both ends, so this means one end contained no example of its class -- common when a score is inflated by records agreeing on empty or null fields. Raise nseeds, supply seeds directly, or use a score column that separates the classes.` | The seeds hold only one class, so no matcher can be trained. | Do what the message says. Check also that the labeler labels correctly, for example that the gold set uses the same ids as the tables. |
| `existing training data at '<path>' has <n>-element feature vectors but this run produces <n>. That file is treated as a resume checkpoint, so it is almost certainly left over from a different dataset or feature set. Delete it, or pass checkpoint_path=<new path> to start fresh.` | The checkpoint file holds labels from another run. | Delete the file, or name another with `checkpoint_path`. |
| `Resuming active learning from <n> labeled examples in '<path>'; the supplied seeds are ignored. Delete that file or pass checkpoint_path=<new path> to start fresh.` | A warning. The checkpoint file holds labels from an earlier run, and labeling continues from them. | Nothing when resuming is intended. Otherwise delete the file or name another. |
| `GoldLabeler: <column>=<value> has type <type> but the gold set holds <column> values of type [...]. Every pair would be labeled 0.0. Normalize the gold and record id dtypes to match.` | The ids of the gold set and of the tables have different types, such as text and integers. | Convert the gold set's ids to the type of the tables' ids. |
| `GoldLabeler: the gold set is empty, so every pair will be labeled 0.0` | A warning. The gold set has no pairs. | Check the path of the gold set. |
| `'<package>' is required for this labeler but is not installed. Install the browser labeling support of Iris with: pip install 'iris[web]'` | `WebUILabeler` needs the browser labeling support, which is not installed. | Run the command in the message. |
| `could not bind the labeler's Flask port <port> on <host>: <error>. A previous labeling session may still be running -- stop it (e.g. ...), or pass a different flask_port to WebUILabeler.` | The port of the browser labeler is in use. | Stop the earlier session, or pass another `flask_port`. |
| `the Streamlit labeling UI exited immediately with code <n>. Check that streamlit is installed (pip install "iris[web]") and that port <port> is free.` | The browser page of the labeler did not start. | Install the browser labeling support, and free the port or pass another `streamlit_port`. |

### Training, applying and evaluating

| What you see | What it means | What to do |
|---|---|---|
| `training needs matches and non-matches, but the labels hold <n> match(es) and <n> non-match(es)` | The labeled pairs hold only one class once the unsure ones are left out. | Label more pairs, including at least one of each class. |
| `<label> holds [...]; a label is 1.0 (match), 0.0 (non-match) or 2.0 (unsure)` | The label column holds other values, such as `-1` or text. | Convert the labels to 1.0, 0.0 and 2.0. |
| `<path>: <function> writes Parquet; name it so that it ends in .parquet` | `save_data`, `save_predictions` or `write_flat_candidates` was given a path whose name ends in another extension, such as `.csv`. | Name the file so that it ends in `.parquet`. |
| `save_predictions needs the columns [...]; pass the result of apply_matcher(..., 'prediction', 'confidence')` | The predictions have no `confidence` column, or lack the ids. | Call `apply_matcher` with a `confidence_col`. |
| `Model must be trained to predict` | `apply_matcher` was given a model that has not been trained. | Pass the model that `train_matcher` returned. |
| `SparkMLModel predicts on Spark: pass the feature vectors as a Spark DataFrame, or use an SKLearnModel on local data` | A Spark ML classifier was given pandas or Arrow data, which run on the local engine. | Use `SKLearnModel` on the local engine, or run the whole workflow on Spark. |

### Spark

| What you see | What it means | What to do |
|---|---|---|
| `this needs Spark, and pyspark is not installed. Install Iris with Spark support, pip install 'iris[spark]' (it also needs Java 17), or run on the local engine by passing pandas DataFrames or Arrow datasets.` | A Spark DataFrame or a Spark function was used, and Spark support is not installed. | Install Iris with Spark support, or pass pandas DataFrames or Arrow datasets. |
| `Iris is running on Java <n>, but PySpark 3.5.x supports Java 8, 11 and 17 only. ...` | A warning. Spark started on an unsupported Java version, and later errors can look unrelated. | Install Java 17 and point `JAVA_HOME` at it. |
| `not enough scratch disk for the pair pass: about <x> GiB of records cross the shuffle (up to <y> GiB on disk), and <directory> has <z> GiB free. Point SPARK_LOCAL_DIRS (or spark.local.dir) at a larger volume, or free space.` | Spark's scratch directory cannot hold the shuffle of `featurize`. | Set `SPARK_LOCAL_DIRS` on every worker to a volume with the space the message names. |
| `<what> would move <n> rows onto the driver, above the limit of <n>. ... Raise the limit with IRIS_MAX_DRIVER_ROWS=<n> only if the driver really can hold this frame.` | A step would collect a large table into the driver's memory. | Follow the remedy in the message, such as writing to an `s3a://` or `hdfs://` URI, which every machine of the cluster can read and write. Raise the limit only when the driver has the memory. |
| `save_data would collect <n> rows to the driver, above the <n>-row limit. Write to a distributed path (hdfs://, s3a://, ...) so the write stays parallel, or raise the limit with IRIS_MAX_DRIVER_ROWS=<n> if the driver really can hold this frame.` | `save_data` was given a path without a scheme on a cluster, so the data would be collected to the driver. | Write to an `s3a://` or `hdfs://` URI, which every machine of the cluster can read and write, such as `s3a://bucket/run/predictions.parquet`. |
