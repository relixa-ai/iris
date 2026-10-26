## Iris tutorial

This tutorial shows how to use Iris in eight parts. Each part is built around a script in the `examples` directory. We recommend that you go through the parts in order, since each part uses what the previous ones introduced. Before starting, install Iris by following [the installation page](installation/README.md). Parts 1 to 6 give the functions pandas DataFrames or Arrow datasets, so they run on the *local engine*, which uses Python processes on this machine. Parts 7 and 8 give them Spark DataFrames, so they run on the *Spark engine*, which needs Java 17 and Iris installed with Spark support, an optional part of Iris that adds PySpark.

A matcher is trained on *labels*, which are pairs marked match (1.0) or non-match (0.0). Labels come from one of four sources.

- *A file.* Labels produced earlier, with Iris or by another tool. The pairs are featurized, and the matcher is trained on them.
- *A person.* `CLILabeler` asks a person to label pairs at the terminal. `WebUILabeler` shows the pairs on a page in a browser.
- *Code.* A subclass of `CustomLabeler` labels pairs by a rule, and can pass the pairs the rule does not decide to another labeler.
- *A gold set.* When the true matches are known, for example from a benchmark, `GoldLabeler` labels from them. A gold set is also what precision and recall are computed against. Every number in this tutorial is computed against one.

The pairs to label are selected either by you or by *active learning*, in which Iris trains a matcher on the labels collected so far, asks for labels on the pairs the matcher is least certain about, and repeats. Active learning is introduced in Part 4.

Three datasets are included with Iris, in the `data` directory. The tutorial uses the first two.

| Dataset | What it is | Size |
| --- | --- | --- |
| `data/dblp_acm` | Two tables of computer-science papers, DBLP (2,616 rows) and ACM (2,294 rows), with a candidate set of 114,700 pairs from a blocker, a gold set of the 2,220 pairs known to be the same paper, and a file of 550 labeled pairs. | 1.3 MB |
| `data/music_1m` | One table of 1,000,000 songs that contains duplicates, with the 33,174,998 candidate pairs produced by blocking the table against itself, and a gold set of the 146,011 known duplicate pairs. | 128 MB |
| `data/abt_buy` | Two tables of electronics products as two stores list them, Abt (1,081 rows) and Buy (1,092 rows), with a gold set of the 1,097 pairs known to be the same product, and a file of 550 labeled pairs. The [quick start](https://www.relixa.ai/quick-start) on the Relixa website uses it. | 249 KB |

The numbers below were obtained on a Mac with 18 cores and 48 GB of memory. The counts are the same on any machine. The precision, recall and F1 can differ slightly on another machine, because active learning can choose differently between pairs whose uncertainty differs only in the last digits of a floating-point number. The running times will differ. Run every script from the `iris` directory, which holds the downloaded repository, after activating the Python environment that the installer created:

```bash
source .venv/bin/activate
```

Each script writes its output under `examples/out/<name>/`. In the scripts, a comment starting with `# ->` after a call describes the value the call returns.

### Part 1: Your first run, from labels you already have

`examples/run_passive.py` is the shortest complete run. It reads labels from a file, trains a matcher on them, and applies it to the candidate set. Run:

```bash
python examples/run_passive.py
```

```
features 35  labels 550  candidates 114700
precision 0.9857  recall 0.9937  f1 0.9897
```

The script makes the following calls, in order:

```python
table_a = iris.read_table('data/dblp_acm/table_a.parquet')
table_b = iris.read_table('data/dblp_acm/table_b.parquet')
candidates = iris.read_pairs('data/dblp_acm/cand.parquet')
gold = iris.read_pairs('data/dblp_acm/gold.parquet')
labeled_pairs = iris.read_pairs('data/dblp_acm/labeled_pairs.parquet', with_label=True)

iris.check_tables(table_a, table_b)
iris.check_candidates(candidates, table_a, table_b)
iris.check_gold_data(gold, table_a, table_b)
iris.check_labeled_data(labeled_pairs, table_a, table_b)

columns = ['title', 'authors', 'venue', 'year']
features = iris.create_features(table_a, table_b, columns, columns)
labeled = iris.featurize(features, table_a, table_b, labeled_pairs[['id1', 'id2']])
labeled = labeled.merge(labeled_pairs[['id1', 'id2', 'label']], on=['id1', 'id2'])

model = iris.SKLearnModel(XGBClassifier, nan_fill=0.0, eval_metric='logloss',
                          objective='binary:logistic', max_depth=6, seed=42)
matcher = iris.train_matcher(model, labeled)

fvs = iris.featurize(features, table_a, table_b, candidates)
predictions = iris.apply_matcher(matcher, fvs, 'feature_vectors', 'prediction', 'confidence')
metrics = iris.evaluate(predictions, gold)
```

The table below explains each call.

| Call | Returns | What it does |
| --- | --- | --- |
| `read_table(path)` | `pandas.DataFrame` | Reads a table into memory. A table has an integer id column, `_id`. A file whose id column has another name is read with `id_col`, which renames it. |
| `read_pairs(path)` | `pandas.DataFrame` | Reads a set of pairs into memory, as the columns `id1` and `id2`, which hold an `_id` of table A and of table B. The id columns `a_id` and `b_id`, and the other names that blockers give them, are recognized, and `columns` names the file's id columns when they have names that are not recognized. The labeled pairs are stored in the grouped form, with one row per record of B, and `read_pairs` turns them into one row per pair, with `with_label=True` reading their labels. |
| `check_tables`, `check_candidates`, `check_gold_data`, `check_labeled_data` | the counts of rows or pairs checked | Check the ids before any work starts, and raise an error that names the invalid ones. |
| `create_features(table_a, table_b, columns, columns)` | `list[Feature]` | Selects the features for the four columns, 35 in all. Part 2 describes them. |
| `featurize(features, table_a, table_b, pairs)` | `pandas.DataFrame` | Computes the feature vector and the score of each pair, in the columns `feature_vectors` and `score`. It is called twice, for the 550 labeled pairs and for the 114,700 candidate pairs. |
| `SKLearnModel(XGBClassifier, ...)` | `SKLearnModel` | Creates an untrained matcher from a classifier class of scikit-learn's interface and its arguments. |
| `train_matcher(model, labeled)` | `SKLearnModel` | Trains the matcher on the feature vectors and the labels of the labeled pairs. |
| `apply_matcher(matcher, fvs, 'feature_vectors', 'prediction', 'confidence')` | `pandas.DataFrame` | Adds to every candidate pair the prediction, 1.0 for a match and 0.0 for a non-match, and its confidence. |
| `evaluate(predictions, gold)` | `dict` | Compares the predictions with the gold set and returns the precision, recall and F1 that the script printed, together with the counts and a threshold curve. |

The 550 labeled pairs hold 218 matches. The matcher trained on them predicts the gold matches of the candidate set with a precision of 0.9857 and a recall of 0.9937. The script ends with `save_features` and `save_predictions`, which write the features and the predicted matches to `examples/out/passive/`. The file of matches has the columns `id1`, `id2` and `probability`, with one row per pair predicted to match.

### Part 2: The features that `create_features` selects

`examples/run_features.py` shows what `create_features` selected in Part 1, how to change the set, and how to save and reload it. Run:

```bash
python examples/run_features.py
```

```
35 features
  title    11
  authors  11
  venue    11
  year     2
  exact_match(title, title)
  exact_match(authors, authors)
  exact_match(venue, venue)
  exact_match(year, year)
  rel_diff(year, year)
  tf_idf_stripped_whitespace_tokens(title, title)
pair (1552, 0): 35 values, score 5.562
pair (1941, 0): 35 values, score 6.792
pair (2123, 0): 35 values, score 24.455
without venue: 24 features
with the extra tokenizers: 107 features
reloaded 35 features, the same: True
```

Every column gets an exact-match feature. The column `year` holds numbers, so it also gets a relative-difference feature, `rel_diff(year, year)`. The columns `title`, `authors` and `venue` each average at least three tokens under the word tokenizer and under the 3-gram tokenizer, so each gets five similarity features per tokenizer: TF/IDF, Jaccard, SIF, overlap coefficient and cosine. That is 1 + 5 + 5 = 11 features for each of the three columns. [How Iris works](how-it-works.md#features) gives the rules.

The calls of this part are the following:

| Call | Returns | What it does |
| --- | --- | --- |
| `featurize(features, table_a, table_b, candidates.head(3))` | `pandas.DataFrame` | Featurizes three pairs. Each feature vector has one value per feature, in the order of the list. The third pair, whose records are the same paper, has the highest score. |
| `[f for f in features if f.a_attr != 'venue']` | `list[Feature]` | Removes the features of one column. A set of features is a list, and can be filtered or reordered before `featurize` is called. |
| `get_base_tokenizers() + get_extra_tokenizers()` | `list[Tokenizer]` | The three base tokenizers and the four extra ones. Given as `tokenizers` to `create_features`, they produce 107 features. |
| `save_features(features, path)` and `load_features(path)` | `None` and `list[Feature]` | Write the features to a file and read them back. A matcher can only be applied to feature vectors of the features it was trained on, so the features are saved with the run. |

The features are chosen before any labels exist, and a matcher can use only what they measure. Part 3 turns to the labels.

### Part 3: Labeling the pairs you select

`examples/run_label.py` labels a set of pairs that you have selected, and then continues as in Part 1. It labels the 550 pairs of the labeled-pairs file again, this time with a labeler. By default the labeler is a `GoldLabeler`, which labels from the gold set. Run:

```bash
python examples/run_label.py
```

```
labeled 550 pairs, 218 matches
precision 0.9857  recall 0.9937  f1 0.9897
```

The labels are the same as those in the file, so the result is that of Part 1. The new call is the following:

```python
labeler = iris.GoldLabeler(gold)
labels = iris.label_pairs(labeler, selected)
```

| Call | Returns | What it does |
| --- | --- | --- |
| `GoldLabeler(gold)` | `GoldLabeler` | A labeler that labels a pair 1.0 when it is in the gold set and 0.0 otherwise. |
| `label_pairs(labeler, selected)` | `pandas.DataFrame` | Asks the labeler about each pair, in the order given, and returns the pairs with the column `label`. |

To label the pairs yourself, run the script with `--cli`. `CLILabeler` then shows each pair at the terminal and reads the label you type:

```
Do these refer to the same concept?
================================================================================
┌───────────┬────────────────────────────────────────────────────┬─────────────────────────────────────────────────────┐
│ Field     │ From A                                             │ From B                                              │
├───────────┼────────────────────────────────────────────────────┼─────────────────────────────────────────────────────┤
│ _id       │ 2123                                               │ 0                                                   │
├───────────┼────────────────────────────────────────────────────┼─────────────────────────────────────────────────────┤
│ title     │ the wasa2 object-oriented workflow management      │ the wasa2 object-oriented workflow management       │
│           │ system                                             │ system                                              │
├───────────┼────────────────────────────────────────────────────┼─────────────────────────────────────────────────────┤
│ authors   │ mathias weske , gottfried vossen                   │ gottfried vossen , mathias weske                    │
├───────────┼────────────────────────────────────────────────────┼─────────────────────────────────────────────────────┤
│ venue     │ sigmod conference                                  │ international conference on management of data      │
├───────────┼────────────────────────────────────────────────────┼─────────────────────────────────────────────────────┤
│ year      │ 1999                                               │ 1999                                                │
└───────────┴────────────────────────────────────────────────────┴─────────────────────────────────────────────────────┘
--------------------------------------------------------------------------------
Enter y[es], n[o], u[nsure], h[elp], or s[top]:
```

`y` labels the pair a match, `n` a non-match, and `u` unsure. `h` selects which columns are shown. `s` stops the labeling, after a confirmation, and the run continues with the labels collected so far. A pair labeled unsure is kept in the result with the label 2.0, and `train_matcher` leaves it out.

With `--web`, `WebUILabeler` shows the pairs on a page in a browser instead, with the same choices as buttons. It needs the browser labeling support of Iris, which is installed with `pip install 'iris[web]'`. When the labeler starts, Streamlit prints the address of the page, `http://localhost:8501` by default. Open it in a browser on the same machine.

Labeling 550 pairs by hand takes time, and the pairs in the file were not selected as the pairs the matcher learns most from. Part 4 lets active learning select the pairs to label.

### Part 4: Letting active learning select the pairs

`examples/run_al_gold.py` starts with no labels. It draws a pool from the candidate set, collects 50 seed labels, and then runs 50 rounds of active learning that collect 10 labels each. A `GoldLabeler` takes the place of a person, so that the script runs unattended. Run:

```bash
python examples/run_al_gold.py
```

```
pool 34504  labels 550  positive 255
precision 0.9941  recall 0.9851  f1 0.9896
```

The new calls are the following:

```python
fvs = iris.featurize(features, table_a, table_b, candidates)
pool = iris.down_sample(fvs, percent=0.3, seed=42)
labeler = iris.GoldLabeler(gold)
seeds = iris.create_seeds(pool, 50, labeler, checkpoint_path=CHECKPOINT)
labeled = iris.label_al(model, 'batch', labeler, pool, seeds, checkpoint_path=CHECKPOINT,
                        batch_size=10, max_iter=50)
```

| Call | Returns | What it does |
| --- | --- | --- |
| `down_sample(fvs, percent=0.3, seed=42)` | `pandas.DataFrame` | Draws the pool, 30% of the candidate set, 34,504 pairs. About half of the pool consists of high-scoring pairs, where the likely matches are, as [How Iris works](how-it-works.md#the-pool) describes. The same seed draws the same pool. |
| `create_seeds(pool, 50, labeler, checkpoint_path=...)` | `pandas.DataFrame` | Asks the labeler about pairs from both ends of the score ranking until it has 50 labels with both classes in them. |
| `label_al(model, 'batch', labeler, pool, seeds, checkpoint_path=..., batch_size=10, max_iter=50)` | `pandas.DataFrame` | In each of 50 rounds, trains the matcher on the labels so far, asks the labeler about the 10 pool pairs the matcher is least certain about, and adds the labels. It returns the 550 labeled pairs with their feature vectors. |

The 550 labels that active learning selected hold 255 matches, against 218 in the file of Part 1. The matcher trained on them reaches an F1 of 0.9896.

The labels are written to the checkpoint file while they are collected. When the script is interrupted and run again, labeling resumes from the file. The script deletes the file at its start so that each run begins afresh. In your own runs, keep the file, since it holds the labels a person has given. Without `checkpoint_path`, the file is `active-matcher-training-data.parquet` in the current directory.

To label the pairs yourself, replace the `GoldLabeler` with a `CLILabeler` or a `WebUILabeler`, as in Part 3.

### Part 5: A labeler of your own

`examples/run_custom_labeler.py` defines a labeler that labels some pairs by a rule and asks another labeler about the rest. The rule labels a pair as a match when the titles and the years are equal. The other pairs go to a `GoldLabeler`, which takes the place of a person. Run:

```bash
python examples/run_custom_labeler.py
```

```
labels 550, 160 by the rule
precision 0.9778  recall 0.9901  f1 0.9839
```

The labeler is a subclass of `CustomLabeler`:

```python
class TitleAndYear(iris.CustomLabeler):

    def __init__(self, table_a, table_b, fallback):
        super().__init__(table_a, table_b)
        self.fallback = fallback
        self.by_rule = 0

    def label_pair(self, row1, row2):
        same_title = str(row1['title']).strip().lower() == str(row2['title']).strip().lower()
        if same_title and row1['year'] == row2['year']:
            self.by_rule += 1
            return 1.0
        return self.fallback(row1['_id'], row2['_id'])

labeler = TitleAndYear(table_a, table_b, iris.GoldLabeler(gold))
```

| Call | Returns | What it does |
| --- | --- | --- |
| `CustomLabeler.__init__(table_a, table_b)` | | Keeps the tables, so that each pair can be looked up by its ids. |
| `label_pair(row1, row2)` | `float` | Receives the record of A and the record of B of a pair, as dictionaries of their columns, and returns 1.0, 0.0, 2.0 for unsure, or -1.0 to stop. |

The labeler is then used with `create_seeds` and `label_al` as in Part 4. Of the 550 labels, 160 came from the rule and the rest from the gold set, so a person would have labeled 390 pairs. The rule labels every pair with equal titles and years as a match. In the candidate set there are 2,016 such pairs, and 41 of them are not in the gold set, so some of the rule's labels disagree with the gold set, and the matcher learns from them too. Precision is 0.9778, against 0.9941 in Part 4. A rule's labels are trusted as they are, so a rule must be one that a person would agree with.

### Part 6: One table against itself, a million records on one machine

`examples/run_dedup.py` finds the duplicates in `data/music_1m`, a table of 1,000,000 songs, by matching the table against itself. The candidate set has 33,174,998 pairs. The parts so far read every input with pandas, which holds it in memory. This part opens the table and the candidate set as Arrow datasets, which the local engine reads from disk a chunk at a time, and writes the feature vectors to disk. Run:

```bash
python examples/run_dedup.py
```

```
pairs 33,174,998  pool 9,999,928  labels 550
precision 0.9991  recall 0.9235  f1 0.9598
```

The script differs from Part 4 in the following calls:

```python
iris.configure(spill_dir=f'{OUT}/scratch')
table = iris.read_table('data/music_1m/table_a.parquet', df_type='arrow')
candidates = iris.read_pairs('data/music_1m/cand.parquet', df_type='arrow')
iris.check_candidates(candidates, table, table, dedup=True)
candidates = iris.clean_candidates(candidates, drop_self_pairs=True, canonicalize_mirrors=True)
features = iris.create_features(table, table, columns, columns)
fvs = iris.featurize(features, table, table, candidates, out=f'{OUT}/fvs.parquet')
labeler = iris.GoldLabeler(gold, symmetric=True)
metrics = iris.evaluate(predictions, gold, dedup=True)
```

| Call | Returns | What it does |
| --- | --- | --- |
| `configure(spill_dir=...)` | `None` | Sets `IRIS_SPILL_DIR`, the directory for the local engine's scratch files. By default they go to the system's temporary directory. |
| `read_table(path, df_type='arrow')`, `read_pairs(path, df_type='arrow')` | `pyarrow.dataset.Dataset` | Open a Parquet file without reading it. The functions that receive it read it a chunk at a time. A CSV, TSV or JSON Lines file, or one whose columns need renaming, is first written once as Parquet to the scratch directory. |
| `check_candidates(candidates, table, table, dedup=True)` | `int` | The input check, with one table as both A and B. |
| `clean_candidates(candidates, drop_self_pairs=True, canonicalize_mirrors=True)` | `pyarrow.dataset.Dataset` | Removes the pairs of a song with itself and keeps each pair in one order, with `id1` smaller than `id2`. The result is written to scratch files. |
| `featurize(features, table, table, candidates, out=...)` | `pyarrow.dataset.Dataset` | Passing the same table object as A and B makes this a deduplication. Worker processes featurize the pairs and write the feature vectors to `out`, and the result reads them from there. |
| `GoldLabeler(gold, symmetric=True)` | `GoldLabeler` | The gold set lists each duplicate pair once, in one order. `symmetric=True` labels (y, x) as it labels (x, y). |
| `evaluate(predictions, gold, dedup=True)` | `dict` | Counts (x, y) and (y, x) as one pair. |

The pool, the seeds, active learning and the matcher are as in Part 4. The pool is an Arrow table in a scratch file, mapped into memory rather than copied into it. `apply_matcher` has worker processes apply the matcher to the feature vectors on disk, a chunk at a time, and write the predictions to scratch files, which it returns as an Arrow dataset. What the run holds in memory is therefore the pool, the labels, and one chunk of pairs per worker process, not the candidate set or its feature vectors.

On the tutorial's machine the run took 3 minutes. Iris ran 18 worker processes, one per core of that machine. `iris info` shows the number for your machine, and the setting `IRIS_WORKERS` changes it. The candidate set contains 98.92% of the 146,011 gold pairs, so no matcher can reach a recall above 0.9892 on it.

### Part 7: The same functions on Spark

`examples/run_spark.py` runs the functions of Part 4 on Spark, with Spark running on this machine only. It needs Iris installed with Spark support, by `bash install.sh --spark`, and Java 17. Run:

```bash
python examples/run_spark.py
```

```
precision 0.9941  recall 0.9851  f1 0.9896
```

The inputs are read with Spark, so every function runs on the Spark engine:

```python
spark = build_session(plan())
table_a = iris.read_table('data/dblp_acm/table_a.parquet', df_type='sparkdf')
table_b = iris.read_table('data/dblp_acm/table_b.parquet', df_type='sparkdf')
candidates = iris.read_pairs('data/dblp_acm/cand.parquet', df_type='sparkdf')
gold = iris.read_pairs('data/dblp_acm/gold.parquet', df_type='sparkdf')
fvs = iris.featurize(features, table_a, table_b, candidates, out=f'{OUT}/fvs.parquet')
```

| Call | Returns | What it does |
| --- | --- | --- |
| `build_session(plan())` | `SparkSession` | Starts Spark on this machine only, in Spark's local mode, with a driver memory and a number of parallel tasks that `plan` sets from the memory and the cores of this machine. Both functions are in `iris.resources`. On a cluster, the session comes from `spark-submit` instead, as in Part 8. |
| `read_table(path, df_type='sparkdf')`, `read_pairs(path, df_type='sparkdf')` | `pyspark.sql.DataFrame` | Read an input with the session as a Spark DataFrame, which Spark splits across its tasks. |
| `featurize(..., out=...)` | `pyspark.sql.DataFrame` | Joins the pairs with the records of A and then of B, featurizes them in Spark's tasks, and writes the feature vectors to `out`. |

The other calls are those of Part 4, unchanged. The labeler runs on the driver, the Python process that runs the script, and `label_al` returns the labels as a Spark DataFrame. `apply_matcher` and `evaluate` run on Spark.

Spark prints its own messages as it starts, such as warnings about the host name, before the script's output. Part 8 submits the functions to a cluster.

### Part 8: A Spark cluster

`examples/run_spark_cluster.py` runs the functions of Part 7 on a Spark cluster. It is submitted to the cluster with `spark-submit`, which starts the script on the driver and the tasks on the workers. Its two arguments name the dataset and a new location for the outputs. On a cluster both are URIs of storage that every machine of the cluster can read and write, such as `s3a://` for Amazon S3.

The script can be tried on one machine first, with `--master 'local[4, 4]'`, which runs Spark on this machine with 4 parallel tasks and up to 4 attempts at each, and `file://` URIs in place of a cluster and its storage. `spark.log.level=WARN` keeps Spark's log on the terminal to its warnings. Run:

```bash
spark-submit --master 'local[4, 4]' --conf spark.log.level=WARN examples/run_spark_cluster.py \
    file://$PWD/data/dblp_acm file://$PWD/examples/out/cluster
```

```
precision 0.9941  recall 0.9851  f1 0.9896
```

The script differs from Part 7 in how it starts and where it reads and writes:

```python
data, out = sys.argv[1].rstrip('/'), sys.argv[2].rstrip('/')
spark = SparkSession.builder.appName('iris').getOrCreate()
table_a = iris.read_table(f'{data}/table_a.parquet', df_type='sparkdf')
fvs = iris.featurize(features, table_a, table_b, candidates, out=f'{out}/fvs.parquet')
seeds = iris.create_seeds(pool, 50, labeler, checkpoint_path=f'{out}/labeled.parquet')
```

| Call | Returns | What it does |
| --- | --- | --- |
| `SparkSession.builder.appName('iris').getOrCreate()` | `SparkSession` | Attaches to the session that `spark-submit` started, with the master and the settings given to `spark-submit` or in the cluster's `spark-defaults.conf`. |
| `featurize(..., out=f'{out}/fvs.parquet')` | `pyspark.sql.DataFrame` | Writes the feature vectors to the storage that `out` names. |
| `create_seeds(..., checkpoint_path=f'{out}/labeled.parquet')` | `pyspark.sql.DataFrame` | The checkpoint of the labels is written through Spark to the same storage, so it is kept when the driver machine is stopped or replaced. |

On a cluster, the same script is submitted to the cluster's master, with the dataset copied to the cluster's storage:

```bash
spark-submit --master spark://HOST:7077 examples/run_spark_cluster.py \
    s3a://BUCKET/datasets/dblp_acm s3a://BUCKET/datasets/dblp_acm/runs/1
```

[How to install Iris on a Spark cluster](installation/install-spark-cluster.md) describes what every machine of the cluster needs, and the script `deploy/spark-node.sh`, which installs Iris, Spark and Java on an Ubuntu machine at any cloud provider or on your own hardware. The directory `deploy/aws` holds the scripts that start a cluster on Amazon Web Services, run a script on it, and stop it.

### Summary

This completes the tutorial. In the eight parts you have checked the inputs, created and changed a set of features, trained a matcher from labels in a file, collected labels from a gold set, a person and your own code, let active learning select the pairs, matched a table of a million records against itself on one machine, and run the same functions on Spark and on a Spark cluster.

## Going further

### A workflow of your own

The functions of Iris can be combined in different orders to create a workflow that fits your data. [Workflows](workflows.md) describes the four ways a run can start and the script for each.

When the result of a run is not as good as expected, the following are the usual causes.

- *Blocking.* A true match that the blocker left out of the candidate set cannot be found.
- *The labels.* The matcher learns what the labeler says. Labels from an inconsistent person train a worse matcher than a smaller set of correct labels.
- *The label budget.* The 550 labels of the tutorial were enough for its data. Data on which matches and non-matches look more alike may need more rounds, set with `max_iter`, or a larger pool.
- *The columns.* A column that is mostly empty, or that does not differ between matches and non-matches, adds noise. Pass only the columns that identify an entity to `create_features`.
- *The threshold.* Read the curve that `evaluate` returns before changing the threshold. Change it only when the curve shows a higher F1 at another threshold.

[How Iris works](how-it-works.md) discusses each of these in more detail.

### Running at scale

As the tables and the candidate set grow, the workflow does not change, but the data is read differently. Read with pandas, every input and every result is held in memory. Read with `df_type='arrow'`, the tables and the candidate set are read from disk a chunk at a time, and `featurize` with `out` writes the feature vectors to disk, as in Part 6. On a Spark cluster, the inputs are Spark DataFrames read from storage that every machine of the cluster can read and write, as in Part 8. The section [Options for large runs](technical-guide.md#options-for-large-runs) of the technical guide describes the settings for both engines.

### What your files need

- *An id column.* Both tables need an id column of unique integers. `read_table` reads it as `_id`, and `id_col` names it when it has another name. Every id in the candidate set and the gold set must exist in its table. The `check_*` functions verify this before any work starts and raise an error that names the ids that do not.
- *Candidate pairs.* A flat set with one column of table A's ids and one of table B's, or the grouped form with one row per record of B and the list of the records of A paired with it. `read_pairs` reads either as `id1` and `id2`. It recognizes the id columns `a_id`/`b_id`, `ltable_id`/`rtable_id` and `left_id`/`right_id`, and `columns` names the two id columns when they have other names, as in `read_pairs('out.csv', columns=('product_a', 'product_b'))`. Ids are integers that are not negative. The section [Tables keyed by strings](technical-guide.md#tables-keyed-by-strings) of the technical guide describes how to give integer ids to a table keyed by strings.
- *Columns to compare.* The columns given to `create_features` must exist in both tables. A column whose values are numbers in at least 90% of the rows is detected as numeric and also gets a relative-difference feature.
- *Labels, a gold set, or a labeler.* Labeled pairs have two id columns and `label`, with `label` 1.0 for a match, 0.0 for a non-match and 2.0 for a pair labeled unsure, and are read with `read_pairs(path, with_label=True)`. A gold set has two id columns and is read with `read_pairs`. Without either, labels are collected from a person or from code, as in Parts 3 to 5.

The files are Parquet, CSV, TSV or JSON Lines files, or directories of Parquet files. The section [File formats](technical-guide.md#file-formats) of the technical guide describes them.

---

- [Technical guide](technical-guide.md). Every public function, with its arguments, return value and steps. It also covers the file formats, the features and tokenizers, the labelers, and the options for large runs.
- [Workflows](workflows.md). The four ways a run can start, and the example script for each.
- [How Iris works](how-it-works.md). The concepts behind each stage, and the two engines.
- [Troubleshooting](troubleshooting.md). The error messages, what they mean, and what to do.
