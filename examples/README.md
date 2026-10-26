## Examples of Running Iris

This directory contains examples of how to run Iris. Each example is a script, and each script is described in one part of the [tutorial](../doc/tutorial.md). Before proceeding, ensure that Iris has been [installed](../doc/installation/README.md).

Run each script from the `iris` directory after activating the Python environment with `source .venv/bin/activate`, for example `python examples/run_passive.py`. Each script writes its output under `examples/out/<name>/`. `run_spark_cluster.py` is the exception. It is submitted to a Spark cluster with `spark-submit` and writes its output to the location that its second argument names.

All scripts except `run_dedup.py` use the dataset `data/dblp_acm`, which consists of two tables of computer-science papers, a candidate set produced by a blocker, a gold set, and a file of 550 labeled pairs. `run_dedup.py` uses `data/music_1m`, which is a table of 1,000,000 songs and the candidate pairs produced by blocking the table against itself.

### `run_passive.py` (Part 1 of the tutorial)

This is the shortest complete workflow. It reads the tables with `read_table` and the candidate set, the gold set and the 550 labeled pairs with `read_pairs`, checks them, creates the features, featurizes the labeled pairs, trains a matcher on them, applies it to the candidate set, evaluates the result against the gold set, and saves the features and the predicted matches.

### `run_features.py` (Part 2)

This example shows what `create_features` selects. It prints the number of features for each column, featurizes three pairs, removes the features of one column, creates a larger set of features with the extra tokenizers, and saves and reloads a set of features.

### `run_label.py` (Part 3)

This example labels a set of pairs that you have selected. It labels the 550 pairs of the labeled-pairs file with `label_pairs`, and then trains, applies and evaluates the matcher as in `run_passive.py`. By default the labels come from a `GoldLabeler`. With the `--cli` flag a person labels the pairs at the terminal, and with `--web` in a browser.

### `run_al_gold.py` (Part 4)

This example uses active learning, starting with no labels. It draws a pool with `down_sample`, selects seed pairs with `create_seeds`, runs `label_al` with a `GoldLabeler`, which labels each pair from the gold set, in place of a person, and then trains, applies and evaluates the matcher.

### `run_custom_labeler.py` (Part 5)

This example defines a labeler of its own, a subclass of `CustomLabeler` that labels a pair as a match when the titles and the years are equal and asks a `GoldLabeler` otherwise. It then runs active learning as in `run_al_gold.py` and reports how many labels the rule gave.

### `run_dedup.py` (Part 6)

This example finds the duplicates in a table of 1,000,000 songs on one machine. It reads the table and the 33,174,998 candidate pairs with `read_table` and `read_pairs` as Arrow datasets, which are read from disk a chunk at a time, removes self-pairs and mirrored pairs with `clean_candidates`, writes the feature vectors to disk, and runs active learning, matching and evaluation as in `run_al_gold.py`.

### `run_spark.py` (Part 7)

This example runs the functions of `run_al_gold.py` on Spark, with Spark running on this machine only. The inputs are read as Spark DataFrames, with `df_type='sparkdf'`, so each function runs on Spark.

### `run_spark_cluster.py` (Part 8)

This example runs the same functions on a Spark cluster. It is submitted with `spark-submit`, and its two arguments name the dataset and a new location for the outputs. On a cluster both are URIs of storage that every machine of the cluster can read and write, such as `s3a://`, `gs://`, `abfss://` or `hdfs://`.
