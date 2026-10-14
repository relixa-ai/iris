## How Iris works

This page explains the ideas behind a matching run. Read it after the [tutorial](tutorial.md) and before the [technical guide](technical-guide.md), which describes the functions one by one. You can skip the first section if you have worked on entity matching before.

### The matching step

Let A and B be two tables to be matched. The goal is to find the pairs (x, y), where x is a row of A and y is a row of B, that refer to the same real-world entity. Such a pair is a *match*. Comparing every row of A with every row of B is not feasible for large tables, since two tables of a million rows each form a trillion pairs. Matching is therefore done in two steps.

*Blocking* uses a fast heuristic, such as requiring the two rows to share enough words, to select the pairs that could plausibly be matches. Its output is the *candidate set* C.

*Matching* examines each pair in C more carefully and decides match or non-match. Iris does this with a supervised classifier called the *matcher*. The matcher takes a numeric description of a pair, called its *feature vector*, and returns whether the pair is a match, with a confidence. The matching step consists of the following:

1. Create a set of *features*. A feature compares the two records of a pair and produces a number. Iris derives the features from the columns of A and B.
2. Convert every pair in C into a feature vector.
3. Obtain *labeled pairs*, which are pairs labeled match or non-match by a person, a gold set, or code, together with their feature vectors. Train the matcher on them.
4. Apply the matcher to every feature vector from step 2.

When there are no labeled pairs, Iris adds a step between steps 2 and 3 that collects labels by *active learning*, which is described below.

### The inputs

Each table has an integer column `_id` that identifies its records. The values of `_id` are unique within a table and not negative. The candidate set refers to the records by these ids. In a *flat* candidate set each row is one pair, with the columns `id1`, which is an `_id` of A, and `id2`, which is an `_id` of B. In a *grouped* candidate set each row holds one record of B in `id2` and the records of A it was paired with in the list `id1_list`. Blockers write the flat form by default, and some write the grouped form. A blocker's own score of a pair, when it gives one, is in the column `blocking_score`, and Iris does not use it. `read_table` and `read_pairs` read the tables and the pairs from Parquet, CSV, TSV or JSON Lines files and give them these column names. `read_pairs` recognises the names that earlier blockers give their id columns, such as `a_id` and `b_id`, and other names are given to it. `check_tables`, `check_candidates` and `check_gold_data` check that the inputs have these shapes and that every id in a pair names a record of its table.

To find the duplicates within one table, the table is matched against itself. This is *deduplication*. Iris treats a run as a deduplication when the same table object is passed as A and as B, as in `featurize(features, table, table, candidates)`. Two separate reads of one file are two tables. A blocker run on one table can pair a record with itself, and can produce a pair in both orders, as (x, y) and (y, x). Such pairs mislead the matcher. A record paired with itself agrees on every feature, so it ranks at the top when the seeds are drawn by score, although it is not a match between two records. A pair in both orders has the same feature vector twice, so it can be labeled twice. In a deduplication, `featurize` therefore drops the pairs of a record with itself and featurizes each pair once, as (smaller id, larger id). `clean_candidates` does the same to the candidate set itself, and turns it into distinct flat pairs. The section [clean_candidates](technical-guide.md#clean_candidates) of the technical guide describes the options.

### Features

A feature compares one column of A with the same column of B and returns a number. `create_features` reads the tables and selects the features by the following rules:

1. A column that is empty in half or more of the rows of A and B, counted together, is not used.
2. Every column used gets an *exact-match* feature, which is 1.0 when the two values are equal, ignoring case, and 0.0 otherwise.
3. A column whose type is numeric, or whose values parse as numbers in at least 90% of the rows, also gets a *relative-difference* feature.
4. A column's text is broken into *tokens* by *tokenizers*. The base tokenizers produce the words of the text, the numbers in it, and its 3-character fragments. For each tokenizer under which a column averages at least three tokens, the column gets five *similarity* features, namely TF/IDF, Jaccard, SIF, overlap coefficient and cosine, each comparing the two sets of tokens.

A column with one word per row has one token under the word tokenizer, and a similarity over one token is equivalent to an exact match, which is why rule 4 requires three tokens per tokenizer. On the tutorial's dataset of papers, the columns title, authors and venue get 11 features each, and the column year gets 2, for 35 in total. More tokenizers are available, such as 5-character fragments. With the alphanumeric word tokenizer, a column that averages at most 10 words also gets the Monge-Elkan, edit-distance and Smith-Waterman similarities. The section [Features and tokenizers](technical-guide.md#features-and-tokenizers) of the technical guide lists all of them.

TF/IDF and SIF weigh a token by how rare it is in the tables, so they need a *vocabulary* of the tokens and their frequencies. `featurize` builds the vocabulary from the records of A and B. In a deduplication each record is counted once, not once as a record of A and again as a record of B.

A matcher depends on the features, with the same features in the same order. A matcher trained on feature vectors of one set of features can only be applied to feature vectors of the same set. `save_features` and `load_features` keep a set of features for later runs.

### Feature vectors and the score

`featurize` computes the feature vector of every pair in C. When either record's value of a column is empty, the features of that column cannot be computed for the pair, and `featurize` sets them to 0.0, or to the value given as `fill_na`. A pair that appears more than once in C is featurized once.

Along with each feature vector, `featurize` computes a *score*, which is the sum of the pair's similarity and exact-match features. Pairs whose records agree have high scores. The score is not a prediction. It is used to rank the pairs when the pool and the seeds are drawn, as described below.

### The pool

Active learning trains a matcher many times and scores the pairs it may ask about each time. Doing this over all of C is not feasible when C is large, and only a few hundred pairs are labeled in any case. Active learning therefore runs over a *pool*, which is a sample of C. `down_sample` draws a pool of a given fraction of C, and of at most 10,000,000 pairs by default. When C has at most 10,000,000 pairs, the pool can be all of C. The feature vectors are then passed to active learning directly, without `down_sample`.

The pool cannot be a uniform random sample of C. Matches are rare in C, and a uniform sample would hold too few of them to learn from. `down_sample` therefore draws a *stratified* sample. It divides C into buckets by a hash of each pair. Within each bucket, the pairs whose scores are in the top share of the bucket form the *high-scoring band*, where the share is the fraction of C requested. From every bucket `down_sample` draws, at random, as many pairs from the band as from the other pairs. About half of the pool therefore consists of high-scoring pairs, where the likely matches are, and the other half of the rest of C. The draw depends only on the pairs, their scores and the seed, so the same seed gives the same pool.

### Active learning

Once the pool has been drawn, a *labeler* labels pairs from it. A labeler labels each pair it is given as a match (1.0) or a non-match (0.0). It can also label a pair as *unsure* (2.0), in which case the pair is set aside and not asked again, or *stop* (−1.0), which ends the labeling and keeps the labels collected so far. Active learning selects which pairs to ask about, in two stages.

1. *Seeds.* `create_seeds` ranks the pool by score and asks for labels from both ends, alternating, until it has the requested number of labels and they include at least one match and one non-match. The high end supplies matches and the low end supplies non-matches. The tutorial uses 50 seeds.
2. *Rounds.* `label_al` trains a matcher on all labels collected so far. It then scores every pair of the pool and selects the 10 pairs whose predictions are the least certain, by the entropy of the predicted probabilities. It asks the labeler for their labels, adds them, and repeats for 50 rounds by default. With 50 seeds, the default budget is therefore 50 + 50 × 10 = 550 labels.

The least certain pairs are selected because a pair the matcher already predicts with a probability of 0.99 or 0.01 would teach it nothing new. The pairs near 0.5 are where the matcher's decision boundary lies, and each label there moves the boundary. This is what allows a few hundred labels to train an accurate matcher. Pairs with the same entropy are ordered by a hash of the pair, so a run with the same labels selects the same pairs. The `selection` option can instead select each batch at random, or half by uncertainty and half at random.

*Batch and continuous mode.* In batch mode, described above, a person labels ten pairs, waits while the matcher is retrained and the pool rescored, and receives the next ten. In continuous mode the matcher is retrained in the background while the person labels, so there is no wait, but some pairs are selected by a matcher that is a few labels out of date. Batch mode is the default.

*The checkpoint.* `create_seeds` and `label_al` write every label to a checkpoint file as it is collected, by default `active-matcher-training-data.parquet` in the current directory. When the file exists, labeling resumes from the labels in it, so a session that was interrupted continues where it stopped. To start afresh, delete the file or name another one.

### Labelers

Iris includes labelers that label from a gold set (`GoldLabeler`), at the terminal (`CLILabeler`), or in a browser (`WebUILabeler`). A subclass of `CustomLabeler` labels with your own code, for example by a rule that labels the pairs it can decide and asks another labeler about the rest. `label_pairs` labels a given set of pairs without active learning. The section [Labelers](technical-guide.md#labelers) of the technical guide describes each one.

### The matcher

The matcher is any classifier with the interface of scikit-learn, such as an XGBoost or a random forest classifier, wrapped in `SKLearnModel` together with its arguments. On Spark a classifier of Spark ML can be used instead, wrapped in `SparkMLModel`. `train_matcher` trains the matcher on labeled feature vectors, leaving out the pairs labeled unsure. `apply_matcher` applies it to feature vectors and adds to each pair a *prediction*, which is 1.0 for a match and 0.0 for a non-match, and the *confidence* of the prediction.

### Evaluation

A *gold set* is a set of pairs that are known to be matches. Given a gold set, `evaluate` counts the predicted matches that are in the gold set (true positives), the predicted matches that are not (false positives), and the gold pairs that were not predicted (false negatives). From these it computes precision `tp / (tp + fp)`, recall `tp / (tp + fn)`, and F1, which is their harmonic mean. In a deduplication the pairs (x, y) and (y, x) count as one pair.

A pair is a predicted match when its probability of being a match is at or above a threshold, 0.5 by default. Raising the threshold trades recall for precision. `evaluate` also reports precision, recall and F1 at every threshold from 0.0 to 1.0 in steps of 0.05, and the threshold with the best F1, so that the trade-off can be read from the curve rather than guessed.

Recall is measured against the whole gold set, including the gold pairs that blocking did not put in C. Those pairs cannot be found by any matcher, so a run's recall is at most the blocker's recall. If recall is lower than you expect, compare it with the blocker's recall on the same gold set before changing anything in the matcher.

A gold set is also what `GoldLabeler` labels from, which is how the tutorial runs without a person. This is a fair test of the run only when the gold set is complete, that is, when it lists every true match. A gold set that lists only some of the true matches teaches the matcher that the others are non-matches, and the result is low recall.

### The two engines

Each function of Iris runs on one of two engines, selected by the kind of data it is given. All the data of one run is passed in one kind, pandas DataFrames or Arrow datasets for the local engine, or Spark DataFrames for the Spark engine.

*The local engine* runs when the functions are given pandas DataFrames or Arrow datasets. It needs neither Spark nor Java. `featurize` packs the records of A and B into a *record store* on disk and sorts the candidate pairs into parts by their ids. *Worker processes* then featurize the parts in parallel, each reading the records its pairs need from the store, and write the feature vectors to disk. Each worker holds one part at a time, so the memory a run needs depends on the number of workers and not on the size of C, and a candidate set can be larger than the memory of the machine. The local engine starts up to one worker per core, and fewer when the memory of the machine cannot hold that many. It writes its scratch files to disk, in the system's temporary directory by default, so a run needs free disk space rather than memory.

*The Spark engine* runs when the functions are given Spark DataFrames, either on one machine, with Spark running in its *local mode* without a cluster, or on a cluster. `featurize` joins the candidate pairs with the records of A and then with the records of B, so that each pair arrives at a Spark task with its two records, and the tasks featurize the pairs in Python. The data and the outputs are named by paths or by URIs of storage that every machine of the cluster can read and write, such as `s3a://` for Amazon S3 or `hdfs://`, so the machines need no shared file system. For the joins, Spark moves the records between the machines in a step called the *shuffle*, which writes scratch files to the disk of every worker of the cluster.

The section [Options for large runs](technical-guide.md#options-for-large-runs) of the technical guide describes the settings of each engine.

### What determines the result

When a run's precision or recall is lower than expected, the possible causes are listed below, in the order in which to check them.

1. *Blocking recall.* A gold pair that is not in C cannot be found. Checking how many gold pairs are in C gives the upper bound on recall.
2. *The candidate set of a deduplication.* A candidate set that holds pairs of a record with itself, or the same pair in both orders, misleads the matcher. `featurize` removes both when the same table is passed as A and as B, and not when the table is passed as two separate reads.
3. *The labels.* Active learning trains on what the labeler says. A gold set that is incomplete, or a person who is inconsistent, trains a worse matcher than a smaller but correct set of labels would.
4. *The label budget.* The default of 550 labels worked on the datasets included with Iris. Data on which matches and non-matches look more alike may need more rounds or a larger pool.
5. *The columns.* A column that is mostly empty, or that does not differ between matches and non-matches, adds noise. Pass only the columns that identify an entity to `create_features`.
6. *The threshold.* Read the curve that `evaluate` returns before changing the threshold. Change it only when the curve shows a higher F1 at another threshold.
