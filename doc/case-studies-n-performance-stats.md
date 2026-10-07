## Case Studies and Performance Statistics

### The datasets included with Iris

The [tutorial](tutorial.md) runs on the two datasets in the `data` directory. The numbers below were obtained on a Mac with 18 cores and 48 GB of memory. On another machine the results can differ slightly, because active learning can choose differently between pairs whose uncertainty differs only in the last digits of a floating-point number.

`data/music_1m` is a table of 1,000,000 songs that is matched against itself to find duplicates. The candidate set from the blocker has 33,174,998 pairs and contains 98.92% of the 146,011 gold duplicate pairs. The script `examples/run_dedup.py` draws a pool of 9,999,928 pairs, collects 550 labels by active learning, and applies the matcher to all 33,174,998 pairs, with precision 0.9991, recall 0.9235 and an F1 of 0.9598. It took 3 minutes. Iris ran 18 worker processes, one per core of that machine.

`data/dblp_acm` consists of two tables of computer-science papers, with 2,616 and 2,294 rows, a candidate set of 114,700 pairs that contains all 2,220 gold matches, and a file of 550 labeled pairs. The results of the tutorial's runs on it are as follows.

| Run | Labels | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Train from labels in a file (`run_passive.py`) | 550 | 0.9857 | 0.9937 | 0.9897 |
| Label selected pairs (`run_label.py`) | 550 | 0.9857 | 0.9937 | 0.9897 |
| Active learning from a gold labeler (`run_al_gold.py`) | 550 | 0.9941 | 0.9851 | 0.9896 |
| A rule labels the pairs it can, a labeler the rest (`run_custom_labeler.py`) | 550 (160 by the rule) | 0.9778 | 0.9901 | 0.9839 |
| Active learning on Spark, running on the same machine without a cluster (`run_spark.py`) | 550 | 0.9941 | 0.9851 | 0.9896 |
