## Iris: Machine-Learning Matching for Entity Matching

Iris is a tool for the matching step of entity matching. Entity matching finds tuples from two tables A and B that match, that is, refer to the same real-world entity. It typically proceeds in two steps. *The blocking step* uses heuristics to quickly identify a relatively small set of tuple pairs that can be matches. *The matching step* examines each surviving pair more carefully and predicts match or non-match. (See [this page](https://anhaidgroup.github.io/magellan/about) for details.)

Iris focuses on the matching step, and is distinguished in four aspects:

* It trains a matcher, such as XGBoost or a random forest, on labeled pairs. When no labels are available, it collects them by active learning. On a table of 1,000,000 songs, 550 labels give an F1 of 0.9598 over 33,174,998 candidate pairs.
* It runs on one machine. It splits the work among Python processes, up to one per core, and can read the tables and the candidate pairs from disk a chunk at a time, so a candidate set can be larger than the memory of the machine.
* The same functions run on a Spark cluster when they are given Spark DataFrames. The data and the outputs are then kept on storage that every machine of the cluster can read and write, such as Amazon S3.
* Labels can be read from a file or a gold set, entered by a person at the terminal or in a browser, or produced by user-written code.

Iris takes as input the candidate set produced by a blocker. For the blocking step, see [Sparkly](https://github.com/anhaidgroup/sparkly) and [Delex](https://github.com/anhaidgroup/delex). Iris builds on MatchFlow, the matcher of the [Magellan project](https://anhaidgroup.github.io/magellan) at the University of Wisconsin-Madison.

### How Iris Works

Let A and B be the two tables to be matched, and let C be the candidate set that a blocker produced from them. Iris first examines the columns of A and B and builds a set of *features*. For each column it creates an exact-match feature, a numeric feature where the values are numbers, and a set of similarity features such as TF/IDF, Jaccard and cosine, computed over the tokenized values. It then converts every pair in C into a *feature vector*, which is a numeric description of how the two records compare.

Iris then trains a *matcher*, a supervised classifier that takes a feature vector and returns the probability that the pair is a match. The matcher is trained on labeled pairs. If labeled pairs are already available, the matcher is trained on them directly. Otherwise Iris draws a sample of pairs from C, called the *pool*, and collects labels by active learning. In each round it trains a matcher on the labels collected so far, selects the pairs of the pool that the matcher is least certain about, and asks a *labeler*, which is a person, a gold set, or code, to label them. Finally, Iris applies the matcher to every pair in C and, when a *gold set* of the pairs known to match is available, evaluates the result against it.

Implementation-wise, Iris is a Python package with a function for each stage of the process, such as checking the inputs, creating the features, featurizing the pairs, drawing the pool, collecting labels, and training, applying and evaluating the matcher. You combine these functions into the workflow that suits your data. Each function runs in one of two ways, which we call *engines*. Given pandas DataFrames or Arrow datasets, it runs on the *local engine*, which uses Python processes on one machine. Given Spark DataFrames, it runs on the *Spark engine*, as Spark jobs on a cluster or on one machine. See [this page](doc/how-it-works.md) for details.

### Case Studies and Performance Statistics

On the datasets included with Iris, it finds the duplicates in a table of 1,000,000 songs, from 33,174,998 candidate pairs and 550 labels, with an F1 of 0.9598, in 3 minutes on a Mac with 18 cores and 48 GB of memory. On two tables of computer-science papers it reaches an F1 of 0.9896 with 550 labels collected by active learning. See [this page](doc/case-studies-n-performance-stats.md) for details.

### Installation

A run needs Python 3.10, 3.11 or 3.12 on Linux, macOS or Windows. A run on Spark also needs Java 17 and Iris installed with Spark support, an optional part of Iris that adds PySpark. See instructions to install Iris on [Linux](doc/installation/install-linux.md), [macOS](doc/installation/install-macos.md), [Windows](doc/installation/install-windows.md) or [a Spark cluster](doc/installation/install-spark-cluster.md). [This page](doc/installation/README.md) explains which installation a run needs.

### How to Use

See [this page](doc/using-iris.md), which points to a tutorial, examples, and the technical guide.

### Further Pointers

See the [technical guide](doc/technical-guide.md) for API documentation.
For questions / comments, contact [our team](mailto:hello@relixa.ai).
