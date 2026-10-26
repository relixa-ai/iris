## How to Use Iris

Iris provides the matching process as a set of Python functions. The functions fall into seven groups, listed below. The [technical guide](technical-guide.md) describes each function in detail.

| Need to do | Functions |
| --- | --- |
| Read, write and check the data | `read_table`, `read_pairs`, `load_data`, `save_data`, `save_predictions`, `check_tables`, `check_candidates`, `check_gold_data`, `check_labeled_data`, `check_pairs`, `clean_candidates` |
| Convert the output of a blocker | `candidates_from_flat`, `candidates_from_result`, `write_flat_candidates` |
| Featurize pairs | `create_features`, `featurize`, `get_base_sim_functions`, `get_base_tokenizers`, `get_extra_tokenizers`, `save_features`, `load_features` |
| Label pairs | `label_pairs`, `down_sample`, `create_seeds`, `label_al`, `GoldLabeler`, `DelayedGoldLabeler`, `CLILabeler`, `WebUILabeler`, `CustomLabeler` |
| Train, apply and evaluate a matcher | `SKLearnModel`, `SparkMLModel`, `train_matcher`, `apply_matcher`, `evaluate` |
| Extend Iris with your own classes | `Labeler`, `Feature`, `Tokenizer`, `Vectorizer`, `MLModel` |
| Configure a run | `configure`, `load_env` |

### How It Works

If you want to start by understanding the conceptual working of Iris, read [How Iris works](how-it-works.md). It describes the matching step, the features, the pool and active learning, the matcher, evaluation, and the two engines that run the functions, one on a single machine and one on Spark.

### Tutorial and Examples

The [tutorial](tutorial.md) is a step-by-step guide to using Iris, in eight parts. Each part is built around a script in the `examples` directory. The [examples page](../examples/README.md) describes each script.

We recommend reading the tutorial first and then running the examples.

### Technical Guide

The [technical guide](technical-guide.md) describes every public function of Iris, in the order a run calls them. For each function it gives the arguments, the steps it performs, the return value, and an example. The guide also describes the file formats, the features and tokenizers, the labelers, the settings, and the options for large runs and for Spark clusters.

### Workflows and Troubleshooting

[Workflows](workflows.md) describes the four ways a run can start, depending on the data you have, and the example script for each. [Troubleshooting](troubleshooting.md) lists the error messages Iris raises, what each means, and what to do.
