## Workflows

A matching run starts in one of four ways, depending on the data you already have. Each way is one script in `examples/`. The [tutorial](tutorial.md) runs these scripts on `data/dblp_acm`, and the [technical guide](technical-guide.md) describes every function they call.

| You have | Workflow | Script |
|---|---|---|
| A file of labeled pairs, such as one labeled earlier with Iris or by another tool | 1. Train from your labels | `examples/run_passive.py` |
| Pairs you want labeled, but no labels yet | 2. Label the pairs you selected | `examples/run_label.py` |
| Only the tables and the candidate set | 3. Let active learning select the pairs | `examples/run_al_gold.py` |
| A rule that decides some of the pairs | 4. Label by a rule, and by a labeler for the rest | `examples/run_custom_labeler.py` |

### 1. Train from your labels

*When to use it.* You have a file of pairs that are already labeled match or non-match, with two id columns and a column `label`.

*What happens.* The labeled pairs are converted into feature vectors and the matcher is trained on them directly. No labeler, pool or active learning is involved.

```python
labeled_pairs = iris.read_pairs('labels.csv', columns=('a_id', 'b_id'), with_label=True)
features = iris.create_features(table_a, table_b, columns, columns)
labeled  = iris.featurize(features, table_a, table_b, labeled_pairs[['id1', 'id2']])
labeled  = labeled.merge(labeled_pairs[['id1', 'id2', 'label']], on=['id1', 'id2'])
matcher  = iris.train_matcher(iris.SKLearnModel(XGBClassifier, nan_fill=0.0), labeled)
fvs      = iris.featurize(features, table_a, table_b, candidates)
preds    = iris.apply_matcher(matcher, fvs, 'feature_vectors', 'prediction', 'confidence')
metrics  = iris.evaluate(preds, gold)
```

*Note.* The matcher learns only what the labels show it. `train_matcher` needs at least one match and one non-match, and it leaves out the pairs labeled unsure.

### 2. Label the pairs you selected

*When to use it.* You know which pairs need labels, but the labels do not exist yet. The pairs may be a sample someone drew, every pair involving one record, or the pairs an earlier run predicted with low confidence.

*What happens.* `label_pairs` shows each pair to a labeler, in the order given, and collects the labels. The run then continues as in workflow 1.

```python
labeler = iris.CLILabeler(table_a, table_b)      # or WebUILabeler, GoldLabeler, or a CustomLabeler
labels  = iris.label_pairs(labeler, selected)
# then featurize, train_matcher, apply_matcher and evaluate as in workflow 1
```

*Note.* A pair labeled unsure stays in the result with the label 2.0, and `train_matcher` leaves it out. A stop ends the labeling, and the result holds the labels collected until then.

### 3. Let active learning select the pairs

*When to use it.* You have the two tables and the candidate set, and nothing else. This is the usual case.

*What happens.* `down_sample` draws a pool from the feature vectors. `create_seeds` collects the first labels from the most and the least similar pairs of the pool. `label_al` then runs in rounds. Each round trains a matcher on the labels collected so far, asks for labels on the pool pairs the matcher is least certain about, and adds them. The tutorial collects 550 labels, 50 seeds and then 50 rounds of 10 pairs.

```python
fvs     = iris.featurize(features, table_a, table_b, candidates)
pool    = iris.down_sample(fvs, percent=0.3, seed=42)
labeler = iris.GoldLabeler(gold)                 # a gold set here, CLILabeler or WebUILabeler for a person
seeds   = iris.create_seeds(pool, 50, labeler)
model   = iris.SKLearnModel(XGBClassifier, nan_fill=0.0)
labeled = iris.label_al(model, 'batch', labeler, pool, seeds, batch_size=10, max_iter=50)
matcher = iris.train_matcher(model, labeled)
# then apply_matcher and evaluate
```

*Note.* `create_seeds` and `label_al` write each label to a checkpoint file as it is collected, `active-matcher-training-data.parquet` in the current directory unless `checkpoint_path` names another. A session that is interrupted resumes from that file when it is run again. Delete the file to start afresh.

### 4. Label by a rule, and by a labeler for the rest

*When to use it.* Some pairs can be decided by a rule that you trust, such as pairs whose titles and years are equal, and the rest need a person or a gold set.

*What happens.* A subclass of `CustomLabeler` receives the two records of each pair. It labels the pairs its rule decides and passes the others to another labeler. Active learning then runs as in workflow 3, and every pair the rule labels is a pair the person does not have to label.

```python
class TitleAndYear(iris.CustomLabeler):
    def __init__(self, table_a, table_b, fallback):
        super().__init__(table_a, table_b)
        self.fallback = fallback

    def label_pair(self, row1, row2):
        if str(row1['title']).lower() == str(row2['title']).lower() and row1['year'] == row2['year']:
            return 1.0
        return self.fallback(row1['_id'], row2['_id'])

labeler = TitleAndYear(table_a, table_b, iris.CLILabeler(table_a, table_b))
# then create_seeds and label_al as in workflow 3
```

*Note.* The rule's labels are trusted as they are. A rule that labels some non-matches as matches teaches the matcher the same mistake.

### After any of the four

- To find the duplicates within one table, pass the same table object as A and as B, as in `featurize(features, table, table, candidates)`. Clean the candidate set with `clean_candidates(candidates, drop_self_pairs=True, canonicalize_mirrors=True)`, create the gold labeler with `symmetric=True`, and evaluate with `dedup=True`. `examples/run_dedup.py` shows this on a table of 1,000,000 songs.
- When the candidate set has at most 10,000,000 pairs, active learning can run over all of it rather than a pool. Skip `down_sample`, and pass the feature vectors to `create_seeds` and `label_al` as the pool. Active learning scores every pair of the pool in each round, so a larger pool makes each round slower, and `label_al` refuses a pool of more than 10,000,000 pairs.
- When the candidate set is larger than the memory of the machine, read the tables and the candidate set with `read_table` and `read_pairs` and `df_type='arrow'`, and pass `out` to `featurize` so that the feature vectors are written to disk. `examples/run_dedup.py` does both.
- To run any workflow on Spark, give the functions Spark DataFrames. `examples/run_spark.py` runs workflow 3 on Spark running on this machine only, and `examples/run_spark_cluster.py` runs it on a cluster.
- `save_features` and `save_data` save the features and the predictions. A saved set of features can be loaded with `load_features` to featurize new pairs with the same features in the same order, which a matcher trained on them requires.
