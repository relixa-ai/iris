"""Label a set of selected pairs, then train, apply and evaluate a matcher on those labels.

Run with --cli to label the pairs at the terminal, or --web to label them in a browser.
"""
import os
import sys

from xgboost import XGBClassifier

import iris

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA = os.path.join(ROOT, 'data', 'dblp_acm')
OUT = os.path.join(ROOT, 'examples', 'out', 'label')
os.makedirs(OUT, exist_ok=True)

table_a = iris.read_table(f'{DATA}/table_a.parquet')
table_b = iris.read_table(f'{DATA}/table_b.parquet')
candidates = iris.read_pairs(f'{DATA}/cand.parquet')
gold = iris.read_pairs(f'{DATA}/gold.parquet')
selected = iris.read_pairs(f'{DATA}/labeled_pairs.parquet')

if '--cli' in sys.argv:
    labeler = iris.CLILabeler(table_a, table_b)
    # -> CLILabeler     shows each pair at the terminal and reads the label typed
elif '--web' in sys.argv:
    labeler = iris.WebUILabeler(table_a, table_b)
    # -> WebUILabeler   shows each pair in a browser and takes the label clicked
else:
    labeler = iris.GoldLabeler(gold)
    # -> GoldLabeler    labels from the gold set, in place of a person

labels = iris.label_pairs(labeler, selected)
# -> pandas.DataFrame   id1, id2, label: one row per pair asked about
print(f'labeled {len(labels)} pairs, {int((labels["label"] == 1.0).sum())} matches')

columns = ['title', 'authors', 'venue', 'year']
features = iris.create_features(table_a, table_b, columns, columns)
labeled = iris.featurize(features, table_a, table_b, labels[['id1', 'id2']])
labeled = labeled.merge(labels[['id1', 'id2', 'label']], on=['id1', 'id2'])

model = iris.SKLearnModel(XGBClassifier, nan_fill=0.0, eval_metric='logloss',
                          objective='binary:logistic', max_depth=6, seed=42)
matcher = iris.train_matcher(model, labeled)
fvs = iris.featurize(features, table_a, table_b, candidates)
predictions = iris.apply_matcher(matcher, fvs, 'feature_vectors', 'prediction', 'confidence')

metrics = iris.evaluate(predictions, gold)
print(f"precision {metrics['precision']:.4f}  recall {metrics['recall']:.4f}  "
      f"f1 {metrics['f1']:.4f}")
iris.save_data(labels, f'{OUT}/labels.parquet')
