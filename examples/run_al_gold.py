"""Collect labels by active learning, with the gold set as the labeler, then train,
apply and evaluate."""
import os

from xgboost import XGBClassifier

import iris

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA = os.path.join(ROOT, 'data', 'dblp_acm')
OUT = os.path.join(ROOT, 'examples', 'out', 'al_gold')
os.makedirs(OUT, exist_ok=True)
CHECKPOINT = f'{OUT}/labeled.parquet'
if os.path.exists(CHECKPOINT):
    os.remove(CHECKPOINT)

table_a = iris.read_table(f'{DATA}/table_a.parquet')
table_b = iris.read_table(f'{DATA}/table_b.parquet')
candidates = iris.read_pairs(f'{DATA}/cand.parquet')
gold = iris.read_pairs(f'{DATA}/gold.parquet')

iris.check_tables(table_a, table_b)
iris.check_candidates(candidates, table_a, table_b)
iris.check_gold_data(gold, table_a, table_b)

columns = ['title', 'authors', 'venue', 'year']
features = iris.create_features(table_a, table_b, columns, columns)
fvs = iris.featurize(features, table_a, table_b, candidates)
# -> pandas.DataFrame   id1, id2, feature_vectors, score: one row per candidate pair

pool = iris.down_sample(fvs, percent=0.3, seed=42)
# -> pandas.DataFrame   the rows of fvs drawn for labeling, across the range of scores
labeler = iris.GoldLabeler(gold)
# -> GoldLabeler  callable(id1, id2) -> 1.0 match, 0.0 non-match; stands in for a person
seeds = iris.create_seeds(pool, 50, labeler, checkpoint_path=CHECKPOINT)
# -> pandas.DataFrame   50 rows of the pool with label; saved to the checkpoint as labeled

model = iris.SKLearnModel(XGBClassifier, nan_fill=0.0, eval_metric='logloss',
                          objective='binary:logistic', max_depth=6, seed=42)
labeled = iris.label_al(model, 'batch', labeler, pool, seeds, checkpoint_path=CHECKPOINT,
                        batch_size=10, max_iter=50)
# -> pandas.DataFrame  id1, id2, feature_vectors, label, labeled_in_iteration; seeds included

matcher = iris.train_matcher(model, labeled)
predictions = iris.apply_matcher(matcher, fvs, 'feature_vectors', 'prediction', 'confidence')
metrics = iris.evaluate(predictions, gold)
print(f"pool {len(pool)}  labels {len(labeled)}  "
      f"positive {int(labeled['label'].sum())}")
print(f"precision {metrics['precision']:.4f}  recall {metrics['recall']:.4f}  "
      f"f1 {metrics['f1']:.4f}")

iris.save_predictions(predictions, f'{OUT}/matches.parquet')
# -> int                       the predicted matches written, as id1, id2 and probability
