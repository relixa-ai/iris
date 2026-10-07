"""Train a matcher on labeled pairs from a file, apply it to the candidate set and evaluate it."""
import os

from xgboost import XGBClassifier

import iris

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA = os.path.join(ROOT, 'data', 'dblp_acm')
OUT = os.path.join(ROOT, 'examples', 'out', 'passive')
os.makedirs(OUT, exist_ok=True)

table_a = iris.read_table(f'{DATA}/table_a.parquet')
# -> pandas.DataFrame   _id and the columns to compare
table_b = iris.read_table(f'{DATA}/table_b.parquet')
candidates = iris.read_pairs(f'{DATA}/cand.parquet')
# -> pandas.DataFrame   id1, id2: one row per candidate pair
gold = iris.read_pairs(f'{DATA}/gold.parquet')
labeled_pairs = iris.read_pairs(f'{DATA}/labeled_pairs.parquet', with_label=True)
# -> pandas.DataFrame   id1, id2, label: the grouped file read as one row per pair

iris.check_tables(table_a, table_b)
# -> {"table_a": int, "table_b": int}  rows checked; raises ValueError on a bad id column
iris.check_candidates(candidates, table_a, table_b)
# -> int                                     candidate pairs checked
iris.check_gold_data(gold, table_a, table_b)
# -> int                                     gold pairs checked
iris.check_labeled_data(labeled_pairs, table_a, table_b)
# -> int                                     labeled pairs checked

columns = ['title', 'authors', 'venue', 'year']
features = iris.create_features(table_a, table_b, columns, columns)
# -> list[Feature]                           one entry per feature; str(f) names it

labeled = iris.featurize(features, table_a, table_b, labeled_pairs[['id1', 'id2']])
# -> pandas.DataFrame   id1, id2, feature_vectors (float32 arrays), score
labeled = labeled.merge(labeled_pairs[['id1', 'id2', 'label']], on=['id1', 'id2'])

model = iris.SKLearnModel(XGBClassifier, nan_fill=0.0, eval_metric='logloss',
                          objective='binary:logistic', max_depth=6, seed=42)
# -> SKLearnModel                            an untrained matcher
matcher = iris.train_matcher(model, labeled)
# -> SKLearnModel                            the matcher, trained on the labeled vectors

fvs = iris.featurize(features, table_a, table_b, candidates)
# -> pandas.DataFrame   id1, id2, feature_vectors, score: one row per candidate pair
predictions = iris.apply_matcher(matcher, fvs, 'feature_vectors', 'prediction', 'confidence')
# -> pandas.DataFrame   the rows of fvs with prediction (1 or 0) and confidence (float32)

metrics = iris.evaluate(predictions, gold)
# -> {"precision", "recall", "f1", "tp", "fp", "fn", "n_gold", "n_predicted",
#     "threshold_curve", "best_threshold", "best_f1"}
print(f"features {len(features)}  labels {len(labeled)}  candidates {len(fvs)}")
print(f"precision {metrics['precision']:.4f}  recall {metrics['recall']:.4f}  "
      f"f1 {metrics['f1']:.4f}")

iris.save_features(features, f'{OUT}/features.pkl')
# -> None
iris.save_data(predictions[['id1', 'id2', 'prediction', 'confidence']],
               f'{OUT}/predictions.parquet')
# -> None
