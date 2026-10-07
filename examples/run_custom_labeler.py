"""Label pairs with a rule where it applies and with another labeler elsewhere, by
active learning."""
import os

from xgboost import XGBClassifier

import iris

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA = os.path.join(ROOT, 'data', 'dblp_acm')
OUT = os.path.join(ROOT, 'examples', 'out', 'custom_labeler')
os.makedirs(OUT, exist_ok=True)
CHECKPOINT = f'{OUT}/labeled.parquet'
if os.path.exists(CHECKPOINT):
    os.remove(CHECKPOINT)


class TitleAndYear(iris.CustomLabeler):
    """Label a pair 1.0 when title and year are equal, and ask `fallback` otherwise."""

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


table_a = iris.read_table(f'{DATA}/table_a.parquet')
table_b = iris.read_table(f'{DATA}/table_b.parquet')
candidates = iris.read_pairs(f'{DATA}/cand.parquet')
gold = iris.read_pairs(f'{DATA}/gold.parquet')

columns = ['title', 'authors', 'venue', 'year']
features = iris.create_features(table_a, table_b, columns, columns)
fvs = iris.featurize(features, table_a, table_b, candidates)
pool = iris.down_sample(fvs, percent=0.3, seed=42)

labeler = TitleAndYear(table_a, table_b, iris.GoldLabeler(gold))
# -> TitleAndYear       callable(id1, id2) -> 1.0 or 0.0, from the rule or the gold set
seeds = iris.create_seeds(pool, 50, labeler, checkpoint_path=CHECKPOINT)
model = iris.SKLearnModel(XGBClassifier, nan_fill=0.0, eval_metric='logloss',
                          objective='binary:logistic', max_depth=6, seed=42)
labeled = iris.label_al(model, 'batch', labeler, pool, seeds, checkpoint_path=CHECKPOINT,
                        batch_size=10, max_iter=50)
# -> pandas.DataFrame   id1, id2, feature_vectors, label, labeled_in_iteration

matcher = iris.train_matcher(model, labeled)
predictions = iris.apply_matcher(matcher, fvs, 'feature_vectors', 'prediction', 'confidence')
metrics = iris.evaluate(predictions, gold)
print(f'labels {len(labeled)}, {labeler.by_rule} by the rule')
print(f"precision {metrics['precision']:.4f}  recall {metrics['recall']:.4f}  "
      f"f1 {metrics['f1']:.4f}")
