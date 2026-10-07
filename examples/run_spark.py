"""Run the matching functions on Spark, in local mode on this machine."""
import os

from xgboost import XGBClassifier

import iris
from iris.resources import build_session, plan

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA = os.path.join(ROOT, 'data', 'dblp_acm')
OUT = os.path.join(ROOT, 'examples', 'out', 'spark')
os.makedirs(OUT, exist_ok=True)
CHECKPOINT = f'{OUT}/labeled.parquet'
if os.path.exists(CHECKPOINT):
    os.remove(CHECKPOINT)

spark = build_session(plan())
# -> SparkSession              local mode, sized to the memory and cores of this machine
table_a = iris.read_table(f'{DATA}/table_a.parquet', df_type='sparkdf')
table_b = iris.read_table(f'{DATA}/table_b.parquet', df_type='sparkdf')
candidates = iris.read_pairs(f'{DATA}/cand.parquet', df_type='sparkdf')
gold = iris.read_pairs(f'{DATA}/gold.parquet', df_type='sparkdf')

iris.check_tables(table_a, table_b)
iris.check_candidates(candidates, table_a, table_b)
iris.check_gold_data(gold, table_a, table_b)

columns = ['title', 'authors', 'venue', 'year']
features = iris.create_features(table_a, table_b, columns, columns)
fvs = iris.featurize(features, table_a, table_b, candidates, out=f'{OUT}/fvs.parquet')
# -> pyspark.sql.DataFrame     id1, id2, feature_vectors, score; also written to out

pool = iris.down_sample(fvs, percent=0.3, seed=42)
# -> pyspark.sql.DataFrame     the rows of fvs drawn for labeling
labeler = iris.GoldLabeler(gold)
seeds = iris.create_seeds(pool, 50, labeler, checkpoint_path=CHECKPOINT)
model = iris.SKLearnModel(XGBClassifier, nan_fill=0.0, eval_metric='logloss',
                          objective='binary:logistic', max_depth=6, seed=42)
labeled = iris.label_al(model, 'batch', labeler, pool, seeds, checkpoint_path=CHECKPOINT,
                        batch_size=10, max_iter=50)
# -> pandas.DataFrame          id1, id2, feature_vectors, label, labeled_in_iteration

matcher = iris.train_matcher(model, labeled)
predictions = iris.apply_matcher(matcher, fvs, 'feature_vectors', 'prediction', 'confidence')
# -> pyspark.sql.DataFrame     the rows of fvs with prediction and confidence
metrics = iris.evaluate(predictions, gold)
print(f"precision {metrics['precision']:.4f}  recall {metrics['recall']:.4f}  "
      f"f1 {metrics['f1']:.4f}")
spark.stop()
