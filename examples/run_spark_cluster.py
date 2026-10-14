"""Run the matching functions on a Spark cluster, with the data and the outputs named by URIs.

Submit with: spark-submit --master spark://HOST:7077 examples/run_spark_cluster.py DATA OUT
DATA holds table_a.parquet, table_b.parquet, cand.parquet and gold.parquet, and OUT is a
new location for the outputs, for example s3a://bucket/datasets/dblp_acm and
s3a://bucket/datasets/dblp_acm/runs/1.
"""
import sys

from pyspark.sql import SparkSession
from xgboost import XGBClassifier

import iris

data, out = sys.argv[1].rstrip('/'), sys.argv[2].rstrip('/')
spark = SparkSession.builder.appName('iris').getOrCreate()
# -> SparkSession              the session spark-submit started on the cluster

table_a = iris.read_table(f'{data}/table_a.parquet', df_type='sparkdf')
table_b = iris.read_table(f'{data}/table_b.parquet', df_type='sparkdf')
candidates = iris.read_pairs(f'{data}/cand.parquet', df_type='sparkdf')
gold = iris.read_pairs(f'{data}/gold.parquet', df_type='sparkdf')

iris.check_tables(table_a, table_b)
iris.check_candidates(candidates, table_a, table_b)
iris.check_gold_data(gold, table_a, table_b)

columns = [c for c in table_a.columns if c != '_id' and c in table_b.columns]
features = iris.create_features(table_a, table_b, columns, columns)
fvs = iris.featurize(features, table_a, table_b, candidates, out=f'{out}/fvs.parquet')
# -> pyspark.sql.DataFrame     id1, id2, feature_vectors, score; written to OUT/fvs.parquet

pool = iris.down_sample(fvs, percent=0.3, seed=42)
labeler = iris.GoldLabeler(gold)
seeds = iris.create_seeds(pool, 50, labeler, checkpoint_path=f'{out}/labeled.parquet')
model = iris.SKLearnModel(XGBClassifier, nan_fill=0.0, eval_metric='logloss',
                          objective='binary:logistic', max_depth=6, seed=42)
labeled = iris.label_al(model, 'batch', labeler, pool, seeds,
                        checkpoint_path=f'{out}/labeled.parquet', batch_size=10, max_iter=50)

matcher = iris.train_matcher(model, labeled)
predictions = iris.apply_matcher(matcher, fvs, 'feature_vectors', 'prediction', 'confidence')
# -> pyspark.sql.DataFrame     the rows of fvs with prediction and confidence
iris.save_predictions(predictions, f'{out}/matches.parquet')
# -> int                       the predicted matches written, as id1, id2 and probability
metrics = iris.evaluate(predictions, gold)
print(f"precision {metrics['precision']:.4f}  recall {metrics['recall']:.4f}  "
      f"f1 {metrics['f1']:.4f}")
spark.stop()
