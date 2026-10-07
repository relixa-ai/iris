"""Find the duplicate songs in a table of 1,000,000 records on one machine, reading the
data from disk a chunk at a time."""
import os
import shutil

from xgboost import XGBClassifier

import iris

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA = os.path.join(ROOT, 'data', 'music_1m')
OUT = os.path.join(ROOT, 'examples', 'out', 'dedup')
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(f'{OUT}/scratch')
iris.configure(spill_dir=f'{OUT}/scratch')
CHECKPOINT = f'{OUT}/labeled.parquet'

table = iris.read_table(f'{DATA}/table_a.parquet', df_type='arrow')
# -> pyarrow.dataset.Dataset  nothing is read yet; the local engine reads it a chunk at a time
candidates = iris.read_pairs(f'{DATA}/cand.parquet', df_type='arrow')
gold = iris.read_pairs(f'{DATA}/gold.parquet')

iris.check_tables(table, table)
iris.check_candidates(candidates, table, table, dedup=True)
iris.check_gold_data(gold, table, table)

candidates = iris.clean_candidates(candidates, drop_self_pairs=True, canonicalize_mirrors=True)
# -> pyarrow.dataset.Dataset   distinct pairs with id1 < id2, self-pairs removed
columns = [c for c in table.schema.names if c != '_id']
features = iris.create_features(table, table, columns, columns)
fvs = iris.featurize(features, table, table, candidates, out=f'{OUT}/fvs.parquet')
# -> pyarrow.dataset.Dataset   id1, id2, feature_vectors, score, in out; the same table
#    as A and B makes this a deduplication

pool = iris.down_sample(fvs, percent=0.3, seed=42)
# -> pyarrow.Table             the pool, memory-mapped from disk
labeler = iris.GoldLabeler(gold, symmetric=True)
# -> GoldLabeler               symmetric: a gold pair matches in either order
seeds = iris.create_seeds(pool, 50, labeler, checkpoint_path=CHECKPOINT)
model = iris.SKLearnModel(XGBClassifier, nan_fill=0.0, eval_metric='logloss',
                          objective='binary:logistic', max_depth=6, seed=42)
labeled = iris.label_al(model, 'batch', labeler, pool, seeds, checkpoint_path=CHECKPOINT,
                        batch_size=10, max_iter=50)

matcher = iris.train_matcher(model, labeled)
predictions = iris.apply_matcher(matcher, fvs, 'feature_vectors', 'prediction', 'confidence')
metrics = iris.evaluate(predictions, gold, dedup=True)
# -> dict                      dedup=True orients the gold pairs as the candidates are
print(f"pairs {fvs.count_rows():,}  pool {pool.num_rows:,}  labels {len(labeled)}")
print(f"precision {metrics['precision']:.4f}  recall {metrics['recall']:.4f}  "
      f"f1 {metrics['f1']:.4f}")
