"""Show the features `create_features` selects, change the set, and save and reload it."""
import os
from collections import Counter


import iris

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA = os.path.join(ROOT, 'data', 'dblp_acm')
OUT = os.path.join(ROOT, 'examples', 'out', 'features')
os.makedirs(OUT, exist_ok=True)

table_a = iris.read_table(f'{DATA}/table_a.parquet')
table_b = iris.read_table(f'{DATA}/table_b.parquet')
candidates = iris.read_pairs(f'{DATA}/cand.parquet')

columns = ['title', 'authors', 'venue', 'year']
features = iris.create_features(table_a, table_b, columns, columns)
# -> list[Feature]   one entry per feature; f.a_attr and f.b_attr name its columns
print(f'{len(features)} features')
for column, n in Counter(f.a_attr for f in features).items():
    print(f'  {column:8} {n}')
for f in features[:6]:
    print(f'  {f}')

three = iris.featurize(features, table_a, table_b, candidates.head(3))
# -> pandas.DataFrame   id1, id2, feature_vectors (float32 arrays), score
for row in three.itertuples():
    print(f'pair ({row.id1}, {row.id2}): {len(row.feature_vectors)} values, score {row.score:.3f}')

without_venue = [f for f in features if f.a_attr != 'venue']
print(f'without venue: {len(without_venue)} features')

tokenizers = iris.get_base_tokenizers() + iris.get_extra_tokenizers()
# -> list[Tokenizer]   the base tokenizers, then the extra ones
larger = iris.create_features(table_a, table_b, columns, columns, tokenizers=tokenizers)
print(f'with the extra tokenizers: {len(larger)} features')

path = f'{OUT}/features.pkl'
iris.save_features(features, path)
# -> None
reloaded = iris.load_features(path)
# -> list[Feature]   the features as saved
print(f'reloaded {len(reloaded)} features, the same: '
      f'{[str(f) for f in reloaded] == [str(f) for f in features]}')
