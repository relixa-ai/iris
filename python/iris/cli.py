"""The `iris` command, with the subcommands `info` and `smoke`."""
from __future__ import annotations

import argparse
import sys


def _cmd_info(args):
    import importlib.metadata
    import shutil
    import tempfile

    from . import __version__
    from ._core.storage import spill_dir
    from ._local.procpool import workers_for
    from .resources import budget

    total_gb, cores, source = budget()
    print(f'iris      {__version__}')
    print(f'python    {sys.version.split()[0]}')
    print(f'machine   {total_gb:.1f} GiB, {cores} cores ({source})')
    print(f'local     up to {workers_for()} worker processes, scratch in '
          f'{spill_dir() or tempfile.gettempdir()}')
    try:
        pyspark = importlib.metadata.version('pyspark')
    except importlib.metadata.PackageNotFoundError:
        print("spark     not installed (pip install 'iris[spark]')")
    else:
        java = shutil.which('java')
        print(f'spark     pyspark {pyspark}, java {java if java else "not found on PATH"}')
    return 0


def _cmd_smoke(args):
    import pandas as pd

    from . import __version__
    from .tools import apply_matcher, create_features, featurize, train_matcher
    from ._core.ml_model import SKLearnModel

    print(f'iris {__version__}')
    print(f'python    {sys.version.split()[0]}')

    spark = None
    if args.engine == 'spark':
        from ._spark.utils import check_java_version
        from .resources import build_session, plan

        spark = build_session(plan(master='local[1]', driver_memory='1g'),
                              app_name='iris-smoke', shuffle_partitions=8)
    try:
        if spark is not None:
            check_java_version(spark)
            java = spark.sparkContext._jvm.System.getProperty('java.version')
            print(f'java      {java}')
            print(f'spark     {spark.version}')
        else:
            from ._local.procpool import workers_for
            print(f'engine    local, up to {workers_for()} worker processes')

        titles = ['deep learning for entity matching', 'a survey of record linkage',
                  'scalable blocking methods', 'probabilistic data integration',
                  'string similarity joins', 'learning to match records',
                  'schema mapping at scale', 'duplicate detection in databases']
        years = [2018, 1999, 2011, 2004, 2016, 2020, 2007, 2012]
        a = pd.DataFrame({'_id': range(len(titles)), 'title': titles, 'year': years})
        b = pd.DataFrame({'_id': range(len(titles)),
                          'title': [t.replace(' ', '  ').upper() for t in titles],
                          'year': years})
        pairs = [(i, j) for i in range(len(titles)) for j in range(len(titles))]
        cand = pd.DataFrame({'id1': [p[0] for p in pairs],
                             'id2': [p[1] for p in pairs],
                             'score': [1.0 if p[0] == p[1] else 0.1 for p in pairs]})

        if spark is not None:
            from ._spark.io import as_spark
            table_a, table_b, candidates = (as_spark(x, spark) for x in (a, b, cand))
        else:
            table_a, table_b, candidates = a, b, cand

        features = create_features(A=table_a, B=table_b, a_cols=['title', 'year'],
                                   b_cols=['title', 'year'])
        print(f'features  {len(features)}')

        fvs = featurize(features=features, A=table_a, B=table_b, candidates=candidates,
                        output_col='feature_vectors', fill_na=0.0)
        labeled = fvs.toPandas() if spark is not None else fvs
        if labeled.empty:
            raise RuntimeError('featurize produced no feature vectors')
        print(f'featurize {len(labeled)} vectors x {len(labeled["feature_vectors"][0])}')

        labeled['label'] = (labeled['id1'] == labeled['id2']).astype(float)
        from xgboost import XGBClassifier
        model = SKLearnModel(model=XGBClassifier, n_estimators=10, random_state=42,
                             nan_fill=0.0)
        trained = train_matcher(model=model, labeled_data=labeled,
                                feature_col='feature_vectors', label_col='label')
        preds = apply_matcher(model=trained, df=labeled, feature_col='feature_vectors',
                              prediction_col='prediction', confidence_col='confidence')
        hits = int(((preds['prediction'] == 1.0) & (preds['label'] == 1.0)).sum())
        print(f'matcher   {hits}/{len(titles)} known matches recovered')
        if hits == 0:
            raise RuntimeError('the trained matcher recovered no known matches')
    finally:
        if spark is not None:
            spark.stop()

    print('\nOK: iris works on this machine.')
    return 0


def main(argv=None):
    """Parse the command line and run the subcommand."""
    p = argparse.ArgumentParser(prog='iris', description='Iris: machine-learning '
                                'matching for entity matching.')
    sub = p.add_subparsers(dest='cmd', required=True)

    i = sub.add_parser('info', help='show versions and the resources each engine sees',
                       description='Show the versions, the resources of the local '
                                   'engine and whether the Spark engine is installed.')
    i.set_defaults(func=_cmd_info)

    s = sub.add_parser('smoke', help='check the installation on this machine',
                       description='Run a small match on generated data.')
    s.add_argument('--engine', choices=['local', 'spark'], default='local',
                   help='local (the default) or spark (checks Java and iris[spark])')
    s.set_defaults(func=_cmd_smoke)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, TypeError, NotImplementedError) as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('interrupted', file=sys.stderr)
        return 130
    except Exception as e:
        print(f'error: {type(e).__name__}: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
