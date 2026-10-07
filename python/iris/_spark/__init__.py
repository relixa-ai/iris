"""The Spark engine, the only package of Iris that imports pyspark."""
try:
    import pyspark
except ImportError as e:
    raise ModuleNotFoundError(
        "this needs Spark, and pyspark is not installed. Install Iris with Spark support, "
        "pip install 'iris[spark]' (it also needs Java 17), or run on the local engine "
        "by passing pandas DataFrames or Arrow datasets.", name='pyspark') from e
