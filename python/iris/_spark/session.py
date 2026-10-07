"""Starting the SparkSession that `iris.resources` plans."""
import os

from pyspark.sql import SparkSession


def build_session(plan_, app_name, shuffle_partitions):
    """Start, or attach to, the SparkSession a plan describes."""
    builder = (SparkSession.builder.master(plan_.master).appName(app_name)
               .config('spark.driver.memory', plan_.driver_memory)
               .config('spark.sql.execution.arrow.pyspark.enabled', 'true')
               .config('spark.sql.shuffle.partitions', str(shuffle_partitions))
               .config('spark.ui.showConsoleProgress', 'false')
               .config('spark.iris.ownedSession', 'true'))
    if plan_.master.startswith('local'):
        host = os.environ.get('SPARK_LOCAL_IP', '127.0.0.1')
        builder = (builder.config('spark.driver.bindAddress', host)
                          .config('spark.driver.host', host))
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')
    return spark
