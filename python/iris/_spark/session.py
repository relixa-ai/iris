"""Starting the SparkSession that `iris.resources` plans, and the check on the task
retries of a local session."""
import os
import re

from pyspark.sql import SparkSession

from .._core.utils import get_logger
from ..resources import TASK_ATTEMPTS

log = get_logger(__name__)

_LOCAL = re.compile(r'^local(?:\[\s*(\*|\d+)\s*\])?$')

_CHECKED = set()


def with_retries(master):
    """Return a local master with `TASK_ATTEMPTS` attempts per task when it has no
    retries of its own, and any other master unchanged."""
    found = _LOCAL.match(master.strip())
    if found is None:
        return master
    return f'local[{found.group(1) or 1}, {TASK_ATTEMPTS}]'


def check_retries(spark):
    """Warn, once per session, when a local session retries no failed task."""
    sc = spark.sparkContext
    master = sc.master
    if sc.applicationId in _CHECKED or with_retries(master) == master:
        return
    _CHECKED.add(sc.applicationId)
    log.warning(
        f"the Spark master {master!r} retries no failed task, so one transient failure, "
        f"such as macOS refusing a socket write with 'No buffer space available', ends "
        f"the run. Start the session with the master {with_retries(master)!r} to try "
        f"each task up to {TASK_ATTEMPTS} times, or with iris.resources.build_session.")


def build_session(plan_, app_name, shuffle_partitions):
    """Start, or attach to, the SparkSession a plan describes."""
    master = with_retries(plan_.master)
    builder = (SparkSession.builder.master(master).appName(app_name)
               .config('spark.driver.memory', plan_.driver_memory)
               .config('spark.sql.execution.arrow.pyspark.enabled', 'true')
               .config('spark.sql.shuffle.partitions', str(shuffle_partitions))
               .config('spark.ui.showConsoleProgress', 'false'))
    if master.startswith('local'):
        host = os.environ.get('SPARK_LOCAL_IP', '127.0.0.1')
        builder = (builder.config('spark.driver.bindAddress', host)
                          .config('spark.driver.host', host))
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')
    return spark
