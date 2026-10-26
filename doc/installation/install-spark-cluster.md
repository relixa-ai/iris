## How to install Iris on a Spark cluster

This guide installs Iris on the machines of a Spark cluster. A cluster has one *master*, which assigns the work, and a number of *workers*, which do it. A script that uses Iris is started with `spark-submit`, which runs the script on a *driver*, here the master machine, and the tasks of each step on the workers. Part 8 of the [tutorial](../tutorial.md) runs such a script.

The data and the outputs of a run are kept on storage that every machine of the cluster can read and write, and are named by URIs, such as `s3a://bucket/path` for Amazon S3 and storage that is compatible with it, or `hdfs://namenode/path` for HDFS. Each worker needs scratch disk for Spark's *shuffle*, the step in which Spark moves records between machines, which `featurize` uses to bring each candidate pair together with its two records.

The script `deploy/spark-node.sh` installs one machine. It installs Java 17 and Iris with Spark support in a Python environment. Spark support includes Spark 3.5.3, in the PySpark package, so every machine runs that Spark, at the version Iris requires. The script then adds the Hadoop library that connects Spark to the storage, such as `hadoop-aws` for S3, configures Spark, and starts the machine as the master or as a worker. It runs on Ubuntu 22.04 or 24.04, on x86 or ARM machines, at any cloud provider or on your own hardware. Everything it installs is under `/opt`, apart from the packages it installs with `apt-get`, and Spark's logs are in `/var/log/spark`.

This guide was tested on Amazon EC2, with a master of type r6i.2xlarge, two workers of type i4i.2xlarge whose NVMe disks held the shuffle, Ubuntu 24.04 and Amazon S3, started by the scripts in `deploy/aws`. It was also tested with a master and a worker running Ubuntu 24.04 on ARM, in Docker containers, with an S3-compatible store in place of Amazon S3.

---

## Step 1: Check the prerequisites

### The machines

Every machine runs Ubuntu 22.04 or 24.04, and you can run commands on it with `sudo`. Every machine can connect to every other machine on every port, and you can connect to the master on ports 22 (SSH), 7077 (the master) and 8080 (the master's web page). The master runs the driver, which holds the pool and the labels during active learning. The workers do the featurizing, the scoring and the applying of the matcher, and their total memory and cores decide how fast a run is.

### The scratch disk

Spark writes its shuffle files to `/mnt/scratch` on each worker. Before it starts, `featurize` estimates the shuffle of the run, logs it, and stops with an error that names the space it needs when a worker's scratch disk is too small. Machines with local NVMe disks, such as the i4i types of Amazon EC2, are suited to this. `deploy/spark-node.sh` formats and mounts blank disks at `/mnt/scratch` when `IRIS_SCRATCH_DEVICES=auto` is set.

### The storage

| Storage | URI | What `spark-node.sh` needs |
|---|---|---|
| Amazon S3, and services compatible with it | `s3a://bucket/path` | `IRIS_STORAGE=s3a`. On Amazon EC2, the machines get access to S3 from an *instance role*, a set of permissions attached to each machine. Elsewhere, the access key, the secret key and the endpoint go in `IRIS_SPARK_CONF`. |
| HDFS | `hdfs://namenode:8020/path` | Nothing more. Spark includes the HDFS client. |
| Google Cloud Storage | `gs://bucket/path` | The Cloud Storage connector, given as `IRIS_JARS`, and its settings in `IRIS_SPARK_CONF`. This has not yet been tested. |
| Azure Data Lake Storage | `abfss://container@account.dfs.core.windows.net/path` | The `hadoop-azure` connector, given as `IRIS_JARS`, and its settings in `IRIS_SPARK_CONF`. This has not yet been tested. |

---

## Step 2: Get Iris onto every machine

`deploy/spark-node.sh` installs Iris from the source given as `IRIS_SOURCE`, which is one of the following:

- a directory that holds the `iris` repository,
- a `.tar.gz` archive of it, as a path or a URL, or
- the URL of the git repository, `https://github.com/relixa-ai/iris.git`, with `IRIS_REF` naming a branch or a tag.

Every machine must install the same version of Iris. A URL that every machine can download, such as a release archive, is the simplest source.

---

## Step 3: Install the master

On the master, copy `deploy/spark-node.sh` to the machine and execute:

```
sudo IRIS_ROLE=master IRIS_SOURCE=<source> IRIS_STORAGE=s3a bash spark-node.sh
```

This command does the following.

- **Installs** Java 17 (Eclipse Temurin) with `apt-get`, from the Adoptium repository, and Iris in `/opt/iris`. It installs Iris with Spark support and its dependencies into the Python environment `/opt/iris-venv` with `pip`, from the Python Package Index. `/opt/spark` points to the Spark of that environment. With `IRIS_STORAGE=s3a`, it adds the S3A connector and committer to Spark, as jar files from Maven Central. This takes about 5 minutes.
- **Prepares the scratch directory** `/mnt/scratch`.
- **Configures Spark** in `/opt/spark/conf`, with the scratch directory, the Python environment, and the memory of the driver and of each *executor*, the process that runs Spark's tasks on a worker, which by default are 35% and 3/8 of the machine's memory.
- **Starts the master.**

The script prints the output of the packages it installs, and its own lines, which start with `[spark-node]`. You should see lines like the following:

```
[spark-node] Java: openjdk version "17.0.20.1" 2026-08-18
[spark-node] Iris 0.3.0 with Python 3.12.3 and Spark 3.5.3
[spark-node] storage: s3a
[spark-node] scratch: /mnt/scratch, 840G free
[spark-node] master: spark://172.18.0.3:7077 (web UI on port 8080)
```

## Step 4: Install the workers

On each worker, execute the same command with `IRIS_ROLE=worker` and the address of the master:

```
sudo IRIS_ROLE=worker IRIS_MASTER=<master address> IRIS_SOURCE=<source> IRIS_STORAGE=s3a IRIS_SCRATCH_DEVICES=auto bash spark-node.sh
```

You should see lines like the following:

```
[spark-node] Java: openjdk version "17.0.20.1" 2026-08-18
[spark-node] Iris 0.3.0 with Python 3.12.3 and Spark 3.5.3
[spark-node] storage: s3a
[spark-node] scratch: /mnt/scratch, 840G free
[spark-node] worker: joined spark://iris-master:7077 with 18 cores
```

The master's web page then lists the worker.

The settings of `deploy/spark-node.sh` are the following.

| Setting | What it does |
|---|---|
| `IRIS_ROLE` | `master` or `worker`. Required. |
| `IRIS_MASTER` | The host name or address of the master. Required for a worker. |
| `IRIS_SOURCE` | Where Iris is installed from, as in Step 2. By default the repository that holds the script. |
| `IRIS_REF` | The branch or tag of a git `IRIS_SOURCE`. |
| `IRIS_STORAGE` | `s3a` to install the S3A connector and committer, or `none`, the default. |
| `IRIS_JARS` | Further jar files for Spark, as URLs separated by spaces, such as a storage connector. |
| `IRIS_SPARK_CONF` | A file of further lines for Spark's `spark-defaults.conf`, such as the settings of the storage. |
| `IRIS_SCRATCH` | The scratch directory, `/mnt/scratch` by default. |
| `IRIS_SCRATCH_DEVICES` | Blank disks to format and mount at the scratch directory, striped when there are several, or `auto` for every disk that has no partition and no file system. |
| `IRIS_EXECUTOR_MEMORY` | The memory of each executor. By default 3/8 of the machine's memory. Set it on the master when the workers are a different type of machine. |
| `IRIS_DRIVER_MEMORY` | The memory of the driver. By default 35% of the machine's memory. |
| `IRIS_USER` | The account that runs Spark. By default `ubuntu` when the machine has that account, and `root` otherwise. |
| `IRIS_MAX_HOURS` | Shuts the machine down after this many hours, so that a cluster that was not stopped stops costing money. |

The script can be run again on a machine. It reinstalls Iris and the configuration, and starts the master or the worker again.

---

## Step 5: Run a script on the cluster

On the master, copy the dataset to the storage and submit the cluster example of the tutorial:

```
source /etc/profile.d/iris.sh
spark-submit --master spark://<master address>:7077 --conf spark.log.level=WARN \
    /opt/iris/examples/run_spark_cluster.py s3a://<bucket>/datasets/dblp_acm s3a://<bucket>/datasets/dblp_acm/runs/1
```

You should see:

```
precision 0.9941  recall 0.9851  f1 0.9896
```

The master's web page, at port 8080 of the master, shows the workers and the running script. The page of the script itself is at port 4040 while it runs.

---

## Clusters on a cloud provider

`deploy/spark-node.sh` is written to run as a machine's startup script, which providers call *user data* or a *startup script*. The settings are given as environment variables set before the script runs. The directory `deploy/aws` holds scripts that do this on Amazon Web Services. `account.sh` prepares the account once, `up.sh` starts a master and the workers and waits until every worker has joined, `run.sh` submits a script, and `down.sh` terminates the machines. Its [README](../../deploy/aws/README.md) describes them.

Managed Spark services, such as Amazon EMR, Google Dataproc, Azure HDInsight and Databricks, provide the cluster themselves. On them, Iris needs Spark 3.5, Java 17, and `iris[spark]` installed with `pip` on every machine, which the services do with *bootstrap actions* or *initialization scripts*. This has not yet been tested.

---

## Troubleshooting

| What you see | What it means | What to do |
|---|---|---|
| `spark-node.sh must run as root` | The script was run without `sudo`. | Run it with `sudo`. |
| `Iris needs Python 3.10, 3.11 or 3.12; this machine has Python 3.13.1` | The machine's Python is not a supported version. | Use Ubuntu 22.04 or 24.04, whose Python is 3.10 or 3.12. |
| `no Iris source at <path>` | `IRIS_SOURCE` names nothing that holds the repository. | Check the path or the URL. |
| A worker never appears on the master's web page | The worker cannot reach port 7077 of the master. | Allow all traffic between the machines of the cluster. |
| `not enough scratch disk for the pair pass: ...` | A worker's scratch disk cannot hold the shuffle. | Use workers with more local disk, more workers, or `IRIS_SCRATCH_DEVICES`. |
| `No FileSystem for scheme "s3a"` | The S3A connector is not installed. | Run `spark-node.sh` with `IRIS_STORAGE=s3a` on every machine. |
| `java.nio.file.AccessDeniedException` on an `s3a://` path | The machines have no credentials for the bucket. | Give the machines an instance role with access to the bucket, or put the keys in `IRIS_SPARK_CONF`. |

---

## How to uninstall

On each machine, stop Spark with `/opt/spark/sbin/spark-daemon.sh stop org.apache.spark.deploy.worker.Worker 1`, or with `org.apache.spark.deploy.master.Master` on the master, and delete `/opt/spark`, `/opt/iris`, `/opt/iris-venv`, `/var/log/spark` and `/etc/profile.d/iris.sh`. A cluster of cloud machines is removed by terminating the machines.
