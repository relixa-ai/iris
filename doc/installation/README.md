## How to install Iris

Iris runs on one machine or on a Spark cluster, and the same functions run on both. Given pandas DataFrames or Arrow datasets, the functions run on the *local engine*, which splits the work among Python processes on one machine and needs neither Spark nor Java. Given Spark DataFrames, they run on the *Spark engine*, as Spark jobs on a cluster or on one machine. The table below gives the guide to follow for each kind of installation.

| You are installing on | What runs there | Guide |
|---|---|---|
| A Linux machine | The local engine. With Spark support, also the Spark engine, with Spark running on this machine only. | [Linux](install-linux.md) |
| A Mac | The local engine. With Spark support, also the Spark engine, with Spark running on this machine only. | [macOS](install-macos.md) |
| A Windows machine | The local engine. | [Windows](install-windows.md) (not yet tested) |
| The machines of a Spark cluster | The Spark engine, with Spark and Iris on every node of the cluster. | [A Spark cluster](install-spark-cluster.md) |

The Spark engine needs Java 17 and Iris installed with *Spark support*, which is an optional part of Iris. Spark support adds PySpark 3.5.3, the Python package that contains Spark, and is installed with `bash install.sh --spark` or `pip install 'iris[spark]'`. If you only want to try Iris on the datasets included with it, one machine without Spark is sufficient. Parts 1 to 6 of the [tutorial](../tutorial.md) run without Spark, and parts 7 and 8 need it.

For one machine:

- Python 3.10, 3.11 or 3.12
- On a Mac with an Intel processor, the Xcode Command Line Tools, to build the dependency `py_stringmatching`. On other machines every dependency installs from a prebuilt package.
- For the Spark engine, Java 17 and Spark support

For a Spark cluster:

- Spark 3.5.3, Java 17, and Iris with Spark support on every node, with Iris in the Python environment that Spark uses on that node
- Storage for the data and the outputs that every node can read and write by a URI, such as Amazon S3 with `s3a://` URIs, and the Hadoop library that connects Spark to it, such as `hadoop-aws` for S3

The script `deploy/spark-node.sh` installs all of this on a machine running Ubuntu, at any cloud provider or on your own hardware, and starts it as the master or a worker of the cluster.

On one machine the installer is the script `install.sh`. It creates a Python virtual environment in the `.venv` directory inside the iris directory, installs Iris into the environment, and runs a short test, which featurizes 64 pairs of two small built-in tables and trains and applies a matcher on them. With the flag `--spark` it also checks Java and installs Spark support. Nothing is installed system-wide.
