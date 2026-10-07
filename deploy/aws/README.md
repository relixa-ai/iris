## A Spark Cluster on Amazon Web Services

These scripts start a Spark cluster for Iris on Amazon EC2, in Spark's standalone mode, which needs no other cluster manager. They also run scripts on it and stop it. The data and the outputs are stored in Amazon S3 and named by `s3a://` URIs, so no file system is shared between the machines. Each machine is installed by [`deploy/spark-node.sh`](../spark-node.sh), the same script that installs a machine at any other cloud provider or on your own hardware.

By default the cluster has a driver of type r6i.2xlarge, which runs the Spark master and the scripts, and six workers of type i4i.2xlarge, whose local NVMe disks hold Spark's shuffle files. That is 56 virtual CPUs in total. The settings are in `config.env`, and each can be overridden by an environment variable of the same name.

You need the AWS CLI with credentials, so that `aws sts get-caller-identity` succeeds, and a quota of on-demand virtual CPUs large enough for the cluster (56 with the defaults). The quota is named "Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances".

```
deploy/aws/account.sh
aws s3 cp --recursive data/dblp_acm s3://iris-ACCOUNT/datasets/dblp_acm/
deploy/aws/up.sh
deploy/aws/run.sh examples/run_spark_cluster.py s3a://iris-ACCOUNT/datasets/dblp_acm s3a://iris-ACCOUNT/datasets/dblp_acm/runs/1
deploy/aws/ssh.sh tail -f run-TIME.log
deploy/aws/status.sh
deploy/aws/down.sh
```

`account.sh` is run once. It creates a private bucket named `iris-ACCOUNT`, where ACCOUNT is the number of the AWS account, an instance role that gives the machines access to that bucket only, an SSH key pair, and a security group that admits all traffic between the machines of the cluster and SSH from your address. Running it again changes nothing that exists.

`up.sh` uploads the committed code of the repository to the bucket, starts the machines, and waits until every worker has joined the master, which takes 5 to 10 minutes. It refuses to start when files under `python/` have uncommitted changes, because the cluster installs the committed code.

`run.sh` copies a script to the driver and submits it with `spark-submit`, followed by the arguments given. The run continues when the terminal is closed, and its output is written to `run-TIME.log` on the driver.

`down.sh` terminates the machines. The bucket, the role, the key pair and the security group remain for the next `up.sh`. Each machine also shuts down and terminates on its own `MAX_HOURS` hours after it starts, 6 by default, so a cluster that was not stopped stops costing money.

The web pages of Spark are reached through an SSH tunnel, which the command that `up.sh` prints opens. Then http://localhost:8080 shows the master, and http://localhost:4040 shows the running script.
