#!/bin/bash
set -Eeuo pipefail
trap 'echo "[spark-node] failed at line $LINENO: $BASH_COMMAND" >&2' ERR

usage() {
  cat >&2 <<'EOF'
Usage: sudo IRIS_ROLE=master|worker [settings] deploy/spark-node.sh

Installs Java 17 and Iris with Spark support on an Ubuntu 22.04 or 24.04 machine, and
starts a standalone Spark master or worker with the Spark that Iris's Spark support installs.
Settings are environment variables:

  IRIS_ROLE             master or worker (required)
  IRIS_MASTER           the master's host name or address (required for a worker)
  IRIS_SOURCE           the Iris source: a directory, a .tar.gz archive (path or URL),
                        or a git repository URL (default: the directory holding deploy/)
  IRIS_REF              the branch or tag of a git IRIS_SOURCE
  IRIS_STORAGE          s3a to add the S3 connector and committer, or none (default)
  IRIS_JARS             further jar URLs for Spark's classpath, separated by spaces
  IRIS_SPARK_CONF       a file of further spark-defaults.conf lines
  IRIS_SCRATCH          the directory for Spark's shuffle (default /mnt/scratch)
  IRIS_SCRATCH_DEVICES  blank disks to format and mount at IRIS_SCRATCH, striped when
                        several, or auto for every disk with no partition or filesystem
  IRIS_EXECUTOR_MEMORY  the executor heap (default: 3/8 of this machine's memory)
  IRIS_DRIVER_MEMORY    the driver heap (default: 35% of this machine's memory)
  IRIS_USER             the account that runs Spark (default ubuntu, else root)
  IRIS_MAX_HOURS        shut this machine down after so many hours
EOF
  exit 2
}

[ "$(id -u)" = 0 ] || { echo "spark-node.sh must run as root" >&2; exit 2; }
ROLE=${IRIS_ROLE:-}
[ "$ROLE" = master ] || [ "$ROLE" = worker ] || usage
[ "$ROLE" = master ] || [ -n "${IRIS_MASTER:-}" ] || usage

HADOOP_VERSION=3.3.4
AWS_SDK_VERSION=1.12.262
MAVEN=https://repo1.maven.org/maven2
HERE=$(cd "$(dirname "$0")/.." 2>/dev/null && pwd || true)
SOURCE=${IRIS_SOURCE:-$HERE}
STORAGE=${IRIS_STORAGE:-none}
SCRATCH=${IRIS_SCRATCH:-/mnt/scratch}
RUN_AS=${IRIS_USER:-$(id -u ubuntu >/dev/null 2>&1 && echo ubuntu || echo root)}
MEM_GB=$(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo)
EXECUTOR_MEMORY=${IRIS_EXECUTOR_MEMORY:-$((MEM_GB * 3 / 8))g}
DRIVER_MEMORY=${IRIS_DRIVER_MEMORY:-$((MEM_GB * 35 / 100))g}
say() { printf '[spark-node] %s\n' "$*"; }

if [ -n "${IRIS_MAX_HOURS:-}" ]; then
  shutdown -h +$((IRIS_MAX_HOURS * 60))
  say "this machine shuts down in $IRIS_MAX_HOURS hours"
fi
if [ "$STORAGE" != s3a ] && [ "$STORAGE" != none ]; then
  echo "IRIS_STORAGE is s3a or none, got $STORAGE; other stores take IRIS_JARS and IRIS_SPARK_CONF" >&2
  exit 2
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q curl gpg git mdadm python3-venv
PY=$(command -v python3)
"$PY" -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 12))' \
  || { echo "Iris needs Python 3.10, 3.11 or 3.12; this machine has $("$PY" --version)" >&2; exit 1; }

if [ ! -d /usr/lib/jvm/temurin-17-jdk-"$(dpkg --print-architecture)" ]; then
  curl -fsSL https://packages.adoptium.net/artifactory/api/gpg/key/public \
    | gpg --dearmor --yes -o /etc/apt/trusted.gpg.d/adoptium.gpg
  echo "deb https://packages.adoptium.net/artifactory/deb $(. /etc/os-release; echo "$VERSION_CODENAME") main" \
    > /etc/apt/sources.list.d/adoptium.list
  apt-get update -q
  apt-get install -y -q temurin-17-jdk
fi
JAVA_HOME=/usr/lib/jvm/temurin-17-jdk-$(dpkg --print-architecture)
say "Java: $("$JAVA_HOME/bin/java" -version 2>&1 | head -1)"

rm -rf /opt/iris
case "$SOURCE" in
  *.git|git@*|git+*)
    git clone -q ${IRIS_REF:+-b "$IRIS_REF"} --depth 1 "${SOURCE#git+}" /opt/iris ;;
  http://*|https://*)
    mkdir -p /opt/iris
    curl -fsSL "$SOURCE" | tar -xz -C /opt/iris ;;
  *.tar.gz|*.tgz)
    mkdir -p /opt/iris
    tar -xzf "$SOURCE" -C /opt/iris ;;
  *)
    [ -f "$SOURCE/pyproject.toml" ] || { echo "no Iris source at $SOURCE" >&2; exit 1; }
    cp -r "$SOURCE" /opt/iris ;;
esac
TOP=(/opt/iris/*)
if [ ! -f /opt/iris/pyproject.toml ] && [ ${#TOP[@]} = 1 ] && [ -d "${TOP[0]}" ]; then
  mv "${TOP[0]}" /opt/iris-src && rm -rf /opt/iris && mv /opt/iris-src /opt/iris
fi
[ -f /opt/iris/pyproject.toml ] || { echo "no Iris source in $SOURCE" >&2; exit 1; }
rm -rf /opt/iris-venv
"$PY" -m venv /opt/iris-venv
/opt/iris-venv/bin/pip install -q -U pip
/opt/iris-venv/bin/pip install -q '/opt/iris[spark]' -c /opt/iris/constraints.txt
SPARK=$(/opt/iris-venv/bin/python -c 'import os, pyspark; print(os.path.dirname(pyspark.__file__))')
SPARK_VERSION=$(/opt/iris-venv/bin/python -c 'import pyspark; print(pyspark.__version__)')
ln -sfn "$SPARK" /opt/spark
say "Iris $(/opt/iris-venv/bin/python -c 'import iris; print(iris.__version__)') with $("$PY" --version) and Spark $SPARK_VERSION"

JARS=${IRIS_JARS:-}
if [ "$STORAGE" = s3a ]; then
  [ -f "$SPARK/jars/hadoop-client-api-$HADOOP_VERSION.jar" ] \
    || { echo "this Spark is not built with Hadoop $HADOOP_VERSION, which the S3A connector here matches" >&2; exit 1; }
  JARS="$JARS $MAVEN/org/apache/hadoop/hadoop-aws/$HADOOP_VERSION/hadoop-aws-$HADOOP_VERSION.jar"
  JARS="$JARS $MAVEN/com/amazonaws/aws-java-sdk-bundle/$AWS_SDK_VERSION/aws-java-sdk-bundle-$AWS_SDK_VERSION.jar"
  JARS="$JARS $MAVEN/org/apache/spark/spark-hadoop-cloud_2.12/$SPARK_VERSION/spark-hadoop-cloud_2.12-$SPARK_VERSION.jar"
fi
for jar in $JARS; do
  name=$(basename "${jar%%\?*}")
  [ -f "$SPARK/jars/$name" ] || curl -fsSL -o "$SPARK/jars/$name" "$jar"
done
say "storage: $STORAGE"

if [ "${IRIS_SCRATCH_DEVICES:-}" = auto ]; then
  IRIS_SCRATCH_DEVICES=$(lsblk -dpno NAME,TYPE | awk '$2=="disk"{print $1}' | while read -r d; do
    if [ -z "$(lsblk -no FSTYPE,MOUNTPOINT "$d" | tr -d '[:space:]')" ] && [ "$(lsblk -no NAME "$d" | wc -l)" = 1 ]; then
      echo "$d"
    fi
  done | tr '\n' ' ')
fi
DEVS=${IRIS_SCRATCH_DEVICES:-}
mkdir -p "$SCRATCH"
if [ -n "${DEVS// /}" ] && ! mountpoint -q "$SCRATCH"; then
  set -- $DEVS
  if [ $# -gt 1 ]; then
    mdadm --create /dev/md0 --run --level=0 --raid-devices=$# "$@"
    DEV=/dev/md0
  else
    DEV=$1
  fi
  mkfs.ext4 -q -F -E nodiscard "$DEV"
  mount -o noatime "$DEV" "$SCRATCH"
  say "scratch: $DEVS mounted at $SCRATCH"
fi
mkdir -p "$SCRATCH/spark" "$SCRATCH/s3a" /var/log/spark
chown -R "$RUN_AS" "$SCRATCH" /var/log/spark
say "scratch: $SCRATCH, $(df -h --output=avail "$SCRATCH" | tail -1 | tr -d ' ') free"

mkdir -p "$SPARK/conf"
cat > "$SPARK/conf/spark-env.sh" <<EOF
export JAVA_HOME=$JAVA_HOME
export SPARK_LOCAL_DIRS=$SCRATCH/spark
export SPARK_LOG_DIR=/var/log/spark
export SPARK_PID_DIR=/var/log/spark
export SPARK_WORKER_CORES=$(nproc)
export SPARK_WORKER_MEMORY=$((MEM_GB > 8 ? MEM_GB - 4 : MEM_GB / 2))g
export PYSPARK_PYTHON=/opt/iris-venv/bin/python
export PYSPARK_DRIVER_PYTHON=/opt/iris-venv/bin/python
EOF
{
  echo "spark.executor.memory                          $EXECUTOR_MEMORY"
  echo "spark.driver.memory                            $DRIVER_MEMORY"
  echo "spark.pyspark.python                           /opt/iris-venv/bin/python"
  echo "spark.ui.showConsoleProgress                   false"
  if [ "$STORAGE" = s3a ]; then
    echo "spark.hadoop.fs.s3a.buffer.dir                 $SCRATCH/s3a"
    echo "spark.hadoop.fs.s3a.committer.name             magic"
    echo "spark.hadoop.fs.s3a.committer.magic.enabled    true"
    echo "spark.sql.sources.commitProtocolClass          org.apache.spark.internal.io.cloud.PathOutputCommitProtocol"
    echo "spark.sql.parquet.output.committer.class       org.apache.spark.internal.io.cloud.BindingParquetOutputCommitter"
  fi
  [ -z "${IRIS_SPARK_CONF:-}" ] || cat "$IRIS_SPARK_CONF"
} > "$SPARK/conf/spark-defaults.conf"
cat > /etc/profile.d/iris.sh <<EOF
export JAVA_HOME=$JAVA_HOME SPARK_HOME=$SPARK PATH=/opt/iris-venv/bin:\$PATH
EOF
chown -R "$RUN_AS" /opt/iris /opt/iris-venv

IP=$(hostname -I | awk '{print $1}')
DAEMON="$SPARK/sbin/spark-daemon.sh"
if [ "$ROLE" = master ]; then
  runuser -u "$RUN_AS" -- env SPARK_HOME="$SPARK" "$DAEMON" start org.apache.spark.deploy.master.Master 1 \
    --host "$IP" --port 7077 --webui-port 8080 >/dev/null
  say "master: spark://$IP:7077 (web UI on port 8080)"
else
  until (exec 3<>"/dev/tcp/$IRIS_MASTER/7077") 2>/dev/null; do sleep 5; done
  runuser -u "$RUN_AS" -- env SPARK_HOME="$SPARK" "$DAEMON" start org.apache.spark.deploy.worker.Worker 1 \
    --webui-port 8081 "spark://$IRIS_MASTER:7077" >/dev/null
  say "worker: joined spark://$IRIS_MASTER:7077 with $(nproc) cores"
fi
touch /var/lib/iris-ready
