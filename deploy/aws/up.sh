#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env
say() { printf '[up] %s\n' "$*"; }
REPO=$(git rev-parse --show-toplevel)
SHA=$(git -C "$REPO" rev-parse --short HEAD)
if [ -n "$(git -C "$REPO" status --porcelain -- python pyproject.toml constraints.txt)" ]; then
  echo "uncommitted changes under python/: the cluster installs the committed code;" \
       "commit them first" >&2
  exit 1
fi
if [ -n "$(aws ec2 describe-instances --filters Name=tag:Project,Values="$CLUSTER" \
           Name=instance-state-name,Values=pending,running --query 'Reservations[].Instances[].InstanceId' \
           --output text)" ]; then
  echo "a $CLUSTER cluster is already running (deploy/aws/status.sh; deploy/aws/down.sh)" >&2
  exit 1
fi

ARCHIVE=$(mktemp -d)/iris.tar.gz
git -C "$REPO" archive --format=tar.gz -o "$ARCHIVE" HEAD
aws s3 cp --quiet "$ARCHIVE" "s3://$BUCKET/code/iris-$SHA.tar.gz"
URL=$(aws s3 presign "s3://$BUCKET/code/iris-$SHA.tar.gz" --expires-in $((MAX_HOURS * 3600)))
say "code: $SHA"

AMI=$(aws ssm get-parameter --query Parameter.Value --output text \
      --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id)
SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values="$SG_NAME" \
     --query 'SecurityGroups[0].GroupId' --output text)

user_data() {
  printf '#!/bin/bash\nexec > /var/log/iris-init.log 2>&1\n'
  printf 'cat > /etc/iris-spark.conf <<EOF\n%s\nEOF\n' \
    'spark.hadoop.fs.s3a.aws.credentials.provider com.amazonaws.auth.InstanceProfileCredentialsProvider'
  printf "export IRIS_ROLE=%s IRIS_MASTER=%s IRIS_SOURCE='%s' IRIS_STORAGE=s3a\n" "$1" "$2" "$URL"
  printf 'export IRIS_SCRATCH_DEVICES=auto IRIS_SPARK_CONF=/etc/iris-spark.conf\n'
  printf 'export IRIS_EXECUTOR_MEMORY=%s IRIS_MAX_HOURS=%s\n' "$EXECUTOR_MEMORY" "$MAX_HOURS"
  tail -n +2 ../spark-node.sh
}

launch() {
  aws ec2 run-instances --image-id "$AMI" --instance-type "$2" --count "$3" \
    --key-name "$KEY_NAME" --security-group-ids "$SG" \
    --iam-instance-profile Name="$ROLE" \
    --instance-initiated-shutdown-behavior terminate \
    --metadata-options HttpTokens=required \
    --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$4,VolumeType=gp3}" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=$CLUSTER},{Key=Role,Value=$1},{Key=Name,Value=$CLUSTER-$1}]" \
    --user-data "$(user_data "$1" "$5")" \
    --query 'Instances[].InstanceId' --output text
}

DRIVER=$(launch master "$DRIVER_TYPE" 1 "$DRIVER_DISK_GB" none)
aws ec2 wait instance-running --instance-ids "$DRIVER"
read -r PRIV PUB < <(aws ec2 describe-instances --instance-ids "$DRIVER" --query \
  'Reservations[0].Instances[0].[PrivateIpAddress,PublicIpAddress]' --output text)
say "driver $DRIVER ($DRIVER_TYPE): $PUB, private $PRIV"
WORKER_IDS=$(launch worker "$WORKER_TYPE" "$WORKERS" "$WORKER_DISK_GB" "$PRIV")
say "workers: $WORKERS x $WORKER_TYPE: $WORKER_IDS"
printf 'DRIVER_ID=%s\nDRIVER_PUBLIC=%s\nDRIVER_PRIVATE=%s\n' "$DRIVER" "$PUB" "$PRIV" > .cluster

SSH="ssh -i $KEY_FILE -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 ubuntu@$PUB"
say "installing Java, Spark and Iris on every node: about 5 to 10 minutes"
DEADLINE=$((SECONDS + 25 * 60))
alive() {
  $SSH "curl -fsS localhost:8080/json/" 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["aliveworkers"])' 2>/dev/null
}
until $SSH test -f /var/lib/iris-ready 2>/dev/null && [ "$(alive)" = "$WORKERS" ]; do
  if [ $SECONDS -gt $DEADLINE ]; then
    echo "not ready after 25 minutes: $(alive || echo 0) of $WORKERS workers joined." \
         "Each node logs its installation to /var/log/iris-init.log (deploy/aws/ssh.sh" \
         "sudo tail /var/log/iris-init.log); deploy/aws/down.sh stops the cluster." >&2
    exit 1
  fi
  sleep 20
done
say "ready: master spark://$PRIV:7077 with $WORKERS workers"
say "web UI: ssh -i $KEY_FILE -L 8080:localhost:8080 -L 4040:localhost:4040 ubuntu@$PUB"
say "every instance terminates itself $MAX_HOURS hours after boot; deploy/aws/down.sh ends it now"
