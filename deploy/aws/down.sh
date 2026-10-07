#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env
IDS=$(aws ec2 describe-instances --filters Name=tag:Project,Values="$CLUSTER" \
      Name=instance-state-name,Values=pending,running,stopping,stopped \
      --query 'Reservations[].Instances[].InstanceId' --output text)
if [ -z "$IDS" ]; then echo "[down] nothing running"; rm -f .cluster; exit 0; fi
aws ec2 terminate-instances --instance-ids $IDS >/dev/null
echo "[down] terminating $IDS"
aws ec2 wait instance-terminated --instance-ids $IDS
rm -f .cluster
echo "[down] all terminated"
