#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env
aws ec2 describe-instances --filters Name=tag:Project,Values="$CLUSTER" \
  Name=instance-state-name,Values=pending,running,stopping,shutting-down \
  --query 'Reservations[].Instances[].[Tags[?Key==`Role`]|[0].Value,InstanceId,InstanceType,State.Name,PublicIpAddress,LaunchTime]' \
  --output table
if [ -f .cluster ]; then
  source ./.cluster
  ssh -i "$KEY_FILE" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 ubuntu@"$DRIVER_PUBLIC" "curl -fsS localhost:8080/json/" \
    2>/dev/null | python3 -c 'import json,sys; m=json.load(sys.stdin); print(
      "master %s: %d workers, %d cores, %d GiB; running apps: %d" % (m["url"],
      m["aliveworkers"], m["cores"], m["memory"] // 1024, len(m["activeapps"])))' || true
fi
