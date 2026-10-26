#!/usr/bin/env bash
set -euo pipefail
[ $# -ge 1 ] && [ -f "$1" ] || { echo "usage: deploy/aws/run.sh SCRIPT [ARGUMENT ...]" >&2; exit 2; }
SCRIPT=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); shift
cd "$(dirname "$0")"
source ./config.env
source ./.cluster
STAMP=$(date +%Y%m%d-%H%M%S)
JOB=jobs/$STAMP-$(basename "$SCRIPT")
SSH=(ssh -i "$KEY_FILE" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "ubuntu@$DRIVER_PUBLIC")
"${SSH[@]}" mkdir -p jobs
scp -q -i "$KEY_FILE" "$SCRIPT" "ubuntu@$DRIVER_PUBLIC:$JOB"
ARGS=
[ $# -eq 0 ] || ARGS=$(printf ' %q' "$@")
"${SSH[@]}" "source /etc/profile.d/iris.sh; setsid nohup spark-submit \
  --master spark://$DRIVER_PRIVATE:7077 $JOB$ARGS > run-$STAMP.log 2>&1 < /dev/null &"
echo "[run] $(basename "$SCRIPT")$ARGS"
echo "[run] follow: deploy/aws/ssh.sh tail -f run-$STAMP.log"
