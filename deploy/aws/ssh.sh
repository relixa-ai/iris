#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env
source ./.cluster
exec ssh -i "$KEY_FILE" -o StrictHostKeyChecking=accept-new ubuntu@"$DRIVER_PUBLIC" "$@"
