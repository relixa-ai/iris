#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env
source ./.cluster
exec ssh -i "$KEY_FILE" ubuntu@"$DRIVER_PUBLIC" "$@"
