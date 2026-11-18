#!/usr/bin/env bash
set -euo pipefail
mkdir -p "${DATA_DIR:-/workspace}"/{models,logs,results}
exec "$@"
