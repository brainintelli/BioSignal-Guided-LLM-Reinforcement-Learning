#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME=/home/researcher/mea-r2-22554/runtime_20261116_01/assets
export PYTHONPATH="$ROOT/project/standalone/mea_online:$ROOT/project:$RUNTIME/overlay:$RUNTIME/site"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONIOENCODING=utf-8
exec "$RUNTIME/python/bin/python3.12" -m mea_online.cli "$@"
