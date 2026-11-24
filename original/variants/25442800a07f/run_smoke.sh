#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ]; then echo 'Usage: bash run_smoke.sh NEW_RUN_NAME' >&2; exit 2; fi
case "$1" in ''|*[!a-zA-Z0-9_-]*) echo 'Use letters, digits, underscores or hyphens for the new run name' >&2; exit 2;; esac
export TASK_ROOT=/home/researcher/mea-r2-22554/smoke_20261116_01
export TASK_RUNTIME=/home/researcher/mea-r2-22554/runtime_20261116_01/assets
export PYTHONPATH="$TASK_RUNTIME/overlay:$TASK_RUNTIME/site:$TASK_ROOT/project"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=GPU-e9739564-e2ec-59c3-10f7-096d780abcbc
export TRITON_CACHE_DIR="$TASK_ROOT/triton_cache" XDG_CACHE_HOME="$TASK_ROOT/cache"
TASK_PYTHON="$TASK_RUNTIME/python/bin/python3.12"
TASK_OUTPUT="$TASK_ROOT/$1"
if [ -e "$TASK_OUTPUT" ]; then echo 'Output already exists; choose a new run name' >&2; exit 2; fi
if [ -n "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)" ]; then echo 'Selected GPU has an active compute process' >&2; exit 3; fi
timeout --signal=TERM --kill-after=30s 1800 "$TASK_PYTHON" "$TASK_ROOT/checkpoint_smoke.py" collect "$TASK_OUTPUT"
timeout --signal=TERM --kill-after=30s 1200 "$TASK_PYTHON" "$TASK_ROOT/checkpoint_smoke.py" verify "$TASK_OUTPUT"
"$TASK_PYTHON" "$TASK_ROOT/check_acceptance.py" "$TASK_OUTPUT"
