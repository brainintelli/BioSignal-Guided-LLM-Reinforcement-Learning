#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ]; then echo 'Usage: bash run_experiment.sh NEW_RUN_NAME' >&2; exit 2; fi
case "$1" in ''|*[!a-zA-Z0-9_-]*) echo 'Run name must use letters, digits, underscores or hyphens' >&2; exit 2;; esac
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_RUNTIME=/home/researcher/mea-r2-22554/runtime_20261116_01/assets
TASK_PYTHON="$TASK_RUNTIME/python/bin/python3.12"
TASK_OUTPUT="$TASK_ROOT/$1"
export PYTHONPATH="$TASK_ROOT:$TASK_RUNTIME/overlay:$TASK_RUNTIME/site:$TASK_ROOT/project"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 TRITON_CACHE_DIR="$TASK_ROOT/triton_cache" XDG_CACHE_HOME="$TASK_ROOT/cache"
if [ -e "$TASK_OUTPUT" ]; then echo 'New experiment output required; existing experiment will not be overwritten' >&2; exit 2; fi
if [ ! -f "$TASK_ROOT/input/manifest.json" ]; then echo 'Actual QC-approved input/manifest.json missing' >&2; exit 2; fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then echo 'GPU compute already occupied' >&2; exit 3; fi
if [ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ne 8 ]; then echo 'Exactly eight registered physical GPUs required' >&2; exit 3; fi
if [ ! -f "$TASK_ROOT/dataset.json" ]; then "$TASK_PYTHON" "$TASK_ROOT/launch.py" build; fi
"$TASK_PYTHON" "$TASK_ROOT/launch.py" validate
CUDA_VISIBLE_DEVICES=0 "$TASK_PYTHON" "$TASK_ROOT/gpu_acceptance.py"
CUDA_VISIBLE_DEVICES=0 "$TASK_PYTHON" "$TASK_ROOT/launch.py" register --root "$TASK_OUTPUT"
"$TASK_PYTHON" "$TASK_ROOT/launch.py" run --root "$TASK_OUTPUT"
