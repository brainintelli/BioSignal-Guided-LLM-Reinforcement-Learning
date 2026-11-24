#!/usr/bin/env bash
set -euo pipefail
cd /home/researcher/wetware-afternoon-22021-20261123
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/home/researcher/mea-r2-22554/runtime_20261116_01/assets/overlay:/home/researcher/mea-r2-22554/runtime_20261116_01/assets/site:${PYTHONPATH:-}
mkdir -p /home/researcher/wetware-afternoon-22021-20261123/results
echo "=== [1/3] Running library_smoke ==="
/home/researcher/mea-r2-22554/runtime_20261116_01/assets/python/bin/python3 -u tools/gpu_existing_data_runner.py --orchestrate library_smoke --model-config /home/researcher/wetware-afternoon-22021-20261123/m9_model_paths.json --godot /home/researcher/mea-r2-22554/runtime_20261116_01/assets/godot/Godot_v4.7.2-stable_linux.x86_64 --project /home/researcher/wetware-afternoon-22021-20261123 --libraries-dir /home/researcher/wetware-afternoon-22021-20261123/output/existing_wetware_libraries --r2-mode dual --gpus 1,2,3,4,5,6,7 --microbatch 1 --eval-worlds 24 --n-boot 10000 --out /home/researcher/wetware-afternoon-22021-20261123/results/smoke
echo "=== [2/3] Running library_stage0 ==="
/home/researcher/mea-r2-22554/runtime_20261116_01/assets/python/bin/python3 -u tools/gpu_existing_data_runner.py --orchestrate library_stage0 --model-config /home/researcher/wetware-afternoon-22021-20261123/m9_model_paths.json --godot /home/researcher/mea-r2-22554/runtime_20261116_01/assets/godot/Godot_v4.7.2-stable_linux.x86_64 --project /home/researcher/wetware-afternoon-22021-20261123 --libraries-dir /home/researcher/wetware-afternoon-22021-20261123/output/existing_wetware_libraries --r2-mode dual --gpus 1,2,3,4,5,6,7 --microbatch 1 --eval-worlds 24 --n-boot 10000 --out /home/researcher/wetware-afternoon-22021-20261123/results/stage0 --seeds 16 --confirm-seeds 16 --ladder-tiers 1,2,3
echo "=== [3/3] Running library_stage3 (22021 clean 233-ROI) ==="
/home/researcher/mea-r2-22554/runtime_20261116_01/assets/python/bin/python3 -u tools/gpu_existing_data_runner.py --orchestrate library_stage3 --model-config /home/researcher/wetware-afternoon-22021-20261123/m9_model_paths.json --godot /home/researcher/mea-r2-22554/runtime_20261116_01/assets/godot/Godot_v4.7.2-stable_linux.x86_64 --project /home/researcher/wetware-afternoon-22021-20261123 --libraries-dir /home/researcher/wetware-afternoon-22021-20261123/output/existing_wetware_libraries --r2-mode dual --gpus 1,2,3,4,5,6,7 --microbatch 1 --eval-worlds 24 --n-boot 10000 --out /home/researcher/wetware-afternoon-22021-20261123/results/stage3 --stage0-report /home/researcher/wetware-afternoon-22021-20261123/results/stage0/stage0_summary.json --tier 1 --seeds 16
/home/researcher/mea-r2-22554/runtime_20261116_01/assets/python/bin/python3 -c "import json,sys; r=json.load(open('/home/researcher/wetware-afternoon-22021-20261123/results/stage3/stage3_summary.json')); sys.exit(0 if r['outcome']['verdict'] in ('weak_positive','dual_weak_positive','stable_positive') else 99)" && echo "=== [4/4] Stage 3 positive -> Running library_stage4 (28080 replication) ===" && /home/researcher/mea-r2-22554/runtime_20261116_01/assets/python/bin/python3 -u tools/gpu_existing_data_runner.py --orchestrate library_stage4 --model-config /home/researcher/wetware-afternoon-22021-20261123/m9_model_paths.json --godot /home/researcher/mea-r2-22554/runtime_20261116_01/assets/godot/Godot_v4.7.2-stable_linux.x86_64 --project /home/researcher/wetware-afternoon-22021-20261123 --libraries-dir /home/researcher/wetware-afternoon-22021-20261123/output/existing_wetware_libraries --r2-mode dual --gpus 1,2,3,4,5,6,7 --microbatch 1 --eval-worlds 24 --n-boot 10000 --out /home/researcher/wetware-afternoon-22021-20261123/results/stage4 --stage0-report /home/researcher/wetware-afternoon-22021-20261123/results/stage0/stage0_summary.json --tier 1 --seeds 16 || echo "=== Stage 3 did not reach weak_positive; skipping Stage 4 ==="
echo "=== ALL REQUESTED PHASES FINISHED ==="
