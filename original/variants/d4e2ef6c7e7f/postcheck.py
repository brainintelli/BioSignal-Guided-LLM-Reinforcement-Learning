"""Read-only final registration, process and checkpoint checks on node 5."""
import json
from pathlib import Path
import subprocess
import sys
root=Path('/home/researcher/meadowmind-gpu/m8b-project');sys.path.insert(0,str(root))
from agent.m8x_kv2_registration import verify
regdir=Path('/home/researcher/meadowmind-gpu/m8x-kv-runs/kv_attempt2')
folder=Path('/home/researcher/meadowmind-gpu/diagnostics/rl_kv2_registered_20261107')
reg=json.loads((regdir/'registration.json').read_text());verify(root,reg)
comparison=json.loads((folder/'comparison.json').read_text())
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
processes=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
active=[p for p in processes if any(x in p for x in ('tools/benchmark_m8x','agent/m8x_train.py','agent.m8x_train','model_server.py')) and 'ps -eo' not in p]
checkpoints=list(regdir.glob('checkpoint_*'))
result=dict(passed=not gpu and not active and not checkpoints and comparison['passed'],
            registration_verified=True,sources=len(reg['source_hashes']),inputs=len(reg['input_files']),
            registration_sha256=reg['registration_sha256'],gpu_compute_processes=gpu,
            matching_training_or_service_processes=active,new_checkpoints=[str(p) for p in checkpoints],
            formal_training_status=reg['formal_training_status'],comparison_passed=comparison['passed'])
(folder/'postcheck.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
sys.exit(0 if result['passed'] else 1)
