"""Run required regressions, then seal; no model training is called here."""
import os
from pathlib import Path
import subprocess
import sys
root=Path('/home/researcher/meadowmind-gpu/m8b-project')
env=dict(os.environ,MEADOW_GODOT='/home/researcher/meadowmind-gpu/godot/Godot_v4.7.2-stable_linux.x86_64',
         HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
         PYTHONPATH='/home/researcher/meadowmind-gpu/m8b-runtime-candidates/causal_conv1d_1_7_0_sourcebuild/site:'+str(root))
log=Path('/home/researcher/meadowmind-gpu/diagnostics/rl_kv2_regression_preseal_fixed.log')
with log.open('x') as stream:
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
print(log.read_text()[-500:],flush=True)
if result.returncode:sys.exit(result.returncode)
result=subprocess.run([sys.executable,'-u','tools/register_m8x_kv2.py'],cwd=root,env=env)
sys.exit(result.returncode)
