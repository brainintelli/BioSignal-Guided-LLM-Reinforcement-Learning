"""Bounded read-only hardware telemetry for owned training workers."""
import json,os,subprocess,time
from pathlib import Path
root=Path(os.environ['TASK_ROOT']);pids=[]
for p in Path('/proc').iterdir():
    if not p.name.isdecimal():continue
    try:
        args=(p/'cmdline').read_bytes().split(b'\0')
        if str(root/'eightgpu_run.py').encode() in args and b'collect' in args and args[0].endswith(b'python3.12'):pids.append(int(p.name))
    except OSError:pass
assert len(pids)==8,pids
with (root/'joint01_gpu_telemetry.jsonl').open('x') as stream:
    deadline=time.monotonic()+7200
    while True:
        alive=[pid for pid in pids if Path(f'/proc/{pid}/cmdline').exists()]
        v=dict(time=time.time(),worker_pids=pids,alive=alive,
            gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True),
            compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True))
        stream.write(json.dumps(v)+'\n');stream.flush()
        if not alive:break
        assert time.monotonic()<deadline,'Telemetry bound exceeded'
        time.sleep(5)
