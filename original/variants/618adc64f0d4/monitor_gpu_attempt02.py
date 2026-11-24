"""Bounded read-only hardware telemetry for owned training workers."""
import argparse,json,os,subprocess,time
from pathlib import Path
root=Path(os.environ['TASK_ROOT']);parser=argparse.ArgumentParser();parser.add_argument('campaign');campaign=parser.parse_args().campaign
def owned(pid):
    try:
        args=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        return str(root/'eightgpu_run.py').encode() in args and str(root/campaign).encode() in args and b'collect' in args and args[0].endswith(b'python3.12')
    except OSError:return False
deadline=time.monotonic()+90
while True:
    pids=[int(p.name) for p in Path('/proc').iterdir() if p.name.isdecimal() and owned(p.name)]
    if len(pids)==8:break
    assert time.monotonic()<deadline,'Eight workers never appeared'
    time.sleep(1)
with (root/(campaign+'_gpu_telemetry.jsonl')).open('x') as stream:
    deadline=time.monotonic()+7200
    while True:
        alive=[pid for pid in pids if owned(pid)]
        v=dict(time=time.time(),worker_pids=pids,alive=alive,
            gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True),
            compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True))
        stream.write(json.dumps(v)+'\n');stream.flush()
        if not alive:break
        assert time.monotonic()<deadline,'Telemetry bound exceeded'
        time.sleep(5)
