"""Sequential fixed-budget child processes; failed cells are not replaced."""
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

from agent.mea_ablation import ARMS


def main():
    p=argparse.ArgumentParser()
    for key in ('root','source','model-config','godot'):p.add_argument('--'+key,required=True)
    a=p.parse_args();root=Path(a.root)
    if (root/'batch_status.json').exists():raise FileExistsError('never overwrite a batch')
    base=[sys.executable,'-m','tools.mea_ablation']
    common=['--root',a.root,'--source',a.source,'--model-config',a.model_config,'--godot',a.godot]
    status=dict(complete=False,cells=[],planned_cells=24)
    orders=list(itertools.permutations(ARMS))
    def persist():
        temporary=root/'batch_status.partial.json'
        temporary.write_text(json.dumps(status,indent=2))
        temporary.replace(root/'batch_status.json')
    persist()
    for block in range(8):
        for arm in orders[block%6]:
            cell=dict(block=block,arm=arm,stages=[]);status['cells'].append(cell);persist()
            for stage in ('cell','verify'):
                cmd=base+[stage]+common+['--block',str(block),'--arm',arm]
                if stage=='verify':cmd+=['--model','--result',str(root/f'verified_b{block:02d}_{arm}.json')]
                started=time.monotonic()
                with (root/f'b{block:02d}_{arm}_{stage}.log').open('w') as log:
                    try:
                        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=600)
                        code=result.returncode;error=None
                    except subprocess.TimeoutExpired:
                        code=-1;error='600-second stage timeout'
                cell['stages'].append(dict(stage=stage,exit_code=code,error=error,seconds=time.monotonic()-started));persist()
                if code:break
    status['complete']=True
    status['passed']=all(len(c['stages'])==2 and all(s['exit_code']==0 for s in c['stages']) for c in status['cells'])
    persist()
    return 0 if status['passed'] else 1


if __name__=='__main__':sys.exit(main())
