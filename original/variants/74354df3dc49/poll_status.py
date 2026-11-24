"""Score-blind progress: never opens evaluation metrics or analysis results."""
import json,subprocess
from datetime import datetime,timezone
from pathlib import Path
r=Path(__file__).resolve().parent;e=r/'experiment01'
def read(p):return json.loads(p.read_text())
result={'utc':datetime.now(timezone.utc).isoformat(),'accepted_cells':0,'blocks':[]}
for p in sorted(e.glob('block_*_status.json')):
    b=read(p);result['accepted_cells']+=b['accepted_cells']
    brief={k:b[k] for k in ('block','gpu','complete','passed','accepted_cells','current_arm','current_stage','error') if k in b}
    log=e/f"b{b['block']:02d}_{b.get('current_arm')}_{b.get('current_stage')}.log"
    if log.exists():
        lines=log.read_text(errors='replace').splitlines()
        for line in reversed(lines):
            try:row=json.loads(line)
            except ValueError:continue
            if isinstance(row,dict) and 'event' in row:
                brief['progress']={k:row[k] for k in ('event','index','round','phase','context') if k in row};break
        if b.get('error'):brief['failure_tail']='\n'.join(lines[-18:])
    result['blocks'].append(brief)
if (e/'parallel_status.json').exists():result['parallel']=read(e/'parallel_status.json')
if (r/'job.json').exists():result['job']=read(r/'job.json')
result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip().splitlines()
print(json.dumps(result),flush=True)
