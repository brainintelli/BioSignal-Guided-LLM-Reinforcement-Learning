"""Read-only score-blind watcher. Stopping this process does not stop training."""
import json,time
from pathlib import Path
r=Path('/home/researcher/mea-stim-offline-20261117');root=r/'experiment01'
while True:
    job=json.loads((r/'job.json').read_text());s=json.loads((root/'batch_status.json').read_text())
    out=dict(epoch=time.time(),complete=s['complete'],passed=s['passed'],accepted=sum(len(c['stages'])==3 and all(q['exit_code']==0 for q in c['stages']) for c in s['cells']))
    if s.get('error'):out['error']=s['error']
    if s['cells']:
        c=s['cells'][-1];out['cell']=f"b{c['block']:02d}_{c['arm']}";stage=['train','evaluate','verify'][min(2,len(c['stages']))];out['stage']=stage
        p=root/(out['cell']+'_'+stage+'.log')
        if p.exists():
            with p.open('rb') as f:f.seek(max(0,p.stat().st_size-16000));lines=f.read().decode(errors='replace').splitlines()
            events=[]
            for line in lines:
                try:q=json.loads(line)
                except ValueError:continue
                if isinstance(q,dict) and 'event' in q:events.append({k:q[k] for k in ['event','index','round','phase','context','seconds','applied_steps'] if k in q})
            out['progress']=events[-2:]
            if out.get('error'):out['failure_tail']='\n'.join(lines[-10:])
    if 'exit_code' in job:out['job']=job
    print(json.dumps(out),flush=True)
    if 'exit_code' in job:break
    time.sleep(45)
