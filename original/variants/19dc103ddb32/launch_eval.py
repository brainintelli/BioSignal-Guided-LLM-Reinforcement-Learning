from pathlib import Path
import json,os,sys,subprocess,time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from resources import PYTHON,environment
q=list(json.loads((ROOT/'design.json').read_text())['seeds'])
busy={};avoid={}
while q or busy:
    for g,(seed,p,log) in list(busy.items()):
        if p.poll() is None:continue
        log.close();del busy[g]
        if p.returncode:
            f=Path(log.name);txt=f.read_text(errors='replace')
            if 'OutOfMemoryError' not in txt:raise RuntimeError((seed,g,p.returncode))
            avoid.setdefault(seed,set()).add(g);f.rename(f.with_suffix('.gpu'+str(g)+'.oom'))
            if len(avoid[seed])>=7:raise RuntimeError('Repeated OOM')
            q.append(seed)
    if q:
        listing=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        for free,g in sorted([(int(s.split(',')[1]),int(s.split(',')[0])) for s in listing.splitlines()],reverse=True):
            if g in busy or free<11500:continue
            seed=next((a for a in q if g not in avoid.get(a,())),None)
            if seed is None:continue
            q.remove(seed);log=(ROOT/(str(seed)+'.log')).open('a')
            cmd=[PYTHON,'-u',ROOT/'freeze.py',str(seed)];p=subprocess.Popen(cmd,env=dict(environment(),CUDA_VISIBLE_DEVICES=str(g)),stdout=log,stderr=subprocess.STDOUT)
            busy[g]=(seed,p,log);print('start',seed,g,free,flush=True)
    time.sleep(4)
print('all frozen evaluations complete',flush=True)
