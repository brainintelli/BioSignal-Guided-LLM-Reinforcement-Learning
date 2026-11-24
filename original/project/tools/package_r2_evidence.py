"""Pack completed offline results and traces; never export model weights or live recordings."""
import argparse,io,json,tarfile,os
from pathlib import Path

def load(p): return json.loads(p.read_bytes())
def pack220(out):
 for tag in ('002','005','010'):
  root=Path.home()/('mea-six-beta-22021-beta'+tag);prefix='22021/beta_'+tag
  with tarfile.open(out/f'evidence22021_{tag}.tgz.part','w:gz',compresslevel=2) as t:
   def add(p,rel): t.add(str(p),arcname=prefix+'/'+str(rel))
   for reg in root.glob('models/[0-9]*/*/registration.json'):
    d=reg.parent; f=d.relative_to(root)
    for n in ('registration.json','replay-audit.json','holdout.jsonl','update.json'): add(d/n,f/n)
    a=[]
    for p in sorted(d.glob('train-*.jsonl')):
     for line in p.open():
      r=json.loads(line);a.append([r['r1_score']['reward'],r['r2'],r['duration'],r['contribution'],r['reward']])
    assert len(a)==load(root/'design.json')['batch']*load(root/'design.json')['updates']; data=json.dumps(a,separators=(',',':')).encode();i=tarfile.TarInfo(prefix+'/'+str(f)+'/reward-audit.json');i.size=len(data);t.addfile(i,io.BytesIO(data))
  (out/f'evidence22021_{tag}.tgz.part').replace(out/f'evidence22021_{tag}.tgz');print(tag,flush=True)
def pack225(out,tag,arm,folder):
 root=Path.home()/'mea-offline-single-pulse/bd-22554-r2beta-20261120';d=root/folder/arm; per=d/'training'; b=root/('budget-beta_'+tag+'.json'); seeds=load(b)['training_seeds']; regs=[per/str(s) for s in seeds if (per/str(s)/'result.json').is_file()]
 dest=out/f'evidence22554_{tag}_{arm}.tgz'; prefix='22554/beta_'+tag
 with tarfile.open(str(dest)+'.part','w:gz',compresslevel=2,dereference=True) as t:
  t.add(str(b),arcname=prefix+'/'+b.name)
  for n in ('exit.json','training/result.json'):
   if (d/n).is_file():t.add(str(d/n),arcname=prefix+'/'+arm+'/'+n)
  for s in regs:
   for n in ('registration.json','result.json','training-trace.json','eval-trace.json','updates.json'):
    t.add(str(s/n),arcname=prefix+'/'+arm+'/training/'+s.name+'/'+n)
 Path(str(dest)+'.part').replace(dest);print(tag,arm,len(regs),dest.stat().st_size,flush=True)
if __name__=='__main__':
 import sys
 out=Path.home()/'r2-evidence-20261120';out.mkdir(exist_ok=True)
 if sys.argv[1]=='22021':pack220(out)
 else:pack225(out,*sys.argv[2:5])
