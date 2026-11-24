from pathlib import Path
import subprocess,sys,json
r=Path('/home/researcher/mea-state2-r2-20261118/run02');base=r.parent
common=['--root',str(r),'--godot','/home/researcher/mea-r2-22554/runtime_20261116_01/assets/godot/Godot_v4.7.2-stable_linux.x86_64','--output',str(base/'terminal_probe16_01')]
stages=[]
for mode in ('collect','verify'):
 with (base/('terminal16_01_'+mode+'.log')).open('x') as log:
  p=subprocess.run([sys.executable,'-u',str(base/'terminal_probe_v2.py'),mode,*common],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
 stages.append({'mode':mode,'exit_code':p.returncode})
 if p.returncode:break
result={'passed':len(stages)==2 and all(s['exit_code']==0 for s in stages),'stages':stages,'technical_only':True}
(base/'terminal16_01_validation_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
