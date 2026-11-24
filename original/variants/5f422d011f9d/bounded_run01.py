from pathlib import Path
import subprocess,json,time
b=Path(__file__).parent;p=b/'project';results=[]
for stage in json.loads((b/'stages_run01.json').read_text()):
 start=time.time();error=None
 try:
  with (b/(stage['name']+'_run01.log')).open('xb') as log:
   r=subprocess.run(stage['command'],cwd=p,stdout=log,stderr=subprocess.STDOUT,timeout=stage['timeout']);code=r.returncode
 except Exception as exc:code=-1;error=str(exc)
 results.append(dict(name=stage['name'],exit_code=code,seconds=time.time()-start,error=error))
 (b/'status_run01.json').write_text(json.dumps(dict(stages=results,complete=False),indent=2)+'\n')
 if code:break
(b/'status_run01.json').write_text(json.dumps(dict(stages=results,complete=True,passed=len(results)==4 and all(s['exit_code']==0 for s in results)),indent=2)+'\n')
