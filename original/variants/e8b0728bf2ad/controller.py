import sys,time,signal,json,traceback,hashlib
from pathlib import Path
work=Path(__file__).resolve().parent
sys.path.insert(0,'/home/researcher/brainDesktop/mea-standby-20261118')
import session as s
b=s.load(work/'binding.json')
assert hashlib.sha256(Path(s.__file__).read_bytes()).hexdigest()==s.load(work/'control.json')['session_sha256']
assert time.time()<s.binding(b)+5700
def save(name,data):
 p=work/(name+'.json');t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2,allow_nan=False));t.replace(p)
def stage(name):save('phase',dict(stage=name,time=time.time()))
for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM):signal.signal(sig,s.expiry)
study=s.Study(); result=dict(completed=False,qualification_passed=None)
try:
 stage('identity-localization-connected-baseline')
 base=study.open(b)
 if any(base['resolved_binding'].get(k)!=val for k,val in b.items()):raise RuntimeError('Acquisition binding changed')
 stage('registered-nine-trial-qualification')
 qual=study.calibrate();save('qualification',qual)
 result.update(completed='error' not in qual,qualification_passed=qual['passed'],qualification_trials=len(qual['trials']),error=qual.get('error'))
except BaseException as exc:
 result.update(error=repr(exc),traceback=traceback.format_exc())
finally:
 try:result['close']=study.close()
 except BaseException as exc:result['close_error']=repr(exc)
 signal.setitimer(signal.ITIMER_REAL,0)
 result['ended']=time.time()
 result['open_h5']=s._server_snapshot()[1]
 save('end',result)
 stage('closed')
