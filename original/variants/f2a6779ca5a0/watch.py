"""Start the bound on-device assay once, only when a new electrode has qualified."""
import json
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from acquire import connect,remote,STAGE,ENV,PYTHON
D=STAGE+'/records/frequency-retest'
QUERY="""from pathlib import Path
import sys,json
p=Path(sys.argv[1]);o=p/'single-site-v1';state={}
for f in ('selected-scan.json','survey/overview.json','survey-confirm/rank.json','single-site-v1/sites.json','single-site-v1/neutral-check.json','single-site-v1/end.json'):
 q=p/f
 if q.exists():
  v=json.loads(q.read_bytes())
  if f.endswith('rank.json'): v={k:v[k] for k in ('mapped','active','pass_hist','new')}
  if f.endswith('overview.json'):v={k:v[k] for k in ('actual_electrodes','screen_candidates','candidates_two_checks')}
  if f.endswith('neutral-check.json'): v={k:v[k] for k in ('settled_rail_counts','quiet_noise_fraction')}
  state[f]=v
state['pulse_trains']=len(list(o.glob('attempt-*.json')))
state['screened']=len(list((p/'survey').glob('field-*/screen.json')))
state['complete_rows']=len(list(o.glob('trial_*-evidence.json')))
for n in ('survey','single'):
 log=p/(n+'.log')
 if log.exists() and 'Traceback' in log.read_text(errors='replace'):state[n+'_error']=log.read_text(errors='replace')[-400:]
print(json.dumps(state))"""
START="""import subprocess,sys,json
from pathlib import Path
p=Path(sys.argv[1])
assert (p/'selected-scan.json').is_file() and not (p/'single-site-v1').exists()
with (p/'single.log').open('xb') as log:
 pr=subprocess.Popen(json.loads(sys.argv[2]),cwd=p,stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
print(pr.pid)"""
if __name__=='__main__':
    launched=False;last=None
    for _ in range(230):
        with connect() as c:
            state=json.loads(remote(c,QUERY,[D]))
            if state!=last:print(json.dumps(state,ensure_ascii=False),flush=True);last=state
            if 'single-site-v1/end.json' in state or state.get('single_error') or state.get('survey_error'):break
            if 'selected-scan.json' in state and not launched:
                print('Recorder',remote(c,START,[D,json.dumps(ENV+[PYTHON,D+'/one_hz.py',D])]),flush=True);launched=True
            if 'survey-confirm/rank.json' in state and not state['survey-confirm/rank.json']['new']:break
            if 'survey/overview.json' in state and not state['survey/overview.json']['screen_candidates']:break
        time.sleep(12)
