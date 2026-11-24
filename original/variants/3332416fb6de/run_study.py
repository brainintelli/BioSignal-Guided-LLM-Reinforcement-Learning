"""Run the registered fits, audit every update, then freeze the models for paired evaluation."""
from pathlib import Path
from resources import PYTHON, environment
import json,hashlib,subprocess,sys,os
H=Path(__file__).resolve().parent; S=H/'stress'
sys.path.insert(0,str(S))
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(o):return hashlib.sha256(json.dumps(o,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def stage(name,script,cwd):
    with (H/(name+'.log')).open('a') as log:
        subprocess.run([PYTHON,'-u',script],cwd=cwd,env=dict(environment(),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false'),stdout=log,stderr=subprocess.STDOUT,check=True)
def main():
    tr=read(H/'design.json');scenes=read(S/'scenes.json');pilot=read(S/'scene-check.json')
    assert sha(S/'evaluate.py')==tr['stress_eval_code_sha'] and sha(S/'analyze.py')==tr['analysis_code_sha']
    stage('fit','schedule-fit.py',H)
    stage('fit-audit','audit_fits.py',H)
    checks=read(H/'training-audit.json');assert len(checks)==96
    spec={k:tr[k] for k in ('arms','training_seeds','worlds','windows','shuffle','horizon','source_sha','roi_sha','normalization','alpha','beta','draws','bootstrap_seed','primary','targeted_secondary','exploration','stable')}
    spec.update(scenes=scenes,pilot_sha=sha(S/'scene-check.json'),game_pins=pilot['pins'],play_sha=sha(H/'play60.py'),runner_sha=sha(S/'evaluate.py'),
            training_design_sha=digest(tr),training_audit_sha=sha(H/'training-audit.json'),states={f"{r['seed']}/{r['arm']}":r['after'] for r in checks})
    f=S/'design.json'
    if f.exists():assert read(f)==spec
    else:f.write_text(json.dumps(spec,indent=2,ensure_ascii=False))
    stage('eval','schedule.py',S)
    stage('analysis','analyze.py',S)
    print('all stages finished',flush=True)
if __name__=='__main__':main()
