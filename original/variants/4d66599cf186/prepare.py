"""Calibrate on archived control training steps and register the matched offline rerun."""
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import sys

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'offline-ppo'
sys.path.insert(0,str(HERE))
from replay_reward import ReplayRewardMix

def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):
    with p.open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    spec=read(PREVIOUS/'design.json')
    gamma=.99985
    old_beta=spec['beta']
    reference=[]
    source_hashes={}
    for seed in spec['training_seeds']:
        for i in range(1,spec['updates']+1):
            path=PREVIOUS/'models'/str(seed)/'r1'/f'train-{i}.jsonl'
            rows=[json.loads(row) for row in path.read_text(encoding='utf8').splitlines()]
            assert len(rows)==spec['batch']
            for r in rows:
                native=r['r1_score']['reward']
                assert r['reward']==native and r['contribution']==0
                duration=math.expm1(math.log(gamma)*r['duration'])/math.expm1(math.log(gamma))
                reference.append(abs(native)/duration)
            source_hashes[path.relative_to(PREVIOUS).as_posix()]=sha(path)
    calibration={'schema':'bounded_native_r1.v1','reference_per_tick':statistics.median(reference),'scale_fraction':.1,
                 'gamma_tick':gamma,'fit_source':'previous same-sample R1-only training decisions','fit_steps':len(reference)}
    assert len(reference)==384
    mix=ReplayRewardMix(calibration['reference_per_tick'],.1,1.,.5,gamma)
    measured=[]
    for seed in spec['training_seeds']:
        for i in range(1,spec['updates']+1):
            for line in (PREVIOUS/'models'/str(seed)/'ordered'/f'train-{i}.jsonl').read_text(encoding='utf8').splitlines():
                r=json.loads(line)
                native=r['r1_score']['reward'];sign=r['r2'];ticks=r['duration']
                scaled,term,total=mix.score(native,sign,ticks)
                old=old_beta*sign*math.expm1(ticks*math.log(gamma))/math.expm1(math.log(gamma))
                assert math.isclose(old,r['contribution'],abs_tol=1e-14) and math.isclose(native+old,r['reward'],abs_tol=1e-14)
                assert abs(term)<=.05*abs(native)+1e-15 and (native==0 or total*native>0)
                measured.append([abs(native),sign,abs(old),abs(term)])
    summary={}
    for i,key in [(2,'historical'),(3,'new_same_steps')]:
        active=[row[i]/row[0] for row in measured if row[0] and row[1]]
        summary[key]={'sum_abs_contribution_over_sum_abs_r1':sum(x[i] for x in measured)/sum(x[0] for x in measured),
            'ratio_median_active':statistics.median(active),'ratio_max_active':max(active),'over_r1_steps':sum(x[i]>x[0] for x in measured)}
    prior=set()
    for f in HERE.parent.parent.rglob('design.json'):
        s=read(f)
        if not isinstance(s,dict):continue
        for k in ('training_seeds','holdout_seeds','worlds','seeds'):
            prior.update(x for x in s.get(k,[]) if isinstance(x,int))
        if isinstance(s.get('episodes'),dict):
            prior.update(x for seq in s['episodes'].values() if isinstance(seq,list) for x in seq if isinstance(x,int))
    worlds=random.Random(220210501).sample(range(186000000,187000000),48)
    assert not prior.intersection(worlds)
    spec.update(study='22021_bounded_r2_offline_pair',alpha=1.,beta=.5,reward_mix=calibration,holdout_seeds=worlds,
                calibration_design_sha=sha(PREVIOUS/'design.json'),effective_r2_limit_relative_to_r1=.05)
    write(HERE/'calibration.json',{'control_train_sha256':source_hashes,'steps':len(measured),'active':sum(bool(x[1]) for x in measured),
                                    'reward_mix':calibration,'old_beta':old_beta,'audit':summary})
    write(HERE/'design.json',spec)
    for name in ('context.json','model_config.json','verify_game.py'):
        shutil.copyfile(PREVIOUS/name,HERE/name)
    print(summary)
if __name__=='__main__':main()
