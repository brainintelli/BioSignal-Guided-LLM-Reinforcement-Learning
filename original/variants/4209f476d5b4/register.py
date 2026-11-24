"""Lock training worlds, hydration scoring, the archived MEA input and the fresh map test."""
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
SESSION=HERE.parent
PREV=SESSION/'offline-r2-bounded-005'
WORK=SESSION.parent.parent
sys.path[:0]=[str(WORK/'standalone/mea_live_pair'),str(WORK/'standalone/mea_state2_r2')]
from prepare_crossover_study import prepare
from mea_state2_r2.features import digest

def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,j):
    with p.open('x',encoding='utf8') as f:json.dump(j,f,ensure_ascii=False,indent=2)
def main():
    old=read(PREV/'design.json');previous_eval=read(PREV/'thirst-60-paired/design.json')
    ctx=prepare(SESSION/'cpp-crossover-v2',read(SESSION/'offline-ppo/protocol.json'))
    assert digest(ctx)==old['context_sha'] and ctx==read(PREV/'context.json')
    assert ctx['source_sha256']==previous_eval['source_sha'] and ctx['chip']=='22021'
    series=ctx['series'];wins=previous_eval['windows']
    assert len(wins)==60 and set(w['trial'] for w in wins)==set(range(13,19))
    assert max(w['frame_end'] for k in ('train','bootstrap') for w in series[k])<=wins[0]['frame_start']
    assert old['reward_mix']['scale_fraction']==.1 and old['alpha']==1 and old['beta']==.5
    train=old.copy()
    seen={179999999}
    for f in SESSION.parent.rglob('design.json'):
        if HERE in f.parents:continue
        obj=read(f)
        if not isinstance(obj,dict):continue
        for key in ('worlds','training_seeds','holdout_seeds','seeds'):
            seen.update(x for x in obj.get(key,[]) if isinstance(x,int))
        for seq in obj.get('episodes',{}).values() if isinstance(obj.get('episodes'),dict) else ():
            if isinstance(seq,list):seen.update(x for x in seq if isinstance(x,int))
    new_worlds=random.Random(220212007).sample(range(212000000,213000000),48)
    train_worlds=[w for seq in old['episodes'].values() for w in seq]
    fallback=random.Random(220212034).sample(range(210000000,211000000),len(train_worlds)*12)
    assert not (set(new_worlds)|set(fallback)) & seen
    assert len(set(train_worlds))==32 and not set(new_worlds)&set(fallback)
    restarts={str(w):fallback[12*i:12*(i+1)] for i,w in enumerate(train_worlds)}
    rule=dict(kind='low_water_successful_drink',water_max=20.0)
    train.update(study='22021_low_water_drink',starting_scene=previous_eval['scene_name'],starting_reserves=previous_eval['scenes'],
        feedback_rule=rule,feedback_stride=1,restart_worlds=restarts,holdout_seeds=new_worlds,
        endpoints=previous_eval['endpoints'],stability=previous_eval['decision'],interval=dict(draws=40000,quantiles=[.05/6,1-.05/6],seed=220212008))
    train.pop('minimum_reserve',None)
    # Training consumes all 48 registered windows. The long paired evaluation has its own tick-indexed input.
    train['eval_decisions']=len(series['holdout'])
    write(HERE/'design.json',train)
    for name in ('context.json','model_config.json','resources.py'):
        shutil.copyfile(PREV/name,HERE/name)
    for name in ('fixed_context_ppo.py','replay_reward.py'):shutil.copyfile(WORK/'standalone/mea_live_pair'/name,HERE/name)
    eval_spec={k:v for k,v in previous_eval.items() if k not in ('training_max_ratio','training_steps','checkpoints','training_algorithm_sha','prior_map_ids_checked','worlds','seed_bootstrap','stride')}
    eval_spec.update(worlds=new_worlds,seed_bootstrap=220212008,feedback_rule=rule,training_design_sha=digest(train),
        training_algorithm_sha=sha(HERE/'fixed_context_ppo.py'),reward_function_sha=sha(HERE/'replay_reward.py'),
        prior_map_ids_checked=len(seen),prior_MEA_window_design_sha=sha(PREV/'thirst-60-paired/design.json'))
    write(HERE/'evaluation/plan.json',eval_spec)
    print('registered',len(new_worlds),len(old['training_seeds']),'source',ctx['source_sha256'])
if __name__=='__main__':main()
