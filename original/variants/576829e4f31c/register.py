"""Register paired maps, held-out MEA inputs, endpoints and checkpoint pins before any games."""
import hashlib
import json
import math
import random
import sys
from bisect import bisect_left
from datetime import datetime
from pathlib import Path

HERE=Path(__file__).resolve().parent
TRAIN=HERE.parent
SESSION=TRAIN.parent
ARCHIVE=SESSION/'offline-mea-stim-ablation/archive-frozen-60tick'
WORK=SESSION.parent.parent
sys.path[:0]=[str(WORK/'standalone/mea_state2_r2'),str(WORK/'standalone/mea_live_pair')]
from prepare_crossover_study import prepare
from mea_state2_r2.features import digest,extract,transform

def read(path):return json.loads(path.read_text(encoding='utf8'))
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def register():
    train=read(TRAIN/'design.json');ctx=read(TRAIN/'context.json')
    source=SESSION/'cpp-crossover-v2'; file=source/'cpp-spikes.jsonl'
    assert prepare(source,read(SESSION/'offline-ppo/protocol.json'))==ctx
    assert sha(file)==ctx['source_sha256'] and digest(ctx)==train['context_sha']
    old_game=read(ARCHIVE/'severe/design.json')
    acks=[datetime.fromisoformat(r['utc']).timestamp() for line in (source/'api_audit.jsonl').read_text().splitlines()
          if (r:=json.loads(line))['kind']=='ack' and r['command'].startswith('sequence_send ') and r['reply']=='OK']
    assert len(acks)==18
    plan=read(source/'design.json');roi=plan['roi'];roiset=set(roi)
    assert digest(roi)==ctx['roi_sha']
    spikes=sorted((ms,ch,amp) for line in file.read_bytes().splitlines() for ms,ch,amp,_ in [json.loads(line)] if ch in roiset)
    times=[s[0] for s in spikes];windows=[]
    for trial in range(13,19):
        assert plan['trials'][trial-1]['block'] in (5,6)
        for offset in range(6,16):
            start=math.ceil(1000*(acks[trial-1]+offset));stop=start+1000
            assert times[0]<start<stop<=times[-1]
            if trial<18:assert stop<1000*acks[trial]
            ev=spikes[bisect_left(times,start):bisect_left(times,stop)]
            raw=list(extract(ev,start,stop,roi,1000))
            windows.append(dict(index=30000+len(windows),frame_start=start,frame_end=stop,trial=trial,offset=offset,
                 count=len(ev),features=raw,normalized=list(transform(raw,ctx['normalization'])),events_sha=digest(ev)))
    assert len(windows)==60 and all(a['frame_end']<=b['frame_start'] for a,b in zip(windows,windows[1:]))
    assert max(w['frame_end'] for k in ('train','bootstrap') for w in ctx['series'][k])<=windows[0]['frame_start']
    for old in ctx['series']['holdout']:
        w=next(w for w in windows if w['frame_start']==old['frame_start'])
        assert all(w[k]==old[k] for k in ('normalized','events_sha','count','features','trial','offset'))
    seen={179999999}
    for f in SESSION.parent.rglob('design.json'):
        if HERE in f.parents:continue
        obj=read(f)
        if not isinstance(obj,dict):continue
        for k in ('worlds','training_seeds','holdout_seeds','seeds'):
            seen.update(v for v in obj.get(k,[]) if isinstance(v,int))
        if isinstance(obj.get('episodes'),dict):
            seen.update(v for seq in obj['episodes'].values() if isinstance(seq,list) for v in seq if isinstance(v,int))
    worlds=random.Random(2202160991).sample(range(191000000,192000000),48)
    assert len(set(worlds))==48 and not seen.intersection(worlds)
    assert (train['alpha'],train['beta'],train['effective_r2_limit_relative_to_r1'])==(1.,.5,.05)
    pins={};ratios={};row_counts={}
    for seed in train['training_seeds']:
        pair=[]
        for arm in train['arms']:
            d=TRAIN/'models'/str(seed)/arm;reg=read(d/'registration.json')
            mix=reg['reward_mix'];beta=.5 if arm=='ordered' else 0.
            assert reg['beta']==beta and reg['alpha']==1 and reg['study_digest']==digest(train)
            assert mix==train['reward_mix'] and mix['scale_fraction']==.1 and mix['gamma_tick']==.99985
            ups=read(d/'update.json');pair.append([reg['initial'],reg['config'],ups['actor_steps'],ups['critic_steps']])
            ck=read(d/'checkpoint/manifest.json');assert ck['registration_sha256']==digest(reg) and ck['actor_sha256']==ups['after']['actor']
            for file_name,h in ck['files'].items():assert sha(d/'checkpoint'/file_name)==h
            max_ratio=0.;n=0
            for i in range(1,train['updates']+1):
                for row in (d/f'train-{i}.jsonl').read_text(encoding='utf8').splitlines():
                    r=json.loads(row);native=r['r1_score']['reward'];part=r['contribution'];n+=1
                    assert abs(part)<=.05*abs(native)+1e-15 and math.isclose(r['reward'],native+part,abs_tol=1e-14)
                    assert math.isclose(part,beta*r['r2_scaled'],abs_tol=1e-15)
                    if native:max_ratio=max(max_ratio,abs(part/native))
            assert n==48
            key=f'{seed}/{arm}';pins[key]=sha(d/'checkpoint/manifest.json');ratios[key]=max_ratio;row_counts[key]=n
        assert pair[0]==pair[1]
    spec=dict(chip=ctx['chip'],sample=ctx['sample'],arms=train['arms'],seeds=train['training_seeds'],worlds=worlds,
        horizon=60,scenes=old_game['scenes'],scene_name=old_game['scene_name'],source_sha=ctx['source_sha256'],windows=windows,
        checkpoints=pins,training_max_ratio=ratios,training_steps=row_counts,prior_map_ids_checked=len(seen),
        runner_sha=sha(ARCHIVE/'freeze.py'),training_algorithm_sha=sha(TRAIN/'fixed_context_ppo.py'),context_sha=train['context_sha'],
        alpha=1.,beta_treatment=.5,cap=.05,stride=train['feedback_stride'],seed_bootstrap=2202160992,draws=40000,
        endpoints=['survived','end_tick','wait_to_drink_60'],ci_each=1-.05/3,
        risk_rule='Restricted time to first successful drink: action start tick after verifying an increase in water during the action. Assign 60 if no drink by the horizon, including death before drink. Also report actual times among drinkers and success probability.',
        decision='Treatment minus R1 within seed and map. Two-sided Bonferroni intervals for three outcomes. Stable improvement across all three requires all intervals entirely favorable and at least six of eight training seeds favorable on each. Support per endpoint also reported. All registered episodes included.',
        replay='Six 10-second observed blocks from trial 13 through 18 in chronological order (post acknowledgement offsets 6 through 15); one window per game tick; identical trace for both arms. Frozen policy; rewards audited but never supplied to the actor.')
    with (HERE/'design.json').open('x',encoding='utf8') as f:json.dump(spec,f,indent=2,ensure_ascii=False)
    print('registered',len(pins),len(worlds),len(windows),len(seen),'source',spec['source_sha'],'maximum training contribution',max(ratios.values()))
if __name__=='__main__':register()
