"""Offline eight-rank game PPO with recorded exogenous R2 and ordered or shuffled state2."""
import argparse
import json
import math
from pathlib import Path
import random

from livepair.protocol import check_window,snapshot
from mea_online.m9_runtime import DEFAULTS,M9Runtime,OFFLINE_PULSE,digest
from mea_state2_r2.protocol import BETA,GAMMA


def write(path,obj):
    with path.open('x',encoding='utf8') as out:json.dump(obj,out,allow_nan=False,ensure_ascii=False)


def windows(data,arm,training):
    study=data['study'];bid=digest(study);rows=data['trials']
    if len(rows)!=9 or len({r['source']['path'] for r in rows})!=9:
        raise ValueError('Nine distinct H5 runs required')
    if [(r['block'],r['site'],r['index']) for r in rows]!=[(b,k,b*3+i) for b,order in enumerate(study['order']) for i,k in enumerate(order)]:
        raise ValueError('Recorded schedule differs')
    for r in rows:
        if r['verified_pulses']!=1:raise ValueError('Missing verified event')
        for key,offset in [('pre',-60000),('post',240000)]:
            w=r[key];check_window(w,bid)
            if w['start']!=r['anchor']+offset:raise ValueError('Invalid measured interval')
    donors={r['index']:r for r in rows}
    if training and arm=='shuffled':
        rng=random.Random(225541119)
        for block in study['train_blocks']:
            group=[r for r in rows if r['block']==block];offset=rng.choice([1,2])
            for j,r in enumerate(group):donors[r['index']]=group[(j+offset)%3]
    include=study['train_blocks'] if training else study['test_blocks']
    return [(r['index'],donors[r['index']]['index'],phase,donors[r['index']][phase]) for r in rows if r['block'] in include for phase in ('pre','post')]


def trajectory(rt,cfg,order,prime,seed,train):
    import torch
    from agent.m9_contract import encode
    def prepared(obs,w,key):
        return rt.prepare(obs,snapshot(w,cfg,digest(encode(obs,DEFAULTS['state_variant'])),key))
    obs=rt.reset(seed);resets=0;rows=[];prior=prime
    phase='train' if train else 'eval'
    for i,(source_index,donor_index,window_type,w) in enumerate(order):
        prep=prepared(obs,prior,f'{phase}-{seed}-{i}')
        torch.manual_seed(seed*10+i)
        action=rt.act(prep);step=rt.step(action['action']);obs=step['observation']
        sign=None if step['terminal'] else 1 if window_type=='pre' else -1
        ticks=step['duration'];add=0. if sign is None else BETA*sign*math.expm1(ticks*math.log(GAMMA))/math.expm1(math.log(GAMMA))
        row=dict(**action,prepared=prep,terminal=step['terminal'],boundary=step['terminal'],duration=ticks,
                 reward=step['reward']+add,r1=step['reward'],r2=sign,r2_contribution=add,
                 transition=step['transition'],context_donor=prior['id'],next_donor=w['id'],
                 r2_trial=source_index,next_state2_trial=donor_index,recorded_phase=window_type)
        rows.append(row);prior=w
        if step['terminal']:
            if not train:break
            if i+1<len(order):
                resets+=1;obs=rt.reset(seed+10000*resets);prior=prime
    if not train or rows[-1]['terminal']:return rows,0.
    return rows,rt.value(prepared(obs,prior,f'train-{seed}-bootstrap'))


def metrics(rows):
    first=rows[0]['transition']['before'];last=rows[-1]['transition']['after']
    return dict(explored=last['visited_count']-first['visited_count'],health=last['health'],
                hunger=last['hunger'],thirst=last['thirst'],death=int(rows[-1]['terminal']),
                decisions=len(rows),r1=sum(r['r1'] for r in rows),return_sum=sum(r['reward'] for r in rows),
                ticks=sum(r['duration'] for r in rows))


def run(args):
    import torch
    data=json.loads(args.data.read_text());study=data['study'];bid=digest(study)
    base=json.loads(args.settings.read_text());prime=json.loads(args.prime.read_text())['prime'];check_window(prime,bid)
    train=windows(data,args.arm,True);test=windows(data,args.arm,False)
    if len(train)!=12 or len(test)!=6 or set(t[0] for t in train)&set(t[0] for t in test):raise ValueError('Invalid split')
    args.out.mkdir(exist_ok=False,parents=True);updates=[]
    for seed in study['training_seeds']:
        out=args.out/str(seed);out.mkdir()
        cfg=dict(source='recorded',experiment=OFFLINE_PULSE,r2_enabled=True,stimulation=False,
                 arm=args.arm,binding_sha256=bid,normalization=data['normalization'],
                 normalization_id=digest(data['normalization']),devices=list(range(8)),batch_size=len(train),
                 seed=seed,training=dict(minibatch=len(train),critic_minibatch=len(train)),**base)
        registration=dict(config=cfg,physical_files=len(data['trials']),replay_digest=digest(data),eval_seeds=study['game_eval_seeds'],
                          r2='recorded binary delivery, independent of game actions; identical schedule for both arms',
                          train_pairs=[(i,j,k) for i,j,k,w in train],test_pairs=[(i,j,k) for i,j,k,w in test])
        write(out/'registration.json',registration);torch.manual_seed(seed)
        rt=M9Runtime(cfg,out)
        try:
            initial=rt.state();rows,boot=trajectory(rt,cfg,train,prime,seed,True)
            write(out/'training-trace.json',[{k:v for k,v in r.items() if k not in ('prepared','transition','reference_verb','reference_target')} for r in rows])
            upd=rt.update(rows,boot,1)
            if not(upd['actor_changed'] and upd['critic_changed'] and upd['world_size']==8):
                raise ValueError('Missing eight-rank actor/critic update')
            saved=rt.checkpoint(out/'checkpoint',digest(registration),1)
            post_state=rt.state();evals=[]
            for gs in study['game_eval_seeds']:
                ev,_=trajectory(rt,cfg,test,prime,gs,False)
                evals.append(dict(seed=gs,**metrics(ev)))
            if rt.state()!=post_state:raise ValueError('Evaluation changed weights')
            result=dict(training_seed=seed,arm=args.arm,initial=initial,after=post_state,update=upd,checkpoint=saved,
                        training_steps=len(rows),eval=evals)
            write(out/'result.json',result);updates.append(result)
            print(json.dumps(dict(arm=args.arm,seed=seed,world=upd['world_size'],eval=evals)),flush=True)
        finally:rt.close()
    write(args.out/'result.json',dict(arm=args.arm,study=bid,replay_digest=digest(data),seeds=updates))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('data','prime','settings','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--arm',choices=('ordered','shuffled'),required=True)
    run(p.parse_args())
