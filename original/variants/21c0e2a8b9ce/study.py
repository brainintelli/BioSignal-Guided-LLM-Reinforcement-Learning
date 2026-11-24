"""File-held-out 22554 PPO and checkpoint-frozen Godot evaluation."""
import hashlib
import json
import sys
from pathlib import Path

import torch
from clock_replay import require, sha
from evaluate_fresh_maps import save, verify_engine
from livepair.protocol import check_window
from mea_online.m9_runtime import M9Runtime, OFFLINE_PULSE, digest
from mea_state2_r2.features import fit
from mea_state2_r2.reward_mix import recording_mix
from train_long_recorded import decisions, prepare, read


def donors(plan, data, seed, arm):
    hold = plan['heldout'][str(seed)]
    train = [k for k in 'HWF' if k!=hold]
    for k,record in data.items():
        require(record['sha256']==plan['source'][k]['sha256'] and len(record['channels'])>=50
                and len(record['windows'])==2, 'Source or readout failed')
        for dt,w in zip(plan['offsets'],record['windows']):
            check_window(w,plan['binding'])
            require(w['origin']==k and w['start']==plan['source'][k]['event']+dt*20000, 'Window origin differs')
    norm=fit([data[k]['windows'][0]['features'] for k in train])
    def make(sources,donor_sites):
        return [(s,d,phase,data[d]['windows'][i]) for s,d in zip(sources,donor_sites)
                for i,phase in enumerate(('pre','post'))]
    return norm,make(train,train if arm=='ordered' else train[::-1]),data[train[0]]['prime'],make([hold],[hold]),data[hold]['prime']


def train(root,out,arm,seed,plan,data):
    norm,order,prime,_,_=donors(plan,data,seed,arm)
    batch=plan['batch']
    cfg=dict(source='recorded',experiment=OFFLINE_PULSE,r2_enabled=True,arm=arm,stimulation=False,
             normalization=norm,normalization_id=digest(norm),binding_sha256=plan['binding'],devices=list(range(8)),
             batch_size=batch,seed=seed,training=dict(minibatch=batch,critic_minibatch=batch),
             reward_mix=recording_mix(sha(root/'paired-windows.json'),plan['mix']),**read(root/'settings.json'))
    reg=dict(config=cfg,plan=sha(root/'paired-protocol.json'),windows=sha(root/'paired-windows.json'),
             hold=plan['heldout'][str(seed)],pairs=[(s,d,p,w['id']) for s,d,p,w in order])
    verify_engine(cfg,plan['engine'])
    save(out/'registration.json',reg)
    torch.manual_seed(seed)
    rt=M9Runtime(cfg,out)
    try:
        initial=rt.state()
        rows,updates,traces=[],[],[]
        for row,tr,obs,prior in decisions(rt,cfg,order,prime,seed,plan['steps'],True):
            require(abs(row['r2_contribution'])<=.05*abs(row['r1'])+1e-14,'R2 magnitude exceeded')
            rows.append(row)
            traces.append(tr)
            if len(rows)==batch:
                uid=len(updates)+1
                boot=0. if row['terminal'] else rt.value(prepare(rt,cfg,obs,prior,f'bootstrap-{seed}-{uid}'))
                upd=rt.update(rows,boot,uid)
                require(upd['world_size']==8 and upd['unique_decisions']==batch and upd['actor_changed'] and upd['critic_changed'],'Incomplete PPO update')
                updates.append(upd)
                rows=[]
                save(out/'progress.json',dict(updates=uid))
        cp=rt.checkpoint(out/'checkpoint',digest(reg),len(updates))
        require(cp['reload_verified'] and len(traces)==96 and len(updates)==8,'Checkpoint incomplete')
        save(out/'training-trace.json',traces)
        save(out/'updates.json',updates)
        save(out/'trained.json',dict(initial=initial,checkpoint=cp,final=rt.state(),steps=len(traces)))
    finally:
        rt.close()


def evaluate(root,out,arm,seed,plan,data):
    if arm=='pure':
        model=plan['pure'][str(seed)]
        cp=Path(model['checkpoint'])
        reg=read(cp.parent/'registration.json')
        require(digest(reg)==model['registration'] and reg['budget']['train_steps']==plan['steps']
                and reg['budget']['rollout_batch']==plan['batch'],'Pure training provenance differs')
        state,manifest=model['state'],model['manifest']
        order,prime=[],None
    else:
        meta=read(out/'trained.json')
        cp=out/'checkpoint'
        state,manifest=meta['final'],meta['checkpoint']['manifest_sha256']
        reg=read(out/'registration.json')
        _,_,_,order,prime=donors(plan,data,seed,arm)
    require(sha(cp/'manifest.json')==manifest,'Checkpoint manifest differs')
    cfg=reg['config']
    require(cfg['seed']==seed and cfg['arm']==arm and not cfg['stimulation'],'Model mismatch')
    verify_engine(cfg,plan['engine'])
    torch.manual_seed(seed)
    rt=M9Runtime(cfg,out,frozen_evaluation=True)
    try:
        rt.core.load_checkpoint(cp,digest(reg))
        require(rt.state()==state,'Loaded weights differ')
        games={}
        for gs in plan['maps']:
            path=out/f'map-{gs}.json'
            if path.exists():
                game=read(path)
                require(game['model']==manifest and game['protocol']==sha(root/'paired-protocol.json') and game['seed']==gs,'Cached map mismatch')
            else:
                obs=rt.reset(gs)
                first=digest(obs)
                prior=prime
                ticks,actions=[],[]
                term=None
                for i in range(max(plan['ticks'])):
                    p=prepare(rt,cfg,obs,prior,f'eval-{gs}-{i}')
                    torch.manual_seed(gs*10+i)
                    a=rt.act(p)['action']
                    step=rt.step(a)
                    tr=step['transition']
                    if not ticks:
                        ini=tr['before']
                        ticks.append([0,ini['health'],ini['hunger'],ini['thirst']])
                    start=len(ticks)-1
                    legal={v['verb'] for branch in p['branches'] for v in branch}
                    require(len(tr['trace'])==step['duration'],'Truncated action trace')
                    for t in tr['trace']:
                        require(t['tick']==ini['tick']+len(ticks),'Game clock mismatch')
                        ticks.append([t['visited_count']-ini['visited_count'],t['health'],t['hunger'],t['thirst']])
                    actions.append([start,len(ticks)-1,a['verb'],'eat' in legal,'drink' in legal])
                    if step['terminal']:
                        term=len(ticks)-1
                        break
                    if len(ticks)>max(plan['ticks']):break
                    obs,prior=step['observation'],None if arm=='pure' else order[i%len(order)][3]
                require(term is not None or len(ticks)>max(plan['ticks']),'Incomplete rollout')
                if len(ticks)<=max(plan['ticks']):ticks.extend([ticks[-1]]*(1+max(plan['ticks'])-len(ticks)))
                game=dict(seed=gs,model=manifest,protocol=sha(root/'paired-protocol.json'),first=first,
                          terminal=term,ticks=ticks[:max(plan['ticks'])+1],actions=actions)
                save(path,game)
            games[gs]=sha(path)
        require(rt.state()==state,'Frozen weights changed')
        save(out/'done.json',dict(arm=arm,seed=seed,state=state,manifest=manifest,games=games))
    finally:
        rt.close()


if __name__=='__main__':
    root=Path(sys.argv[1])
    mode,arm=sys.argv[2:4]
    seed=int(sys.argv[4])
    out=root/'runs'/f'{arm}-{seed}'
    out.mkdir(parents=True,exist_ok=True)
    p,d=read(root/'paired-protocol.json'),read(root/'paired-windows.json')
    (train if mode=='train' else evaluate)(root,out,arm,seed,p,d)
