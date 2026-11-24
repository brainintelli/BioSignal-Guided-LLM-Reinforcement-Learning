"""Technical-only native terminal coverage for the frozen new collector; no efficacy use."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import os
from types import SimpleNamespace

from agent.m9_env import GodotEnv
from agent.m9_contract import Reward
from agent.mea_contract import observation_key
from tools.mea_offline import evidence,policy_statistics,validate_policy_statistics,_value
from mea_state2_r2.data import Dataset
from mea_state2_r2.features import digest
from mea_state2_r2.protocol import GAMMA,Rewards,cell
from mea_state2_r2.policy import new_policy
from mea_state2_r2.train import collect,prepare,stored,write
from mea_state2_r2.trajectory import audit,finish
from mea_state2_r2.verify import reward_audit

WORLD=4126101


def environment(args):
    return GodotEnv(str(args.godot),str(args.root/'project'),map_mode='infinite_freeze',navigation_steps=4)


def prefix(env,count):
    obs=env.reset(WORLD)['observation'];hashes=[]
    for _ in range(count):
        nxt=env.step({'verb':'wait'})
        assert not nxt['terminated'],'Fixture already terminal'
        hashes.append(digest(nxt['transition']));obs=nxt['observation']
    return obs,hashes


def run(args):
    import torch
    data=Dataset(args.root/'data')
    policy=new_policy(args.root/'model_config.json','C',0,data.normalization_id,16)
    args.output.mkdir(exist_ok=False)
    attempts=[];fixture=None
    with environment(args) as env:
        for steps in (488,484,480):
            obs,hashes=prefix(env,steps);start=obs;actions=[]
            for index in range(24):
                _,item=prepare(policy,data,obs,index,'C',0)
                torch.manual_seed(cell(0,'C')['action_seed']+index)
                action=policy.act([item],with_reference=True)['actions'][0]
                nxt=env.step(action);actions.append(action);obs=nxt['observation']
                if nxt['terminated']:break
            attempts.append(dict(prefix_steps=steps,count=len(actions),terminal=nxt['terminated']))
            if nxt['terminated'] and len(actions)>=2:
                fixture=dict(prefix_steps=steps,prefix_hashes=hashes,start_observation=start,actions=actions,
                    count=len(actions),world_seed=WORLD,attempts=attempts)
                break
    if fixture is None:raise RuntimeError('No eligible terminal fixture in preregistered technical scout')
    write(args.output/'fixture.json',fixture)
    reports={}
    for case,extra in [('tail',0),('middle',2)]:
        rd=args.output/case;rd.mkdir()
        with environment(args) as env:
            obs,hashes=prefix(env,fixture['prefix_steps'])
            assert obs==fixture['start_observation'] and hashes==fixture['prefix_hashes']
            rewards=Rewards(True)
            rows,boot,_,_,_=collect(policy,data,env,obs,Reward(obs,'bottleneck',GAMMA),rewards,
                WORLD,'C',0,0,fixture['count']+extra,rd)
        assert [r['index'] for r in rows if r['terminal']]==[fixture['count']-1]
        assert [r['action'] for r in rows[:fixture['count']]]==fixture['actions']
        terminal=rows[fixture['count']-1]
        assert terminal['reward_record']['r2_contribution']==0 and terminal['next_snapshot'] is None
        assert 'reset' in terminal and terminal['reset']['seed']==WORLD+1
        if case=='tail':
            assert boot==dict(terminal=True,value=0.,snapshot=None,prepared=None)
            modified=deepcopy(rows);finish(modified,1e9,GAMMA,.999)
            assert all(a['advantage']==b['advantage'] and a['return']==b['return'] for a,b in zip(rows,modified))
        else:
            first=rows[fixture['count']];player=first['next_observation']['player']
            fresh=Rewards(True).score(first['r1_score']['reward'],first['duration'],first['terminal'],
                dict(H=player['health'],F=player['hunger'],W=player['thirst']),observation_key(first['next_observation']))
            assert first['seed']==WORLD+1 and first['reward_record']==fresh
            assert boot['terminal'] is False
        active=()
        for row in rows:active=reward_audit(row,active,True)
        error=validate_policy_statistics(policy_statistics(policy,[r['prepared'] for r in rows],
            [r['verb'] for r in rows],[r['target'] for r in rows]),rows)
        write(rd/'rows.json',[stored(r) for r in rows]);write(rd/'bootstrap.json',boot)
        reports[case]=dict(passed=True,decisions=len(rows),terminal_index=fixture['count']-1,statistic_error=error)
    write(args.output/'result.json',dict(passed=True,technical_only=True,efficacy_included=False,ppo_update_performed=False,
        hardware_commands_sent=0,collector_pid=os.getpid(),cases=reports))
    print(json.dumps(reports))


def verify(args):
    data=Dataset(args.root/'data')
    policy=new_policy(args.root/'model_config.json','C',0,data.normalization_id,16)
    fixture=json.loads((args.output/'fixture.json').read_text())
    report=json.loads((args.output/'result.json').read_text())
    assert os.getpid()!=report['collector_pid'] and report['passed'] is True
    checks={}
    for case in ('tail','middle'):
        rd=args.output/case;rows=json.loads((rd/'rows.json').read_text());boot=json.loads((rd/'bootstrap.json').read_text())
        prepared=[];active=();seed=WORLD
        with environment(args) as env:
            obs,hashes=prefix(env,fixture['prefix_steps'])
            assert obs==fixture['start_observation'] and hashes==fixture['prefix_hashes']
            tracker=Reward(obs,'bottleneck',GAMMA)
            for index,row in enumerate(rows):
                assert row['observation']==obs and row['index']==index and row['seed']==seed
                snap,item=prepare(policy,data,obs,index,'C',0)
                assert snap==row['snapshot'] and evidence(item)==row['prepared']
                assert item['branches'][row['verb']][row['target']]==row['action']
                nxt=env.step(row['action']);score=tracker.score(nxt['transition'],nxt['observation'])
                assert nxt['transition']==row['transition'] and nxt['observation']==row['next_observation']
                assert nxt['terminated']==row['terminal'] and score==row['r1_score']
                active=reward_audit(row,active,True);prepared.append(item);obs=nxt['observation']
                if row['terminal']:
                    seed+=1;obs=env.reset(seed)['observation'];tracker=Reward(obs,'bottleneck',GAMMA)
                    assert row['reset']==dict(seed=seed,observation=obs)
            if rows[-1]['terminal']:
                assert boot==dict(terminal=True,value=0.,snapshot=None,prepared=None)
                audit(rows,1e9,GAMMA,.999)
            else:
                snap,item=prepare(policy,data,obs,len(rows),'C',0)
                assert snap==boot['snapshot'] and evidence(item)==boot['prepared']
                assert abs(_value(policy,item)-boot['value'])<1e-6
        audit(rows,boot['value'],GAMMA,.999)
        error=validate_policy_statistics(policy_statistics(policy,prepared,[r['verb'] for r in rows],[r['target'] for r in rows]),rows)
        checks[case]=dict(passed=True,native_replay=True,statistic_error=error)
    write(args.output/'independent_verification.json',dict(passed=True,cases=checks,verifier_pid=os.getpid(),hardware_commands_sent=0))
    print(json.dumps(checks))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('collect','verify'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--godot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();(run if args.mode=='collect' else verify)(args)
