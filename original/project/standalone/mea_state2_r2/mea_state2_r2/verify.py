"""Fresh-process model reload, native game replay, and independent reward audit."""
import json
import math
import os
from pathlib import Path
from agent.m9_contract import Reward
from agent.m9_env import GodotEnv
from agent.mea_contract import observation_key
from agent.mea_long_ablation import ORIGINAL_TENSORS
from tools.mea_offline import evidence, _value, policy_statistics, validate_policy_statistics
from .data import Dataset
from .features import digest
from .protocol import GAMMA, BETA, cell
from .trajectory import audit
from .train import prepare, write, metrics
from .integrity import check_baseline_identity,baseline_padded_critic_hash,check_round_summary,check_mirrors


def read(path):
    return json.loads(Path(path).read_text())


def equal(actual, expected, name):
    if actual != expected:
        raise ValueError('Independent verification differs: '+name)


def reward_audit(row, active, enabled):
    record=row['reward_record']
    r1=row['r1_score']['reward']
    p=row['next_observation']['player']
    reserves={'H':p['health'],'W':p['thirst'],'F':p['hunger']}
    if row['terminal']:
        active=()
        expected_sign=None
        contribution=0.
        expected_site=None
    else:
        active=tuple(k for k in ('H','W','F') if reserves[k] < (90 if k in active else 85))
        expected_site=min(active,key=lambda k:(reserves[k],('H','W','F').index(k))) if active else None
        expected_sign=(-1 if active else 1) if enabled else None
        contribution=(BETA*expected_sign*math.fsum(GAMMA**k for k in range(row['duration']))) if enabled else 0.
    equal(record['intent']['site'],expected_site,'encoded demand site')
    equal(record['r2_raw'],expected_sign,'binary reward sign')
    equal(record['r2_contribution'],contribution,'time-aware auxiliary contribution')
    equal(record['total_reward'],r1+contribution,'combined reward')
    equal(row['reward'],r1+contribution,'PPO reward')
    equal(record['real_output_verified'],False,'never real output')
    equal(record['hardware_commands_sent'],0,'no hardware')
    return active


def check_eval(policy,data,args,path,phase):
    rows=[json.loads(line) for line in (path/'rollout.jsonl').read_text().splitlines()]
    if not rows or len(rows)>150 or (len(rows)<150 and not rows[-1]['terminal']):
        raise ValueError('Incomplete evaluation')
    prepared=[]
    with GodotEnv(str(args.godot),str(args.project),map_mode='infinite_freeze',navigation_steps=4) as env:
        obs=env.reset(cell(args.block,args.arm)['eval_world_seed'])['observation']
        tracker=Reward(obs,'bottleneck',GAMMA)
        for index,row in enumerate(rows):
            equal(row['index'],index,'evaluation position')
            equal(row['observation'],obs,'evaluation observation')
            snap,item=prepare(policy,data,obs,index,args.arm,args.block,'eval',phase)
            equal(row['snapshot'],snap,'evaluation data')
            equal(row['prepared'],evidence(item),'evaluation inputs')
            equal(item['branches'][row['verb']][row['target']],row['action'],'legal sampled action')
            nxt=env.step(row['action'])
            score=tracker.score(nxt['transition'],nxt['observation'])
            for key,value in [('transition',nxt['transition']),('next_observation',nxt['observation']),
                              ('terminal',nxt['terminated']),('r1_score',score),('duration',score['elapsed_ticks'])]:
                equal(row[key],value,'evaluation '+key)
            if row['terminal'] and index!=len(rows)-1:
                raise ValueError('Evaluation continues after death')
            prepared.append(item)
            obs=nxt['observation']
    if args.arm=='A0':
        # Preserve the original 108-D baseline's single-row value reduction.
        statistics=[policy_statistics(policy,[p],[r['verb']],[r['target']])[0] for p,r in zip(prepared,rows)]
    else:
        statistics=policy_statistics(policy,prepared,[r['verb'] for r in rows],[r['target'] for r in rows])
    error=validate_policy_statistics(statistics,rows)
    expected=dict(full=metrics(rows),first16=metrics(rows,16))
    equal(read(path/'metrics.json'),expected,'reported evaluation metrics')
    return dict(passed=True,decisions=len(rows),statistic_error=error,metrics=expected)


def verify(args):
    import torch
    from agent.m9_policy import verify_checkpoint
    from tools.mea_shuffle_train import state
    from .policy import new_policy,load_checked
    data=Dataset(args.source)
    args.output.mkdir(parents=True,exist_ok=False)
    policy=new_policy(args.model_config,args.arm,args.block,data.normalization_id,16 if args.smoke else 128)
    initial=state(policy)
    training_summary=None
    if args.arm=='A0':
        check_baseline_identity(initial,baseline_padded_critic_hash(policy))
        training_result=None
    else:
        equal(initial,ORIGINAL_TENSORS,'original warm start')
        training_result=read(args.train_dir/'training_result.json')
        registration=read(args.train_dir/'registration.json')
        if registration['collection_pid']==os.getpid():
            raise ValueError('Fresh verification process required')
        equal(registration['data'],data.summary(),'data/normalization registration')
        equal(registration['schedule'],cell(args.block,args.arm),'seed registration')
        count=16 if args.smoke else 128
        rounds=1 if args.smoke else 3
        equal(len(training_result['rounds']),rounds,'round count')
        equal(training_result['initial'],initial,'training initial summary')
        equal(training_result['smoke'],args.smoke,'technical/effect separation')
        load_checked(policy,args.train_dir/'initial_checkpoint')
        equal(state(policy),initial,'initial checkpoint reload')
        verified_pre=None
        if not args.smoke:
            verified_pre=check_eval(policy,data,args,args.train_dir/'pre_after','after')['metrics']
        previous=args.train_dir/'initial_checkpoint'
        active=()
        audits=[];round_reports=[];verified_rounds=[]
        optimizer_before={'actor':0,'critic':0}
        with GodotEnv(str(args.godot),str(args.project),map_mode='infinite_freeze',navigation_steps=4) as env:
            seed=cell(args.block,args.arm)['world_seed']
            obs=env.reset(seed)['observation']
            tracker=Reward(obs,'bottleneck',GAMMA)
            for number in range(1,rounds+1):
                rd=args.train_dir/f'round_{number:02d}'
                report=read(rd/'report.json')
                rows=[json.loads(line) for line in (rd/'rollout.jsonl').read_text().splitlines()]
                boot=read(rd/'bootstrap.json')
                equal(len(rows),count,'round budget')
                load_checked(policy,previous)
                equal(state(policy),report['before'],'previous parameters')
                prepared=[]
                for offset,row in enumerate(rows):
                    index=(number-1)*count+offset
                    equal(row['index'],index,'training sequence')
                    equal(row['seed'],seed,'training episode')
                    equal(row['observation'],obs,'native training observation')
                    snap,item=prepare(policy,data,obs,index,args.arm,args.block)
                    equal(row['snapshot'],snap,'training snapshot')
                    equal(row['prepared'],evidence(item),'tokens/features')
                    equal(item['branches'][row['verb']][row['target']],row['action'],'model action')
                    nxt=env.step(row['action'])
                    score=tracker.score(nxt['transition'],nxt['observation'])
                    for key,value in [('transition',nxt['transition']),('next_observation',nxt['observation']),
                              ('terminal',nxt['terminated']),('r1_score',score),('duration',score['elapsed_ticks'])]:
                        equal(row[key],value,'training '+key)
                    expected_next=None if nxt['terminated'] else data.snapshot(index+1,'after',observation_key(nxt['observation']),args.arm,args.block,'train')
                    equal(row['next_snapshot'],expected_next,'next observation snapshot')
                    active=reward_audit(row,active,args.arm in ('B','C','D'))
                    obs=nxt['observation']
                    if row['terminal']:
                        seed+=1
                        obs=env.reset(seed)['observation']
                        tracker=Reward(obs,'bottleneck',GAMMA)
                        equal(row.get('reset'),dict(seed=seed,observation=obs),'terminal reset')
                    elif 'reset' in row:
                        raise ValueError('Unexpected reset')
                    prepared.append(item)
                old_error=validate_policy_statistics(policy_statistics(policy,prepared,[r['verb'] for r in rows],[r['target'] for r in rows]),rows)
                if rows[-1]['terminal']:
                    equal(boot,dict(terminal=True,value=0.,snapshot=None,prepared=None),'exact batch-end terminal')
                else:
                    snap,item=prepare(policy,data,obs,number*count,args.arm,args.block)
                    equal(boot['snapshot'],snap,'bootstrap snapshot')
                    equal(boot['prepared'],evidence(item),'bootstrap input')
                    if abs(boot['value']-_value(policy,item))>1e-6:
                        raise ValueError('Bootstrap value changed')
                audit(rows,boot['value'],GAMMA,.999)
                load_checked(policy,rd/'checkpoint')
                equal(state(policy),report['after'],'updated parameter evidence')
                post_error=validate_policy_statistics(policy_statistics(policy,prepared,[r['verb'] for r in rows],[r['target'] for r in rows]),read(rd/'post_statistics.json'))
                stored_opts=torch.load(rd/'checkpoint/optimizers.pt',map_location='cpu',weights_only=False)
                applied=report['applied_steps']
                if not 1<=applied['actor']<=2 or applied['critic']!=16:
                    raise ValueError('Optimizer steps out of bounds')
                actual_steps={}
                for key in ('actor','critic'):
                    previous_steps=optimizer_before[key]
                    optimizer_before[key]+=applied[key]
                    counts=[int(s['step'].item()) for s in stored_opts[key]['state'].values() if 'step' in s]
                    if not counts or set(counts)!={optimizer_before[key]}:
                        raise ValueError('Stored optimizer step continuity failed')
                    actual_steps[key]=counts[0]-previous_steps
                verified_rounds.append(check_round_summary(report,rows,actual_steps))
                round_reports.append(report)
                for key in ('base','reference'):
                    equal(report['after'][key],initial[key],'frozen '+key)
                if not all(report['checks'].values()):
                    raise ValueError('Producer checks failed')
                audits.append(dict(round=number,decisions=len(rows),old_stat_error=old_error,post_stat_error=post_error,
                                   terminal_indices=[r['index'] for r in rows if r['terminal']]))
                previous=rd/'checkpoint'
        equal(state(policy),training_result['final'],'final checkpoint')
        check_mirrors(training_result,round_reports,None if args.smoke else read(args.train_dir/'pre_metrics.json'),verified_pre)
        training_summary=dict(decisions=sum(r['decisions'] for r in verified_rounds),
            positive=sum(r['positive'] for r in verified_rounds),negative=sum(r['negative'] for r in verified_rounds),
            terminals=sum(r['terminals'] for r in verified_rounds),applied_steps=[r['applied_steps'] for r in verified_rounds],
            pre_metrics=verified_pre,seconds=training_result['seconds'])
        write(args.output/'training_verification.json',dict(passed=True,rounds=audits,optimizer_totals=optimizer_before))
    results={}
    if args.eval_dir:
        for phase in ('after','before'):
            results[phase]=check_eval(policy,data,args,args.eval_dir/phase,phase)
        report=read(args.eval_dir/'evaluation.json')
        equal(report['evaluated'],state(policy),'frozen evaluated model')
        equal(report['metrics'],{p:x['metrics'] for p,x in results.items()},'evaluation summary')
    write(args.output/'verified.json',dict(passed=True,block=args.block,arm=args.arm,smoke=args.smoke,
           training_decisions=0 if args.arm=='A0' else (16 if args.smoke else 384),evaluation=results,
           training_summary=training_summary,verifier_pid=os.getpid(),hardware_commands_sent=0,real_output_verified=False))
