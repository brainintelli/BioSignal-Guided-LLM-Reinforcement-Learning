"""M8 recorded-action replay. Never samples a model or opens new test worlds.

Release authorization remains the comparator's responsibility. Final identity
schema must provide exact CORE_FILES; missing identity fails closed.
"""
import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.rl_audit_io import file_sha256,iter_trace_lines
from tools.audit_m6_run import assert_equivalent
from agent.rl_contract import transition_at_horizon, LABELS
from agent.m6_stochastic_eval import validate_probabilities
from agent.m8b_eval import episode_seed


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_sources(registration,project):
    from agent.m8b_registration import verify_registration
    verify_registration(project,registration)
    from agent.m8b_identity import CORE_FILES
    sources=registration['source_sha256']
    if set(sources)!=set(CORE_FILES):raise ValueError('Exact frozen M8 source identity required')
    for name,digest in sources.items():
        part=Path(name)
        if part.is_absolute() or '..' in part.parts or file_sha256(Path(project)/part)!=digest:
            raise ValueError('Changed or invalid frozen source: '+name)
    return sources


def validate_environment_identity(value,registration,*,arm=None,profile=None):
    arm=value.get('arm') if arm is None else arm
    if arm not in ('original_v2','balanced_parent','candidate'):raise ValueError('Unknown assessment arm')
    original=arm=='original_v2'
    expected=dict(world_version=2 if original else 4,environment_version=None if original else 'm8_balanced_v1',
        balance_sha256=None if original else registration['balance_sha256'])
    for name,wanted in expected.items():
        if name not in value or value[name]!=wanted:raise ValueError('World identity differs: '+name)
    if (value.get('profile') if profile is None else profile)!=('open1' if original else 'open2'):raise ValueError('Profile differs')


def validate_step(row,observation,transition,tracker,contract,*,assessment=False,rng_prefix=None,horizon=1800):
    assert_equivalent(observation,row['observation'],'observation')
    salt=row.get('salt','m6')
    if salt!='m6':raise AssertionError('Unexpected encoding salt')
    encoded=contract.encode(observation,salt=salt)
    assert_equivalent(encoded['messages'],row['model_input'],'model_input')
    v,t=row['verb_index'],row['target_index']
    if type(v) is not int or type(t) is not int or not 0<=v<len(encoded['branches']):raise AssertionError('Invalid action index')
    branch=encoded['branches'][v]
    if not 0<=t<len(branch):raise AssertionError('Invalid target index')
    assert_equivalent(branch[t],row['action'],'action')
    assert_equivalent(contract.target_messages(encoded,v) if len(branch)>1 else None,row['target_input'],'target_input')
    if row['source']!='model':raise AssertionError('Non-model fallback')
    if assessment:
        if row['scored_horizon']!=horizon:raise AssertionError('Changed scoring horizon')
        expected=episode_seed(rng_prefix,observation['profile'],row['seed'],row['replicate'])
        if row['rng_seed']!=expected or type(row['rng_seed']) is not int:raise AssertionError('Changed episode RNG')
        validate_probabilities(row['verb_probabilities'],encoded['verb_mask'])
        validate_probabilities(row['target_probabilities'],[i<len(branch) for i in range(len(LABELS))])
        if row['verb_probabilities'][v]<=0 or row['target_probabilities'][t]<=0:raise AssertionError('Zero probability selected')
        for key in ('rng_before_sha256','rng_after_sha256'):
            value=row[key]
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise AssertionError('Invalid RNG digest')
    assert_equivalent(transition,row['transition'],'transition')
    scored=transition_at_horizon(transition,horizon) if assessment else transition
    reward=tracker.score(scored)
    assert_equivalent(reward,row['reward'],'reward')
    terminal=bool(scored['after']['dead'])
    # Training resets only on natural death; 1800 is an assessment window only.
    truncated=assessment and scored['after']['tick']>=horizon and not terminal
    assert_equivalent(terminal,row['terminal'],'terminal')
    assert_equivalent(truncated,row['truncated'],'truncated')
    return terminal or truncated,scored,reward


def audit(training_run,trace,godot,project,report_path=None,allow_prefix=False):
    training_run=Path(training_run);trace=Path(trace)
    reg=read(training_run/'preregistration.json');c=reg['config'];sources=verify_sources(reg,project)
    report=read(report_path) if report_path else None
    if report and allow_prefix:raise ValueError('Incomplete assessment cannot pass')
    if report:
        from agent.m8b_eval import DECODING
        expected_contract='open_world_hierarchical_v2' if report['arm']=='original_v2' else 'open_world_balanced_v1'
        if report['contract']!=expected_contract or report['temperature']!=1 or report['decoding']!=DECODING:raise ValueError('Changed decoding/contract')
        if report['source_sha256']!=sources or report['rng_prefix']!=c['rng_prefix']:raise ValueError('Changed assessment identity')
        if not ((report['seeds']==c['validation_seeds'] and report['replicates']==[0]) or (report['seeds']==c['test_seeds'] and report['replicates']==[0])):raise ValueError('Unregistered assessment matrix')
        validate_environment_identity(report,reg)
        if report['horizon']!=1800:raise ValueError('Changed horizon')
        if file_sha256(trace)!=report['trace_sha256']:raise ValueError('Trace identity differs')
        from tools.m8b_metrics import summarize
        summarize(report['episodes'])
        matrix={(s,r) for s in report['seeds'] for r in report['replicates']}
        expected={(e['seed'],e['replicate']):e for e in report['episodes']}
        if set(expected)!=matrix:raise ValueError('Incomplete assessment matrix')
        module='agent.m6_contract' if report['arm']=='original_v2' else 'agent.m8b_contract'
    else:matrix=None;module='agent.m8b_contract'
    original=bool(report and report['arm']=='original_v2')
    profile='open1' if original else 'open2'
    GodotEnv=importlib.import_module('agent.rl_env' if original else 'agent.m8b_env').GodotEnv
    contract=importlib.import_module(module)
    contexts={};updates=Counter();count=ticks=0;last_update=1
    reset_cursor=c['envs'];next_seeds={i:c['train_seed_start']+i for i in range(c['envs'])}
    try:
        for line in iter_trace_lines(trace):
            if not line.strip():continue
            row=json.loads(line);key=(row['seed'],row['replicate']) if report else row['env']
            if report and (key not in matrix or row['arm']!=report['arm']):raise AssertionError('Wrong arm/matrix')
            if report:
                validate_environment_identity(row,reg,arm=report['arm'],profile=row['observation']['profile'])
                for field in ('contract','temperature','decoding','actor_sha256'):
                    assert_equivalent(row[field],report[field],'trace.'+field)
            if not report:
                if type(key) is not int or not 0<=key<c['envs']:raise AssertionError('Invalid env slot')
                if not c['train_seed_start']<=row['seed']<c['train_seed_start']+c['train_seed_count']:raise AssertionError('Nontraining seed')
                u=row['update']
                if type(u) is not int or not 1<=u<=24 or u<last_update or u>last_update+1:raise AssertionError('Update order/budget')
                updates[u]+=1;last_update=u
            if key not in contexts:contexts[key]=dict(env=GodotEnv(godot,project),state=None,ended=True)
            ctx=contexts[key]
            if ctx['ended']:
                if report and ctx['state'] is not None:raise AssertionError('Repeated episode')
                if not report and row['seed']!=next_seeds[key]:raise AssertionError('Training reset sequence changed')
                if row['observation']['tick']!=0:raise AssertionError('Reset starts after zero')
                ctx.update(state=ctx['env'].reset(row['seed'],profile),seed=row['seed'])
                ctx['tracker']=contract.SurvivalReward(ctx['state']['observation'])
                ctx['decisions']=0;ctx['actions']=Counter()
            elif ctx['seed']!=row['seed']:raise AssertionError('Reset before death')
            if report and 'rng_after' in ctx and ctx['rng_after']!=row['rng_before_sha256']:raise AssertionError('RNG chain broken')
            nxt=ctx['env'].step(row['action'])
            ended,scored,reward=validate_step(row,ctx['state']['observation'],nxt['transition'],ctx['tracker'],contract,
                assessment=report is not None,rng_prefix=report['rng_prefix'] if report else None)
            ctx.update(state=nxt,ended=ended);count+=1;ticks+=reward['elapsed_ticks'];ctx['decisions']+=1;ctx['actions'][row['action']['verb']]+=1
            if not report and ended:
                next_seeds[key]=c['train_seed_start']+reset_cursor%c['train_seed_count'];reset_cursor+=1
            if report:
                ctx['rng_after']=row['rng_after_sha256']
                if ended:
                    ctx['env'].close();final=scored['after']
                    actual=dict(decisions=ctx['decisions'],actions=dict(ctx['actions']),source_counts={'model':ctx['decisions'],'fallback':0},
                        survival_ticks=final['tick'],terminal=bool(final['dead']),alive_at_horizon=not final['dead'],final=final)
                    for name,value in actual.items():assert_equivalent(value,expected[key][name],'episode.'+name)
        if not count:raise AssertionError('Empty trace')
        if report:
            if set(contexts)!=matrix or not all(ctx['ended'] for ctx in contexts.values()):raise AssertionError('Incomplete replay')
        elif allow_prefix:
            if count>49152 or any(n!=2048 for u,n in updates.items() if u!=last_update) or not 1<=updates[last_update]<=2048:raise AssertionError('Invalid prefix')
        else:
            summary=read(training_run/'training_summary.json')
            if count!=49152 or dict(updates)!={i:2048 for i in range(1,25)}:raise AssertionError('Incomplete full budget')
            assert_equivalent(count,summary['decisions'],'summary.decisions');assert_equivalent(ticks,summary['world_ticks'],'summary.ticks')
        return dict(passed=True,assessment=report is not None,prefix_only=allow_prefix,complete_training_budget=not report and not allow_prefix,
            replayed_decisions=count,scored_world_ticks=ticks,episodes=len(matrix) if matrix else None,source_sha256=sources,
            trace_sha256=file_sha256(trace),report_sha256=file_sha256(Path(report_path)) if report else None,
            registration_sha256=file_sha256(training_run/'preregistration.json'),
            actor_sha256=report['actor_sha256'] if report else None,
            arm=report['arm'] if report else None,contract=contract.VERSION,rng_sampling_reproduced=False,
            is_new_model_rollout=False,release_authorized=False)
    finally:
        for ctx in contexts.values():ctx['env'].close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training-run','trace','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--godot',required=True);p.add_argument('--project',default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--report',type=Path);p.add_argument('--allow-prefix',action='store_true');a=p.parse_args()
    result=audit(a.training_run,a.trace,a.godot,a.project,a.report,a.allow_prefix)
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
