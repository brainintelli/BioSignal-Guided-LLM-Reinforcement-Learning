"""Fail-closed M8 publication/selection review; --request JSON --output JSON.

Request: training_run, training_audit, training_finished, training_ledger,
selection (original_v2, semantic_parent, checkpoint_0008/0016/0024/0032;
each value {report,replay_audit,rng_audit}), phase=selection|test.
Test additionally requires test (three arms, same bundle structure),
test_finished, test_ledger. Attempt 2 requires previous_request and
previous_result, recursively reproduced. Paths refer to preserved evidence.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.rl_audit_io import file_sha256
from tools.m8_metrics import qualify_selection,compare_test,summarize
from agent.m8_registration import verify_registration
from agent.m8_identity import read_ledger,config_sha,BASE_WEIGHT_SHA
from agent.m8_policy_identity import PARENT_ACTOR_SHA256
from agent.m8_eval import DECODING,episode_seed


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition,message):
    if not condition:raise ValueError(message)


def check_ledger(path,registration_sha,attempt,phase,evidence,checkpoint=None,actor=None):
    events=read_ledger(path)
    require(len(events)==2 and [x['status'] for x in events]==['started','completed'],'Phase not exactly once and successfully completed')
    for row in events:
        require(row['registration_sha256']==registration_sha and row['attempt']==attempt and row['phase']==phase,'Ledger identity differs')
        require(row.get('selected_checkpoint')==checkpoint and row.get('selected_actor_sha256')==actor,'Ledger selected policy differs')
    require(events[-1]['evidence_sha256']==evidence,'Ledger evidence differs')


def check_audit(audit,report,report_sha,registration_sha,*,rng=False):
    require(audit.get('passed') is True and audit.get('trace_sha256')==report['trace_sha256'],'Missing/failed/mismatched audit')
    require(audit.get('actor_sha256')==report['actor_sha256'],'Audit actor differs')
    require(audit.get('episodes')==len(report['episodes']),'Audit matrix incomplete')
    count=sum(e['decisions'] for e in report['episodes'])
    require(audit.get('replayed_decisions')==count,'Audit action count differs')
    require(audit.get('report_sha256')==report_sha and audit.get('registration_sha256')==registration_sha,'Audit report/registration binding differs')
    if rng:
        require(audit.get('rng_sampling_reproduced') is True,'CUDA RNG has not been reproduced')
    else:
        require(audit.get('source_sha256')==report['source_sha256'] and audit.get('prefix_only') is False,'Replay source/prefix differs')


def build_result(request,project=None,_depth=0):
    require(_depth<=1,'Attempt recursion exceeds registered maximum')
    project=Path(project or Path(__file__).resolve().parents[1]);run=Path(request['training_run'])
    tracked={}
    def load(path):
        path=Path(path);value=read(path);tracked[str(path)]=file_sha256(path);return value
    def digest(path):
        path=Path(path);value=file_sha256(path);tracked[str(path)]=value;return value
    reg=load(run/'registration.json');reg_sha=tracked[str(run/'registration.json')]
    require(digest(run/'preregistration.json')==reg_sha,'Registration compatibility copy changed')
    verify_registration(project,reg);c=reg['config'];sources=reg['source_hashes'];execution=reg['execution_identity']
    require(execution['config_sha256']==config_sha(c) and execution['base_weight_sha256']==BASE_WEIGHT_SHA,'Execution identity changed')
    require(bool(execution['runtime']) and bool(execution['tokenizer_files_sha256']) and bool(execution['parent_restore_sha256']),'Incomplete execution identity')
    require(reg['training_reset']=='natural death only','Training reset differs')
    if c['attempt']==2:
        prior_request=load(request['previous_request']);prior=load(request['previous_result'])
        rebuilt=build_result(prior_request,project,_depth+1)
        require(prior==rebuilt and prior['attempt']==1 and prior['decision']=='reject_attempt' and prior['evidence_complete'],'Attempt 1 rejection not independently reproduced')
    else:require(_depth in (0,1),'Invalid first attempt')
    summary=load(run/'training_summary.json');training=load(request['training_audit'])
    require(summary['decisions']==32768 and summary['initial_actor_sha256']==PARENT_ACTOR_SHA256,'Training budget/initial actor differs')
    require(summary['source_counts']=={'model':32768,'fallback':0,'scripted':0},'Non-model training actions')
    require(summary['weight_change_verified'] is True and summary['final_actor_sha256']!=PARENT_ACTOR_SHA256,'No trained actor change')
    trace_sha=digest(run/'transitions.jsonl')
    require(training.get('passed') is True and training.get('complete_training_budget') is True and training.get('prefix_only') is False,'Full training replay required')
    require(training['replayed_decisions']==32768 and training['scored_world_ticks']==summary['world_ticks'],'Training replay budget mismatch')
    require(training['trace_sha256']==trace_sha and training['source_sha256']==sources and training['registration_sha256']==reg_sha,'Training replay identity mismatch')
    finished=load(request['training_finished']);require(type(finished.get('exit_code')) is int and finished['exit_code']==0,'Training process did not finish successfully')
    metrics=[json.loads(line) for line in (run/'training.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    digest(run/'training.jsonl')
    require(len(metrics)==32 and all(m['update']==i and m['decisions']==i*1024 for i,m in enumerate(metrics,1)),'Optimizer budget incomplete')
    require(metrics[-1]['ticks']==summary['world_ticks'],'Optimized ticks differ')
    required_audits={'training':training};report_hashes={}
    def bundle(spec,arm,seeds,reps,actor=None):
        path=Path(spec['report']);r=load(path);rsha=tracked[str(path)]
        report_hashes[path.name]=rsha
        expected_contract='open_world_hierarchical_v2' if arm=='original_v2' else 'open_world_semantic_v4'
        require(r['arm']==arm and r['contract']==expected_contract,'Wrong assessment interface/arm')
        require(r['profile']=='open1' and r['horizon']==1800 and r['temperature']==1 and r['decoding']==DECODING,'Assessment rules differ')
        require(r['source_sha256']==sources and r['rng_prefix']==c['rng_prefix'],'Assessment source/RNG differs')
        require(r['seeds']==seeds and r['replicates']==reps,'Wrong registered matrix')
        require({(e['seed'],e['replicate']) for e in r['episodes']}=={(s,p) for s in seeds for p in reps},'Incomplete matrix')
        stats=summarize(r['episodes'])
        require(r['mean_survival_ticks']==stats['rmst'] and r['alive_at_horizon']==stats['successes'],'Reported statistics differ')
        if actor is not None:require(r['actor_sha256']==actor,'Wrong assessment actor')
        for e in r['episodes']:
            require(e['rng_seed']==episode_seed(c['rng_prefix'],'open1',e['seed'],e['replicate']),'Changed episode RNG')
            require(type(e['decisions']) is int and e['decisions']>0 and e['source_counts']=={'model':e['decisions'],'fallback':0},'Fallback or empty episode')
            require(sum(e['actions'].values())==e['decisions'],'Action totals differ')
        require(digest(path.with_suffix('.jsonl'))==r['trace_sha256'],'Raw trace changed')
        replay=load(spec['replay_audit']);rng=load(spec['rng_audit'])
        check_audit(replay,r,rsha,reg_sha);check_audit(rng,r,rsha,reg_sha,rng=True)
        require(rng.get('runtime')==execution['runtime'],'RNG runtime differs')
        required_audits[str(path)]={'replay':replay,'rng':rng}
        return r
    selection=request['selection'];keys={'original_v2','semantic_parent',*(f'checkpoint_{u:04d}' for u in (8,16,24,32))}
    require(set(selection)==keys,'All six selection assessments required')
    parents={arm:bundle(selection[arm],arm,c['validation_seeds'],[0,1],PARENT_ACTOR_SHA256) for arm in ('original_v2','semantic_parent')}
    candidates={};identities={}
    for update in (8,16,24,32):
        checkpoint=run/f'checkpoint_{update:04d}';meta=load(checkpoint/'policy.json')
        require(meta['contract']=='open_world_semantic_v4' and meta['reference_actor_sha256']==PARENT_ACTOR_SHA256,'Checkpoint contract/parent differs')
        # Full artifact bindings preserve adapter, critic and metadata together.
        for path in sorted(checkpoint.rglob('*')):
            if path.is_file():digest(path)
        report=bundle(selection[f'checkpoint_{update:04d}'],'candidate',c['validation_seeds'],[0,1],meta['actor_sha256'])
        candidates[update*1024]=report['episodes'];identities[update*1024]=meta['actor_sha256']
    require(identities[32768]==summary['final_actor_sha256'],'Final summary actor differs')
    result=qualify_selection(candidates,parents['semantic_parent']['episodes'],parents['original_v2']['episodes'])
    selected=result['selected_decisions']//1024;actor=identities[result['selected_decisions']]
    saved=load(run/'selection.json')
    for key,value in result.items():require(saved.get(key)==json.loads(json.dumps(value)),'Saved selection not independently reproduced: '+key)
    require(saved.get('selected_actor_sha256')==actor and str(saved.get('selected_checkpoint','')).replace('\\','/').rsplit('/',1)[-1]==f'checkpoint_{selected:04d}','Saved selected actor/checkpoint differs')
    selection_sha=tracked[str(run/'selection.json')]
    train_evidence={**report_hashes,'transitions.jsonl':trace_sha,'training_summary.json':tracked[str(run/'training_summary.json')],'selection.json':selection_sha}
    check_ledger(request['training_ledger'],reg_sha,c['attempt'],'train',train_evidence)
    digest(request['training_ledger'])
    phase=request['phase'];require(phase in ('selection','test'),'Unknown phase')
    selection_result=result;decision='eligible_for_test' if result['eligible_for_test'] else 'reject_attempt';release=False
    if phase=='test':
        require(result['eligible_for_test'],'Selection did not unlock test')
        require(set(request['test'])=={'candidate','semantic_parent','original_v2'},'Three final arms required')
        report_hashes.clear()
        final={arm:bundle(request['test'][arm],arm,c['test_seeds'],[0],actor if arm=='candidate' else PARENT_ACTOR_SHA256) for arm in ('original_v2','semantic_parent','candidate')}
        done=load(request['test_finished']);require(type(done.get('exit_code')) is int and done['exit_code']==0,'Assessment process failed')
        check_ledger(request['test_ledger'],reg_sha,c['attempt'],'test',dict(report_hashes),f'checkpoint_{selected:04d}',actor)
        digest(request['test_ledger'])
        result=compare_test(final['candidate']['episodes'],final['semantic_parent']['episodes'],final['original_v2']['episodes'])
        release=result['numerical_gate_passed'];decision='accept' if release else 'reject_attempt'
    output=dict(result,protocol='m8_reliable_v1',phase=phase,attempt=c['attempt'],decision=decision,
        release_authorized=release,evidence_complete=True,selected_checkpoint=f'checkpoint_{selected:04d}',selected_actor_sha256=actor,
        parent_actor_sha256=PARENT_ACTOR_SHA256,registration_sha256=reg_sha,selection_sha256=selection_sha,
        selection_review=selection_result,source_sha256=sources,execution_identity=execution,config=c,
        required_audits=required_audits,evidence_sha256=tracked,publication_request=request,
        confidence_scope='Per-attempt descriptive intervals; two attempts have no familywise correction; sample 90% is not a population guarantee')
    # Reports are persisted and independently rebuilt, including A2's A1 chain.
    # Canonical JSON keys prevent integer-key selection summaries from changing
    # identity only because the original result passed through a JSON file.
    return json.loads(json.dumps(output,ensure_ascii=False,allow_nan=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--project',type=Path);a=p.parse_args();result=build_result(read(a.request),a.project)
    with a.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'decision':result['decision'],'release_authorized':result['release_authorized']}))


if __name__=='__main__':main()
