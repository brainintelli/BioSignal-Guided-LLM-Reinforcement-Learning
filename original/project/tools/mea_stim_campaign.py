"""Matched before/after offline campaign. No device tools; all data sealed."""
import argparse,gc,itertools,math,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from agent.mea_stim_offline import StimRecording,PROTOCOL,cell_config,assignment,analyze,validate_budget
from agent.mea_protocol import digest,canonical
from agent.mea_replay import RecordingSnapshot
from tools import mea_stim_train as trainlib
from tools.mea_ablation import episode,replay_episode,load_episode
from tools.mea_offline import evidence,policy_statistics,validate_policy_statistics,_value
from tools.mea_p0 import ROOT,DEFAULT_GODOT,read_json,write_json,sha,_same

def identity(project,godot,model_config):
    i=trainlib.identity(project,godot,model_config)
    i['sources'].update({'code/tools/'+n:sha(Path(__file__).with_name(n)) for n in ('mea_stim_campaign.py','mea_ablation.py','mea_stim_export.py')})
    return i

def registration(root,source,pin,model_config,project,godot):
    reg=read_json(root/'registration.json')
    _same(reg['protocol'],PROTOCOL,'registered protocol');_same(reg['source_pin'],pin,'dataset pin')
    _same(reg['identity'],identity(project,godot,model_config),'registered code/engine/model configuration')
    for context in ('before','after'):
        data=StimRecording(source,pin=pin,phase=context)
        _same(reg['source'][context],data.summary(),'source provenance')
    trainlib.validate_common_initial(reg['common_initial'])
    return reg

def register(root,source,pin,model_config,project,godot):
    data={p:StimRecording(source,pin=pin,phase=p).summary() for p in ('before','after')}
    policy=trainlib.new_policy(model_config,cell_config(0,'none'));initial=trainlib.state(policy)
    trainlib.validate_common_initial(initial);root.mkdir(parents=True,exist_ok=False)
    reg=dict(protocol=PROTOCOL,source_pin=pin,source=data,common_initial=initial,
        identity=identity(project,godot,model_config),created_utc=datetime.now(timezone.utc).isoformat())
    write_json(root/'registration.json',reg);return dict(status='registered',registration_sha256=digest(reg),initial=initial)

def cell_path(root,block,arm):
    cell_config(block,arm);return root/f'b{block:02d}_{arm}'

def train(root,source,pin,model_config,project,godot,block,arm):
    reg=registration(root,source,pin,model_config,project,godot)
    out=cell_path(root,block,arm);out.mkdir(exist_ok=False)
    write_json(out/'binding.json',dict(block=block,arm=arm,registration_sha256=digest(reg),collection_pid=os.getpid()))
    trainlib.train(out/'train',source,pin,model_config,project,godot,cell_config(block,arm))
    report=read_json(out/'train/report.json');validate_budget(report['decisions'],[x['applied_steps'] for x in report['rounds']])
    _same(report['initial'],reg['common_initial'],'same starting tensors')

def evaluation_schedule(block,arm,context):
    cfg=cell_config(block,arm)
    if context not in ('before','after'):raise ValueError('Heldout context')
    return dict(arm=cfg['policy_arm'],order=assignment('eval'),world_seed=cfg['eval_world_seed'],action_seed=cfg['eval_action_seed'])

def evaluate(root,source,pin,model_config,project,godot,block,arm):
    from agent.mea_ablation_policy import load_arm
    reg=registration(root,source,pin,model_config,project,godot);out=cell_path(root,block,arm)
    trainlib.check_manifest(out/'train','training_manifest.json')
    policy=trainlib.new_policy(model_config,cell_config(block,arm));_same(trainlib.state(policy),reg['common_initial'],'reconstructed initial')
    metrics={};identities={}
    for phase,checkpoint in [('pre',out/'train/initial_checkpoint'),('post',out/'train/round_03/checkpoint')]:
        load_arm(policy,checkpoint);identities[phase]=trainlib.state(policy);metrics[phase]={}
        for context in ('before','after'):
            data=StimRecording(source,pin=pin,phase=context,split='eval')
            rows,boot,metrics[phase][context]=episode(policy,data,**evaluation_schedule(block,arm,context),
                output=out/(phase+'_'+context),godot=godot,project=project)
            _same(trainlib.state(policy),identities[phase],'evaluation changed parameters')
            del rows,boot;policy.clear_prefix_cache();gc.collect()
            trainlib.emit('evaluation_finished',block=block,arm=arm,phase=phase,context=context)
    _same(identities['pre'],reg['common_initial'],'pre checkpoint identity')
    _same(identities['post'],read_json(out/'train/report.json')['final'],'final checkpoint identity')
    report=dict(block=block,arm=arm,metrics=metrics,checkpoint_tensors=identities,evaluation_pid=os.getpid(),
        registration_sha256=digest(reg),alpha=1.,r2_enabled=False,device_commands_sent=0,realtime_training=False)
    write_json(out/'evaluation.json',report);trainlib.seal(out,'evaluation_manifest.json')

def verify(root,source,pin,model_config,project,godot,block,arm):
    import torch
    from agent.mea_ablation_policy import load_arm
    reg=registration(root,source,pin,model_config,project,godot);out=cell_path(root,block,arm)
    trainlib.check_manifest(out,'evaluation_manifest.json')
    binding=read_json(out/'binding.json');evaluation=read_json(out/'evaluation.json')
    for k,v in dict(block=block,arm=arm,registration_sha256=digest(reg)).items():
        _same(binding[k],v,'cell binding');_same(evaluation[k],v,'eval binding')
    if os.getpid() in (binding['collection_pid'],evaluation['evaluation_pid']):raise ValueError('Fresh verifier required')
    trainlib.verify(out/'train',source,pin,model_config,project,godot,cell_config(block,arm))
    training=read_json(out/'train/report.json');validate_budget(training['decisions'],[x['applied_steps'] for x in training['rounds']])
    gc.collect();torch.cuda.empty_cache();policy=trainlib.new_policy(model_config,cell_config(block,arm))
    _same(trainlib.state(policy),reg['common_initial'],'common initial');errors={};metrics={}
    for phase,checkpoint in [('pre',out/'train/initial_checkpoint'),('post',out/'train/round_03/checkpoint')]:
        load_arm(policy,checkpoint);_same(trainlib.state(policy),evaluation['checkpoint_tensors'][phase],'checkpoint tensors')
        metrics[phase]={}
        for context in ('before','after'):
            data=StimRecording(source,pin=pin,phase=context,split='eval');schedule=evaluation_schedule(block,arm,context)
            folder=out/(phase+'_'+context);rows,boot,metrics[phase][context]=load_episode(folder)
            _same(read_json(folder/'assignment.json'),schedule,'heldout assignment')
            if any(r['action_source']!='model' for r in rows):raise ValueError('Model-only evaluation')
            replay_episode(rows,boot,metrics[phase][context],data,schedule['arm'],schedule['order'],schedule['world_seed'],godot,project)
            prepared=policy.prepare([r['observation'] for r in rows],[None if arm=='none' else RecordingSnapshot.from_dict(r['snapshot']) for r in rows])
            for p,r in zip(prepared,rows):_same(evidence(p),r['prepared'],'saved exact input tokens')
            errors[phase+'_'+context]=validate_policy_statistics(policy_statistics(policy,prepared,[r['verb'] for r in rows],[r['target'] for r in rows]),rows)
            if not boot['terminal']:
                s=RecordingSnapshot.from_dict(boot['snapshot']);bp=policy.prepare([rows[-1]['next_observation']],[None if arm=='none' else s])[0]
                _same(evidence(bp),boot['prepared'],'bootstrap input')
                if abs(_value(policy,bp)-boot['value'])>1e-6:raise ValueError('Bootstrap value mismatch')
            _same(trainlib.state(policy),evaluation['checkpoint_tensors'][phase],'verification altered weights')
            del rows,prepared;policy.clear_prefix_cache();gc.collect();trainlib.emit('heldout_verified',phase=phase,context=context,error=errors[phase+'_'+context])
    _same(metrics,evaluation['metrics'],'replayed metrics')
    _same(evaluation['checkpoint_tensors']['pre'],reg['common_initial'],'original evaluation identity')
    _same(evaluation['checkpoint_tensors']['post'],training['final'],'trained evaluation identity')
    for k,v in dict(alpha=1.,r2_enabled=False,device_commands_sent=0,realtime_training=False).items():_same(evaluation[k],v,'constraint')
    trainlib.check_manifest(out,'evaluation_manifest.json',extra=('train/independent_verification.json',))
    trainlib.seal(out,'cell_manifest.json')
    receipt=dict(passed=True,block=block,arm=arm,registration_sha256=digest(reg),cell_manifest_sha256=sha(out/'cell_manifest.json'),
        source_pin=pin,training_decisions=384,eval_errors=errors,fresh_process=True,verification_pid=os.getpid(),
        training_receipt_sha256=sha(out/'train/independent_verification.json'))
    write_json(root/f'verified_b{block:02d}_{arm}.json',receipt);return receipt

def analysis(root,source,pin,model_config,project,godot):
    reg=registration(root,source,pin,model_config,project,godot);cells=[]
    for b,a in itertools.product(range(8),PROTOCOL['arms']):
        out=cell_path(root,b,a);receipt=root/f'verified_b{b:02d}_{a}.json'
        if not receipt.exists():continue
        q=read_json(receipt)
        for k,v in dict(passed=True,block=b,arm=a,registration_sha256=digest(reg),source_pin=pin,training_decisions=384,fresh_process=True).items():_same(q.get(k),v,'receipt')
        if set(q['eval_errors'])!={'pre_before','pre_after','post_before','post_after'} or any(not math.isfinite(v) or not 0<=v<=1e-6 for v in q['eval_errors'].values()):raise ValueError('Eval receipt errors')
        _same(q['cell_manifest_sha256'],sha(out/'cell_manifest.json'),'cell seal');trainlib.check_manifest(out,'cell_manifest.json')
        _same(q['training_receipt_sha256'],sha(out/'train/independent_verification.json'),'training receipt seal')
        e=read_json(out/'evaluation.json')
        cells.append(dict(block=b,arm=a,**{p:{c:e['metrics'][p][c]['discounted_r1'] for c in ('before','after')} for p in ('pre','post')}))
    result=analyze(cells)|dict(cells=cells,registration_sha256=digest(reg),source_pin=pin)
    if result['status']=='complete':
        indexed={(c['block'],c['arm']):c for c in cells}
        for b in range(8):_same(indexed[b,'before']['pre'],indexed[b,'after']['pre'],'identical common initial evaluation')
    write_json(root/'analysis.json',result);return result

def batch(a):
    registration(a.root,a.source,a.pin,a.model_config,a.project,a.godot)
    if (a.root/'batch_status.json').exists():raise FileExistsError('No campaign overwrite/restart')
    status=dict(complete=False,passed=False,planned_cells=24,cells=[],started_utc=datetime.now(timezone.utc).isoformat())
    def persist():
        tmp=a.root/'batch_status.partial.json';tmp.write_text(canonical(status)+'\n');tmp.replace(a.root/'batch_status.json')
    persist();orders=list(itertools.permutations(PROTOCOL['arms']))
    common=['--root',str(a.root),'--source',str(a.source),'--pin',a.pin,'--model-config',str(a.model_config),'--project',str(a.project),'--godot',str(a.godot)]
    for b in range(8):
        for arm in orders[b%6]:
            cell=dict(block=b,arm=arm,stages=[]);status['cells'].append(cell);persist()
            for stage in ('train','evaluate','verify'):
                started=time.monotonic()
                with (a.root/f'b{b:02d}_{arm}_{stage}.log').open('x') as log:
                    try:
                        p=subprocess.run([sys.executable,'-m','tools.mea_stim_campaign',stage,*common,'--block',str(b),'--arm',arm],stdout=log,stderr=subprocess.STDOUT,timeout=7200)
                        code=p.returncode;error=None
                    except subprocess.TimeoutExpired:code=-1;error='stage timeout7200s'
                cell['stages'].append(dict(stage=stage,exit_code=code,error=error,seconds=time.monotonic()-started));persist()
                if code:
                    status['complete']=True;status['error']='Technical failure; no automatic restart or replacement';persist();return status
    status['stage']='aggregate_verification';persist()
    try:
        result=analysis(a.root,a.source,a.pin,a.model_config,a.project,a.godot)
        if result.get('status')!='complete':raise ValueError('Final aggregate incomplete')
    except Exception as exc:
        status.update(complete=True,passed=False,error='Final analysis failed: '+repr(exc),finished_utc=datetime.now(timezone.utc).isoformat());persist()
        return status
    status.update(complete=True,passed=True,stage='complete',finished_utc=datetime.now(timezone.utc).isoformat());persist()
    return dict(passed=True,analysis_status=result['status'],positive_evidence=result['positive_evidence'])

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','train','evaluate','verify','batch','analyze'])
    for k in ['root','source','model-config']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--pin',required=True);p.add_argument('--project',type=Path,default=ROOT);p.add_argument('--godot',type=Path,default=DEFAULT_GODOT)
    p.add_argument('--block',type=int);p.add_argument('--arm',choices=PROTOCOL['arms']);a=p.parse_args()
    common=(a.root,a.source,a.pin,a.model_config,a.project,a.godot)
    if a.action=='batch':result=batch(a)
    elif a.action=='analyze':result=analysis(*common)
    elif a.action=='register':result=register(*common)
    else:result=globals()[a.action](*common,a.block,a.arm)
    print(canonical(result),flush=True)

if __name__=='__main__':main()
