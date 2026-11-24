"""Registered chip28080 train/evaluate/replay campaign; no device connection."""
import argparse
import gc
import itertools
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone

from agent.mea_long_ablation import PROTOCOL,PIN,ARMS,ORIGINAL_TENSORS,cell_config,assignment,analyze
from agent.mea_h5_replay import H5MappedRecording
from agent.mea_replay import RecordingSnapshot
from agent.mea_protocol import digest,canonical
from tools import mea_long_train as long
from tools.mea_ablation import episode,replay_episode,load_episode
from tools.mea_offline import evidence,policy_statistics,validate_policy_statistics,_value
from tools.mea_p0 import ROOT,DEFAULT_GODOT,read_json,write_json,sha,_same


def identity(project,godot,model_config):
    result=long.identity(project,godot,model_config)
    result['sources'].update({'code/tools/'+name:sha(Path(__file__).with_name(name))
        for name in ('mea_long_ablation.py','mea_ablation.py')})
    return result


def validate_common_initial(actual):
    _same(actual,ORIGINAL_TENSORS,'independently pinned original actor/critic/base/reference')


def registration(root,source,model_config,project,godot):
    reg=read_json(root/'registration.json')
    _same(reg['protocol'],PROTOCOL,'registered campaign protocol')
    _same(reg['identity'],identity(project,godot,model_config),'registered code/engine/config')
    validate_common_initial(reg['common_initial'])
    data=H5MappedRecording(source,manifest_sha256=PIN,expected_chip_id='28080')
    _same(data.summary(),reg['source'],'registered source')
    return reg,data


def register(root,source,model_config,project,godot):
    data=H5MappedRecording(source,manifest_sha256=PIN,expected_chip_id='28080')
    long.window_split(data.window_count)
    policy=long.new_policy(model_config,cell_config(0,'none'))
    initial=long.state(policy)
    validate_common_initial(initial)
    root.mkdir(parents=True,exist_ok=False)
    reg=dict(protocol=PROTOCOL,identity=identity(project,godot,model_config),source=data.summary(),
        common_initial=initial,created_utc=datetime.now(timezone.utc).isoformat())
    write_json(root/'registration.json',reg)
    return dict(registration_sha256=digest(reg),common_initial=initial)


def cell_path(root,block,arm):
    cell_config(block,arm)
    return root/f'b{block:02d}_{arm}'


def train(root,source,model_config,project,godot,block,arm):
    reg,_=registration(root,source,model_config,project,godot)
    out=cell_path(root,block,arm);out.mkdir(exist_ok=False)
    write_json(out/'binding.json',dict(block=block,arm=arm,registration_sha256=digest(reg),collection_pid=os.getpid()))
    long.train(out/'train',source,PIN,model_config,project,godot,cell_config(block,arm))
    _same(read_json(out/'train/initial_parameters.json'),reg['common_initial'],'common initial tensors')


def evaluation_schedule(block,arm):
    return dict(arm=arm,order=assignment(arm,'eval',block),world_seed=6400100+block,action_seed=6500100+10000*block)


def evaluate(root,source,model_config,project,godot,block,arm):
    from agent.mea_ablation_policy import load_arm
    reg,data=registration(root,source,model_config,project,godot);out=cell_path(root,block,arm)
    long.check_manifest(out/'train','training_manifest.json')
    _same(read_json(out/'train/initial_parameters.json'),reg['common_initial'],'common initial tensors')
    policy=long.new_policy(model_config,cell_config(block,arm))
    _same(long.state(policy),reg['common_initial'],'reconstructed original')
    schedule=evaluation_schedule(block,arm);metrics={};identities={}
    for phase,checkpoint in [('pre',out/'train/initial_checkpoint'),('post',out/'train/round_08/checkpoint')]:
        load_arm(policy,checkpoint);identities[phase]=long.state(policy)
        rows,boot,metrics[phase]=episode(policy,data,**schedule,output=out/phase,godot=godot,project=project)
        _same(long.state(policy),identities[phase],'evaluation changed parameters')
        del rows,boot;policy.clear_prefix_cache();gc.collect()
        long.emit('evaluation_finished',block=block,arm=arm,phase=phase,metrics=metrics[phase])
    _same(identities['pre'],reg['common_initial'],'pre evaluation checkpoint')
    _same(identities['post'],read_json(out/'train/report.json')['final'],'post evaluation checkpoint')
    report=dict(block=block,arm=arm,metrics=metrics,checkpoint_tensors=identities,
        evaluation_pid=os.getpid(),registration_sha256=digest(reg),alpha=1.,r2_enabled=False,
        stimulation=False,device_commands_sent=0,realtime_training=False)
    write_json(out/'evaluation.json',report);long.seal(out,'evaluation_manifest.json')


def validate_receipt(receipt,block,arm,registration_sha,manifest_sha):
    for k,v in dict(passed=True,block=block,arm=arm,registration_sha256=registration_sha,
        cell_manifest_sha256=manifest_sha,source_manifest_sha256=PIN,training_decisions=1024,fresh_process=True).items():
        _same(receipt.get(k),v,'receipt '+k)
    errors=receipt.get('eval_model_errors',{})
    if set(errors)!={'pre','post'} or any(type(v) not in (float,int) or not math.isfinite(v) or not 0<=v<=1e-6 for v in errors.values()):
        raise ValueError('receipt missing full evaluation statistics')


def verify(root,source,model_config,project,godot,block,arm):
    from agent.mea_ablation_policy import load_arm
    reg,data=registration(root,source,model_config,project,godot);out=cell_path(root,block,arm)
    long.check_manifest(out,'evaluation_manifest.json')
    binding=read_json(out/'binding.json');evaluation=read_json(out/'evaluation.json')
    for k,v in dict(block=block,arm=arm,registration_sha256=digest(reg)).items():
        _same(binding[k],v,'cell binding');_same(evaluation[k],v,'evaluation binding')
    if os.getpid() in [binding['collection_pid'],evaluation['evaluation_pid']]:raise ValueError('fresh verifier required')
    long.verify(out/'train',source,PIN,model_config,project,godot,cell_config(block,arm))
    import torch
    gc.collect();torch.cuda.empty_cache()
    policy=long.new_policy(model_config,cell_config(block,arm))
    _same(long.state(policy),reg['common_initial'],'common original policy')
    schedule=evaluation_schedule(block,arm);errors={};metrics={}
    for phase,checkpoint in [('pre',out/'train/initial_checkpoint'),('post',out/'train/round_08/checkpoint')]:
        load_arm(policy,checkpoint)
        _same(long.state(policy),evaluation['checkpoint_tensors'][phase],'evaluation checkpoint tensors')
        rows,boot,metrics[phase]=load_episode(out/phase)
        _same(read_json(out/phase/'assignment.json'),schedule,'heldout assignment')
        if any(r['action_source']!='model' for r in rows):raise ValueError('model-only heldout evaluation required')
        replay_episode(rows,boot,metrics[phase],data,arm,schedule['order'],schedule['world_seed'],godot,project)
        prepared=policy.prepare([r['observation'] for r in rows],
            [None if arm=='none' else RecordingSnapshot.from_dict(r['snapshot']) for r in rows])
        for p,r in zip(prepared,rows):_same(evidence(p),r['prepared'],'heldout exact inputs/tokens')
        errors[phase]=validate_policy_statistics(policy_statistics(policy,prepared,[r['verb'] for r in rows],
            [r['target'] for r in rows]),rows)
        if not boot['terminal']:
            s=RecordingSnapshot.from_dict(boot['snapshot'])
            p=policy.prepare([rows[-1]['next_observation']],[None if arm=='none' else s])[0]
            _same(evidence(p),boot['prepared'],'heldout boundary tokens')
            if abs(_value(policy,p)-boot['value'])>1e-6:raise ValueError('heldout bootstrap value differs')
        _same(long.state(policy),evaluation['checkpoint_tensors'][phase],'verification changed weights')
        del prepared,rows;policy.clear_prefix_cache();gc.collect()
        long.emit('heldout_verified',block=block,arm=arm,phase=phase,error=errors[phase])
    _same(metrics,evaluation['metrics'],'independent heldout metrics')
    _same(evaluation['checkpoint_tensors']['pre'],reg['common_initial'],'pre identity')
    _same(evaluation['checkpoint_tensors']['post'],read_json(out/'train/report.json')['final'],'post identity')
    for k,v in dict(alpha=1.,r2_enabled=False,stimulation=False,device_commands_sent=0,realtime_training=False).items():
        _same(evaluation[k],v,'constraint '+k)
    # Only the fresh training receipt was added since the frozen evaluation manifest.
    long.check_manifest(out,'evaluation_manifest.json',extra=('train/independent_verification.json',))
    long.seal(out,'cell_manifest.json')
    receipt=dict(passed=True,block=block,arm=arm,registration_sha256=digest(reg),cell_manifest_sha256=sha(out/'cell_manifest.json'),
        source_manifest_sha256=PIN,training_decisions=1024,eval_model_errors=errors,fresh_process=True,
        verifier_pid=os.getpid(),training_verification_sha256=sha(out/'train/independent_verification.json'))
    validate_receipt(receipt,block,arm,digest(reg),sha(out/'cell_manifest.json'))
    write_json(root/f'verified_b{block:02d}_{arm}.json',receipt)
    return receipt


def analysis(root,source,model_config,project,godot):
    reg,_=registration(root,source,model_config,project,godot);cells=[];details=[]
    for block,arm in itertools.product(range(8),ARMS):
        out=cell_path(root,block,arm);path=root/f'verified_b{block:02d}_{arm}.json'
        if not path.exists():continue
        validate_receipt(read_json(path),block,arm,digest(reg),sha(out/'cell_manifest.json'))
        long.check_manifest(out,'cell_manifest.json')
        _same(sha(out/'train/independent_verification.json'),read_json(path)['training_verification_sha256'],'training receipt pin')
        report=read_json(out/'evaluation.json')
        cells.append(dict(block=block,arm=arm,**{p:report['metrics'][p]['discounted_r1'] for p in ('pre','post')}))
        details.append(report)
    result=analyze(cells)|dict(cells=details,registration_sha256=digest(reg),source_manifest_sha256=PIN)
    write_json(root/'analysis.json',result)
    return result


def batch(args):
    root=args.root
    if (root/'batch_status.json').exists():raise FileExistsError('never overwrite a campaign')
    registration(root,args.source,args.model_config,args.project,args.godot)
    status=dict(complete=False,cells=[],planned_cells=24,started_utc=datetime.now(timezone.utc).isoformat())
    def persist():
        tmp=root/'batch_status.partial.json';tmp.write_text(canonical(status)+'\n',encoding='utf-8');tmp.replace(root/'batch_status.json')
    persist();orders=list(itertools.permutations(ARMS))
    common=['--root',str(root),'--source',str(args.source),'--model-config',str(args.model_config),
        '--project',str(args.project),'--godot',str(args.godot)]
    for block in range(8):
        for arm in orders[block%6]:
            cell=dict(block=block,arm=arm,stages=[]);status['cells'].append(cell);persist()
            for stage in ('train','evaluate','verify'):
                started=time.monotonic()
                with (root/f'b{block:02d}_{arm}_{stage}.log').open('x') as log:
                    try:
                        proc=subprocess.run([sys.executable,'-m','tools.mea_long_ablation',stage,*common,
                            '--block',str(block),'--arm',arm],stdout=log,stderr=subprocess.STDOUT,timeout=7200)
                        code=proc.returncode;error=None
                    except subprocess.TimeoutExpired:code=-1;error='stage timeout7200s'
                cell['stages'].append(dict(stage=stage,exit_code=code,error=error,seconds=time.monotonic()-started));persist()
                if code:break
    status['complete']=True
    status['passed']=all(len(c['stages'])==3 and all(s['exit_code']==0 for s in c['stages']) for c in status['cells'])
    status['finished_utc']=datetime.now(timezone.utc).isoformat();persist()
    result=analysis(root,args.source,args.model_config,args.project,args.godot)
    return dict(batch_passed=status['passed'],analysis_status=result['status'],positive_evidence=result['positive_evidence'])


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','train','evaluate','verify','batch','analyze'])
    for name in ('root','source','model-config'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--project',type=Path,default=ROOT);p.add_argument('--godot',type=Path,default=DEFAULT_GODOT)
    p.add_argument('--block',type=int);p.add_argument('--arm',choices=ARMS);a=p.parse_args()
    common=(a.root,a.source,a.model_config,a.project,a.godot)
    if a.action=='register':result=register(*common)
    elif a.action=='batch':result=batch(a)
    elif a.action=='analyze':result=analysis(*common)
    else:result=globals()[a.action](*common,a.block,a.arm)
    print(canonical(result),flush=True)


if __name__=='__main__':main()
