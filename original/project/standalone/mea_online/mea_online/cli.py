"""check/offline-verify are hardware-free; live requires a newly bound session."""
import argparse
import os
from pathlib import Path
import sys
import time
from .common import canonical,digest,read,write
from .config import Budget,claim_session,started_wall,template,validate_config

def offline_verify(output):
    from .offline import exercise
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    cases=[('normal',{}),('terminal',dict(terminal_every=4)),('mid_terminal',dict(terminal_every=2)),
           ('disconnect',dict(fault_at=3)),('expired',dict(expire_at=3)),('r2_violation',dict(runtime_reward_violation=True))]
    results={}
    for name,options in cases:
        folder=out/name;folder.mkdir();r=exercise(output=folder/'run',**options)
        expected='completed' if name in ('normal','terminal','mid_terminal') else 'failed'
        passed=r['status']==expected and (r['updates']==2 if expected=='completed' else r['updates']==0)
        results[name]=dict(passed=passed,result=r)
    report=dict(schema='mea.online.offline-verification.v1',passed=all(x['passed'] for x in results.values()),
      real_mea=False,real_m9=False,real_gpu=False,real_stimulation=False,r2_enabled=False,cases=results)
    write(out/'verification.json',report)
    print(canonical(dict(passed=report['passed'],output=str(out),real_mea=False,real_m9=False)))
    return 0 if report['passed'] else 1

def worker(config_path,output):
    # Parent owns exclusive session claim and process-group hard stop.
    from .adapter import WindowClient
    from .calibration import collect
    from .m9_runtime import M9Runtime
    from .trainer import Trainer
    out=Path(output);config=validate_config(read(config_path),live=True,allow_pending_calibration=True)
    launch=read(out/'launch.json')
    if launch['config_sha256']!=digest(config) or launch['parent_pid']!=os.getppid():raise ValueError('Worker launch binding')
    budget=Budget(started_wall(config['binding']));budget.require(config['source']['timeout'])
    source=None;runtime=None
    try:
        source=WindowClient(binding=config['binding'],**config['source'])
        if config['calibration'] is None:
            config['calibration']=collect(source,config['binding'],out/'calibration',budget,live=True)
        config=validate_config(config,live=True)
        write(out/'effective_config.json',config)
        budget.require(config['update_reserve_seconds'])
        runtime_config={**config,'source':'live','binding_sha256':digest(config['binding']),
                        'normalization_id':config['calibration']['normalization_id']}
        runtime=M9Runtime(runtime_config,out/'runtime')
        budget.require(config['source']['timeout'])
        result=Trainer(config,source,runtime,out/'run',budget,live=True).run()
        return 0 if result['status']=='completed' else 1
    except BaseException as e:
        write(out/'worker_failure.json',dict(error=type(e).__name__+': '+str(e),r2_enabled=False,stimulation=False))
        raise
    finally:
        for obj in (source,runtime):
            if obj is not None:obj.close()

def live(config_path,output):
    from .supervisor import run
    config=validate_config(read(config_path),live=True,allow_pending_calibration=True)
    budget=Budget(started_wall(config['binding']));budget.require(config['update_reserve_seconds'])
    if sys.platform!='linux':raise ValueError('Live supervisor requires Linux target 192.0.2.16')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    claim=claim_session(config['registry'],config['binding'])
    write(out/'requested_config.json',config)
    write(out/'launch.json',dict(config_sha256=digest(config),parent_pid=os.getpid(),claim_path=str(claim),r2_enabled=False,stimulation=False))
    result=run([sys.executable,'-m','mea_online.cli','_worker','--config',str(out/'requested_config.json'),'--output',str(out)],out,budget)
    print(canonical(result));return 0 if result['status']=='completed' else 1

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    c=sub.add_parser('check');c.add_argument('--config',required=True)
    o=sub.add_parser('offline-verify');o.add_argument('--output',required=True)
    t=sub.add_parser('template');t.add_argument('--output',required=True)
    for name in ('live','_worker'):
        q=sub.add_parser(name,help=argparse.SUPPRESS if name=='_worker' else 'Start a newly bound unstimulated online session')
        q.add_argument('--config',required=True);q.add_argument('--output',required=True)
    args=p.parse_args(argv)
    if args.mode=='template':write(args.output,template());return 0
    if args.mode=='check':
        value=read(args.config)
        # Validate static template without interpreting live calibration as simulation.
        live_bound=value.get('binding') is not None
        config=validate_config(value,live=live_bound,allow_pending_calibration=True)
        ready=live_bound and config['calibration'] is not None
        print(canonical(dict(status='configuration_checked',identity_bound=live_bound,calibration_present=config['calibration'] is not None,
          live_configuration_complete=ready,hardware_verified=False,gpu_runtime_verified=False,r2_enabled=False,stimulation=False)))
        return 0
    if args.mode=='offline-verify':return offline_verify(args.output)
    if args.mode=='_worker':return worker(args.config,args.output)
    return live(args.config,args.output)

if __name__=='__main__':raise SystemExit(main())
