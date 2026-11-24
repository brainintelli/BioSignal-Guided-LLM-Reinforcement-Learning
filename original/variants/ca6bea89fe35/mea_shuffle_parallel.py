"""One independent paired block per GPU; no shared gradients or hardware access."""
import argparse,concurrent.futures,json,os,signal,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from agent.mea_shuffle import PROTOCOL,PIN
from tools import mea_shuffle_campaign as campaign

def arm_order(block):
    if type(block) is not int or not 0<=block<8:raise ValueError('Block')
    arms=PROTOCOL['arms'];offset=block%4
    return arms[offset:]+arms[:offset]

def validate_blocks(statuses):
    if len(statuses)!=8 or {s.get('block') for s in statuses}!=set(range(8)):raise ValueError('Incomplete or duplicated blocks')
    if any(s.get('passed') is not True or type(s.get('accepted_cells')) is not int or s['accepted_cells']!=4 for s in statuses):raise ValueError('Failed or incomplete block')

def persist(path,value):
    tmp=path.with_suffix('.partial.json');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(path)

def run_stage(command,log_path,*,timeout=7200,cwd=None,env=None):
    began=time.monotonic()
    with Path(log_path).open('x',encoding='utf-8') as log:
        p=subprocess.Popen(command,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=timeout);error=None
        except subprocess.TimeoutExpired:
            if os.name=='posix':os.killpg(p.pid,signal.SIGKILL)
            else:p.kill()
            p.wait();code=-1;error='timeout'
    return dict(exit_code=code,error=error,seconds=time.monotonic()-began,pid=p.pid)

def run_block(args,block):
    status=dict(block=block,gpu=block,complete=False,passed=False,accepted_cells=0,stages=[])
    path=args.root/f'block_{block:02d}_status.json';persist(path,status)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(block))
    common=['--root',str(args.root),'--source',str(args.source),'--pin',PIN,'--model-config',str(args.model_config),
        '--project',str(args.project),'--godot',str(args.godot),'--block',str(block)]
    try:
        for arm in arm_order(block):
            status.update(current_arm=arm)
            for stage in ('train','evaluate','verify'):
                status['current_stage']=stage;persist(path,status)
                result=run_stage([sys.executable,'-u','-m','tools.mea_shuffle_campaign',stage,*common,'--arm',arm],
                    args.root/f'b{block:02d}_{arm}_{stage}.log',cwd=args.project,env=env)
                status['stages'].append(dict(arm=arm,stage=stage,**result));persist(path,status)
                if result['exit_code']:raise RuntimeError(f'{arm}/{stage} failed')
            receipt=json.loads((args.root/f'verified_b{block:02d}_{arm}.json').read_text(encoding='utf-8'))
            if receipt.get('passed') is not True:raise ValueError('Rejected cell')
            status['accepted_cells']+=1;persist(path,status)
        status.update(complete=True,passed=True,current_stage='complete')
    except Exception as exc:status.update(complete=True,passed=False,error=repr(exc))
    persist(path,status);return status

def main():
    p=argparse.ArgumentParser()
    for k in ('root','source','model-config','project','godot'):p.add_argument('--'+k,required=True,type=Path)
    a=p.parse_args();campaign.registration(a.root,a.source,PIN,a.model_config,a.project,a.godot)
    if (a.root/'parallel_status.json').exists() or list(a.root.glob('block_*_status.json')):raise FileExistsError('No restart/overwrite')
    status=dict(passed=False,complete=False,planned_cells=32,started_utc=datetime.now(timezone.utc).isoformat(),r2_enabled=False)
    persist(a.root/'parallel_status.json',status)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            states=list(pool.map(lambda block:run_block(a,block),range(8)))
        validate_blocks(states)
        status['stage']='aggregate';persist(a.root/'parallel_status.json',status)
        result=campaign.analysis(a.root,a.source,PIN,a.model_config,a.project,a.godot)
        if result.get('status')!='complete' or len(result.get('cells',[]))!=32:raise ValueError('Incomplete aggregate')
        status.update(passed=True,complete=True,accepted_cells=32,stage='complete')
    except Exception as exc:status.update(passed=False,complete=True,error=repr(exc))
    status['finished_utc']=datetime.now(timezone.utc).isoformat();persist(a.root/'parallel_status.json',status)
    print(json.dumps(status),flush=True)
    if not status['passed']:raise SystemExit(1)

if __name__=='__main__':main()
