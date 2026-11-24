"""Stage and monitor the isolated offline single-pulse study on the two existing GPU nodes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.tools'/'mea_ssh_deps'))
import paramiko

HOSTS={'192.0.2.16':'ordered','192.0.2.23':'shuffled'}
SERVERS=json.loads((ROOT/'output/mea_spatial_gate_20261118/servers.json').read_text(encoding='utf-8-sig'))

def command(cli,code,args=()):
    _,out,err=cli.exec_command(shlex.join(['/usr/bin/python3','-c',code,*args]),timeout=110)
    result=out.read().decode();errors=err.read().decode(errors='replace')
    if out.channel.recv_exit_status():raise RuntimeError(errors[-2400:])
    return result.strip()

def bundle():
    f=io.BytesIO()
    with zipfile.ZipFile(f,'w',zipfile.ZIP_DEFLATED) as z:
        def add(name,contents):
            entry=zipfile.ZipInfo(name,date_time=(2026,1,1,0,0,0))
            z.writestr(entry,contents,compress_type=zipfile.ZIP_DEFLATED)
        for folder in ('mea_online/mea_online','mea_live_pair/livepair','mea_binary_r2/mea_binary_r2','mea_state2_r2/mea_state2_r2'):
            src=ROOT/'standalone'/folder
            for p in sorted(src.rglob('*')):
                if p.is_file() and p.suffix in ('.py','.json'):
                    add(str(p.relative_to(src.parent).as_posix()),p.read_bytes())
        add('train_recorded_pulses.py',(ROOT/'standalone/brain_device_session/train_recorded_pulses.py').read_bytes())
        add('train_long_recorded.py',(ROOT/'standalone/brain_device_session/train_long_recorded.py').read_bytes())
    return f.getvalue()

JOB="""import json,os,pathlib,subprocess,sys,time,traceback
root=pathlib.Path(__file__).resolve().parent
arm=sys.argv[1];trial=sys.argv[2];out=root/trial
env=dict(os.environ);env.update(json.loads((pathlib.Path.home()/'mea-state2-r2-20261118/run02/environment.json').read_text()))
env['PYTHONPATH']=str(root/'code')+os.pathsep+env.get('PYTHONPATH','')
py=str(pathlib.Path.home()/'mea-r2-22554/runtime_20261116_01/assets/python/bin/python')
command=[py,'-u',str(root/'code/train_recorded_pulses.py'),'--arm',arm,'--out',str(out/'training'),
         '--data',str(root/'replay.json'),'--prime',str(root/'preceding-quiet.json'),'--settings',str(root/'settings.json')]
if (root/'budget.json').is_file():
    command[2]=str(root/'code/train_long_recorded.py')
    command.extend(['--budget',str(root/'budget.json')])
if len(sys.argv)>3 and sys.argv[3]:
    command.extend(['--reuse-complete',sys.argv[3]])
try:
    p=subprocess.run(command,env=env)
    state=dict(exit_code=p.returncode,finished=time.time())
except Exception:
    state=dict(exit_code=1,error=traceback.format_exc(),finished=time.time())
(out/'exit.json').write_text(json.dumps(state))
sys.exit(state['exit_code'])
"""

def put(sftp,path,blob):
    try:
        with sftp.open(path,'rb') as f:existing=f.read()
        if existing!=blob:raise ValueError('Mismatched existing artifact '+path)
        return
    except FileNotFoundError:pass
    with sftp.open(path,'wx') as f:f.write(blob)

def run_host(host,args):
    arm=args.arm or HOSTS[host];root='/home/researcher/mea-offline-single-pulse/'+args.session
    with paramiko.SSHClient() as cli:
        cli.load_system_host_keys();cli.connect(host,username='researcher',password=os.environ.get('MEA_SSH_'+host.replace('.','_')),timeout=8)
        with cli.open_sftp() as s:
            if args.mode=='stage':
                command(cli,"import pathlib,sys;pathlib.Path(sys.argv[1]).mkdir(parents=True,exist_ok=True)",[root])
                blob=bundle();sha=hashlib.sha256(blob).hexdigest()
                put(s,root+'/bundle.zip',blob);put(s,root+'/job.py',JOB.encode())
                paths={k:SERVERS[host][k] for k in ('model_paths','project','godot')}
                put(s,root+'/settings.json',json.dumps(paths,sort_keys=True).encode())
                put(s,root+'/preceding-quiet.json',(args.study/'preceding-quiet.json').read_bytes())
                if (args.study/'budget.json').is_file():
                    put(s,root+'/budget.json',(args.study/'budget.json').read_bytes())
                code="""import sys,pathlib,zipfile,hashlib,subprocess,json,os
r=pathlib.Path(sys.argv[1]);z=r/'bundle.zip'
if hashlib.sha256(z.read_bytes()).hexdigest()!=sys.argv[2]:raise ValueError('Package differs')
with zipfile.ZipFile(z) as a:a.extractall(r/'code')
env=dict(os.environ);env.update(json.loads((pathlib.Path.home()/'mea-state2-r2-20261118/run02/environment.json').read_text()))
env['PYTHONPATH']=str(r/'code')+os.pathsep+env.get('PYTHONPATH','')
py=str(pathlib.Path.home()/'mea-r2-22554/runtime_20261116_01/assets/python/bin/python')
entry='train_long_recorded.py' if (r/'budget.json').exists() else 'train_recorded_pulses.py'
subprocess.run([py,str(r/'code'/entry),'--help'],check=True,env=env,stdout=subprocess.DEVNULL)
print(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
"""
                return dict(arm=arm,free_mib=command(cli,code,[root,sha]).splitlines(),bundle_sha256=sha)
            if args.mode=='start':
                put(s,root+'/replay.json',(args.study/'replay.json').read_bytes())
                gate="""import json,pathlib,subprocess,sys,time,os
threshold=int(sys.argv[1]);out=pathlib.Path(sys.argv[2]);previous=sys.argv[3];primary=int(sys.argv[4]);cmd=sys.argv[5:]
if previous:
    done=pathlib.Path(previous)/'exit.json'
    while not done.exists():time.sleep(5)
    state=json.loads(done.read_text())
    if state['exit_code']!=0:
        (out/'exit.json').write_text(json.dumps(dict(exit_code=1,error='Prior job failed: '+previous)))
        sys.exit(1)
consecutive=0
while threshold and consecutive<3:
    text=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    devices=[line.split(',') for line in text.strip().splitlines()]
    free=[int(v) for _,v in devices]
    consecutive=consecutive+1 if len(free)==8 and min(free)>=threshold and max(free)>=primary else 0
    if consecutive<3:time.sleep(15)
if primary:
    order=sorted(devices,key=lambda x:-int(x[1]))
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(uuid.strip() for uuid,_ in order)
(out/'resource-ready.json').write_text(json.dumps(dict(free_mib=free if threshold else None,start=time.time(),
                   device_order=os.environ.get('CUDA_VISIBLE_DEVICES'))))
os.execv(cmd[0],cmd)
"""
                code="""import pathlib,sys,subprocess,json,os
r=pathlib.Path(sys.argv[1]);out=r/sys.argv[3];out.mkdir()
f=(out/'job.log').open('xb')
cmd=[sys.executable,str(r/'job.py'),sys.argv[2],sys.argv[3]]
if sys.argv[7]:cmd.append(sys.argv[7])
if int(sys.argv[4]) or sys.argv[6]:cmd=[sys.executable,'-c',sys.argv[5],sys.argv[4],str(out),sys.argv[6],sys.argv[8],*cmd]
p=subprocess.Popen(cmd,cwd=r,stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
(out/'pid').write_text(str(p.pid));print(p.pid)
"""
                previous='/home/researcher/mea-offline-single-pulse/'+args.after if args.after else ''
                reuse='/home/researcher/mea-offline-single-pulse/'+args.reuse_complete+'/training' if args.reuse_complete else ''
                return dict(arm=arm,pid=int(command(cli,code,[root,arm,args.attempt,str(args.min_free_mib),gate,previous,reuse,str(args.primary_free_mib)])))
            code="""import pathlib,sys,json
p=pathlib.Path(sys.argv[1])/sys.argv[2];t=p/'training';e=p/'exit.json';f=p/'job.log'
with f.open('rb') as log:
 log.seek(max(0,f.stat().st_size-3000));tail=log.read().decode(errors='replace')
stages={v.name:[name for name in ('registration.json','training-trace.json','checkpoint/manifest.json','result.json') if (v/name).exists()] for v in t.glob('*') if v.is_dir()}
progress={v.parent.name:json.loads(v.read_text()) for v in t.glob('*/progress.json')}
print(json.dumps(dict(exit=json.loads(e.read_text()) if e.exists() else None,seeds=[v.parent.name for v in t.glob('*/result.json')],stages=stages,progress=progress,tail=tail[-1200:])))
"""
            status=json.loads(command(cli,code,[root,args.attempt]))
            if args.mode=='collect' and status['exit'] and status['exit']['exit_code']==0:
                remote=root+'/'+args.attempt+'/training'
                local=args.study/arm;local.mkdir(exist_ok=True)
                s.get(remote+'/result.json',str(local/'result.json'))
                for seed in status['seeds']:
                    d=local/seed;d.mkdir(exist_ok=True)
                    for name in ('registration.json','training-trace.json'):
                        s.get(remote+'/'+seed+'/'+name,str(d/name))
                    if (args.study/'budget.json').is_file():
                        for name in ('eval-trace.json','updates.json'):
                            s.get(remote+'/'+seed+'/'+name,str(d/name))
            return status

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['stage','start','status','collect']);p.add_argument('study',type=Path)
    p.add_argument('--session',required=True);p.add_argument('--attempt',default='run01');p.add_argument('--host',choices=list(HOSTS))
    p.add_argument('--arm',choices=['ordered','shuffled','pure','stateonly','r2only'])
    p.add_argument('--after',help='Wait for a successful session/attempt on this host')
    p.add_argument('--min-free-mib',type=int,default=0)
    p.add_argument('--primary-free-mib',type=int,default=0)
    p.add_argument('--reuse-complete',help='Verify and link finished seeds from session/attempt on this host')
    a=p.parse_args()
    if a.arm and not a.host:p.error('--arm requires one --host')
    if a.after and (len(a.after.split('/'))!=2 or not all(
            n.replace('-','').isalnum() for n in a.after.split('/'))):p.error('Invalid prior session/attempt')
    if a.reuse_complete and (len(a.reuse_complete.split('/'))!=2 or not all(
            n.replace('-','').isalnum() for n in a.reuse_complete.split('/'))):p.error('Invalid seed source')
    if a.primary_free_mib and not a.min_free_mib:p.error('Primary free-memory budget requires a per-card minimum')
    if not all(x.replace('-','').isalnum() for x in (a.session,a.attempt)):raise ValueError('Invalid run name')
    hosts=[a.host] if a.host else list(HOSTS)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={h:pool.submit(run_host,h,a) for h in hosts};out={}
        for h,f in futures.items():
            try:out[h]=f.result()
            except Exception as exc:out[h]={'error':str(exc)}
    print(json.dumps(out,ensure_ascii=False))
    sys.exit(int(any('error' in v for v in out.values())))
