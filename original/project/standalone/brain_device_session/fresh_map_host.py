"""Register and run the 22554 frozen-checkpoint validation on its two original hosts."""
import argparse
import hashlib
import io
import json
import os
import random
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from replay_hosts import ROOT, bundle, command, paramiko, put

SOURCE = ROOT/'output/22554-20261118T090720Z/exploratory/extended-96x48'
OUT = ROOT/'output/22554_fresh_maps_20261119'
REMOTE = '/home/researcher/mea-offline-single-pulse/22554-fresh-maps-20261119'
HERE = Path(__file__).resolve().parent
ARMS = ('pure','stateonly','r2only','ordered','shuffled')


def read(path):
    return json.loads(path.read_bytes())


def protocol():
    prior = SOURCE/'reward-mix-tick48'
    base = read(prior/'protocol.json')
    models = {}
    for co, seeds in base['cohorts'].items():
        for arm in ARMS:
            study = {'pure':'three-arm-six-seeds','stateonly':'factorial-tick48'}.get(arm,'reward-mix-tick48')
            result = read(SOURCE/study/co/arm/'result.json')
            for seed, record in zip(seeds,result['seeds'],strict=True):
                assert record['training_seed']==seed and record['arm']==arm
                if arm in ('pure','stateonly'):
                    host = '16' if co=='original' else '23'
                else:
                    host = base['assignments'][co][arm]
                cp = record['checkpoint']
                models[f'{arm}-{seed}'] = dict(arm=arm,training_seed=seed,host=host,checkpoint=cp['path'],
                       manifest=cp['manifest_sha256'],registration=cp['registration_sha256'],state=record['after'])
    prime = read(prior/'original/preceding-quiet.json')['prime']
    # Same canonical JSON representation as mea_online.m9_runtime.digest.
    prime_hash = hashlib.sha256(json.dumps(prime,sort_keys=True,separators=(',',':'),ensure_ascii=False,
                                           allow_nan=False).encode()).hexdigest()
    maps = random.Random(2026111901).sample(range(40000000,50000000),24)
    return dict(chip='22554', frozen=True, source_sha=base['source_sha'], prime_digest=prime_hash, models=models,
                maps=maps, old_maps=base['game_eval_seeds'], ticks=[48,192], reward_mix=base['reward_mix'],
                engine=read(SOURCE/'three-arm-six-seeds/game-runtime-hashes.json')['192.0.2.23'],
                primary='Tick 48 satiety, ordered minus both pure and shuffled. Weak candidate: positive mean and at least 4/6 training seeds positive on each contrast. Stronger support: both paired 95% crossed-bootstrap lower bounds above zero.',
                secondary='Explored cells, health, water and deaths; tick 192 persistence; eat/drink choices when legally available and corresponding reserve below 60. Record all directions, including costs.',
                pairing='Same 6 frozen seeds and 24 fresh maps for all arms. Paired differences, equal-weight training seeds and maps. Percentile crossed bootstrap over both axes with 20000 draws, seed 2026111902. Old maps only for reload probes.')


def archive():
    data = io.BytesIO(bundle())
    with zipfile.ZipFile(data,'a') as f:
        for name in ('clock_replay.py','evaluate_fresh_maps.py'):
            f.writestr(zipfile.ZipInfo(name,date_time=(2026,1,1,0,0,0)),(HERE/name).read_bytes())
    return data.getvalue()


DISPATCH = """import concurrent.futures,json,os,pathlib,subprocess,sys,time
r=pathlib.Path(sys.argv[1]);host=sys.argv[2];code=sys.argv[3];plan=json.loads((r/'protocol.json').read_text())
jobs=[k for k,v in plan['models'].items() if v['host']==host and not (r/'runs'/k/'done.json').exists()]
env=dict(os.environ);env.update(json.loads((pathlib.Path.home()/'mea-state2-r2-20261118/run02/environment.json').read_text()))
env.update(PYTHONPATH=code+os.pathsep+env.get('PYTHONPATH',''),OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
py=str(pathlib.Path.home()/'mea-r2-22554/runtime_20261116_01/assets/python/bin/python')
def lane(gpu):
    for job in jobs[gpu::8]:
        while True:
            mem=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']))
            if mem>=10500:break
            time.sleep(12)
        with (r/(job+'.log')).open('ab') as log:
            proc=subprocess.run([py,'-u',str(pathlib.Path(code)/'evaluate_fresh_maps.py'),str(r),job],
                     stdout=log,stderr=log,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)))
        if proc.returncode:
            (r/(job+'.error')).write_text(str(proc.returncode))
        else:
            (r/(job+'.error')).unlink(missing_ok=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(lane,range(8)))
"""


def host_action(host,mode):
    address='192.168.77.'+host
    with paramiko.SSHClient() as c:
        c.load_system_host_keys()
        c.connect(address,username='researcher',password=os.environ['MEA_SSH_'+address.replace('.','_')],timeout=10)
        with c.open_sftp() as s:
            if mode=='start':
                blob=archive()
                key=hashlib.sha256(blob).hexdigest()
                code=REMOTE+'/code-'+key[:12]
                command(c,'import pathlib,sys;pathlib.Path(sys.argv[1]).mkdir(parents=True,exist_ok=True)',[code])
                put(s,code+'/payload.zip',blob)
                put(s,code+'/dispatch.py',DISPATCH.encode())
                for name in ('replay.json','preceding-quiet.json'):
                    put(s,REMOTE+'/'+name,(SOURCE/'reward-mix-tick48/original'/name).read_bytes())
                put(s,REMOTE+'/protocol.json',(OUT/'protocol.json').read_bytes())
                cmd="""import hashlib,json,pathlib,subprocess,sys,zipfile
r,code=map(pathlib.Path,sys.argv[1:3]);host,key=sys.argv[3:5];z=code/'payload.zip';pid=r/'dispatcher.pid'
if pid.exists():
 old=pathlib.Path('/proc')/pid.read_text()/'stat'
 if old.exists() and old.read_text().split()[2]!='Z':raise ValueError('Dispatcher already running')
if hashlib.sha256(z.read_bytes()).hexdigest()!=key:raise ValueError('Changed code')
with zipfile.ZipFile(z) as f:f.extractall(code)
plan=json.loads((r/'protocol.json').read_text());missing=[k for k,v in plan['models'].items() if v['host']==host and not pathlib.Path(v['checkpoint'],'manifest.json').is_file()]
if missing:raise ValueError('Missing checkpoints '+str(missing))
with (r/'dispatch.log').open('ab') as log:p=subprocess.Popen([sys.executable,str(code/'dispatch.py'),str(r),host,str(code)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
pid.write_text(str(p.pid));print(p.pid)
"""
                return command(c,cmd,[REMOTE,code,host,key])
            status=json.loads(command(c,"""import json,pathlib,sys
r=pathlib.Path(sys.argv[1]);out={};plan=json.loads((r/'protocol.json').read_text())
for k,v in plan['models'].items():
 if v['host']!=sys.argv[2]:continue
 d=r/'runs'/k;e=r/(k+'.error');log=r/(k+'.log')
 out[k]=dict(games=len(list(d.glob('[0-9]*.json'))),done=(d/'done.json').is_file(),error=log.read_text(errors='replace')[-1500:] if e.exists() else None)
print(json.dumps(out))""",[REMOTE,host]))
            if mode=='collect':
                for job,st in status.items():
                    if not st['done']:continue
                    local=OUT/'runs'/job;local.mkdir(parents=True,exist_ok=True)
                    if (local/'done.json').is_file():continue
                    remote=REMOTE+'/runs/'+job+'/'
                    with s.open(remote+'done.json') as f:meta=json.load(f)
                    for seed,expected in meta['game_hashes'].items():
                        dst=local/(seed+'.json');tmp=dst.with_suffix('.partial');s.get(remote+dst.name,str(tmp))
                        if hashlib.sha256(tmp.read_bytes()).hexdigest()!=expected:raise ValueError('Game transfer differs')
                        tmp.replace(dst)
                    (local/'done.json').write_text(json.dumps(meta))
            return status


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['register','start','status','collect'])
    parser.add_argument('--hosts',nargs='+',default=['16','23'])
    args=parser.parse_args()
    if args.mode=='register':
        OUT.mkdir(exist_ok=True)
        payload=json.dumps(protocol(),indent=2).encode()
        path=OUT/'protocol.json'
        if path.exists() and path.read_bytes()!=payload:raise ValueError('Registration changed')
        path.write_bytes(payload)
        print(path)
    else:
        with ThreadPoolExecutor(len(args.hosts)) as pool:
            for h,res in zip(args.hosts,pool.map(lambda h:host_action(h,args.mode),args.hosts)):
                print(h,json.dumps(res),flush=True)
