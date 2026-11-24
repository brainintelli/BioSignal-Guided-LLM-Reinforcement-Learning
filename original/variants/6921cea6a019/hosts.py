"""Schedule one original-budget model per available GPU; collect verified archives."""
import base64
import concurrent.futures
import hashlib
import io
import json
import os
import runpy
import sys
import tarfile
import time
import zipfile
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT/'standalone/brain_device_session'))
from replay_hosts import command,paramiko,put
REMOTE='/home/researcher/mea-22021-1119-five-arm-cross-sample-v2'
ARMS=['masked','baseline','sham','regular','irregular']
SPEC=json.loads((OUT/'design.json').read_bytes())
JOBS=[f'{ARMS[(i+j)%5]}-{seed}' for j in range(5) for i,seed in enumerate(SPEC['training_seeds'])]
HOSTS=['16','23']

def ssh(h):
    address='192.168.77.'+h
    key=os.environ.get('MEA_SSH_'+address.replace('.','_'))
    if not key:
        old=Path(os.environ['TEMP'])/'codex_22554_factorial_host.py'
        if old.exists():key=runpy.run_path(str(old))['keys']()[h]
    c=paramiko.SSHClient()
    c.load_system_host_keys()
    c.connect(address,username='researcher',password=key,timeout=12)
    return c

SUPPORT="""import pathlib,sys,io,zipfile,base64
h=pathlib.Path.home()
orig=h/'cpp28080-crossover-20261118-replay/start-scene-paired/game'
pkg=h/'mea-predictability-check-369c6af196e0'
code=h/'mea-state2-r2-20261118/run02/project'
f=io.BytesIO()
with zipfile.ZipFile(f,'w') as z:
 for label,base,parts in [('game',orig,['scripts','configs','godot_tests','project.godot']),('pkg',pkg,['livepair','mea_binary_r2','mea_state2_r2','mea_online','mea_stimulation_v2']),('code',code,['agent','tools','configs','scripts'])]:
  for part in parts:
   folder=base/part
   assert folder.exists(),folder
   for p in ([folder] if folder.is_file() else sorted(folder.rglob('*'))):
    if p.is_file() and p.suffix in ('.py','.gd','.json','.godot','.tres','.tscn') and '__pycache__' not in p.parts:
     entry=zipfile.ZipInfo(label+'/'+p.relative_to(base).as_posix(),(2026,1,1,0,0,0))
     z.writestr(entry,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED)
print(base64.b64encode(f.getvalue()).decode())"""

def stage():
    with ssh('16') as c:buf=io.BytesIO(base64.b64decode(command(c,SUPPORT)))
    with zipfile.ZipFile(buf,'a') as z:
        for f in (['worker.py','design.json','train.py','verify_game.py','model_config.json']+
                  [a+'.json' for a in ARMS if a!='baseline']+
                  ['pre-stim-baseline/'+n for n in ['design.json','baseline.json','train.py','verify_game.py','model_config.json']]):
            z.writestr(zipfile.ZipInfo(f,(2026,1,1,0,0,0)),(OUT/f).read_bytes(),compress_type=zipfile.ZIP_DEFLATED)
    payload=buf.getvalue();sha=hashlib.sha256(payload).hexdigest()
    def install(h):
        with ssh(h) as c,c.open_sftp() as s:
            command(c,'from pathlib import Path;import sys;Path(sys.argv[1]).mkdir(parents=True,exist_ok=True)',[REMOTE])
            put(s,REMOTE+'/payload-'+sha[:16]+'.zip',payload)
            return command(c,"""import sys,json,hashlib,zipfile
from pathlib import Path
r=Path(sys.argv[1]);z=r/('payload-'+sys.argv[2][:16]+'.zip')
assert hashlib.sha256(z.read_bytes()).hexdigest()==sys.argv[2]
with zipfile.ZipFile(z) as f:f.extractall(r)
assert all(Path(v).is_dir() for v in json.loads((r/'model_config.json').read_bytes()).values())
print(sys.argv[2])""",[REMOTE,sha])
    with concurrent.futures.ThreadPoolExecutor(2) as ex:print('Package',list(ex.map(install,HOSTS)),flush=True)

STATUS="""import sys,subprocess,pathlib,json
r=pathlib.Path(sys.argv[1]);jobs={};occupied=[]
for p in (r/'jobs').glob('*'):
 if not p.is_dir():continue
 gpu=int((p/'gpu').read_text());done=p/'done.json';err=p/'error.json'
 a,seed=p.name.split('-');s=(r/'pre-stim-baseline' if a=='baseline' else r)/'models'/seed/a
 stage='done' if done.exists() else 'error' if err.exists() else 'fit'
 if stage=='fit':
  pid=(p/'pid').read_text().strip();stat=pathlib.Path('/proc')/pid/'stat'
  if not stat.exists() or stat.read_text().split()[2]=='Z':stage='error'
  elif (s/'result.json').exists():stage='audit'
  elif (s/'update.json').exists():stage='evaluate'
  if stage in ('fit','evaluate','audit'):occupied.append(gpu)
 jobs[p.name]=[stage,gpu]
free=[int(x) for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()]
print(json.dumps([jobs,[i for i,m in enumerate(free) if m>=16400 and i not in occupied]]))"""

START="""import os,sys,subprocess,pathlib
r=pathlib.Path(sys.argv[1])
for pair in sys.argv[2:]:
 j,g=pair.split(':');arm,seed=j.split('-');p=r/'jobs'/j;p.mkdir(parents=True)
 (p/'gpu').write_text(g)
 with (p/'run.log').open('xb') as log:
  proc=subprocess.Popen([sys.executable,str(r/'worker.py'),arm,seed,g],stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True,cwd=r)
 (p/'pid').write_text(str(proc.pid))
print(len(sys.argv)-2)"""

def status(h):
    with ssh(h) as c:return json.loads(command(c,STATUS,[REMOTE]))

def run():
    last=None
    while True:
        with concurrent.futures.ThreadPoolExecutor(2) as ex:states=dict(zip(HOSTS,ex.map(status,HOSTS)))
        combined={k:v for js,_ in states.values() for k,v in js.items()}
        if len(combined)!=sum(len(js) for js,_ in states.values()):raise RuntimeError('Duplicate fit')
        failed=[(k,h) for h,(js,_) in states.items() for k,v in js.items() if v[0]=='error']
        if failed:
            for k,h in failed:
                with ssh(h) as c:
                    print(h,k,command(c,'import pathlib,sys;print(pathlib.Path(sys.argv[1]).read_text(errors="replace")[-1800:])',[REMOTE+'/jobs/'+k+'/run.log']))
            raise RuntimeError('Worker failure')
        pending=[k for k in JOBS if k not in combined]
        if pending:
            allocation={h:[] for h in HOSTS}
            for i in range(8):
                for h in HOSTS:
                    if pending and i in states[h][1]:allocation[h].append(pending.pop(0)+':'+str(i))
            for h,new in allocation.items():
                if new:
                    with ssh(h) as c:command(c,START,[REMOTE,*new])
        progress={h:{st:sum(v[0]==st for v in js.values()) for st in ('fit','evaluate','audit','done')} for h,(js,_) in states.items()}
        if progress!=last:print(time.strftime('%H:%M:%S'),progress,flush=True);last=progress
        if len(combined)==40 and all(v[0]=='done' for v in combined.values()):return
        time.sleep(12)

def fetch(task):
    h,j=task;arm,seed=j.split('-')
    dest=(OUT/'pre-stim-baseline' if arm=='baseline' else OUT)/'models'/seed/arm
    if (dest/'transfer.json').is_file():return
    dest.mkdir(parents=True,exist_ok=True)
    base=REMOTE+'/jobs/'+j+'/'
    local=dest/'transfer.tar.partial'
    with ssh(h) as c,c.open_sftp() as s:
        with s.open(base+'done.json') as f:meta=json.load(f)
        s.get(base+'result.tar',str(local),max_concurrent_prefetch_requests=48)
    with local.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==meta['sha256']
    with tarfile.open(local) as tar:
        for item in tar:
            assert item.isfile() and (dest/item.name).resolve().is_relative_to(dest.resolve())
            tar.extract(item,dest,filter='data')
    local.unlink()
    (dest/'transfer.json').write_text(json.dumps(meta|dict(host=h)),encoding='utf8')

def collect():
    tasks=[(h,j) for h in HOSTS for j,v in status(h)[0].items() if v[0]=='done']
    with concurrent.futures.ThreadPoolExecutor(6) as ex:list(ex.map(fetch,tasks))
    print('Verified transfers',len(tasks),flush=True)

if __name__=='__main__':
    {'stage':stage,'run':run,'collect':collect,'status':lambda:print({h:status(h) for h in HOSTS})}[sys.argv[1]]()
