import concurrent.futures
import hashlib
import io
import json
import os
import runpy
import sys
import tarfile
import zipfile
from pathlib import Path

out=Path(__file__).resolve().parent
root=out.parents[1]
sys.path.insert(0,str(root/'standalone/brain_device_session'))
from fresh_map_host import archive
from replay_hosts import command,put,paramiko,SERVERS
remote='/home/researcher/mea-offline-single-pulse/22554-hwf-1119-clean'

def action(host,mode):
    address='192.168.77.'+host
    key=os.environ.get('MEA_SSH_'+address.replace('.','_'))
    if not key:
        loader=runpy.run_path(str(Path(os.environ['TEMP'])/'codex_22554_factorial_host.py'))
        key=loader['keys']()[host]
    with paramiko.SSHClient() as c:
        c.load_system_host_keys()
        c.connect(address,username='researcher',password=key,timeout=10)
        with c.open_sftp() as s:
            if mode=='start':
                z=io.BytesIO(archive())
                with zipfile.ZipFile(z,'a') as f:
                    for n in ('study.py','dispatch.py'):
                        f.writestr(zipfile.ZipInfo(n,(2026,1,1,0,0,0)),(out/n).read_bytes())
                blob=z.getvalue()
                sha=hashlib.sha256(blob).hexdigest()
                code=remote+'/code-'+sha[:12]
                command(c,'from pathlib import Path;import sys;Path(sys.argv[1]).mkdir(parents=True,exist_ok=True)',[code])
                put(s,code+'/code.zip',blob)
                for n in ('protocol.json','windows.json'):
                    put(s,remote+'/paired-'+n,(out/('clean-'+n)).read_bytes())
                put(s,remote+'/settings.json',json.dumps({k:SERVERS[address][k] for k in ('godot','project','model_paths')},sort_keys=True).encode())
                return command(c,"""import pathlib,subprocess,sys,zipfile,hashlib
r,code=map(pathlib.Path,sys.argv[1:3]);pid=r/'dispatcher.pid'
if pid.exists():
 p=pathlib.Path('/proc')/pid.read_text()/'stat'
 if p.exists() and p.read_text().split()[2]!='Z':raise RuntimeError('Already running')
z=code/'code.zip'
assert hashlib.sha256(z.read_bytes()).hexdigest()==sys.argv[4]
with zipfile.ZipFile(z) as a:a.extractall(code)
with (r/'dispatcher.log').open('ab') as f:p=subprocess.Popen([sys.executable,str(code/'dispatch.py'),str(r),sys.argv[3]],start_new_session=True,stdin=subprocess.DEVNULL,stdout=f,stderr=f)
pid.write_text(str(p.pid));print(p.pid)""",[remote,code,host,sha])
            status=json.loads(command(c,"""import pathlib,sys,json
r=pathlib.Path(sys.argv[1]);ret={}
for d in sorted((r/'runs').glob('*')):
 p=d/'progress.json'; errs=list(d.glob('*.error'))
 ret[d.name]=[json.loads(p.read_bytes())['updates'] if p.exists() else 0,(d/'trained.json').is_file(),len(list(d.glob('map-*.json'))),(d/'done.json').is_file(),[(q.stem,(d/(q.stem+'.log')).read_text(errors='replace')[-1600:]) for q in errs]]
print(json.dumps(ret))""",[remote]))
            if mode=='collect':
                for k,st in status.items():
                    if not st[3]:continue
                    d=out/'runs'/k
                    if (d/'done.json').exists():continue
                    d.mkdir(parents=True,exist_ok=True)
                    base=remote+'/runs/'+k+'/'
                    with s.open(base+'done.json') as f:done=json.load(f)
                    for name,h in done['games'].items():
                        n='map-'+name+'.json'
                        with s.open(base+n) as f:raw=f.read()
                        if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Corrupt transfer')
                        (d/n).write_bytes(raw)
                    if st[1]:
                        for n in ('trained.json','training-trace.json','registration.json','updates.json'):s.get(base+n,str(d/n))
                    (d/'done.json').write_text(json.dumps(done))
            return status

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        for h,res in zip(('16','23'),pool.map(lambda h:action(h,sys.argv[1]),('16','23'))):
            print(h,json.dumps(res),flush=True)
