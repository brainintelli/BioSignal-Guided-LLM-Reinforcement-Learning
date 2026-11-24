"""Copy stable study evidence and closed raw recordings with both-end hashes."""
from pathlib import Path
import concurrent.futures
import json
import sys
import tempfile
import zipfile
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent))
from acquire import connect,remote,STAGE
from archive import download

D=STAGE+'/records/frequency-retest'
META="""import sys,pathlib,zipfile,tempfile
p=pathlib.Path(sys.argv[1]); files=[x for x in p.rglob('*') if x.is_file() and x.suffix in ('.json','.jsonl','.cfg','.log') and not x.name.endswith('cpp-spikes.jsonl')]
with tempfile.NamedTemporaryFile(dir=p,suffix='.zip',delete=False) as tmp:
 with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
  for f in files:z.write(f,f.relative_to(p))
 print(tmp.name)"""
RAW="""import sys,pathlib,json,hashlib
p=pathlib.Path(sys.argv[1]);known=json.loads(sys.argv[2]);finished=sys.argv[3]=='final';out={}
for f in p.rglob('*'):
 if f.suffix!='.h5' and f.name!='cpp-spikes.jsonl':continue
 if not finished:
  if f.suffix!='.h5':continue
  directory=f.parent.parent
  if f.name=='quiet.raw.h5' and not ((directory/'rank.json').exists() or (directory/'rank-initial.json').exists() or (f.parent/'screen.json').exists()):continue
  if f.name=='neutral.raw.h5' and not (directory/'neutral-check.json').exists():continue
  if f.name.startswith('trial_') and not (directory/(f.parent.name+'-evidence.json')).exists():continue
 rel='frequency-retest/'+f.relative_to(p).as_posix()
 if rel in known:continue
 h=hashlib.sha256()
 with f.open('rb') as s:
  for b in iter(lambda:s.read(8*1024**2),b''):h.update(b)
 out[rel]={'sha256':h.hexdigest(),'bytes':f.stat().st_size}
print(json.dumps(out))"""
def collect(mode):
    manifest=OUT/'verified-recordings.json'
    known=json.loads(manifest.read_bytes()) if manifest.exists() else {}
    with connect() as c:
        source=remote(c,META,[D])
        with c.open_sftp() as s,tempfile.TemporaryFile() as file:
            s.getfo(source,file);file.seek(0)
            with zipfile.ZipFile(file) as z:z.extractall(OUT)
            s.remove(source)
        missing=json.loads(remote(c,RAW,[D,json.dumps(list(known)),mode],timeout=180))
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        for job in concurrent.futures.as_completed([pool.submit(download,x) for x in missing.items()]):
            k,v=job.result();known[k]=v
            t=manifest.with_suffix('.partial');t.write_text(json.dumps(known,indent=2),encoding='utf8');t.replace(manifest)
            print(k,v['bytes'],flush=True)
    print('archived',len(known),flush=True)
if __name__=='__main__':collect(sys.argv[1])
