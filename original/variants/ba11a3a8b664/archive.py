"""Copy only closed records, verifying SHA-256 on the acquisition host and locally."""
import concurrent.futures
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import sys
import time
from acquire import OUT, STAGE, connect, remote

MANIFEST="""import pathlib, hashlib, json,sys,os
p=pathlib.Path(sys.argv[1]);r=p/'records';final=sys.argv[2]=='final'
out={}
opened=set()
for proc in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  if (proc/'comm').read_text().strip()=='mxwserver':
   opened.update(os.readlink(x) for x in (proc/'fd').iterdir() if x.is_symlink())
 except (FileNotFoundError,ProcessLookupError):pass
for f in sorted(r.rglob('*')):
 if not f.is_file() or str(f) in opened:continue
 if not final:
  if f.suffix!='.h5': continue
  if f.name=='quiet.raw.h5':ok=(f.parent/'site-candidates.json').exists()
  elif f.name=='warmup.raw.h5':ok=(f.parent.parent/'warmup.json').exists()
  else:ok=(f.parent.parent/(f.parent.name+'-evidence.json')).exists()
  if not ok: continue
 rel=f.relative_to(r).as_posix()
 if rel in json.loads(sys.argv[3]): continue
 with f.open('rb') as fh:
  h=hashlib.sha256()
  for chunk in iter(lambda:fh.read(4*1024**2),b''):h.update(chunk)
 out[rel]={'bytes':f.stat().st_size,'sha256':h.hexdigest()}
print(json.dumps(out))"""

def localpath(rel):
    name, *rest = PurePosixPath(rel).parts
    if name.endswith('-site-selection'): name='baseline'
    return OUT.joinpath(name,*rest)

def download(item):
    rel,meta=item
    dest=localpath(rel);dest.parent.mkdir(parents=True,exist_ok=True)
    with connect() as c:
        partial=dest.with_suffix(dest.suffix+'.partial')
        for attempt in range(2):
            h=hashlib.sha256()
            with c.get_transport().open_session(window_size=8*1024**2,timeout=15) as ch:
                ch.settimeout(55)
                ch.exec_command(shlex.join(['cat','--',STAGE+'/records/'+rel]));ch.shutdown_write()
                with partial.open('wb') as f:
                    while chunk:=ch.recv(1024**2):
                        f.write(chunk);h.update(chunk)
                ok=ch.recv_exit_status()==0 and partial.stat().st_size==meta['bytes'] and h.hexdigest()==meta['sha256']
            if ok:break
        if not ok:raise ValueError('Checksum mismatch: '+rel)
        partial.replace(dest)
    return rel,meta

def run(mode):
    receipt=OUT/'transfer.json'
    known=json.loads(receipt.read_bytes()) if receipt.exists() else {}
    with connect() as c:
        batch=json.loads(remote(c,MANIFEST,[STAGE,mode,json.dumps(list(known))],timeout=180))
        if mode=='final':
            with c.open_sftp() as s:
                for n in ['pre-session.cfg','sample-route.cfg','session-route.json','baseline.log','route.log','crossover.log']:
                    try:s.get(STAGE+'/'+n,str(OUT/n))
                    except FileNotFoundError:pass
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for rel,meta in pool.map(download,batch.items()):
            known[rel]=meta
            tmp=receipt.with_suffix('.tmp')
            tmp.write_text(json.dumps(known,indent=2),encoding='utf8');tmp.replace(receipt)
    final=mode=='final'
    if final:
        folder=OUT/'cpp-crossover-v1'
        if folder.exists():
            hashes={k.split('/',1)[1]:v for k,v in known.items() if k.startswith('cpp-crossover-v1/') and k.endswith('.h5')}
            (folder/'raw-hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
    print('Verified files',len(known),'raw H5',sum(k.endswith('.h5') for k in known),flush=True)
if __name__=='__main__':
    run(sys.argv[1])
