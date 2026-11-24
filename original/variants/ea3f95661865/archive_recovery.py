from pathlib import Path
import concurrent.futures
import json
import sys
import time
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent.parent))
from acquire import connect,remote,STAGE
from archive import download

QUERY="""import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);known=json.loads(sys.argv[2]);control=sys.argv[3];files={}
for f in p.glob('*/*.h5'):
 rel='frequency-retest/recovery/'+f.relative_to(p).as_posix()
 if rel not in known and (f.parent/'qc.json').is_file():
  h=hashlib.sha256()
  with f.open('rb') as src:
   for block in iter(lambda:src.read(8*1024**2),b''):h.update(block)
  files[rel]=dict(bytes=f.stat().st_size,sha256=h.hexdigest())
small={f.relative_to(p).as_posix():f.read_text() for f in p.rglob('*') if f.is_file() and f.suffix in ('.json','.log') and f.name not in ('verified.json','原始信号复核.json')}
print(json.dumps([files,small,(p/control).exists()]))"""

def run(control):
    man=OUT/'verified.json';known=json.loads(man.read_bytes()) if man.exists() else {}
    for _ in range(150):
        with connect() as c:files,small,done=json.loads(remote(c,QUERY,[STAGE+'/records/frequency-retest/recovery',json.dumps(list(known)),control],timeout=90))
        for key,data in small.items():
            f=OUT/key;f.parent.mkdir(exist_ok=True,parents=True);f.write_text(data,encoding='utf8')
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            for key,meta in pool.map(download,files.items()):
                known[key]=meta;print(key,meta['bytes'],flush=True)
        man.write_text(json.dumps(known,indent=2),encoding='utf8')
        if done:break
        time.sleep(10)
if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else 'end.json')
