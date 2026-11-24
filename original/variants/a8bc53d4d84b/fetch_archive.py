"""Prefetched, hash-checked task archive download; password only arrives on stdin."""
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'.tools/mea_ssh_deps'))
import paramiko

request=json.loads(sys.stdin.readline())
name=request['name']
if Path(name).name!=name or '/' in name or '\\' in name or not name.endswith('.zip'):
    raise ValueError('Expected task archive basename')
destination=Path(__file__).with_name(name)
if destination.exists():raise FileExistsError(destination)
client=paramiko.SSHClient();client.load_host_keys(str(Path.home()/'.ssh/known_hosts'))
client.connect('192.0.2.16',username='researcher',password=request.pop('password'),look_for_keys=False,allow_agent=False,timeout=12)
client.get_transport().set_keepalive(20)
started=time.monotonic();count=0;reported=0;h=hashlib.sha256()
try:
    with client.open_sftp() as sftp:
        source='/home/researcher/mea-state2-r2-20261118/'+name
        size=sftp.stat(source).st_size
        if size!=request['size']:raise ValueError('Export archive size changed')
        with sftp.open(source,'rb') as incoming,destination.with_suffix('.zip.partial').open('xb') as outgoing:
            incoming.prefetch(file_size=size,max_concurrent_requests=32)
            while count<size:
                data=incoming.read(min(1024*1024,size-count))
                if not data:raise EOFError('Archive truncated')
                outgoing.write(data);h.update(data);count+=len(data)
                if count-reported>=8*1024*1024:
                    reported=count;print(json.dumps({'bytes':count,'total':size}),flush=True)
    if h.hexdigest()!=request['sha256']:raise ValueError('Archive digest mismatch')
    destination.with_suffix('.zip.partial').rename(destination)
    result={'passed':True,'bytes':count,'sha256':h.hexdigest(),'seconds':time.monotonic()-started}
    destination.with_suffix('.download.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
finally:client.close()
