"""Copy only the two existing runtime/model archives between authorized hosts."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'.tools/mea_ssh_deps'))
import paramiko


class Pinned23(paramiko.MissingHostKeyPolicy):
    def missing_host_key(self,client,hostname,key):
        actual=base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('=')
        if hostname!='192.0.2.23' or actual!='h3XRGbb5gpkMNOdZh0Lka8FIZfckGn772zPUmqpLH4E':
            raise ValueError('Second host identity changed')
        client.get_host_keys().add(hostname,key.get_name(),key)


auth=json.loads(sys.stdin.readline())
clients=[]
try:
    for host,key in [('192.0.2.16','first'),('192.0.2.23','second')]:
        c=paramiko.SSHClient();c.load_host_keys(str(Path.home()/'.ssh/known_hosts'))
        c.set_missing_host_key_policy(Pinned23())
        c.connect(host,username='researcher',password=auth.pop(key),timeout=12,look_for_keys=False,allow_agent=False)
        c.get_transport().set_keepalive(20);clients.append(c)
    del auth
    receipt=[]
    with clients[0].open_sftp() as src,clients[1].open_sftp() as dst:
        for source,name in [('/home/researcher/mea-r2-22554/runtime_20261116_01/runtime.tar','runtime.tar'),
                            ('/home/researcher/mea-r2-22554/import_20261116_01/model_bundle.tar','model_bundle.tar')]:
            target='/home/researcher/mea-state2-r2-20261118/bootstrap/'+name
            size=src.stat(source).st_size;copied=0;reported=0;h=hashlib.sha256();started=time.monotonic()
            with src.open(source,'rb') as incoming,dst.open(target+'.partial','wx') as outgoing:
                incoming.prefetch(file_size=size,max_concurrent_requests=32)
                outgoing.set_pipelined(True)
                while copied<size:
                    chunk=incoming.read(min(1024*1024,size-copied))
                    if not chunk:raise EOFError('Source truncated')
                    outgoing.write(chunk);h.update(chunk);copied+=len(chunk)
                    if copied-reported>=512*1024*1024:
                        reported=copied;print(json.dumps(dict(file=name,copied=copied,total=size)),flush=True)
            dst.rename(target+'.partial',target)
            receipt.append(dict(file=name,source=source,target=target,bytes=copied,sha256=h.hexdigest(),seconds=time.monotonic()-started))
            print(json.dumps(receipt[-1]),flush=True)
    path=Path(__file__).with_name('transfer_result.json')
    with path.open('x',encoding='utf-8') as f:json.dump(dict(passed=True,files=receipt),f,indent=2)
finally:
    for c in clients:c.close()
