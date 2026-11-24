"""Stream a closed H5 over SSH, verify source SHA256 and retain a receipt."""
import argparse,hashlib,json,shlex,sys,time
from pathlib import Path
import paramiko
ROOT=Path(__file__).resolve().parent;REMOTE='/home/researcher/mea_full_28080_20261117_1406'
p=argparse.ArgumentParser();p.add_argument('folder',choices=['recording','recording_resume1','recording_resume2','multi_bounded_recording','multi_bounded_resume1','multi_bounded_resume2'])
p.add_argument('name',choices=['identity.raw.h5','baseline.raw.h5','stimulus_post.raw.h5']);a=p.parse_args()
password=sys.stdin.readline().strip()
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.0.2.101',username='researcher',password=password,timeout=10,look_for_keys=False,allow_agent=False)
assert hashlib.sha256(c.get_transport().get_remote_server_key().asbytes()).hexdigest()=='286d75e321b511dff43cc6c35404d56cabc6fc6110b13816cc848ecfd6c5cddb'
path=REMOTE+'/'+a.folder+'/'+a.name;s=c.open_sftp();source=s.stat(path)
servers=[]
for entry in s.listdir('/proc'):
    if not entry.isdigit():continue
    try:
        with s.open('/proc/'+entry+'/comm') as f:name=f.read().decode().strip()
        if name=='mxwserver':servers.append(entry)
    except FileNotFoundError:continue
if len(servers)!=1:raise RuntimeError('Need one inspectable current server')
for fd in s.listdir('/proc/'+servers[0]+'/fd'):
    try:target=s.readlink('/proc/'+servers[0]+'/fd/'+fd)
    except FileNotFoundError:continue
    if target==path:raise RuntimeError('Refusing an open H5')
out=ROOT/'backup'/a.folder;out.mkdir(parents=True,exist_ok=True);dest=out/a.name
if dest.exists():raise FileExistsError(str(dest))
part=dest.with_suffix('.h5.part');start=time.monotonic();h=hashlib.sha256();count=0
channel=c.get_transport().open_session(window_size=8*1024*1024,max_packet_size=256*1024)
channel.settimeout(30);channel.exec_command('cat -- '+shlex.quote(path))
with part.open('xb') as f:
    while True:
        block=channel.recv(1024*1024)
        if not block:break
        f.write(block);h.update(block);count+=len(block)
code=channel.recv_exit_status()
if code!=0 or count!=source.st_size:raise RuntimeError('Incomplete copy')
_,stdout,stderr=c.exec_command('sha256sum -- '+shlex.quote(path),timeout=120)
expected=stdout.read().decode().split()[0];actual=h.hexdigest()
if actual!=expected or s.stat(path).st_size!=count:raise RuntimeError('SHA256 or size mismatch')
part.replace(dest);receipt=dict(source=path,local=str(dest),bytes=count,sha256=actual,source_verified=True,
                             seconds=time.monotonic()-start,mb_per_second=count/1e6/(time.monotonic()-start))
(out/(a.name+'.verified.json')).write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
channel.close();s.close();c.close()
