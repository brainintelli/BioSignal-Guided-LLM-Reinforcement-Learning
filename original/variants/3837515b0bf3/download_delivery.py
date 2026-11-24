import hashlib,json,sys,time
from pathlib import Path
import paramiko
r=Path(__file__).resolve().parent;b='/home/researcher/mea-stim-offline-20261117'
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('192.0.2.15',username='researcher',password=sys.stdin.readline().strip(),timeout=10,look_for_keys=False,allow_agent=False)
assert hashlib.sha256(c.get_transport().get_remote_server_key().asbytes()).hexdigest()=='70b27913e62299123c7d8a979589a47faa5ea04a5668671876357f5b68f6a59c'
s=c.open_sftp();s.get(b+'/archive_receipt.json',str(r/'archive_receipt.json'));q=json.loads((r/'archive_receipt.json').read_text())
part=r/'campaign_delivery.zip.part';dest=r/'campaign_delivery.zip'
if dest.exists():raise FileExistsError('Do not overwrite delivery')
channel=c.get_transport().open_session(window_size=64*1024*1024,max_packet_size=256*1024);channel.settimeout(45)
channel.exec_command('cat -- '+b+'/campaign_delivery.zip');h=hashlib.sha256();count=0;start=time.monotonic()
with part.open('xb') as f:
    while True:
        data=channel.recv(1024*1024)
        if not data:break
        f.write(data);h.update(data);count+=len(data)
if channel.recv_exit_status()!=0 or count!=q['bytes'] or h.hexdigest()!=q['archive_sha256']:raise RuntimeError('Incomplete or mismatched delivery')
part.replace(dest);result=dict(passed=True,bytes=count,sha256=h.hexdigest(),seconds=time.monotonic()-start)
(r/'download_verified.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));channel.close();s.close();c.close()
