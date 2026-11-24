import hashlib,json,sys
from pathlib import Path
import paramiko
root=Path(__file__).resolve().parent
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.0.2.101',username='researcher',password=sys.stdin.readline().strip(),timeout=10,look_for_keys=False,allow_agent=False)
assert hashlib.sha256(c.get_transport().get_remote_server_key().asbytes()).hexdigest()=='286d75e321b511dff43cc6c35404d56cabc6fc6110b13816cc848ecfd6c5cddb'
s=c.open_sftp();result={}
for path in ['/pub/soft/MaxLab/bin/mxwserver.sh','/tmp/MaxLab/researcher/mxwserver/20261116.log','/proc/3694171/cmdline']:
    try:
        with s.open(path) as f:
            if path.endswith('.log'):f.seek(max(0,s.stat(path).st_size-16000))
            data=f.read()
        target=root/'bounded_failure'/Path(path).name;target.write_bytes(data)
        result[path]=data.decode(errors='replace').replace('\u0000',' ')
    except FileNotFoundError:result[path]='absent'
print(json.dumps(result,indent=2));s.close();c.close()
