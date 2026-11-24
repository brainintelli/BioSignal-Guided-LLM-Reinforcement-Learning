import hashlib,sys
from pathlib import Path
import paramiko
r=Path(__file__).resolve().parent;b='/home/researcher/mea-stim-offline-20261117'
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('192.0.2.15',username='researcher',password=sys.stdin.readline().strip(),timeout=10,look_for_keys=False,allow_agent=False)
assert hashlib.sha256(c.get_transport().get_remote_server_key().asbytes()).hexdigest()=='70b27913e62299123c7d8a979589a47faa5ea04a5668671876357f5b68f6a59c'
c.get_transport().set_keepalive(30);s=c.open_sftp();s.put(str(r/'watch.py'),b+'/watch.py');s.close()
_,o,e=c.exec_command('/home/researcher/meadowmind-gpu/m6-venv/bin/python -u '+b+'/watch.py')
for line in iter(o.readline,''):
    print(line.rstrip(),flush=True)
    with (r/'watch_history.jsonl').open('a',encoding='utf-8') as f:f.write(line)
print(e.read().decode(),flush=True);c.close()
