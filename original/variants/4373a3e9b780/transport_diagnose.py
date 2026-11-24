import hashlib,json,sys
from pathlib import Path
import paramiko
r=Path(__file__).resolve().parent;c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.0.2.15',username='researcher',password=sys.stdin.readline().strip(),timeout=10,look_for_keys=False,allow_agent=False)
assert hashlib.sha256(c.get_transport().get_remote_server_key().asbytes()).hexdigest()=='70b27913e62299123c7d8a979589a47faa5ea04a5668671876357f5b68f6a59c'
code='''
import subprocess,json
from pathlib import Path
p=Path('/home/researcher/mea-stim-offline-20261117/campaign_delivery.zip')
ps=subprocess.check_output(['ps','-eo','pid,ppid,stat,pcpu,wchan:30,args'],text=True)
print(json.dumps(dict(archive_bytes=p.stat().st_size,processes=[x for x in ps.splitlines() if 'campaign_delivery.zip' in x or 'seal_delivery.py' in x],network=subprocess.check_output(['ss','-tin','sport','=',':22'],text=True)),indent=2))
'''
i,o,e=c.exec_command('/usr/bin/python3 -',timeout=15);i.write(code);i.channel.shutdown_write();out=o.read().decode();err=e.read().decode()
(r/'transport_diagnosis.json').write_text(out,encoding='utf-8');print(out);print(err);c.close()
