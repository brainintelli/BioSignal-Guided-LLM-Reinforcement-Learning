import hashlib,json,sys
from pathlib import Path
import paramiko
root=Path(__file__).resolve().parent;base='/home/researcher/mea_full_28080_20261117_1406'
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.0.2.101',username='researcher',password=sys.stdin.readline().strip(),timeout=10,look_for_keys=False,allow_agent=False)
assert hashlib.sha256(c.get_transport().get_remote_server_key().asbytes()).hexdigest()=='286d75e321b511dff43cc6c35404d56cabc6fc6110b13816cc848ecfd6c5cddb'
s=c.open_sftp();s.put(str(root/'restore_service_remote.py'),base+'/restore_service_remote.py')
_,o,e=c.exec_command('/home/researcher/chenyuzhe/2026-9-8/BrainAI/.venv/bin/python '+base+'/restore_service_remote.py',timeout=90)
answer=o.read().decode()+e.read().decode();code=o.channel.recv_exit_status();(root/'service_restore.log').write_text(answer)
for name in ['service_restore.json','restored_server.log']:
    try:s.get(base+'/'+name,str(root/name))
    except FileNotFoundError:pass
print(json.dumps({'exit':code,'output':answer},indent=2));s.close();c.close();raise SystemExit(code)
