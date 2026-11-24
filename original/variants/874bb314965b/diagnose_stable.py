import hashlib,json,sys
from pathlib import Path
import paramiko
root=Path(__file__).resolve().parent;base='/home/researcher/mea_full_28080_20261117_1406'
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.0.2.101',username='researcher',password=sys.stdin.readline().strip(),timeout=10,look_for_keys=False,allow_agent=False)
assert hashlib.sha256(c.get_transport().get_remote_server_key().asbytes()).hexdigest()=='286d75e321b511dff43cc6c35404d56cabc6fc6110b13816cc848ecfd6c5cddb'
s=c.open_sftp();s.put(str(root/'diagnose_stable_remote.py'),base+'/diagnose_stable_remote.py')
_,o,e=c.exec_command('HDF5_PLUGIN_PATH=/pub/soft/MaxLab/so /home/researcher/chenyuzhe/2026-9-8/BrainAI/.venv/bin/python '+base+'/diagnose_stable_remote.py',timeout=60)
answer=o.read().decode()+e.read().decode();code=o.channel.recv_exit_status();print(answer)
(root/'stable_failure_diagnosis.log').write_text(answer)
s.get(base+'/multi_bounded_resume2/zero_failure_diagnosis.json',str(root/'multi_bounded_resume2/zero_failure_diagnosis.json'))
s.close();c.close();raise SystemExit(code)
