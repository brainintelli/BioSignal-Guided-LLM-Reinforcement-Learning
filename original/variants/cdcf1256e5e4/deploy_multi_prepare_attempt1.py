"""Deploy and run strictly software-only preparation; never runs stimulation."""
import hashlib,json,sys
from pathlib import Path
import paramiko
root=Path(__file__).resolve().parent;src=root.parents[1]/'tools/mea_multi_recording'
remote='/home/researcher/mea_full_28080_20261117_1406'
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.0.2.101',username='researcher',password=sys.stdin.readline().strip(),timeout=10,look_for_keys=False,allow_agent=False)
assert hashlib.sha256(c.get_transport().get_remote_server_key().asbytes()).hexdigest()=='286d75e321b511dff43cc6c35404d56cabc6fc6110b13816cc848ecfd6c5cddb'
s=c.open_sftp();s.mkdir(remote+'/multi_code')
for name in ['multi_protocol.py','multi_prepare.py']:s.put(str(src/name),remote+'/multi_code/'+name)
s.put(str(root/'multipoint_candidates.json'),remote+'/multipoint_candidates.json')
command='/home/researcher/chenyuzhe/2026-9-8/BrainAI/.venv/bin/python '+remote+'/multi_code/multi_prepare.py '+remote
_,stdout,stderr=c.exec_command(command,timeout=150)
text=stdout.read().decode();err=stderr.read().decode();code=stdout.channel.recv_exit_status()
(root/'multi_prepare_run.log').write_text(text+err)
for name in ['preparation.json','progress.json','failure_cleanup.json']:
    try:s.get(remote+'/multi_software_prepare/'+name,str(root/('multi_'+name)))
    except FileNotFoundError:pass
print(json.dumps(dict(exit_code=code,stdout=text,stderr=err),indent=2));s.close();c.close()
raise SystemExit(code)
