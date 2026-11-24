"""Task-local SSH transport; credentials stay in process memory, never in files."""
import getpass, hashlib, json, shlex, sys
from pathlib import Path
import paramiko

host = sys.argv[1]
client = paramiko.SSHClient()
client.load_host_keys(str(Path.home()/'.ssh/known_hosts'))
client.set_missing_host_key_policy(paramiko.RejectPolicy())
client.connect(host, username='researcher', password=getpass.getpass('SSH password: '), timeout=15,
               look_for_keys=False, allow_agent=False)
print(json.dumps({'connected':host,'host_key_sha256':hashlib.sha256(client.get_transport().get_remote_server_key().asbytes()).hexdigest()}),flush=True)
sftp = client.open_sftp()
for line in sys.stdin:
    try:
        request=json.loads(line)
        if request['op']=='close':break
        if request['op']=='exec':
            _,out,err=client.exec_command(request['command'],timeout=request.get('timeout',60))
            stdout=out.read().decode('utf-8',errors='replace');stderr=err.read().decode('utf-8',errors='replace')
            result={'exit_code':out.channel.recv_exit_status(),'stdout':stdout,'stderr':stderr}
            if request.get('save'):Path(request['save']).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        elif request['op']=='put':
            sftp.put(request['local'],request['remote']);result={'uploaded':request['remote']}
        elif request['op']=='get':
            sftp.get(request['remote'],request['local']);result={'downloaded':request['local']}
        else:raise ValueError('Unknown operation')
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except Exception as exc:print(json.dumps({'error':repr(exc)}),flush=True)
sftp.close();client.close()
