"""Task-local SSH transport. Credentials arrive on stdin and are never persisted."""
import hashlib
import argparse
import base64
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '.tools' / 'mea_ssh_deps'))
import paramiko

parser = argparse.ArgumentParser()
parser.add_argument('--host', choices=('192.0.2.16','192.0.2.23'), default='192.0.2.16')
host = parser.parse_args().host
client = paramiko.SSHClient()
client.load_host_keys(str(Path.home() / '.ssh' / 'known_hosts'))
if host == '192.0.2.23' and host not in client.get_host_keys():
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print('WAITING_FOR_AUTH', flush=True)
auth = json.loads(sys.stdin.readline())
client.connect(host, username='researcher', password=auth.pop('password'),
               timeout=12, auth_timeout=12, look_for_keys=False, allow_agent=False)
del auth
client.get_transport().set_keepalive(20)
key = client.get_transport().get_remote_server_key()
print(json.dumps({'connected': True, 'host': host,
                  'host_key_sha256': base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('=')}), flush=True)
try:
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request['op'] == 'close':
                break
            if request['op'] == 'python':
                stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -', timeout=request.get('timeout', 60))
                stdin.write(request['code'])
                stdin.channel.shutdown_write()
                result = {'stdout': stdout.read().decode(), 'stderr': stderr.read().decode(),
                          'exit_code': stdout.channel.recv_exit_status()}
            elif request['op'] in ('put', 'get'):
                with client.open_sftp() as sftp:
                    local = Path(request['local'])
                    if request['op'] == 'put':
                        data = local.read_bytes()
                        with sftp.open(request['remote'], 'wx') as stream:
                            stream.write(data)
                    else:
                        with sftp.open(request['remote'], 'rb') as stream:
                            data = stream.read()
                        local.parent.mkdir(parents=True, exist_ok=True)
                        with local.open('xb') as stream:
                            stream.write(data)
                result = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            else:
                raise ValueError('Unknown operation')
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({'error': type(exc).__name__ + ': ' + str(exc)}), flush=True)
finally:
    client.close()
print('CLOSED', flush=True)
