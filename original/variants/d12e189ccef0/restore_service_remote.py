"""Restore the crashed original service, then acknowledge all outputs off. No stimuli."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
BASE=Path('/home/researcher/mea_full_28080_20261117_1406')
def pids(name):
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if (p/'comm').read_text().strip()==name:found.append(int(p.name))
        except FileNotFoundError:pass
    return found
if pids('mxwserver'):raise RuntimeError('A live MaxLab server already exists; refusing duplicate launch')
stopped=[]
for pid in pids('crashsender'):
    cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes()
    if b'MaxLab' not in cmd:continue
    os.kill(pid,signal.SIGTERM);stopped.append(pid)
env=dict(os.environ)
try:
    data=Path('/proc/1551317/environ').read_bytes().split(b'\0')
    for entry in data:
        if b'=' not in entry:continue
        k,v=entry.split(b'=',1)
        if k in (b'DISPLAY',b'XAUTHORITY',b'DBUS_SESSION_BUS_ADDRESS'):env[k.decode()]=v.decode()
except FileNotFoundError:pass
with (BASE/'restored_server.log').open('x') as log:
    p=subprocess.Popen(['/bin/bash','/pub/soft/MaxLab/bin/mxwserver.sh'],cwd='/pub/soft/MaxLab/bin',stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
time.sleep(3)
servers=pids('mxwserver')
if len(servers)!=1:raise RuntimeError('Expected one restarted server, got '+repr(servers))
sys.path.insert(0,str(BASE/'corrected/mea_next_recording'))
from device import MaxLabBackend
from brain_stimulation import maxlab as mx
out=BASE/'service_restore_off';out.mkdir(exist_ok=False)
d=MaxLabBackend(json.loads((BASE/'binding.json').read_text()),out);d.cleaning=True
with d.sdk_context():
    version=d._raw('wellplate_query_version');d.zero()
result=dict(status='passed',server_pid=servers[0],wrapper_pid=p.pid,stopped_crash_reporters=stopped,
    maxone_version=version,all32_off_and_dac512_acknowledged=True,sequence_sends=0,wall=time.time())
(BASE/'service_restore.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
