"""Run the registered three-condition assay on the newly bound acquisition host."""
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import shlex
import sys
import zipfile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
LIVE = ROOT / 'standalone/mea_live_pair'
helper = runpy.run_path(str(Path(os.environ['TEMP']) / 'codex_live22554_1119.py'))
connect, remote = helper['connect'], helper['execute']
STAGE = '/data/local/disk1/mea_live_pair/predict27701-20261119T072625Z'
CODE = STAGE + '/code'
PYTHON = '/usr/bin/python3'
ENV = ['env', 'HDF5_PLUGIN_PATH=/pub/soft/MaxLab/so', 'PYTHONUNBUFFERED=1', 'OPENBLAS_NUM_THREADS=1',
       'PYTHONPATH=' + ':'.join([CODE] + [CODE + '/' + x for x in
          ('mea_stimulation_v2', 'mea_online', 'mea_state2_r2', 'mea_binary_r2')] +
          ['/pub/soft/MaxLab/python/lib/python3.10/site-packages'])]

def stage(c):
    request = json.loads((OUT/'session-request.json').read_bytes())
    host = json.loads((OUT/'host-preflight.json').read_bytes())
    assert request['host']==helper['IP'] and host['host']=='researcher-biolab-1' and not host['servers'][0]['open_h5']
    b=dict(acquisition_host=request['host'], chip_id=request['chip_expected'], organoid_id=request['organoid_id'],
           session_id=STAGE.split('/')[-1], started_at=request['started_at'], hard_stop=request['hard_stop'],
           stop_new_at='2026-11-19T17:01:25+08:00', device_model='MaxOne', replicate=0,
           record_root=STAGE+'/records',device_hostname=host['host'],threshold_h5_dataset='settings/spike_threshold',
           threshold_value=5,bad_channels=[])
    package=io.BytesIO()
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(LIVE.glob('*.py'))+sorted((LIVE/'livepair').glob('*.py')):
            z.write(f,'code/'+f.relative_to(LIVE).as_posix())
        for name,pkgs in [('mea_stimulation_v2',['mea_stim_v2','vendor']),('mea_online',['mea_online']),
                ('mea_binary_r2',['mea_binary_r2']),('mea_state2_r2',['mea_state2_r2'])]:
            for pkg in pkgs:
                for f in sorted((LIVE.parent/name/pkg).rglob('*.py')):
                    z.write(f,'code/'+f.relative_to(LIVE.parent).as_posix())
        z.writestr('baseline-binding.json',json.dumps(b,indent=2))
    data=package.getvalue()
    remote(c,"from pathlib import Path;import sys;p=Path(sys.argv[1]);p.mkdir();(p/'records').mkdir()",[STAGE])
    with c.open_sftp() as s:
        s.putfo(io.BytesIO(data),STAGE+'/package.zip')
    remote(c,"import sys,hashlib,zipfile;from pathlib import Path;p=Path(sys.argv[1]);f=p/'package.zip';assert hashlib.sha256(f.read_bytes()).hexdigest()==sys.argv[2];zipfile.ZipFile(f).extractall(p)",[STAGE,hashlib.sha256(data).hexdigest()])
    (OUT/'baseline-binding.json').write_text(json.dumps(b,indent=2),encoding='utf8')

def execute(c, argv, timeout):
    command=shlex.join(ENV+[PYTHON]+argv)
    _,out,err=c.exec_command(command,timeout=timeout)
    result=out.read().decode(errors='replace')
    error=err.read().decode(errors='replace')
    if out.channel.recv_exit_status():
        raise RuntimeError(error[-2000:]+result[-300:])
    return result

def launch(c,step):
    source={'baseline':'quiet_baseline.py','route':'current_chip_route.py','crossover':'current_chip_crossover.py'}[step]
    argv=ENV+[PYTHON,CODE+'/'+source]+([] if step=='baseline' else [STAGE])
    code="""from pathlib import Path
import subprocess,sys,json
p=Path(sys.argv[1]); s=sys.argv[2]
with (p/(s+'.log')).open('xb') as log, (p/'baseline-binding.json').open('rb') as b:
 pr=subprocess.Popen(json.loads(sys.argv[3]),cwd=p,stdin=b if s=='baseline' else subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
(p/(s+'.pid')).write_text(str(pr.pid));print(pr.pid)"""
    print(step,remote(c,code,[STAGE,step,json.dumps(argv)]))

PROBE="""from vendor.brain_stimulation import maxlab as mx,transport
from mea_stim_v2.live import _server_snapshot
import os,socket,time,json,fcntl,pathlib,shutil
from current_chip_crossover import run
p=pathlib.Path.cwd()
pid,opened=_server_snapshot(); assert not opened
lock=pathlib.Path.home()/'.local/state/mea_photo_full/device.lock'
with lock.open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 with transport.bind_sdk(mx,transport.Endpoint(timeout=4)):
  version=mx.send_raw('wellplate_query_version').strip()
  lsb=float(mx.query_DAC_lsb_mV())
  cfg=mx.send_raw('mea_array_command online\\nmea_array_get_config').strip()
  assert version=='0' and 0<lsb<4
  (pathlib.Path(STAGE)/'pre-session.cfg').write_text(cfg)
with socket.create_connection(('127.0.0.1',5555),timeout=4) as client:
 client.settimeout(6);raw=b''
 until=time.monotonic()+8
 while raw.count(b'\\n')<4 and time.monotonic()<until:raw+=client.recv(4096)
 rows=[r.split(b',') for r in raw.strip().splitlines() if r.count(b',')==2]
 assert rows
 stamp=int(rows[-1][0]); age=abs(time.time()-stamp/1000); assert age<2
print(json.dumps(dict(hostname=socket.gethostname(),server_pid=pid,open_h5=opened,maxone_version=version,dac_lsb_mv=lsb,cpp_probe_rows=len(rows),cpp_age_seconds=age,free_gib=shutil.disk_usage(STAGE).free/2**30)))"""

STATUS="""from pathlib import Path
import json,sys
p=Path(sys.argv[1]);f=p/'records/cpp-crossover-v1';result={}
for label in ['baseline','route','crossover']:
 n=p/(label+'.pid')
 if n.exists():
  stat=Path('/proc')/n.read_text()/'stat'; result[label]=stat.read_text().split()[2] if stat.exists() else 'exited'
  result[label+'_tail']=(p/(label+'.log')).read_text(errors='replace')[-500:]
q=next((p/'records').glob('*selection'),None)
result['selection']=bool(q and (q/'site-candidates.json').is_file())
result['trial_evidence']=len(list(f.glob('trial_*-evidence.json')))
result['sent']=len(list(f.glob('trial_*-sent.json')))
result['nonzero_attempts']=len(list(f.glob('attempt-*.json')))
if (f/'end.json').exists():result['end']=json.loads((f/'end.json').read_text())
print(json.dumps(result))"""

if __name__=='__main__':
    step=sys.argv[1]
    with connect() as c:
        if step=='stage':stage(c)
        elif step=='probe':
            obj=json.loads(execute(c,['-c','STAGE='+repr(STAGE)+'\n'+PROBE],30))
            (OUT/'instrument-preflight.json').write_text(json.dumps(obj,indent=2),encoding='utf8');print(obj)
        elif step=='status': print(remote(c,STATUS,[STAGE]))
        else: launch(c,step)
