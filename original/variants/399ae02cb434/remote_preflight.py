"""Read-only host, filesystem and known MaxLab query checks."""
import datetime,hashlib,json,os,re,socket,subprocess,time
from pathlib import Path
import numpy,h5py

BASE=Path('/home/researcher/mea_full_28080_20261117_1406')
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'epoch':time.time(),
        'numpy':numpy.__version__,'h5py':h5py.__version__,'read_only_hardware':True}
opened=[];servers=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        if (p/'comm').read_text().strip()!='mxwserver':continue
        servers.append(int(p.name))
        for fd in (p/'fd').iterdir():
            try:target=os.readlink(fd)
            except FileNotFoundError:continue
            if '.h5' in target:opened.append(target)
    except FileNotFoundError:continue
report['servers']=servers;report['open_h5']=opened
if len(servers)!=1 or opened:raise RuntimeError('Device busy or unknown server inventory')
def query(command):
    with socket.create_connection(('127.0.0.1',7215),timeout=3) as s:
        s.sendall(command.encode('latin1')+b'\r\n###');data=b''
        while b'\r\n###' not in data:
            block=s.recv(65536)
            if not block:raise RuntimeError('API disconnected')
            data+=block
        return data.split(b'\r\n###',1)[0].decode('latin1')
report['maxone_version']=query('wellplate_query_version')
report['dac_lsb_mv']=query('system_query_dac_lsb_mv')
online=query('mea_array_command online\nmea_array_get_config')
(BASE/'online_readonly.cfg').write_text(online)
def mapping(text):return sorted((int(a),int(b)) for a,b in re.findall(r'(\d+)\((\d+)\)',text.split()[0]))
current=mapping(online);report['online_channels']=len(current)
routes=[]
for p in sorted(Path('/home/researcher/configs/261117').glob('*.cfg')):
    data=p.read_text();m=mapping(data)
    routes.append({'path':str(p),'mtime':p.stat().st_mtime,'channels':len(m),
                   'matches_online':m==current,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
report['routes']=routes
# Decompression compatibility uses an already closed historical file, not the new specimen.
os.environ.setdefault('HDF5_PLUGIN_PATH','/pub/soft/MaxLab/so')
old=Path('/home/researcher/mea_photo_22021_20261117/results/20261117-123517-5062b55d/h5/stimulation.raw.h5')
with h5py.File(old,'r') as f:
    report['plugin_old_file_slice']=list(f['data_store/data0000/groups/routed/raw'][0,:10].astype(int))
report['plugin_old_file_slice']=[int(x) for x in report['plugin_old_file_slice']]
(BASE/'read_only_preflight.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
