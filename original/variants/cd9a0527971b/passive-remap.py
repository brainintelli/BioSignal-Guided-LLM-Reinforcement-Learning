import datetime, fcntl, hashlib, json, os, socket, sys, time
from pathlib import Path
work=Path('/home/researcher/mea-live-pair-prep/run-20261118T070238Z')
first=Path('/data/local/disk1/mea_live_pair/pair28073-20261118T070238Z-site-selection')
bundle=work/'bundle'
sys.path[:0]=[str(bundle/x) for x in ('','mea_stimulation_v2','mea_state2_r2','mea_online','mea_binary_r2')]+['/pub/soft/MaxLab/python/lib/python3.10/site-packages']
os.environ['HDF5_PLUGIN_PATH']='/pub/soft/MaxLab/so'
from quiet_baseline import baseline
from vendor.brain_stimulation import maxlab as mx, transport
from mea_stim_v2.live import _server_snapshot
from livepair.protocol import binding
b=json.loads((work/'binding-routeB.json').read_text())
assert socket.gethostname()==b['device_hostname'] and 0<time.time()-binding(b)<7200-260
assert json.loads((first/'baseline-audit.json').read_text())['chip_id']==b['chip_id']
assert json.loads((first/'prospective-routes.json').read_text())['plans'][0]['sites']['13122']['verified']
route=first/'proposal-sparse.cfg'
assert hashlib.sha256(route.read_bytes()).hexdigest()==b['route_sha256']
with (Path.home()/'.local/state/mea_photo_full/device.lock').open('a') as lease:
 fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert _server_snapshot()[1]==[]
 with transport.bind_sdk(mx,transport.Endpoint(timeout=15)):
  assert mx.send_raw('wellplate_query_version').strip()=='0'
  for u in range(32):mx.send(mx.StimulationUnit(u).power_up(False).connect(False))
  mx.send(mx.DAC(0,512))
  a=mx.Array('recording_only_28073_20261118_2')
  try:
   assert str(a.load_config(str(route))).strip().lower()=='ok'
   for u in range(32):
    v=a.query_amplifier_at_stimulation(u).strip()
    if v not in ('','-1'): a.disconnect_amplifier_from_stimulation(int(v))
   assert all(a.query_amplifier_at_stimulation(u).strip() in ('','-1') for u in range(32))
   mx.activate([0]);a.download([0]);time.sleep(5);mx.offset();time.sleep(5)
   assert _server_snapshot()[1]==[]
  finally:a.close()
 with (work/'routeB-made-passive.json').open('x') as out: json.dump({'nonzero_outputs':0,'all_stimulation_links_disconnected_before_passive_download':True,'time':time.time(),'session':b['session_id'],'baseline_clock':b['started_at']},out)
print(json.dumps(baseline(b)))