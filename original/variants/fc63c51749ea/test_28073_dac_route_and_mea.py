import json,sys,tempfile,hashlib,os
from pathlib import Path
base=Path(sys.argv[1]).resolve()
sys.path[:0]=[str(base/x) for x in ('','mea_stimulation_v2','mea_online','mea_state2_r2','mea_binary_r2')]
from livepair.protocol import PROFILE
from livepair.device import Device
from types import SimpleNamespace
from copy import deepcopy
b=json.loads(Path(sys.argv[2]).read_text(encoding='utf-8-sig'))
with tempfile.TemporaryDirectory() as temp:
 d=SimpleNamespace(b=b,run_dir=Path(temp),lsb=b['device_readback_reference']['dac_lsb_mv'],sites={s:r['channel'] for s,r in b['device_readback_reference']['readback'].items()},units={s:r['unit'] for s,r in b['device_readback_reference']['readback'].items()},config=SimpleNamespace(sites=b['electrodes']))
 Device.verify_mea_readback(d)
 assert json.loads((Path(temp)/'device-output-readback.json').read_text())['measured_load_peak_mv'] is None
 for k in ('lsb','sites','units'):
  bad=deepcopy(d)
  if k=='lsb': bad.lsb+=.1
  else: getattr(bad,k)['F']+=1
  try:Device.verify_mea_readback(bad)
  except ValueError:continue
  raise AssertionError('Unmatched '+k+' was accepted')
print('Nominal verification passed. Three mismatched registrations rejected. No device I/O.')
