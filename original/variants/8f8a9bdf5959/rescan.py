"""Acquire a passive wide-field rescan under the original sample clock."""
from pathlib import Path
from types import SimpleNamespace
import fcntl
import json
import signal
import socket
import sys
import time
import uuid
from quiet_baseline import baseline
from livepair.device import write_once
from livepair.protocol import binding
from mea_stim_v2.live import _server_snapshot, mx
from mea_stim_v2.recording import H5Recording
from vendor.brain_stimulation import transport

def rescan(stage):
    b=json.loads((stage/'baseline-binding.json').read_bytes())
    parent=Path(b['record_root'])
    folder=parent/'frequency-retest'
    assert socket.gethostname()==b['device_hostname'] and 0<=time.time()-binding(b)<5700-180
    assert json.loads((parent/'cpp-crossover-v1/end.json').read_bytes())['closed']
    with H5Recording(parent/(b['session_id']+'-site-selection')/'quiet.raw.h5',SimpleNamespace(**b)) as rec:
        assert rec.metadata['chip_id']==b['chip_id']=='27701' and not rec.events
        electrodes=sorted(e for _,e in rec.mapping)
    b=dict(b,record_root=str(folder))
    with (Path.home()/'.local/state/mea_photo_full/device.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not _server_snapshot()[1]
        with transport.bind_sdk(mx,transport.Endpoint(timeout=60)):
            assert mx.send_raw('wellplate_query_version').strip()=='0'
            for unit in range(32):mx.send(mx.StimulationUnit(unit).power_up(False).connect(False))
            for dac in range(3):mx.send(mx.DAC(dac,512))
            mx.send(mx.Core().enable_stimulation_power(False))
            array=mx.Array('rescan_'+uuid.uuid4().hex)
            try:
                array.reset();array.clear_selected_electrodes()
                array.select_electrodes(electrodes);array.route()
                mx.activate([0]);array.download([0])
                time.sleep(5);mx.offset();time.sleep(5)
            finally:array.close()
    write_once(folder/'baseline-binding.json',b)
    result=baseline(b)
    print(json.dumps(dict(chip=result['chip'],candidates=len(result['candidates']))),flush=True)

if __name__=='__main__':
    def stop(*args):raise TimeoutError('Passive rescan timed out')
    signal.signal(signal.SIGALRM,stop);signal.alarm(300)
    rescan(Path(sys.argv[1]))
