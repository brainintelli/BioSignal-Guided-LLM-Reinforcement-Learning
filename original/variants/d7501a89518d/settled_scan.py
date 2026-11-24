"""Compact recording map with gain, filter and offset applied before settling."""
import fcntl
import json
from pathlib import Path
import signal
import socket
import sys
import time
import uuid
from livepair.device import write_once
from livepair.protocol import binding
from mea_stim_v2.live import _server_snapshot, mx
from rank_sites import rank
from vendor.brain_stimulation import transport

def run(root):
    b=json.loads((root/'baseline-binding.json').read_bytes())
    assert socket.gethostname()==b['device_hostname'] and 0<=time.time()-binding(b)<5700-300
    sub=root/'compact';sub.mkdir()
    dest=sub/(b['session_id']+'-site-selection');dest.mkdir()
    center=6257;cx=center%220;cy=center//220
    pool=sorted([220*y+x for x in range(cx-13,cx+14) for y in range(cy-12,cy+13)])
    write_once(sub/'passive-plan.json',dict(center=center,pool=pool,settle_s=75,record_s=65,binding=b))
    lock=Path.home()/'.local/state/mea_photo_full/device.lock'
    with lock.open('a') as f,transport.bind_sdk(mx,transport.Endpoint(timeout=90)):
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pid,opened=_server_snapshot();assert not opened and mx.send_raw('wellplate_query_version').strip()=='0'
        for u in range(32):mx.send(mx.StimulationUnit(u).power_up(False).connect(False))
        mx.send(mx.Core().enable_stimulation_power(False))
        for dac in range(3):mx.send(mx.DAC(dac,512))
        mx.set_gain(512);mx.hpf('300Hz')
        a=mx.Array('compact_'+uuid.uuid4().hex)
        try:
            a.reset();a.clear_selected_electrodes();a.select_electrodes(pool);a.route()
            mx.activate([0]);a.download([0]);time.sleep(5);mx.offset();time.sleep(75)
            assert _server_snapshot()==(pid,[])
            s=mx.Saving();s.open_directory(str(dest));s.set_legacy_format(False)
            s.start_file('quiet')
            try:
                s.group_delete_all();s.group_define(0,'routed',list(range(1024)));s.start_recording([0])
                try:time.sleep(65)
                finally:s.stop_recording()
            finally:
                try:s.stop_file()
                finally:s.group_delete_all()
        finally:a.close()
        assert _server_snapshot()==(pid,[])
    path=dest/'quiet.raw.h5';v=rank(path,b);write_once(sub/'rank.json',v)
    print(v['mapped'],v['active'],v['pass_hist'],v['new'][:5],flush=True)
    if v['new']:write_once(root/'selected-scan.json',dict(scan='compact',file=str(path),target=v['new'][0]))
if __name__=='__main__':
    def timeout(*_):raise TimeoutError('Compact passive scan deadline')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(350)
    run(Path(sys.argv[1]))
