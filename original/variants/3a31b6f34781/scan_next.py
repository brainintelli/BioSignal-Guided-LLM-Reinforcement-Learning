"""Scan two additional prespecified passive arrays, keeping the same sample and clock."""
from pathlib import Path
import fcntl
import json
import sys
import time
import signal
import uuid
from types import SimpleNamespace
from quiet_baseline import baseline
from rank_sites import rank
from livepair.device import write_once
from livepair.protocol import binding
from mea_stim_v2.live import _server_snapshot, mx
from mea_stim_v2.recording import H5Recording
from vendor.brain_stimulation import transport

def explore(root):
    b=json.loads((root/'baseline-binding.json').read_bytes())
    assert b['chip_id']=='27701' and b['record_root']==str(root)
    first=root/(b['session_id']+'-site-selection')/'quiet.raw.h5'
    with H5Recording(first,SimpleNamespace(**b)) as rec:
        assert rec.metadata['chip_id']=='27701' and not rec.events
        old={e for c,e in rec.mapping}
        original=sorted(rec.mapping,key=lambda ce:-int((rec.spike_channels==ce[0]).sum()))
        centers=[e for _,e in original[:20]]+[6257,7158,6026,3621]
    def distance(e):
        return min((e%220-c%220)**2+(e//220-c//220)**2 for c in centers)
    near=sorted(set(range(26400))-old,key=lambda e:(distance(e),e))[:950]
    wide=[220*y+x for y in range(2,120,5) for x in range(3,220,6)]
    for kind,pool in [('near',near),('wide',wide)]:
        sub=root/kind;sub.mkdir()
        sb=dict(b,record_root=str(sub))
        write_once(sub/'baseline-binding.json',sb)
        assert 0<=time.time()-binding(b)<5700-240
        lock=Path.home()/'.local/state/mea_photo_full/device.lock'
        with lock.open('a') as f,transport.bind_sdk(mx,transport.Endpoint(timeout=90)):
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            assert not _server_snapshot()[1] and mx.send_raw('wellplate_query_version').strip()=='0'
            for i in range(32):mx.send(mx.StimulationUnit(i).power_up(False).connect(False))
            mx.send(mx.Core().enable_stimulation_power(False));mx.send(mx.DAC(0,512))
            a=mx.Array('passive_'+uuid.uuid4().hex)
            try:
                a.reset();a.clear_selected_electrodes()
                a.select_electrodes(sorted(set(pool)|{6257}));a.route()
                mx.activate([0]);a.download([0]);time.sleep(5);mx.offset();time.sleep(5)
            finally:a.close()
        try:baseline(sb)
        except ValueError as exc:
            if str(exc) not in ('At least three stable active electrodes required','Too few consistently active channels'):raise
        path=sub/(b['session_id']+'-site-selection')/'quiet.raw.h5'
        result=rank(path,sb);write_once(sub/'rank.json',result)
        print(kind,result['mapped'],len(result['new']),flush=True)
        if result['new']:
            write_once(root/'selected-scan.json',dict(scan=kind,file=str(path),target=result['new'][0]))
            return
if __name__=='__main__':
    def timeout(*_):raise TimeoutError('Passive scan deadline')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(550)
    explore(Path(sys.argv[1]))
