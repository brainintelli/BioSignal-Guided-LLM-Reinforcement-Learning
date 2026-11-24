"""Qualify the as-yet-unmeasured part of the same full-array shortlist."""
import fcntl
import json
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace
from livepair.device import write_once
from livepair.protocol import binding
from mea_stim_v2.live import _server_snapshot, mx
from mea_stim_v2.recording import H5Recording
from rank_sites import rank
from survey import OLD,route_and_record
from vendor.brain_stimulation import transport

def run(root):
    b=json.loads((root/'baseline-binding.json').read_bytes())
    if (root/'selected-scan.json').exists():return
    first=root/'survey-confirm'
    assert not json.loads((first/'rank.json').read_bytes())['new']
    name=b['session_id']+'-site-selection'
    eligible={r['electrode'] for f in (root/'survey').glob('field-*/screen.json') for r in json.loads(f.read_bytes())['rank']}
    with H5Recording(first/name/'quiet.raw.h5',SimpleNamespace(**b)) as r:
        covered={e for _,e in r.mapping}
    pending=eligible-covered
    summaries=[]
    with (Path.home()/'.local/state/mea_photo_full/device.lock').open('a') as lock,transport.bind_sdk(mx,transport.Endpoint(timeout=60)):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pid,busy=_server_snapshot();assert not busy and mx.send_raw('wellplate_query_version').strip()=='0'
        for u in range(32):mx.send(mx.StimulationUnit(u).power_up(False).connect(False))
        for k in range(3):mx.send(mx.DAC(k,512))
        mx.send(mx.Core().enable_stimulation_power(False));mx.set_gain(512);mx.hpf('300Hz')
        for i in range(1,5):
            if not pending or time.time()-binding(b)>5700-680:break
            sub=root/('survey-confirm-more-%02d'%i);sub.mkdir()
            pool=sorted(pending)
            if len(pool)<160:
                extra=sorted(covered-OLD-eligible)
                if len(extra)<160:extra=sorted(set(range(26400))-eligible-OLD)
                pool+=extra[::max(1,len(extra)//200)][:160-len(pool)]
            write_once(sub/'screen-electrodes.json',pool)
            file=route_and_record(sub/name,pool,65,60,pid)
            result=rank(file,b)
            with H5Recording(file,SimpleNamespace(**b)) as r:mapped={e for _,e in r.mapping}
            result['new']=[r for r in result['new'] if r['electrode'] in pending]
            write_once(sub/'rank.json',result)
            added=len(pending&mapped);pending-=mapped;covered|=mapped
            summaries.append(dict(scan=sub.name,newly_confirmed=added,remaining=len(pending),new=len(result['new'])))
            print(summaries[-1],flush=True)
            if result['new']:
                write_once(root/'selected-scan.json',dict(scan=sub.name,file=str(file),target=result['new'][0]));break
            if not added:break
    write_once(root/'confirmation-coverage.json',dict(shortlist=len(eligible),unique_confirmed=len(eligible-pending),unmapped=sorted(pending),additional=summaries))
if __name__=='__main__':
    def timeout(*_):raise TimeoutError('Original acquisition time limit')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(860)
    run(Path(sys.argv[1]))
