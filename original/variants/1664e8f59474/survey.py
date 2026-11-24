"""Search on the physical array, then qualify on a fresh 65-second record."""
import fcntl
import json
from pathlib import Path
import signal
import socket
import sys
import time
from types import SimpleNamespace
import uuid
import numpy as np
from livepair.protocol import binding
from livepair.device import write_once
from mea_stim_v2.live import _server_snapshot, mx
from mea_stim_v2.recording import H5Recording
from mea_stim_v2.qc import stability
from rank_sites import rank
from vendor.brain_stimulation import transport

OLD={7158,6026,3621}
def route_and_record(folder,electrodes,seconds,settle,expected_pid):
    folder.mkdir();assert _server_snapshot()==(expected_pid,[])
    a=mx.Array('survey_'+uuid.uuid4().hex)
    try:
        a.reset();a.clear_selected_electrodes();a.select_electrodes(electrodes);a.route()
        mx.activate([0]);a.download([0]);time.sleep(3);mx.offset();time.sleep(settle)
        s=mx.Saving();s.open_directory(str(folder));s.set_legacy_format(False);s.start_file('quiet')
        try:
            s.group_delete_all();s.group_define(0,'routed',list(range(1024)));s.start_recording([0])
            try:time.sleep(seconds)
            finally:s.stop_recording()
        finally:
            try:s.stop_file()
            finally:s.group_delete_all()
    finally:a.close()
    assert _server_snapshot()==(expected_pid,[])
    return folder/'quiet.raw.h5'

def screen(file,b):
    with H5Recording(file,SimpleNamespace(**b)) as r:
        assert r.metadata['chip_id']==b['chip_id'] and r.fs==20000 and not r.events and r.metadata['frame_gaps']==0 and r.duration>=12
        m=dict(r.mapping); channels=sorted(m)
        counts=[np.bincount(r.spike_channels[(r.spike_times>=s)&(r.spike_times<s+4)],minlength=1024) for s in (0,4,8)]
        minimum=np.min(counts,axis=0); total=np.sum(counts,axis=0)
        ranked=[]
        for i in range(0,len(channels),64):
            cs=channels[i:i+64]
            a=r.sample(cs,1); v=r.sample(cs,6);w=r.sample(cs,10)
            checks=[stability(a,x) for x in (v,w)]
            for j,c in enumerate(cs):
                if m[c] in OLD or total[c]<2:continue
                if any(x['before_saturated'][j] or x['after_saturated'][j] for x in checks):continue
                passed=sum(q['channel_pass'][j] for q in checks)
                ranked.append(dict(electrode=m[c],clean_comparisons=passed,min_spikes=int(minimum[c]),total=int(total[c]),
                                   shift=float(max(q['median_shifts_adc'][j] for q in checks))))
        return dict(chip=r.metadata['chip_id'],routed=sorted(m.values()),spikes=len(r.spike_channels),seconds=r.duration,rank=ranked)

def run(root):
    b=json.loads((root/'baseline-binding.json').read_bytes());start=binding(b)
    assert b['chip_id']=='27701' and socket.gethostname()==b['device_hostname']
    base=root/'survey';base.mkdir()
    phase=[(i%6,(i//6+i%6)%5) for i in range(30)]
    assert len(set(phase))==30
    plan=dict(phase=phase,record_seconds=12,grid=[220,120],exclude=sorted(OLD),
              confirmation_seconds=65,processing='Screen only. Independent original three-by-20s and full voltage stability qualify sites.')
    write_once(base/'plan.json',plan)
    rows=[];coverage=set();guard=Path.home()/'.local/state/mea_photo_full/device.lock'
    with guard.open('a') as f, transport.bind_sdk(mx,transport.Endpoint(timeout=60)):
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pid,opened=_server_snapshot();assert not opened and mx.send_raw('wellplate_query_version').strip()=='0'
        for u in range(32):mx.send(mx.StimulationUnit(u).power_up(False).connect(False))
        for k in range(3):mx.send(mx.DAC(k,512))
        mx.send(mx.Core().enable_stimulation_power(False));mx.set_gain(512);mx.hpf('300Hz')
        for i,(dx,dy) in enumerate(phase):
            if time.time()-start>5700-750:break
            pool=[220*y+x for y in range(dy,120,5) for x in range(dx,220,6)]
            folder=base/('field-%02d'%i)
            file=route_and_record(folder,pool,12.25,4,pid)
            stat=screen(file,b);write_once(folder/'screen.json',stat);rows+=stat['rank'];coverage.update(stat['routed'])
            print(i+1,len(coverage),len(rows),flush=True)
        assert 0<=time.time()-start<5700-600
        unique={}
        def key(row):
            return (row['clean_comparisons'],row['min_spikes']>0,min(row['min_spikes'],100),-row['shift'],row['total'],-row['electrode'])
        for row in rows:
            e=row['electrode']
            if e not in unique or key(row)>key(unique[e]):unique[e]=row
        candidates=sorted(unique.values(),key=key,reverse=True)
        write_once(base/'overview.json',dict(layouts=i+1,actual_electrodes=len(coverage),screen_candidates=len(candidates),
                   candidates_two_checks=sum(r['clean_comparisons']==2 for r in candidates),top=candidates[:40]))
        if not candidates:return
        pool=[r['electrode'] for r in candidates[:720]]
        if len(pool)<180:
            extras=[e for e in sorted(coverage) if e not in set(pool)|OLD]
            pool += extras[::max(1,len(extras)//180)][:180-len(pool)]
        sub=root/'survey-confirm';sub.mkdir()
        write_once(sub/'screen-electrodes.json',pool)
        path=route_and_record(sub/(b['session_id']+'-site-selection'),pool,65,60,pid)
        result=rank(path,b);write_once(sub/'rank.json',result)
        print('independent confirmation',result['mapped'],result['active'],len(result['new']),flush=True)
        if result['new']:
            write_once(root/'selected-scan.json',dict(scan='survey-confirm',file=str(path),target=result['new'][0]))
if __name__=='__main__':
    def expire(*_):raise TimeoutError('Original acquisition window')
    root=Path(sys.argv[1]);b=json.loads((root/'baseline-binding.json').read_bytes())
    signal.signal(signal.SIGALRM,expire);signal.alarm(max(1,int(binding(b)+5700-time.time())))
    run(root)
