"""Same physical route and acquisition clock throughout. This program creates no pulse sequences."""
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import socket
import sys
import time
from types import SimpleNamespace
import uuid
import numpy as np
from livepair.device import write_once
from livepair.protocol import binding
from mea_stim_v2.config import route_mapping
from mea_stim_v2.live import mx,_server_snapshot
from mea_stim_v2.qc import stability
from mea_stim_v2.recording import H5Recording
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rank_sites import rank
from vendor.brain_stimulation import transport

def read(f):return json.loads(f.read_bytes())

def off():
    cmds=[mx.StimulationUnit(u).power_up(False).connect(False) for u in range(32)]
    cmds += [mx.Core().enable_stimulation_power(False)]+[mx.DAC(u,512) for u in range(3)]
    acks=[mx.send(x).strip().lower() for x in cmds]
    assert all(x=='ok' for x in acks),acks
    return len(acks)

def capture(folder,pid):
    folder.mkdir()
    assert _server_snapshot()==(pid,[])
    s=mx.Saving();s.open_directory(str(folder));s.set_legacy_format(False);s.start_file('quiet')
    try:
        s.group_delete_all();s.group_define(0,'routed',list(range(1024)));s.start_recording([0])
        try:time.sleep(65)
        finally:s.stop_recording()
        time.sleep(2)
    finally:
        try:s.stop_file()
        finally:s.group_delete_all()
    assert _server_snapshot()==(pid,[])
    return folder/'quiet.raw.h5'

def assess(f,b,roi,mapping,gain):
    with H5Recording(f,SimpleNamespace(**b)) as r:
        m=r.metadata
        assert m['chip_id']==b['chip_id']=='27701' and r.duration>=65 and m['gain']==gain and m['hpf']==300 and m['fs']==20000 and m['threshold_value']==5
        assert not r.events and m['frame_gaps']==0 and sorted(r.mapping)==sorted(mapping) and set(roi)<=set(r.channels.tolist())
        group=[[],[],[]];noise=[];rails=[]
        for i in range(0,len(roi),64):
            cs=roi[i:i+64];a=r.sample(cs,3)
            for j,t in enumerate((23,43,57)):
                k=stability(a,r.sample(cs,t));group[j].extend(k['channel_pass'])
                if j==0:noise.extend(k['before_mad_adc']);rails.extend(k['before_saturated'])
        counts=[sum(q) for q in group]
        spikes=set(r.spike_channels.tolist())&set(roi)
        return dict(metadata=m,roi_count=len(roi),spiking_channels=len(spikes),stable_counts=counts,
                    stable_all=sum(all(row) for row in zip(*group)),first_window_rails=sum(rails),
                    first_window_median_noise_adc=float(np.median(noise)),
                    passed=bool(gain==512 and len(spikes)>=50 and all(100*n>=95*len(roi) for n in counts)))

def run(root):
    b=read(root/'baseline-binding.json');t0=binding(b);dest=root/'recovery'
    assert b['chip_id']=='27701' and b['device_hostname']==socket.gethostname() and 0<=time.time()-t0<5700-660
    stage=root.parents[1];old=root.parent/'cpp-crossover-v1'
    cfg=stage/'sample-route.cfg';assert hashlib.sha256(cfg.read_bytes()).hexdigest()==read(stage/'session-route.json')['route_sha256']
    mapping=route_mapping(cfg);roi=read(old/'design.json')['roi']
    assert len(roi)==158 and not (root/'selected-scan.json').exists()
    guard=Path.home()/'.local/state/mea_photo_full/device.lock'
    state=dict(started=time.time(),roi=roi,phases=[])
    with guard.open('a') as lock,transport.bind_sdk(mx,transport.Endpoint(timeout=45)):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pid,opened=_server_snapshot();assert not opened and mx.send_raw('wellplate_query_version').strip()=='0'
        a=None
        try:
            off()
            def install(gain,init=False):
                nonlocal a
                if init:
                    mx.initialize();time.sleep(2);off()
                mx.activate([0]);mx.hpf('300Hz');mx.set_gain(gain)
                if a is not None:a.close()
                a=mx.Array('drift_'+uuid.uuid4().hex);a.load_config(str(cfg))
                for u in range(32):
                    ch=a.query_amplifier_at_stimulation(u).strip()
                    if ch not in ('','-1'):a.disconnect_amplifier_from_stimulation(int(ch))
                a.download([0]);time.sleep(5);mx.offset();time.sleep(30)
                assert all(a.query_amplifier_at_stimulation(u).strip() in ('','-1') for u in range(32))
            def phase(name,gain):
                path=capture(dest/name,pid);qc=assess(path,b,roi,mapping,gain);write_once(path.parent/'qc.json',qc)
                state['phases'].append(name)
                print(name,qc['stable_counts'],qc['spiking_channels'],'gain',qc['metadata']['gain'],flush=True)
                return path,qc
            def confirm(path,qc):
                if not qc['passed']:return False
                first=rank(path,b)
                check,cq=phase('independent-'+path.parent.name,512)
                if not cq['passed']:return False
                res=rank(check,b);old_sites={v['electrode'] for v in first['new']}
                res['new']=[v for v in res['new'] if v['electrode'] in old_sites]
                write_once(check.parent/'rank.json',res)
                if not res['new']:return False
                write_once(root/'selected-scan.json',dict(scan='recovery/'+check.parent.name,file=str(check),target=res['new'][0]))
                return True
            install(512)
            p,q=phase('as-found-512',512)
            if not confirm(p,q):
                install(512,True);p,q=phase('reinitialized-512',512)
                if not confirm(p,q):
                    install(112);phase('diagnostic-112',112)
                    install(512);p,q=phase('restored-512',512)
                    confirm(p,q)
        except BaseException as e:
            import traceback
            state['error']=traceback.format_exc()
        finally:
            if a is not None:a.close()
            mx.set_gain(512)
            state.update(off_acks=off(),closed=_server_snapshot()==(pid,[]),finished=time.time())
            write_once(dest/'end.json',state);print(state.get('error','closed'),flush=True)

if __name__=='__main__':
    def expire(*args):raise TimeoutError('Original sample clock')
    for sig in (signal.SIGTERM,signal.SIGALRM):signal.signal(sig,expire)
    signal.alarm(940)
    run(Path(sys.argv[1]))
