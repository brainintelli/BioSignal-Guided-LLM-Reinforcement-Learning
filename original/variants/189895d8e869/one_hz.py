"""Matched zero-output and 1 Hz records on one newly selected electrode."""
import fcntl
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import uuid
import numpy as np
from device_stream import Receiver
from livepair.device import write_once
from livepair.protocol import binding, PROFILE
from mea_binary_r2.encoding import pulse_shape
from mea_stim_v2.config import route_mapping
from mea_stim_v2.live import MaxLabBackend, _server_snapshot, mx
from mea_stim_v2.qc import stability
from mea_stim_v2.recording import H5Recording
from vendor.brain_stimulation import transport

def load(p):return json.loads(p.read_bytes())
def rail(x):return (x==0)|(x==1023)

class Config:
    def __init__(self,b,route,target):
        self.b=b
        self.stim_electrode=target
        self.route_file=str(route)
        self.sha=hashlib.sha256(route.read_bytes()).hexdigest()
        self.device_hostname=b['device_hostname']
        self.threshold_h5_dataset=b['threshold_h5_dataset']
        self.session_id=b['session_id']
    def validate_binding(self):
        p=Path(self.route_file)
        assert hashlib.sha256(p.read_bytes()).hexdigest()==self.sha
        pairs=route_mapping(p)
        assert self.stim_electrode in dict(pairs).values()
        return pairs
    def validate_live(self):
        assert binding(self.b)>0 and self.b['chip_id']=='27701' and self.b['device_model']=='MaxOne'
        assert self.device_hostname==socket.gethostname() and self.b['acquisition_host'] in subprocess.check_output(['hostname','-I'],text=True).split()

def make_route(scan,cfg,source,target,watch):
    with H5Recording(source,SimpleNamespace(**scan)) as r:
        assert r.metadata['chip_id']==scan['chip_id'] and not r.events
        pairs=dict(r.mapping)
        active=sorted(r.channels.tolist(),key=lambda c:-np.count_nonzero(r.spike_channels==c))
        selected=set(pairs[c] for c in active[:220])|set(watch)|{target}
    with transport.bind_sdk(mx,transport.Endpoint(timeout=90)):
        a=mx.Array('single_'+uuid.uuid4().hex)
        try:
            a.reset();a.clear_selected_electrodes()
            a.select_electrodes(sorted(selected));a.select_stimulation_electrodes([target]);a.route()
            for u in range(32):
                ch=a.query_amplifier_at_stimulation(u).strip()
                if ch not in ('','-1'):a.disconnect_amplifier_from_stimulation(int(ch))
            a.connect_electrode_to_stimulation(target)
            unit=int(a.query_stimulation_at_electrode(target))
            assert 0<=unit<32
            channel=int(a.query_amplifier_at_stimulation(unit))
            a.disconnect_amplifier_from_stimulation(channel)
            current=mx.send_raw('mea_array_command '+a.token+'\nmea_array_get_config').strip()
            with cfg.open('x') as out:out.write(current)
            mapped=dict((e,c) for c,e in route_mapping(cfg))
            assert mapped[target]==channel and len(mapped)>=51
            return mapped
        finally:a.close()

def sites(scanfile,b,target,available=None):
    with H5Recording(scanfile,SimpleNamespace(**b)) as r:
        position=dict((e,r.channel_xy_um[c]) for c,e in r.mapping)
        counts=np.bincount(r.spike_channels,minlength=1024)
        pool=[e for c,e in r.mapping if counts[c]>0 and e not in (target,7158,6026,3621)
              and (available is None or e in available)]
        assert len(pool)>=2
        chosen=[target]
        for _ in range(2):
            e=max(pool,key=lambda p:(min(np.linalg.norm(np.asarray(position[p])-position[q]) for q in chosen),-p))
            chosen.append(e);pool.remove(e)
        return chosen,{str(e):position[e] for e in chosen}

def verify_raw(r,dev):
    m=r.metadata
    assert m['chip_id']==dev.config.b['chip_id'] and m['fs']==20000 and m['gain']==512 and m['hpf']==300
    assert m['frame_gaps']==0 and m['threshold_value']==5 and sorted(r.mapping)==dev.config.validate_binding()

def ensure_single(dev):
    with dev._sdk():
        links={u:ch for u in range(32) if (ch:=dev.array.query_amplifier_at_stimulation(u).strip()) not in ('','-1')}
    assert links=={dev.stim_unit:str(dev.stim_channel)}

def neutral(dev,chs,folder,wire):
    dev.open_record('neutral')
    time.sleep(9)
    dev.marker(801,{'phase':'connect'});dev.connect(True)
    time.sleep(6);dev.marker(802,{'phase':'disconnect'});dev.connect(False)
    time.sleep(6);dev.marker(803,{'phase':'hold'});dev.connect(True)
    time.sleep(35)
    with H5Recording(dev.close_record()['path'],dev.config) as r:
        verify_raw(r,dev)
        assert [e['id'] for e in r.events]==[801,802,803]
        t=r.events[-1]['time_s'];ref=r.sample(chs,t+8)
        observed={}
        for e in r.events:
            x=r.sample(chs,e['time_s'],4)
            observed[e['message']['phase']]=[int(v) for v in rail(x).sum(axis=1)]
        quiet=[int(rail(r.sample(chs,t+i)).sum()) for i in (8,15,25,32)]
        roi=sorted(set(r.spike_channels.tolist())-set(chs))
        assert len(roi)>=50
        late=stability(r.sample(roi,t+8),r.sample(roi,t+32))
        result=dict(chip=r.metadata['chip_id'],mapped=len(r.mapping),roi=roi,transition_rail_counts=observed,
                    settled_rail_counts=quiet,quiet_noise_fraction=late['fraction_channels_passed'],events=r.events)
        write_once(folder/'neutral-check.json',result)
        if any(quiet):raise ValueError('The settled zero-output route still clips on a monitored channel')
        if (rail(r.sample(roi,t+32)).any(axis=1)).mean()>.05:raise ValueError('Settled zero-output ROI saturation')
        wire.fresh()
        return roi,r.last_frame

def record_trial(dev,lsb,wire,row,chs,roi,folder,last):
    i=row['index'];active=row['condition']=='one_hz';stem='trial_%02d'%i
    dev.open_record(stem);time.sleep(9);dev.assert_ready();wire.fresh()
    assert dev.connected and dev.record_path.stat().st_size>65536
    base=4*i
    with dev._sdk():
        assert float(mx.query_DAC_lsb_mV())==lsb
        codes=[]
        if active:
            for k in range(3):
                if k:
                    full,remainder=divmod(20000-2*(PROFILE.phase_us//50),4000)
                    codes.extend([('delay',4000)]*full)
                    if remainder:codes.append(('delay',remainder))
                    codes.append(('marker',mx.Event(0,1,base+k,f'trial {i} idx {k}')))
                codes.extend(pulse_shape(lsb,PROFILE))
        ack=dev._sequence(base,dict(trial=i,idx=0,mode=row['condition']),codes)
    write_once(folder/(stem+'-sent.json'),dict(row=row,ack=ack,cpp=wire.count))
    time.sleep(18.2)
    with H5Recording(dev.close_record()['path'],dev.config) as r:
        verify_raw(r,dev); assert r.first_frame>last
        ev=r.events;anchor=ev[0]['frame']; t=ev[0]['time_s']
        assert [e['id'] for e in ev]==list(range(base,base+(3 if active else 1)))
        assert [e['frame']-anchor for e in ev]==([0,20000,40000] if active else [0])
        assert all(int(e['message']['trial'])==i and int(e['message']['idx'])==k for k,e in enumerate(ev))
        assert r.duration-t>=17 and not rail(r.sample(chs,t-3)).any()
        traces=r.sample(chs,t-.004,2.71)
        rails=rail(traces)
        onset=[];late=[]
        for k in range(3):
            ix=80+20000*k
            onset.append([j for j,v in enumerate(rails) if not v[ix-8:ix].any() and v[ix:ix+40].any()])
            late.append(rails[:,ix+12920:ix+13920].sum(axis=1).tolist())
        recovery=stability(r.sample(roi,t-3),r.sample(roi,t+14))
        later=rail(r.sample(chs,t+14)).sum(axis=1).tolist()
        sham_peak=float(np.max(abs(traces.astype(float)-np.median(r.sample(chs,t-3),axis=1)[:,None]))*r.metadata['lsb_uv'])
        physically_correct=onset==([[0]]*3 if active else [[],[],[]])
        clean=physically_correct and not np.any(late) and not any(later) and (active or sham_peak<1000)
        counts=[int(np.sum(np.isin(r.spike_channels,roi)&(r.spike_times>=t+a)&(r.spike_times<t+z))) for a,z in [(-6,-1),(4,9),(10,15)]]
        write_once(folder/(stem+'-evidence.json'),dict(row=row,events=ev,cs=counts,onsets=onset,late_counts=late,
                   late_14s=later,recovery=recovery['fraction_channels_passed'],peak_uv=sham_peak,clean=bool(clean),
                   state2_admitted=bool(clean and recovery['passed']),chip=r.metadata['chip_id'],frames=r.frame_count))
        if not clean: raise ValueError('Trial onset, zero-output artifact or late recovery gate failed: '+stem)
        print(stem,row['condition'],counts,recovery['fraction_channels_passed'],flush=True)
        return r.last_frame

def run(root):
    b=load(root/'baseline-binding.json'); plan=load(root/'design.json');sel=load(root/'selected-scan.json')
    prior=root.parent/'cpp-crossover-v1';end=load(prior/'end.json')
    assert load(prior/'binding.json')['organoid_id']==plan['sample']==b['organoid_id']
    assert b['started_at']==plan['original_start'] and end['closed'] and not end['cleanup']
    assert len(list(prior.glob('attempt-*.json')))==end['attempts_total']==1
    assert 0<=time.time()-binding(b)<5700-600
    assert shutil.disk_usage(root).free>35_000_000_000
    target=sel['target']['electrode']
    assert target not in plan['excluded_output_electrodes'] and sel['target']==load(root/sel['scan']/'rank.json')['new'][0]
    scan=Path(sel['file']);el,xy=sites(scan,b,target)
    folder=root/'single-site-v1';folder.mkdir()
    orders=['one_hz','zero']; seq=[True,False]*3
    random.Random(277011119).shuffle(seq)
    rows=[dict(block=j+1,index=2*j+i+1,condition=a) for j,normal in enumerate(seq) for i,a in enumerate(orders if normal else orders[::-1])]
    write_once(folder/'schedule.json',rows)
    guard=Path.home()/'.local/state/mea_photo_full/device.lock'
    with guard.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pid,opened=_server_snapshot();assert not opened
        path=folder/'route.cfg';mapped=make_route(b,path,scan,target,el[1:])
        el,xy=sites(scan,b,target,mapped);chs=[mapped[e] for e in el]
        dev=MaxLabBackend(Config(b,path,target),folder,SimpleNamespace(wall=time.time,mono=time.monotonic,sleep=time.sleep))
        wire=Receiver(folder);wire.thread.start()
        state=dict(completed=0,prior_attempts=1,started=time.time())
        def reserve(req):
            wire.fresh()
            used=len(list(prior.glob('attempt-*.json')))+len(list(folder.glob('attempt-*.json')))
            if used>=plan['session_cap_trains'] or time.time()-binding(b)>=5700:raise RuntimeError('Original sample budget reached')
            write_once(folder/('attempt-%02d.json'%(used+1)),dict(request=req,time=time.time(),three_pulses=True))
        dev.before_sequence_send=reserve
        try:
            lsb=dev.preflight()['lsb_mv']
            dev.prepare();ensure_single(dev)
            write_once(folder/'sites.json',dict(electrodes=el,channels=chs,xy=xy,unit=dev.stim_unit,lsb_mv=lsb))
            roi,last=neutral(dev,chs,folder,wire)
            for row in rows:
                if time.time()-binding(b)>5700-35:raise TimeoutError('Original stop-new time')
                last=record_trial(dev,lsb,wire,row,chs,roi,folder,last)
                state['completed']+=1
        except BaseException:
            state['error']=traceback.format_exc()
        finally:
            state['cleanup']=dev.cleanup();wire.stop.set();wire.thread.join(5)
            state.update(finished=time.time(),closed=_server_snapshot()==(pid,[]),cpp_rows=wire.count,cpp_error=wire.error,
                         new_attempts=len(list(folder.glob('attempt-*.json'))))
            write_once(folder/'end.json',state);print(json.dumps(state),flush=True)

if __name__=='__main__':
    def expire(*_):raise TimeoutError('Timed acquisition ended')
    for sig in (signal.SIGTERM,signal.SIGHUP,signal.SIGALRM):signal.signal(sig,expire)
    signal.alarm(760)
    run(Path(sys.argv[1]))
