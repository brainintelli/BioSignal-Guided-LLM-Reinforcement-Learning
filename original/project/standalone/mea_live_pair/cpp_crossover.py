"""Counterbalanced three-condition assay using closed H5 and a live C++ receiver."""
import argparse
import fcntl
import hashlib
import itertools
import json
from pathlib import Path
import random
import shutil
import signal
import time
import traceback

import numpy as np
from cpp_response_probe import Receiver
from livepair.device import Device,write_once
from livepair.protocol import PROFILE,binding
from livepair.readout import read_window
from mea_binary_r2.encoding import pulse_shape
from mea_stim_v2.live import mx,_server_snapshot
from mea_stim_v2.qc import stability
from mea_stim_v2.recording import H5Recording
from vendor.brain_stimulation import transport


SITES=('H','W','F')
CONDITIONS=('regular','irregular','sham')
WINDOWS=((-6,-1),(2,7),(8,13))


def design(seed):
    rng=random.Random(seed)
    orders=list(itertools.permutations(CONDITIONS))
    rng.shuffle(orders)
    rows=[]
    for block,order in enumerate(orders):
        for condition in order:
            events=[]
            if condition=='regular':
                events=list(zip(('H','H','H'),(0,4000,8000)))
            if condition=='irregular':
                mid=4000+rng.choice((-1,1))*rng.randint(1000,2000)
                events=list(zip(rng.sample(SITES,3),(0,mid,8000)))
            rows.append(dict(block=block+1,index=len(rows)+1,condition=condition,events=events))
    return rows


def set_units(dev,sites):
    for dac in range(3):
        mx.send(mx.DAC(dac,512))
    for dac,site in enumerate(SITES):
        on=site in sites
        mx.send(mx.StimulationUnit(dev.units[site]).power_up(on).connect(on).set_voltage_mode().dac_source(dac))


def trial(dev,wire,row,roi):
    name='trial_%02d'%row['index']
    dev.late(1500+30)
    dev.open_record(name)
    init_size=dev.record_path.stat().st_size if dev.record_path.exists() else 0
    time.sleep(9)
    if not dev.record_path.exists() or dev.record_path.stat().st_size<=max(init_size,65536):
        raise ValueError('Raw file is not growing')
    dev.assert_ready()
    eid=4*row['index']
    events=row['events']
    with dev._sdk():
        if float(mx.query_DAC_lsb_mV())!=dev.lsb:raise ValueError('DAC step changed')
        # Bind the same three routes at neutral for every condition, including sham.
        set_units(dev,SITES)
        time.sleep(1)
        wire.fresh()
        codes=[]
        for i,(s,offset) in enumerate(events):
            if i:
                codes.append(('delay',offset-events[i-1][1]-2*(PROFILE.phase_us//50)))
                codes.append(('marker',mx.Event(0,1,eid+i,f'trial {row["index"]} idx {i} site {s}')))
            for k,v in pulse_shape(dev.lsb,PROFILE):
                codes.append(('site_dac',(SITES.index(s),v)) if k=='dac' else (k,v))
        e=dict(trial=row['index'],idx=0,site=events[0][0] if events else 'quiet')
        ack=dev._sequence(eid,e,codes)
        host=time.time()
        dev.last_send=time.monotonic()
        write_once(dev.run_dir/(name+'-sent.json'),dict(ack=ack,ack_end=host,plan=row,cpp_rows=wire.count))
        time.sleep(.5)
        set_units(dev,())
    time.sleep(14.5)
    path=dev.close_record()['path']
    with H5Recording(path,dev.config) as rec:
        expected=dict(chip_id=dev.b['chip_id'],fs=20000.,gain=512.,hpf=300.,frame_gaps=0,threshold_value=dev.b['threshold_value'])
        if any(rec.metadata[k]!=v for k,v in expected.items()) or sorted(rec.mapping)!=dev.config.validate_binding():
            raise ValueError('Chip, recording settings, mapping or frame clock changed')
        if dev.last_frame is not None and rec.first_frame<=dev.last_frame:raise ValueError('Frame regression')
        dev.last_frame=rec.last_frame
        manifest=events or [('quiet',0)]
        ev=rec.events
        if len(ev)!=len(manifest):raise ValueError('Wrong number of events')
        anchor=ev[0]['frame']; t=ev[0]['time_s']
        if t<6 or rec.duration-t<13:raise ValueError('Incomplete response windows')
        for i,((site,off),m) in enumerate(zip(manifest,ev)):
            if (m['frame']!=anchor+off or m['eventtype']!=1 or m['id']!=eid+i or
                int(m['message']['trial'])!=row['index'] or int(m['message']['idx'])!=i or m['message']['site']!=site):
                raise ValueError('Wrong event site, offset or trial number')
        pre=rec.sample(roi,t-3)
        out=dict(plan=row,frames=[x['frame'] for x in ev],source=rec.source,metadata=rec.metadata,ack_end=host)
        out['state2']=read_window(rec,anchor+12*20000,roi,dev.bid,name)
        out['pre_state2']=read_window(rec,anchor-3*20000,roi,dev.bid,name+'-pre')
        out['recovery_fraction']=stability(pre,rec.sample(roi,t+12))['fraction_channels_passed']
        chans=[dev.sites[s] for s in SITES]
        raw=rec.sample(chans,t-.004,.7)
        med=np.median(rec.sample(chans,t-3),axis=1)
        rails=(raw==0)|(raw==1023)
        edges=[]
        for site,off in manifest:
            at=80+off
            hit=[s for i,s in enumerate(SITES) if not rails[i,at-8:at].any() and rails[i,at:at+40].any()]
            edges.append(hit)
        out['rails_at_onsets']=edges
        out['late_site_rail']=bool(rails[:,-1000:].any())
        out['pattern_verified']=all(a==[s] for a,(s,_) in zip(edges,events)) if events else not bool((np.abs(raw-med[:,None])*rec.metadata['lsb_uv']>=1000).any())
        out['intended_edges_present']=all(s in a for a,(s,_) in zip(edges,events)) if events else out['pattern_verified']
        out['windows']=[]
        for lo,hi in WINDOWS:
            take=(rec.spike_times>=t+lo)&(rec.spike_times<t+hi)&np.isin(rec.spike_channels,roi)
            out['windows'].append(dict(spikes=int(take.sum()),channels=len(set(rec.spike_channels[take].tolist()))))
        write_once(dev.run_dir/(name+'-evidence.json'),out)
        if out['late_site_rail'] or not out['intended_edges_present']:
            raise ValueError('Missing intended onset, unexpected sham output, or persistent rail in '+name)
    wire.fresh()
    print(name,row['condition'],out['windows'],flush=True)


def run(stage):
    b=json.loads((stage/'baseline-binding.json').read_text())
    root=Path(b['record_root']);previous=root/'cpp-regular-01'
    original=json.loads((previous/'binding.json').read_text())
    plan=design(280801611)
    start=binding(b)
    if not 0<=time.time()-start<95*60-120:raise ValueError('Stimulation window unavailable')
    lease=Path.home()/'.local/state/mea_photo_full/device.lock'
    with lease.open('a') as l:
        fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pid,opened=_server_snapshot()
        old=list(root.rglob('attempt-*.json'))
        if opened or len(old)+12>29 or shutil.disk_usage(root).free<35_000_000_000:
            raise ValueError('Owner, exposure budget or disk constraint')
        frozen=Path(original['route_file'])
        with transport.bind_sdk(mx,transport.Endpoint(timeout=4)):
            if mx.send_raw('wellplate_query_version').strip()!='0' or mx.send_raw('mea_array_command online\nmea_array_get_config').strip()!=frozen.read_text():
                raise ValueError('Instrument changed')
        out=root/'cpp-crossover-v1';out.mkdir(exist_ok=False)
        b=dict(original,prior_nonzero_attempts=len(old),evidence_mode='randomized_crossover_cpp')
        roi=json.loads((previous/'technical-plan.json').read_text())['roi']
        write_once(out/'binding.json',b)
        write_once(out/'design.json',dict(seed=280801611,trials=plan,roi=roi,windows=WINDOWS,prior_attempts=len(old),
            metric='post2_7_minus_pre6_1_spikes',source='cpp_primary_h5_crosscheck',state2=[12,13],training_split='rounds_1_4_train_5_6_held_out'))
        dev=Device(b,out);wire=Receiver(out);status=dict(completed=0,started=time.time())
        def reserve(e):
            wire.fresh()
            if dev.attempts>=len(old)+12:raise ValueError('Crossover budget exhausted')
            dev.reserve(dict(e,cpp_rows=wire.count,host_time_ns=time.time_ns()))
        dev.before_sequence_send=reserve;wire.thread.start()
        try:
            dev.verify_selection();dev.lsb=dev.preflight()['lsb_mv'];dev.prepare_sites()
            write_once(out/'route.json',dict(units=dev.units,channels=dev.sites,lsb_mv=dev.lsb))
            dev.open_record('warmup')
            time.sleep(34)
            with H5Recording(dev.close_record()['path'],dev.config) as rec:
                if (rec.metadata['chip_id']!=b['chip_id'] or sorted(rec.mapping)!=dev.config.validate_binding()
                        or rec.events or rec.duration<30 or rec.metadata['frame_gaps']):
                    raise ValueError('Current quiet recording identity, coverage or route mismatch')
                sample=rec.sample(roi,28)
                if np.any((sample==0)|(sample==1023),axis=1).mean()>.05:
                    raise ValueError('Too many channels at ADC rails before output')
                write_once(out/'warmup.json',dict(source=rec.source,metadata=rec.metadata,
                    fractions=[stability(rec.sample(roi,3),rec.sample(roi,t))['fraction_channels_passed'] for t in (14,28)],
                    windows=[read_window(rec,rec.first_frame+i*20000,roi,dev.bid,'warmup') for i in range(5,29)]))
                dev.last_frame=rec.last_frame
            wire.fresh()
            for row in plan:
                trial(dev,wire,row,roi)
                status['completed']+=1
        except BaseException:
            status['error']=traceback.format_exc()
        finally:
            status['cleanup']=dev.cleanup()
            wire.stop.set();wire.thread.join(4)
            status.update(cpp_error=wire.error,cpp_rows=wire.count,attempts_total=dev.attempts,finished=time.time(),closed=_server_snapshot()==(pid,[]))
            write_once(out/'end.json',status);print(json.dumps(status),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',type=Path);args=p.parse_args()
    def stop(*_):
        raise TimeoutError('Stopped by deadline or signal')
    for sig in (signal.SIGTERM,signal.SIGHUP,signal.SIGALRM):signal.signal(sig,stop)
    signal.alarm(1000)
    run(args.stage)
