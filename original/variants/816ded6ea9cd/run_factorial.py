"""Run the randomized crossover using a new sample's software route and a physical quiet readback."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import time
import traceback

import numpy as np
from cpp_crossover import SITES, WINDOWS, design, set_units, trial
from device_stream import Receiver
from livepair.device import Device, write_once
from livepair.protocol import binding
from livepair.readout import read_window
from mea_stim_v2.live import _server_snapshot
from mea_stim_v2.qc import stability
from mea_stim_v2.recording import H5Recording


def run(stage, output):
    original=json.loads((stage/'baseline-binding.json').read_text())
    route=json.loads((stage/'session-route.json').read_text())
    if any(original[k]!=route[k] for k in ('chip_id','organoid_id','session_id')):
        raise ValueError('Route belongs to a different sample')
    age=time.time()-binding(original)
    if not 0<=age<95*60-780:
        raise ValueError('Not enough time before the registered stimulation cutoff')
    root=Path(original['record_root'])
    prior=root/'cpp-crossover-v1'
    study=json.loads((stage/'protocol.json').read_bytes())
    if route['electrodes']!=study['sites'] or study['session']!=original['session_id'] or study['chip']!=original['chip_id']:
        raise ValueError('New site plan is not bound to this sample')
    import hashlib
    if hashlib.sha256((prior/'design.json').read_bytes()).hexdigest()!=study['prior_plan_sha']:
        raise ValueError('Prior experiment has changed')
    previous=json.loads((prior/'end.json').read_bytes())
    assert previous['closed'] and previous['completed']==18 and not previous['cleanup'] and not previous.get('error') and previous['attempts_total']==12
    oldrows=[json.loads(f.read_bytes()) for f in sorted(prior.glob('trial_*-evidence.json'))]
    assert len(oldrows)==18 and all(v['pattern_verified'] and not v['late_site_rail'] for v in oldrows)
    assert sum(len(v['plan']['events']) for v in oldrows)==36
    lastprior=max(v['metadata']['last_frame'] for v in oldrows)
    if output!='site-rhythm-v1':raise ValueError('Wrong output location')
    b=dict(original,electrodes=route['electrodes'],route_file=route['route_file'],route_sha256=route['route_sha256'],
           material='Pt-Black',prior_nonzero_attempts=12,evidence_mode='within_sample_factorial_followup')
    out=root/output
    order=study['trials'];seed=study['seed']
    assert len(order)==20 and [o['index'] for o in order]==list(range(1,21)) and study['max_total_sequences']==28
    kinds={'anchor_regular','anchor_jitter','new_regular','new_jitter','sham'}
    for k,mid in enumerate(study['jitter_mid_ms'],1):
        g=order[5*k-5:5*k];assert {o['condition'] for o in g}==kinds and all(o['block']==k for o in g)
        for o in g:
            kind=o['condition'];site='H' if kind.startswith('anchor_') else 'W'
            offsets=[0,4000 if kind.endswith('regular') else mid*20,8000]
            assert o['events']==([] if kind=='sham' else [[site,i] for i in offsets])
    assert sorted(study['jitter_mid_ms'])==[120,140,260,280] and sum(len(o['events']) for o in order)==48
    lease=Path.home()/'.local/state/mea_photo_full/device.lock'
    with lease.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pid,opened=_server_snapshot()
        expected={prior/('attempt-%02d.json'%i) for i in range(1,13)}
        if set(root.rglob('attempt-*.json'))!=expected or opened or shutil.disk_usage(root).free<35_000_000_000:
            raise ValueError('Ownership, exposure budget or space failed')
        out.mkdir(exist_ok=False)
        write_once(out/'binding.json',b)
        write_once(out/'plan.json',study)
        dev=Device(b,out)
        wire=Receiver(out)
        status=dict(completed=0,started=time.time())
        def reserve(event):
            wire.fresh()
            if dev.attempts>=28 or time.time()>=binding(original)+95*60 or len(list(root.rglob('attempt-*.json')))!=dev.attempts:
                raise RuntimeError('Current-session budget reached')
            dev.reserve(dict(event,cpp_rows=wire.count,host_time_ns=time.time_ns()))
        dev.before_sequence_send=reserve
        wire.thread.start()
        try:
            dev.lsb=dev.preflight()['lsb_mv']
            if dev.lsb!=json.loads((prior/'route.json').read_bytes())['lsb_mv']:raise ValueError('DAC step changed since original assay')
            dev.prepare_sites()
            write_once(out/'route.json',dict(units=dev.units,channels=dev.sites,lsb_mv=dev.lsb))
            with H5Recording(prior/'warmup/warmup.raw.h5',dev.config) as ref:
                if ref.metadata['chip_id']!=b['chip_id']:raise ValueError('Prior chip mismatch')
            dev.open_record('warmup')
            time.sleep(35)
            with dev._sdk():
                set_units(dev,SITES)
                time.sleep(3)
                set_units(dev,())
            time.sleep(26)
            with H5Recording(dev.close_record()['path'],dev.config) as rec:
                metadata=rec.metadata
                expected=dict(chip_id=b['chip_id'],fs=20000.,gain=512.,hpf=300.,frame_gaps=0,threshold_value=b['threshold_value'])
                if (any(metadata[k]!=v for k,v in expected.items()) or sorted(rec.mapping)!=dev.config.validate_binding()
                    or rec.events or rec.duration<60 or rec.first_frame<=lastprior):
                    raise ValueError('New raw quiet recording did not verify the downloaded route')
                for site,ch in dev.sites.items():
                    if tuple(route['coordinates'][site])!=tuple(rec.channel_xy_um[ch]):
                        raise ValueError('Physical coordinates differ from original chip selection')
                oldspec=json.loads((prior/'design.json').read_bytes())
                with H5Recording(prior/'warmup/warmup.raw.h5',dev.config) as ref:
                    previously_observed={dict(ref.mapping)[ch] for ch in oldspec['roi']}
                active=set(rec.spike_channels.tolist())
                roi=sorted(ch for ch,el in rec.mapping if el in previously_observed and ch in active and ch not in dev.sites.values())
                if len(roi)<50:
                    raise ValueError('Too few active independent recording channels')
                sub=rec.sample(roi,28.)
                if np.any((sub==0)|(sub==1023),axis=1).mean()>.05:
                    raise ValueError('Excess quiet reference saturation')
                site_noise=rec.sample(list(dev.sites.values()),36,1)
                if np.any((site_noise==0)|(site_noise==1023)):
                    raise ValueError('Zero-output reference reached an ADC rail at a target site')
                warm=[read_window(rec,rec.first_frame+i*20000,roi,dev.bid,'warmup') for i in range(5,29)]
                write_once(out/'warmup.json',dict(source=rec.source,metadata=metadata,
                    fractions=[stability(rec.sample(roi,3),rec.sample(roi,t))['fraction_channels_passed'] for t in (14,28,55)],windows=warm,
                    route_confirmed=True,zero_reference_site_rails=False))
                dev.last_frame=rec.last_frame
            wire.fresh()
            write_once(out/'design.json',dict(seed=seed,trials=order,roi=roi,windows=WINDOWS,prior_attempts=12,
                       metric='post2_7_minus_pre6_1_spikes',interpretation='four paired blocks; acquisition only'))
            for row in order:
                trial(dev,wire,row,roi)
                receipt=json.loads((out/('trial_%02d-evidence.json'%row['index'])).read_text())
                if not receipt['pattern_verified']:
                    raise ValueError('An unregistered site also showed an onset; stopping')
                status['completed']+=1
        except BaseException:
            status['error']=traceback.format_exc()
        finally:
            status['cleanup']=dev.cleanup()
            wire.stop.set()
            wire.thread.join(4)
            status.update(new_nonzero_attempts=dev.attempts-12,cpp_error=wire.error,cpp_rows=wire.count,attempts_total=dev.attempts,finished=time.time(),closed=_server_snapshot()==(pid,[]))
            write_once(out/'end.json',status)
            print(json.dumps(status),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',type=Path)
    p.add_argument('--output',default='cpp-crossover-v1')
    args=p.parse_args()
    def stop(*_):
        raise TimeoutError('Session interrupted')
    for sig in (signal.SIGTERM,signal.SIGHUP,signal.SIGALRM):
        signal.signal(sig,stop)
    signal.alarm(1100)
    run(args.stage,args.output)
