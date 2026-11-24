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
    baseline=root/(original['session_id']+'-site-selection')/'site-candidates.json'
    selection=json.loads(baseline.read_text())
    if selection['chip']!=original['chip_id'] or selection['source']!=route['baseline'] or {s:v['electrode'] for s,v in selection['suggested'].items()}!=route['electrodes']:
        raise ValueError('Fresh physical electrodes do not match baseline')
    b=dict(original,electrodes=route['electrodes'],route_file=route['route_file'],route_sha256=route['route_sha256'],
           material='Pt-Black',prior_nonzero_attempts=0,evidence_mode='independent_randomized_response')
    if Path(output).name!=output or not output.startswith('cpp-crossover-'):
        raise ValueError('Expected a new crossover run directory name')
    out=root/output
    seed=220211737
    order=design(seed)
    lease=Path.home()/'.local/state/mea_photo_full/device.lock'
    with lease.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pid,opened=_server_snapshot()
        if opened or list(root.rglob('attempt-*.json')) or shutil.disk_usage(root).free<35_000_000_000:
            raise ValueError('Ownership, exposure budget or space failed')
        out.mkdir(exist_ok=False)
        write_once(out/'binding.json',b)
        write_once(out/'plan.json',dict(seed=seed,trials=order,windows=WINDOWS,training='first four rounds train, last two hold out'))
        dev=Device(b,out)
        wire=Receiver(out)
        status=dict(completed=0,started=time.time())
        def reserve(event):
            wire.fresh()
            if dev.attempts>=12:
                raise RuntimeError('Current-session budget reached')
            dev.reserve(dict(event,cpp_rows=wire.count,host_time_ns=time.time_ns()))
        dev.before_sequence_send=reserve
        wire.thread.start()
        try:
            dev.lsb=dev.preflight()['lsb_mv']
            dev.prepare_sites()
            write_once(out/'route.json',dict(units=dev.units,channels=dev.sites,lsb_mv=dev.lsb))
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
                    or rec.events or rec.duration<60):
                    raise ValueError('New raw quiet recording did not verify the downloaded route')
                for site,ch in dev.sites.items():
                    if tuple(route['coordinates'][site])!=tuple(rec.channel_xy_um[ch]):
                        raise ValueError('Physical coordinates differ from original chip selection')
                roi=sorted((set(rec.channels.tolist())&set(rec.spike_channels.tolist()))-set(dev.sites.values()))
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
            write_once(out/'design.json',dict(seed=seed,trials=order,roi=roi,windows=WINDOWS,prior_attempts=0,
                       metric='post2_7_minus_pre6_1_spikes',training_split='rounds_1_4_train_5_6_held_out'))
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
            status.update(cpp_error=wire.error,cpp_rows=wire.count,attempts_total=dev.attempts,finished=time.time(),closed=_server_snapshot()==(pid,[]))
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
