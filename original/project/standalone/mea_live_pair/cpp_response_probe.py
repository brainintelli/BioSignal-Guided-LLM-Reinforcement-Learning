"""Run the spatial qualification with Python output and a concurrent legacy C++ stream."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import signal
import time
import traceback

import numpy as np
from mea_binary_r2.encoding import pulse_shape
from livepair.device import Device, write_once
from livepair.device_rpc import Session
from livepair.protocol import PROFILE, RATE, binding
from mea_stim_v2.recording import H5Recording
from mea_stim_v2.live import _server_snapshot, mx
from vendor.brain_stimulation import transport
from device_stream import Receiver


def technical_regular(dev, wire, root, session):
    """One sham followed by one fixed-site pattern. No game label or optimizer."""
    target='H'
    stim_channel=dev.sites[target]
    with H5Recording(root/(session+'-site-selection')/'quiet.raw.h5',dev.config) as baseline:
        roi=sorted(set(baseline.spike_channels.tolist())-set(dev.sites.values()))
    write_once(dev.run_dir/'technical-plan.json',dict(target=target,channel=stim_channel,
        electrode=dev.config.sites[target],unit=dev.units[target],roi=roi,profile=PROFILE.digest,
        nominal_limit_mv=PROFILE.target_mv,phase_us=PROFILE.phase_us,frame_offsets=[0,4000,8000],
        before_seconds=[-6,-1],after_seconds=[2,7],late_seconds=[8,13],clinical_or_game_reward=False))
    if len(roi)<50:
        raise ValueError('Insufficient prior active independent recording channels')
    time.sleep(30)
    for index,name in enumerate(('sham','regular')):
        nonzero=name=='regular'
        dev.open_record(name)
        opened=dev.record_path.stat().st_size if dev.record_path.exists() else 0
        time.sleep(12)
        if not dev.record_path.exists() or dev.record_path.stat().st_size<=max(opened,65536):
            raise RuntimeError('Recording failed to grow before '+name)
        dev.assert_ready()
        wire.fresh()
        with dev._sdk():
            if float(mx.query_DAC_lsb_mV())!=dev.lsb:
                raise RuntimeError('DAC readback changed')
            mx.send(mx.DAC(0,512))
            for k,u in dev.units.items():
                mx.send(mx.StimulationUnit(u).power_up(k==target).connect(k==target).set_voltage_mode().dac_source(0))
            time.sleep(1)
            wire.fresh()
            eid=100+4*index
            codes=[]
            if nonzero:
                shape=pulse_shape(dev.lsb,PROFILE)
                for i in range(3):
                    if i:
                        codes.append(('delay',4000-2*(PROFILE.phase_us//50)))
                        codes.append(('marker',mx.Event(0,1,eid+i,f'phase regular idx {i} site H')))
                    codes.extend(shape)
            start=time.time_ns()
            ack=dev._sequence(eid,dict(phase=name,idx=0,site=target),codes)
            write_once(dev.run_dir/(name+'-send.json'),dict(begin_ns=start,ack_ns=time.time_ns(),
                ack=ack,lsb_mv=dev.lsb,codes=[x for x in codes if x[0]!='marker'],cpp_rows=wire.count))
            time.sleep(.5)
            for u in dev.units.values():
                mx.send(mx.StimulationUnit(u).power_up(False).connect(False))
            mx.send(mx.DAC(0,512))
        time.sleep(16)
        path=dev.close_record()['path']
        with dev.inspect(path) as rec:
            ev=rec.events
            frames=[e['frame'] for e in ev]
            offsets=[0,4000,8000] if nonzero else [0]
            if (not ev or frames!=[frames[0]+off for off in offsets] or
                any(e['id']!=eid+i or e['eventtype']!=1 or e['message'].get('phase')!=name
                    or int(e['message']['idx'])!=i or e['message']['site']!=target for i,e in enumerate(ev))):
                raise ValueError('Closed H5 marker mismatch for '+name)
            t=ev[0]['time_s']
            med=np.median(rec.sample([stim_channel],t-3.))
            raw=rec.sample([stim_channel],t-.003,.62)[0]
            high=np.abs(raw.astype(float)-med)*rec.metadata['lsb_uv']>=1000.
            onsets=[bool(not high[60+i-8:60+i].any() and high[60+i:60+i+40].any()) for i in offsets]
            rows=[]
            for lo,hi in ((-6,-1),(2,7),(8,13)):
                keep=(rec.spike_times>=t+lo)&(rec.spike_times<t+hi)&np.isin(rec.spike_channels,roi)
                rows.append(dict(window_seconds=[lo,hi],spikes=int(keep.sum()),channels=len(set(rec.spike_channels[keep].tolist()))))
            samples=rec.sample(roi,t-3.)
            rail_free=float((~np.any((samples==0)|(samples==1023),axis=1)).mean())
            target_rails=bool(np.any((raw==0)|(raw==1023)))
            evidence=dict(source=rec.source,chip=rec.metadata['chip_id'],frames=frames,frame_gaps=rec.metadata['frame_gaps'],
                          duration=rec.duration,stim_channel=stim_channel,roi=roi,windows=rows,raw_artifact_onsets=onsets,
                          site_excursion_uv=float(np.max(abs(raw.astype(float)-med))*rec.metadata['lsb_uv']),
                          pre_fraction_rail_free=rail_free,site_rails=target_rails)
            write_once(dev.run_dir/(name+'-evidence.json'),evidence)
            if not nonzero and (high.any() or target_rails or rail_free<.95 or not sum(r['spikes'] for r in rows)):
                raise ValueError('No clean, active sham reference at the bound route')
        print('closed',name,'frames',frames,'spikes',[r['spikes'] for r in rows],flush=True)
    return dict(technical_test_complete=True)


def run(stage, attempt, technical=False):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',attempt):
        raise ValueError('Invalid attempt directory')
    identity = json.loads((stage/'baseline-binding.json').read_text())
    start = binding(identity)
    if not 0 <= time.time()-start < 95*60-180:
        raise ValueError('Outside original stimulation window')
    root = Path(identity['record_root'])
    out = root/attempt
    lock_path = Path.home()/'.local/state/mea_photo_full/device.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pid,opened = _server_snapshot()
        if opened or shutil.disk_usage(root).free < 40_000_000_000:
            raise ValueError('Active recording or insufficient space for the bounded qualification')
        selection=json.loads((root/(identity['session_id']+'-site-selection')/'site-candidates.json').read_text())
        frozen=stage/'cpp-before.cfg'
        with transport.bind_sdk(mx,transport.Endpoint(timeout=4)):
            raw=mx.send_raw('mea_array_command online\nmea_array_get_config').strip()
            if raw!=frozen.read_text() or mx.send_raw('wellplate_query_version').strip()!='0':
                raise ValueError('Current instrument or online configuration changed')
        previous = list(root.glob('cpp-*/attempt-*.json'))
        if previous:
            raise RuntimeError('Nonzero attempt already reserved in this sample; no automatic repeat')
        out.mkdir(exist_ok=False)
        b=dict(identity, material='Pt-Black', electrodes={k:v['electrode'] for k,v in selection['suggested'].items()},
               route_file=str(frozen),route_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
               prior_nonzero_attempts=0,evidence_mode='python_output_cpp_return_with_H5',replicate=0)
        write_once(out/'binding.json',b)
        dev=Device(b,out)
        wire=Receiver(out)
        status=dict(hardware_output_calibration=False, model_updates=0,server_pid=pid)
        def reserve(event):
            wire.fresh()
            if dev.attempts>=9:
                raise RuntimeError('Nine qualification attempts exhausted')
            dev.reserve(dict(event,cpp_count=wire.count,host_time_ns=time.time_ns()))
        dev.before_sequence_send=reserve
        wire.thread.start()
        try:
            dev.verify_selection()
            dev.lsb=dev.preflight()['lsb_mv']
            dev.prepare_sites()
            print('route_downloaded',dev.units,flush=True)
            if technical:
                status.update(technical_regular(dev,wire,root,b['session_id']))
            else:
                dev.prime('baseline',True)
                wire.fresh()
                print('recorded_baseline',len(dev.roi),wire.count,flush=True)
                session=Session()
                session.dev=dev
                report=session.calibrate()
                status.update(qualification_passed=report['passed'],trials=len(report['trials']),error=report.get('error'))
        except BaseException as exc:
            status.update(error=str(exc),trace=traceback.format_exc())
        finally:
            status['cleanup_errors']=dev.cleanup()
            wire.stop.set()
            wire.thread.join(4)
            server, opened = _server_snapshot()
            status.update(attempts=dev.attempts,cpp_count=wire.count,cpp_error=wire.error,
                          closed=(not opened and server==pid),finished_at=time.time())
            write_once(out/'end.json',status)
            print(json.dumps(status),flush=True)
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',type=Path)
    p.add_argument('attempt')
    p.add_argument('--technical',action='store_true')
    a=p.parse_args()
    def interrupted(*_):
        raise TimeoutError('Job interrupted')
    for sig in (signal.SIGTERM,signal.SIGHUP,signal.SIGALRM):
        signal.signal(sig,interrupted)
    signal.alarm(900)
    result=run(a.stage,a.attempt,a.technical)
    raise SystemExit(int(bool(result.get('error') or result['cleanup_errors'] or not result['closed'])))
