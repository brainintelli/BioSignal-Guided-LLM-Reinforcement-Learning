"""Task-specific supervised wrapper around the frozen tested MaxLab runner.

It closes and qualifies the 600 s connected baseline before permitting the
one nonzero sequence. Main post-stimulus data use a separate H5 file.
"""
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'BrainStimulation'))
from brain_stimulation import maxlab as mx,runner
from brain_stimulation.config import load_config
from brain_stimulation.transport import Api,Endpoint,bind_sdk
from trial_guard import TrialGuard
from trial_qc import inspect,mapping


def main():
    reg=json.loads((ROOT/'registration.json').read_text())
    if reg.get('short_protocol_authorized') is not True:
        raise RuntimeError('Short recording protocol has not been selected by user')
    if (reg['host'],reg['chip_id'],reg['alpha'],reg['r2_reward_enabled'],reg['realtime_training']) != ('192.0.2.101','22021',1,False,False):
        raise RuntimeError('Identity or training boundaries changed')
    if json.loads((ROOT/'identity_qc.json').read_text())['status']!='passed':
        raise RuntimeError('Fresh identity qualification missing')
    expected=(ROOT/'qualified_route.cfg').read_bytes()
    if hashlib.sha256(expected).hexdigest()!='dc32f17300ab8540252883c9a1e606215cead85bb95a14e35ca94afc3e04318b':
        raise RuntimeError('Frozen route changed')
    profile=json.loads((ROOT/'photo_parameters.template.json').read_text())
    if {k:profile[k] for k in ['pulses','ipi','pulse_trains','ipti','amplitude_range','phase','randomize','control','pre_stim_period','post_stim_period','record_only_spikes']} != dict(pulses=3,ipi=10,pulse_trains=5,ipti=.2,amplitude_range=[600,600,1],phase=200,randomize=False,control=False,pre_stim_period=10,post_stim_period=10,record_only_spikes=False):
        raise RuntimeError('Photo parameters changed')
    profile['electrodes_selection']={'recording':[e for ch,e in mapping(expected.decode())],'stimulation':[7587]}
    profile['assay_path']=str(ROOT/'qualified_route.cfg')
    (ROOT/'bound_profile.json').write_text(json.dumps(profile,indent=2))
    params=load_config(ROOT/'bound_profile.json')
    if abs(params.stimulation_seconds()-.8254)>1e-9 or len(params.amplitude_schedule())!=5:
        raise RuntimeError('Unexpected compiled duration')
    guard=TrialGuard(ROOT,started_wall=dt.datetime.fromisoformat(reg['session_started_at']).timestamp(),wall_now=time.time(),monotonic_now=time.monotonic())
    def remaining(): return guard.remaining(wall_now=time.time(),monotonic_now=time.monotonic())
    if guard.claim.exists() or remaining()<600+60+.8254+600+120+60:
        raise RuntimeError('Previous attempt or insufficient complete session budget')
    # Both claim and launch survive interruption; neither is removed on error.
    with (ROOT/'trial_launch.json').open('x') as f:
        json.dump({'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'reserved_for_this_trial':15},f)
        f.flush();os.fsync(f.fileno())
    opened=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if (p/'comm').read_text().strip()=='mxwserver':
                opened.extend(os.readlink(fd) for fd in (p/'fd').iterdir() if '.h5' in os.readlink(fd))
        except (FileNotFoundError,PermissionError):pass
    if opened:raise RuntimeError('An H5 is already open')
    progress={'status':'starting','stimulation_sent':False,'stimulus_attempted':False,'alpha':1,'r2_reward_enabled':False,'realtime_training':False,'cleanup_errors':[]}
    def save():
        progress['utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        temp=ROOT/'progress.json.tmp';temp.write_text(json.dumps(progress,indent=2));temp.replace(ROOT/'progress.json')
    log=(ROOT/'api_audit.jsonl').open('x',buffering=1)
    original_send=Api.send
    original_signals=runner.handle_signals
    token=None; marker_token='photo_22021_event_only_7001'; h5dir=None
    qualified=False; route_checked=False; unit=None; marker_created=False
    def audit(kind,message,**extra):
        log.write(json.dumps({'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'kind':kind,
                              'command':message,'sha256':hashlib.sha256(message.encode()).hexdigest(),**extra})+'\n')
        log.flush()
    def observed(api,message,timeout=None):
        nonlocal token,h5dir,unit
        if message.startswith('sequence_send '):
            sending=message.split()[1]
            if sending==marker_token:
                if progress['status']!='connected_baseline':raise RuntimeError('Unexpected marker phase')
            elif sending.startswith('standalone_'):
                if not route_checked or not qualified:raise RuntimeError('Unqualified stimulation attempt')
                guard.reserve(wall_now=time.time(),monotonic_now=time.monotonic(),qualified=qualified)
                progress.update(status='stimulus_send_attempt',stimulus_attempted=True);save()
                audit('reserved_15_pulses',message);os.fsync(log.fileno())
            else:raise RuntimeError('Unexpected sequence send token')
        audit('request',message)
        try:reply=original_send(api,message,timeout)
        except BaseException as exc:
            audit('error',message,error=repr(exc));raise
        audit('ack',message,reply=reply[:200])
        if message=='system_initialize':
            # Explicitly disable every output unit before enabling the single
            # chosen one later in the runner; no inherited enabled outputs.
            for number in range(32):
                mx.send(mx.StimulationUnit(number).power_up(False).connect(False))
            mx.send(mx.DAC(0,512))
        if message=='system_query_dac_lsb_mv' and abs(float(reply)-2.981901)>1e-9:
            raise RuntimeError('DAC calibration query changed; rebuild preview before output')
        if message.startswith('mea_array_command standalone_') and '\nmea_array_download' in message:
            token=message.splitlines()[0].split()[1]
        if '\nmea_array_query_stimulation_at_electrode 7587' in message:
            unit=int(reply.strip())
        if message.startswith('saving_open_dir '): h5dir=Path(message.split(' ',1)[1])
        if message.startswith('sequence_send standalone_'):
            if reply.strip().lower()!='ok':raise RuntimeError('Unknown stimulation acknowledgement')
            progress.update(stimulation_sent=True,status='post_stimulus_recording');save()
        return reply
    def sleep_record(seconds):
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            if remaining()<120:raise RuntimeError('Session closeout deadline reached')
            progress['phase_seconds_remaining']=round(max(0,until-time.monotonic()),1);save()
            time.sleep(min(10,max(0,until-time.monotonic())))
    def bounded_wait(seconds):
        nonlocal route_checked,qualified,marker_created
        if remaining()<seconds+120:raise RuntimeError('Insufficient wait/closeout time')
        if not route_checked:
            if token is None or unit is None:raise RuntimeError('Missing downloaded route or unit')
            current=mx.send_raw('mea_array_command '+token+'\nmea_array_get_config')
            (ROOT/'downloaded_owned_array.cfg').write_text(current)
            if mapping(current)!=mapping(expected.decode()):raise RuntimeError('Downloaded map mismatch')
            ch=mx.send_raw('mea_array_command '+token+'\nmea_array_query_amplifier_at_electrode 7587')
            reverse=mx.send_raw('mea_array_command '+token+'\nmea_array_query_amplifier_at_stimulation '+str(unit))
            if int(ch.strip())!=691 or int(reverse.strip())!=691 or not 0<=unit<32:
                raise RuntimeError('Bidirectional stimulation routing mismatch')
            route_checked=True;progress.update(stimulation_unit=unit,selected_channel=691);save()
        if seconds==600 and not qualified:
            progress.update(status='connected_baseline',baseline_file=str(h5dir/'stimulation.raw.h5'));save()
            marker_created=True
            marker=runner.ManagedSequence(token=marker_token,initial_delay=0,persistent=True)
            if marker.token!=marker_token:raise RuntimeError('Unexpected marker sequence token')
            marker.append(mx.DelaySamples(100));marker.append(mx.Event(0,1,7001,'mode connected_baseline_no_nonzero_DAC'))
            marker.send()
            sleep_record(600)
            saving=mx.Saving();saving.stop_recording();saving.stop_file()
            qc=inspect(h5dir/'stimulation.raw.h5',600,connected=True)
            (ROOT/'connected_baseline_qc.json').write_text(json.dumps(qc,indent=2))
            progress['connected_baseline_qc']=qc['status'];save()
            if qc['status']!='passed':raise RuntimeError('Connected baseline rejected: '+','.join(qc['errors']))
            if remaining()<60+.8254+600+120+20:raise RuntimeError('Insufficient full post-recording budget')
            qualified=True
            saving.start_file('stimulus_post')
            for key,value in params.assay_inputs().items():saving.write_assay_input(key,value)
            saving.group_delete_all();saving.group_define(0,'routed',list(range(1024)))
            saving.start_recording([0])
            progress.update(status='qualified_pre_stimulus_60s',main_file=str(h5dir/'stimulus_post.raw.h5'));save()
            sleep_record(60)
        else:
            sleep_record(seconds)
    @contextmanager
    def signals(handler):
        if handler==signal.SIG_IGN:signal.setitimer(signal.ITIMER_REAL,0)
        with original_signals(handler):yield
    def expire(*args):raise KeyboardInterrupt('Absolute biological-session deadline')
    signal.signal(signal.SIGALRM,expire)
    signal.setitimer(signal.ITIMER_REAL,remaining()-120)
    Api.send=observed;runner.handle_signals=signals
    save()
    try:
        code=runner.run_experiment(params,Endpoint(timeout=10),ROOT/'results',wait=bounded_wait)
        progress.update(status='completed' if code==0 else 'failed',exit_code=code)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        Api.send=original_send;runner.handle_signals=original_signals
        if marker_created:
            try:
                with bind_sdk(mx,Endpoint(timeout=2)):mx.send_raw('sequence_delete '+marker_token)
            except Exception as exc:progress['cleanup_errors'].append(repr(exc))
        save();log.close()
    return code


if __name__=='__main__':
    if sys.argv[1:]!=['--run']:raise SystemExit('Explicit --run required')
    raise SystemExit(main())
