"""One bounded zero-voltage diagnostic, not a retry of the 600 mV trial."""
import datetime as dt
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'settled_package/BrainStimulation'))
from brain_stimulation import maxlab as mx,runner
from brain_stimulation.config import load_config
from brain_stimulation.transport import Api,Endpoint
from trial_qc import mapping


def main():
    deadline=dt.datetime.fromisoformat('2026-11-17T05:28:40+00:00').timestamp()
    end_mono=time.monotonic()+deadline-time.time()
    if min(deadline-time.time(),end_mono-time.monotonic())<300:
        raise RuntimeError('Insufficient time for zero-only check')
    with (ROOT/'zero_check_launch.json').open('x') as f:
        json.dump({'mode':'zero_voltage_only','nonzero_pulses':0,'utc':dt.datetime.now(dt.timezone.utc).isoformat()},f)
        f.flush();os.fsync(f.fileno())
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if (p/'comm').read_text().strip()=='mxwserver':
                if any('.h5' in os.readlink(fd) for fd in (p/'fd').iterdir()):raise RuntimeError('Existing open H5')
        except (FileNotFoundError,PermissionError):pass
    profile=json.loads((ROOT/'photo_parameters.template.json').read_text())
    profile.update(pulses=1,pulse_trains=1,amplitude_range=[0,0,1],pre_stim_period=2,post_stim_period=0)
    profile['electrodes_selection']={'recording':[e for ch,e in mapping((ROOT/'qualified_route.cfg').read_text())],'stimulation':[7587]}
    profile['assay_path']=str(ROOT/'qualified_route.cfg')
    (ROOT/'zero_only_profile.json').write_text(json.dumps(profile,indent=2))
    params=load_config(ROOT/'zero_only_profile.json')
    assert params.amplitude_schedule()==[0.0]
    log=(ROOT/'zero_only_api_audit.jsonl').open('x',buffering=1)
    original=Api.send
    original_signals=runner.handle_signals
    def observed(api,command,timeout=None):
        # Enforce zero at the wire boundary, both direct and queued commands.
        for line in command.splitlines():
            if line.startswith('mea_set_dac '):
                parts=line.split()
                if any(int(code)!=512 for code in parts[2:]):raise RuntimeError('Nonzero DAC forbidden in this diagnostic')
        log.write(json.dumps({'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'command':command})+'\n')
        reply=original(api,command,timeout)
        log.write(json.dumps({'reply':reply[:160]})+'\n')
        if command=='system_initialize':
            for unit in range(32):mx.send(mx.StimulationUnit(unit).power_up(False).connect(False))
            mx.send(mx.DAC(0,512))
        return reply
    def bounded(seconds):
        if min(deadline-time.time(),end_mono-time.monotonic())<seconds+120:raise RuntimeError('Deadline')
        time.sleep(seconds)
    def stop(*args):raise KeyboardInterrupt('Zero diagnostic deadline')
    @contextmanager
    def cleanup_signals(handler):
        if handler==signal.SIG_IGN:signal.alarm(0)
        with original_signals(handler):yield
    signal.signal(signal.SIGALRM,stop);signal.alarm(240)
    Api.send=observed;runner.handle_signals=cleanup_signals
    try:code=runner.run_experiment(params,Endpoint(timeout=5),ROOT/'zero_settle_results',wait=bounded)
    finally:
        signal.alarm(0);Api.send=original;runner.handle_signals=original_signals;log.close()
    print(json.dumps({'mode':'zero_voltage_only','exit':code,'nonzero_stimulation_sent':False}),flush=True)
    return code


if __name__=='__main__':
    if sys.argv[1:]!=['--run-zero-only']:raise SystemExit('Explicit zero-only flag required')
    raise SystemExit(main())
