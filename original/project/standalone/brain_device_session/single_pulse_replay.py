"""Acquire explicitly exploratory single-pulse recordings after a closed fixed-panel validation."""
import fcntl
import json
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

import session as s
from livepair.qualification import classify_blocks
from mea_state2_r2.features import fit,transform


def run(parent):
    if parent.resolve().parent!=s.DATA:raise ValueError('Invalid original session')
    registered=s.load(parent/'fixed-readout-binding.json')
    start=s.binding(registered)
    deadline=time.monotonic()+start+5700-time.time()
    s.available_time(start,time.time(),deadline,1050)
    follow=s.DATA/registered['session_id']
    validation=s.load(follow/'panel-validation.json')
    if validation['passed'] or s.load(follow/'closed.json')['errors']:raise ValueError('Unexpected followup state')
    original=s.load(parent/'binding.json')
    if original['started_at']!=registered['started_at'] or original['chip_id']!=registered['chip_id']:
        raise ValueError('Original session mismatch')
    if s.load(follow/'closed.json')['attempts']!=3 or list(follow.glob('attempt-*.json')):
        raise ValueError('Original pulse ledger differs')
    closed=s.load(parent/'session-end.json')
    if closed['attempts_reserved']!=3 or set(closed['completed_sites'])!=set('HWF'):
        raise ValueError('Expected three confirmed initial localization pulses')
    plan=s.load(parent/'site-plan.json');roi=registered['fixed_readout_channels']
    if roi!=validation['roi'] or set(roi)&{r['channel'] for r in plan['suggested'].values()}:
        raise ValueError('Changed independent panel')
    dest=parent/'exploratory-single-pulse'
    rng=random.Random(225541118)
    schedule=[rng.sample(list('HWF'),3) for _ in range(3)]
    study=dict(chip=registered['chip_id'],old_session=parent.name,started_at=registered['started_at'],
               panel_sha256=registered['panel_sha256'],roi=roi,order=schedule,prior_attempts=3,
               max_extra_pulses=9,train_blocks=[0,1],test_blocks=[2],trial_unit='one independent native run/H5',
               pre_window_s=[-3,-2],post_window_s=[12,13],normalization='earlier independent quiet baseline only',
               training_seeds=[2255401,2255402,2255403],game_eval_seeds=list(range(22554901,22554909)),
               reward='verified no-stim +1, verified pulse -1; exogenous replay, identical for both arms',
               paired_shuffle='phase-preserving nonidentity permutation of whole trials within each training block; common ordered held-out inputs',
               live_training_admitted=False)
    signal.signal(signal.SIGALRM,s.expiry)
    signal.signal(signal.SIGTERM,s.expiry)
    signal.setitimer(signal.ITIMER_REAL,max(.01,start+7200-time.time()-120))
    with (s.REG/'device.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if s._server_snapshot()[1] or shutil.disk_usage(s.DATA).free<45_000_000_000:raise RuntimeError('Acquisition busy or volume unavailable')
        if subprocess.check_output(['ss','-Htn','state','established','sport','=',':5555'],text=True).strip():
            raise RuntimeError('5555 client already present')
        dest.mkdir(exist_ok=False)
        s.write_once(dest/'study.json',study)
        calendar=s.digest(study)
        baseline=follow/'baseline'/'baseline.raw.h5'
        with s.H5Recording(baseline,s.CFG) as rec:
            if rec.metadata['chip_id']!=study['chip'] or rec.events or rec.mapping!=[tuple(q) for q in plan['mapping']]:
                raise ValueError('Wrong normalization recording')
            rows=[s.read_window(rec,rec.first_frame+t*20000,roi,calendar,'quiet-calibration')['features'] for t in range(5,55)]
            norm=fit(rows)
        summary=dict(study=study,normalization=norm,normalization_source=str(baseline),trials=[])
        acq=s.start_stream(dest);wire=s.Receiver(dest);wire.thread.start()
        try:
            for block,order in enumerate(schedule):
                folder=dest/f'block-{block}';folder.mkdir()
                params=s.configs(plan,folder)
                for site in order:
                    s.available_time(start,time.time(),deadline,30)
                    if s._server_snapshot()[1] or len(list(dest.glob('block-*/attempt-*.json')))>=9:
                        raise RuntimeError('Recording or cumulative dose mismatch')
                    s.run_point(site,params[site],folder,plan,original,s.transport.Endpoint(timeout=5),acq,start,deadline,wire)
                    evidence=s.load(folder/(site+'-result.json'))
                    if not evidence['onset_visible'] or not evidence['stim_channel_recovered_at_12s']:
                        raise RuntimeError('Unconfirmed stimulus or recovery')
                    with s.H5Recording(evidence['source']['path'],s.CFG) as rec:
                        ev=rec.events[0];frame=ev['frame'];t=ev['time_s']
                        pre=s.read_window(rec,frame-60000,roi,calendar,f'{block}-{site}-quiet')
                        post=s.read_window(rec,frame+240000,roi,calendar,f'{block}-{site}-pulse')
                        q=s.stability(rec.sample(roi,t-3),rec.sample(roi,t+12))
                        for when in (t-3,t+12):
                            raw=rec.sample(roi,when)
                            if not ((raw>0)&(raw<1023)).all():raise ValueError('Independent feature window contains an ADC rail')
                        row=dict(index=len(summary['trials']),site=site,block=block,anchor=frame,
                                 source=evidence['source'],pre=pre,post=post,verified_pulses=1,
                                 independent_stability=q['fraction_channels_passed'])
                    summary['trials'].append(row)
                    s.write_once(folder/(site+'-features.json'),row)
                    print(json.dumps(dict(block=block,site=site,frame=frame)),flush=True)
            def classify(key):
                x=[list(transform(r[key]['features'],norm)) for r in summary['trials']]
                return classify_blocks(schedule,[x[i:i+3] for i in (0,3,6)])
            summary['exploratory_classification']={k:classify(k) for k in ('pre','post')}
            s.write_once(dest/'replay.json',summary)
        finally:
            wire.stop.set();wire.thread.join(4)
            s.write_once(dest/'closed.json',dict(prior_attempts=3,new_attempts=len(list(dest.glob('block-*/attempt-*.json'))),
                         verified=len(summary['trials']),tcp_spikes=wire.count,tcp_error=wire.error,openfiles=s._server_snapshot()[1],time=time.time()))

if __name__=='__main__':run(s.DATA/sys.argv[1])
