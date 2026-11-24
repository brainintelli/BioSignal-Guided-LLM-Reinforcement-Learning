"""Fixed prospective protocol; no hardware imports or network operations."""
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re

PROTOCOL = 'photo_full_v1'
SEQUENCE_SECONDS = .8254
POST_SECONDS = 1800
CLOSEOUT_SECONDS = 120
REPLY_MARGIN_SECONDS = 20
SEND_RESERVE = SEQUENCE_SECONDS + POST_SECONDS + CLOSEOUT_SECONDS + REPLY_MARGIN_SECONDS
PROFILE = dict(pulses=3, ipi=10, pulse_trains=5, ipti=.2,
               amplitude_range=[600,600,1], randomize=False, control=False,
               phase=200, pre_stim_period=10, post_stim_period=30,
               record_only_spikes=False, assay_path=None,
               electrodes_selection={'recording':[], 'stimulation':[]})


def validate_binding(b, now):
    required={'chip_id','session_started_at','acquisition_host','route_cfg','alpha',
              'r2_mode','realtime_training','offline_training','protocol'}
    if set(b)!=required: raise ValueError('Binding has missing or unknown fields')
    if b['protocol']!=PROTOCOL: raise ValueError('Unknown protocol')
    if type(b['alpha']) not in (int,float) or b['alpha']!=1 or b['r2_mode']!='record_only':
        raise ValueError('alpha=1 and R2 record_only required')
    if b['realtime_training'] is not False or b['offline_training'] is not False:
        raise ValueError('Training must remain disabled')
    for key in ('chip_id','acquisition_host','route_cfg','session_started_at'):
        if not isinstance(b[key],str) or not b[key].strip(): raise ValueError('Unbound '+key)
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',b['chip_id']): raise ValueError('Invalid chip ID')
    start=datetime.fromisoformat(b['session_started_at'])
    if start.tzinfo is None or start.utcoffset() is None: raise ValueError('Start requires timezone')
    stamp=start.timestamp()
    if not math.isfinite(now) or stamp>now: raise ValueError('Future or invalid session start')
    if now>=stamp+7200: raise RuntimeError('Biological session expired')
    return stamp


def atomic_json(path, value, *, exclusive=False):
    path=Path(path)
    dest=path if exclusive else path.with_suffix(path.suffix+'.tmp')
    with dest.open('x' if exclusive else 'w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    if not exclusive: os.replace(dest,path)
    # Persist directory entries too on the Linux acquisition host.
    if os.name=='posix':
        fd=os.open(str(path.parent),os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)


def session_key(binding, start):
    # Journal is fixed on the acquisition host. IP aliases, chip labels and
    # output locations cannot grant another attempt on its localhost device.
    return hashlib.sha256(f"localhost:7215:{start:.6f}".encode()).hexdigest()[:24]


class Deadline:
    def __init__(self,start,clock):
        self.clock=clock; self.wall_end=start+7200
        self.mono_end=clock.monotonic()+self.wall_end-clock.time()
    def remaining(self):
        return min(self.wall_end-self.clock.time(),self.mono_end-self.clock.monotonic())
    def require(self,seconds):
        if not math.isfinite(self.remaining()) or self.remaining()<seconds:
            raise RuntimeError(f'Insufficient session budget: need {seconds:.4f} s')
