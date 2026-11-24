#!/usr/bin/env python3
"""Default: offline preview. Actual run requires an explicitly bound next session."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from protocol import PROFILE,validate_binding


@contextmanager
def deadline_signals(seconds,backend):
    names=(signal.SIGINT,signal.SIGTERM,signal.SIGALRM)
    old={sig:signal.getsignal(sig) for sig in names}
    def stop(signum,frame):raise KeyboardInterrupt('Stop/deadline signal '+str(signum))
    def disarm():
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig in names:signal.signal(sig,signal.SIG_IGN)
    for sig in names:signal.signal(sig,stop)
    signal.setitimer(signal.ITIMER_REAL,max(.001,seconds))
    backend.disarm=disarm
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig,handler in old.items():signal.signal(sig,handler)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',nargs='?',choices=['preview','simulate','run'],default='preview')
    p.add_argument('--binding',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--scenario',choices=['success','baseline-failure','lost-reply'],default='success')
    args=p.parse_args(argv)
    if args.mode=='preview':
        print(json.dumps(dict(protocol='photo_full_v1',hardware_connected=False,profile=PROFILE,
                              identity_seconds=60,baseline_seconds=600,post_file_leadin_seconds=60,
                              post_seconds=1800,session_limit_seconds=7200,
                              pending=['new_chip_id','biological_start_with_timezone','current_route_cfg'],
                              alpha=1,r2_mode='record_only',realtime_training=False),indent=2))
        return 0
    if args.output is None:p.error('--output is required')
    if args.mode=='simulate':
        from simulation import simulate
        print(json.dumps(simulate(args.output,args.scenario),indent=2));return 0
    if args.binding is None:p.error('--binding is required for actual run')
    b=json.loads(args.binding.read_text(encoding='utf-8-sig'))
    started=validate_binding(b,time.time()) # Before SDK, platform, route or connection.
    if sys.platform!='linux':raise RuntimeError('Run only on the Linux acquisition machine')
    if not Path(b['route_cfg']).is_absolute():raise ValueError('route_cfg must be an absolute local file')
    out=args.output.resolve()
    if out.exists():raise FileExistsError('Output must be new: '+str(out))
    os.environ.setdefault('HDF5_PLUGIN_PATH','/pub/soft/MaxLab/so')
    from device import MaxLabBackend
    from session import Session
    backend=MaxLabBackend(b,out)
    journal=Path.home()/'.local/state/mea_photo_full/sessions'
    with backend.sdk_context(),deadline_signals(started+7200-time.time()-120,backend):
        result=Session(b,backend,time,out,journal).run()
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['status']=='completed' else 1


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,OSError) as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr);raise SystemExit(2)
