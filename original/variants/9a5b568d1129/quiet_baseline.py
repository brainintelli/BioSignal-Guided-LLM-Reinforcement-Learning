"""On the bound acquisition Linux host, save a passive pre-routing baseline for site selection."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

from vendor.brain_stimulation import maxlab as mx, transport
from mea_stim_v2.live import _server_snapshot
from mea_stim_v2.recording import H5Recording
from livepair.protocol import binding
from livepair.device import write_once
from select_sites import candidates


def baseline(b):
    import shutil
    start=binding(b)
    if not 0<=time.time()-start<7200-220:
        raise ValueError('Insufficient original session time for baseline and close')
    if socket.gethostname()!=b['device_hostname'] or b['acquisition_host'] not in subprocess.check_output(
            ['hostname','-I'],text=True).split():
        raise ValueError('Wrong acquisition machine')
    folder=Path(b['record_root']).resolve()/(b['session_id']+'-site-selection')
    if not str(folder).isascii() or not folder.parent.is_dir() or shutil.disk_usage(folder.parent).free<383385600000:
        raise ValueError('Recording volume/path unavailable')
    lock=Path.home()/'.local/state/mea_photo_full/device.lock'
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pid,opened=_server_snapshot()
        if opened:raise RuntimeError('Acquisition server has an open recording')
        folder.mkdir(exist_ok=False)
        write_once(folder/'identity.json',b)
        s=mx.Saving()
        record=file=False
        errors=[]
        with transport.bind_sdk(mx,transport.Endpoint(timeout=3)):
            if mx.send_raw('wellplate_query_version').strip()!='0':raise ValueError('MaxOne expected')
            for u in range(32):
                mx.send(mx.StimulationUnit(u).power_up(False).connect(False))
            mx.send(mx.DAC(0,512))
            mx.set_gain(512)
            mx.hpf('300Hz')
            try:
                s.open_directory(str(folder));s.set_legacy_format(False)
                file=True;s.start_file('quiet')
                s.group_delete_all();s.group_define(0,'routed',list(range(1024)))
                record=True;s.start_recording([0])
                time.sleep(65)
            finally:
                for enabled,fn in ((record,s.stop_recording),(file,s.stop_file),(file,s.group_delete_all)):
                    if enabled:
                        try:fn()
                        except Exception as exc:errors.append(str(exc))
        if errors or _server_snapshot()!=(pid,[]):raise RuntimeError('Quiet file close could not be verified')
        with H5Recording(folder/'quiet.raw.h5',SimpleNamespace(threshold_h5_dataset=b['threshold_h5_dataset'])) as rec:
            if rec.metadata['gain']!=512 or rec.metadata['hpf']!=300 or rec.metadata['threshold_value']!=b['threshold_value']:
                raise ValueError('Current acquisition settings differ')
            output=candidates(rec,b['chip_id'],b.get('bad_channels',[]))
        write_once(folder/'site-candidates.json',output)
        return output


if __name__=='__main__':
    def stop(*args):raise TimeoutError('Acquisition interrupted')
    for sig in (signal.SIGTERM, signal.SIGHUP):signal.signal(sig,stop)
    print(json.dumps(baseline(json.load(sys.stdin))))
