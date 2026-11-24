"""Run on the recording station against closed files, with no device calls."""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ['HDF5_PLUGIN_PATH'] = '/pub/soft/MaxLab/so'
sys.path.insert(0, '/home/researcher/brainDesktop/mea-standby-20261118')
import session as station
import numpy as np
from livepair.readout import read_window
from mea_stim_v2.qc import _stats


def hash_file(path):
    sha = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4194304), b''):
            sha.update(block)
    return sha.hexdigest()


def main(plan):
    source = plan['source']
    bid = plan['binding']
    for obj in source.values():
        if hash_file(obj['path']) != obj['sha256']:
            raise ValueError('Source hash mismatch')
    with station.H5Recording(Path(source['quiet']['path']), station.CFG) as rec:
        assert not rec.events and rec.metadata['chip_id'] == '22554'
        active = set(map(int,rec.spike_channels))
        reference = {e for c,e in rec.mapping if c in active}
    out = {}
    sites = list(plan['sites'].values())
    for label in 'HWF':
        with station.H5Recording(Path(source[label]['path']), station.CFG) as rec:
            assert len(rec.events)==1 and rec.events[0]['frame']==source[label]['event']
            assert rec.metadata['frame_gaps']==0 and rec.fs==20000 and rec.metadata['chip_id']=='22554'
            ev = rec.events[0]
            roi = sorted(c for c,e in rec.mapping if e in reference and all(
                (17.5*(e%220)-s['x_um'])**2+(17.5*(e//220)-s['y_um'])**2 >= 200**2 for s in sites))
            ref_offset = plan.get('ref_offset',-12)
            ref = _stats(rec.sample(roi,ev['time_s']+ref_offset))
            keep = (ref['mad']>0) & ~ref['rails']
            counts = {}
            for offset in plan['offsets']:
                s = _stats(rec.sample(roi,ev['time_s']+offset))
                ratio = np.divide(s['mad'],ref['mad'],out=np.zeros(len(roi)),where=ref['mad']>0)
                ok = (ref['mad']>0) & ~ref['rails'] & ~s['rails'] & (ratio>=.67) & (ratio<=1.50) & (abs(s['median']-ref['median'])<=5)
                keep &= ok
                counts[offset] = int(sum(ok))
            panel = [c for c, admit in zip(roi,keep) if admit]
            record = dict(candidates=len(roi), per_window=counts, channels=panel, windows=[], sha256=source[label]['sha256'])
            if len(panel)>=50:
                record['prime']=read_window(rec,ev['frame']+ref_offset*20000,panel,bid,label)
                record['windows']=[read_window(rec,ev['frame']+dt*20000,panel,bid,label) for dt in plan['offsets']]
            out[label]=record
    print(json.dumps(out,allow_nan=False))


if __name__=='__main__':
    main(json.loads(sys.argv[1]))
