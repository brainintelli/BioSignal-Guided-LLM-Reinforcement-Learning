"""Build a software-only route from the newly recorded chip's physical electrodes."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
import uuid

from livepair.device import write_once
from mea_stim_v2.live import mx, _server_snapshot
from mea_stim_v2.recording import H5Recording
from select_sites import candidates
from vendor.brain_stimulation import transport


def plan(stage):
    b=json.loads((stage/'baseline-binding.json').read_text())
    src=Path(b['record_root'])/(b['session_id']+'-site-selection')
    original=json.loads((src/'site-candidates.json').read_text())
    with H5Recording(src/'quiet.raw.h5',SimpleNamespace(threshold_h5_dataset=b['threshold_h5_dataset'])) as rec:
        measured=candidates(rec,b['chip_id'],b['bad_channels'])
        if original!=measured or rec.events or rec.metadata['frame_gaps']:
            raise ValueError('Original site selection did not reproduce on this chip')
        selected={k:r['electrode'] for k,r in measured['suggested'].items()}
        coords={k:[r['x_um'],r['y_um']] for k,r in measured['suggested'].items()}
        all_e=sorted(e for c,e in rec.mapping)
    contract=stage/'session-route.json'
    lease=Path.home()/'.local/state/mea_photo_full/device.lock'
    with lease.open('a') as guard, transport.bind_sdk(mx,transport.Endpoint(timeout=90)):
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if _server_snapshot()[1] or mx.send_raw('wellplate_query_version').strip()!='0':
            raise ValueError('Instrument busy or not MaxOne')
        for n,pool in enumerate([set(r['electrode'] for r in measured['candidates'])|set(all_e[::5]),set(all_e)]):
            a=mx.Array('design_'+uuid.uuid4().hex)
            try:
                a.reset();a.clear_selected_electrodes()
                a.select_electrodes(sorted(pool));a.select_stimulation_electrodes(list(selected.values()));a.route()
                for u in range(32):
                    ch=a.query_amplifier_at_stimulation(u).strip()
                    if ch not in ('','-1'):a.disconnect_amplifier_from_stimulation(int(ch))
                units={}
                for k,e in selected.items():
                    a.connect_electrode_to_stimulation(e)
                    units[k]=int(a.query_stimulation_at_electrode(e).strip())
                if len(set(units.values()))!=3 or any(not 0<=u<32 for u in units.values()):continue
                cur=mx.send_raw('mea_array_command '+a.token+'\nmea_array_get_config').strip()
                mapping={int(e):int(c) for c,e in re.findall(r'(\d+)\((\d+)\)',cur.split()[0])}
                if len(mapping)<51 or any(e not in mapping for e in selected.values()):continue
                if any(a.query_amplifier_at_stimulation(units[k]).strip()!=str(mapping[e]) for k,e in selected.items()):continue
                for e in selected.values():a.disconnect_amplifier_from_stimulation(mapping[e])
                cfg=stage/'sample-route.cfg'
                with cfg.open('x') as f:f.write(mx.send_raw('mea_array_command '+a.token+'\nmea_array_get_config').strip())
                write_once(contract,dict(chip_id=b['chip_id'],organoid_id=b['organoid_id'],session_id=b['session_id'],
                    baseline=measured['source'],electrodes=selected,coordinates=coords,software_units=units,
                    route_file=str(cfg),route_sha256=hashlib.sha256(cfg.read_bytes()).hexdigest(),channels=len(mapping),
                    strategy=n,hardware_downloaded=False))
                print(contract.read_text(),flush=True)
                return
            finally:
                a.close()
    raise ValueError('Three individually addressable baseline electrodes could not be routed together')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',type=Path)
    plan(p.parse_args().stage)
