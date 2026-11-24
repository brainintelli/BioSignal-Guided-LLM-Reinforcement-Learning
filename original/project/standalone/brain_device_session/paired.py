"""Single workstation entry for BrainDevice localization, qualification, and the existing two-host study."""
import argparse
from pathlib import Path
import json
import sys
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'.tools'/x) for x in ('mea_ssh_deps','mea_baseline_deps')]
sys.path[:0]=[str(ROOT/'standalone'/x) for x in ('mea_live_pair','mea_state2_r2','mea_binary_r2','mea_online','mea_stimulation_v2')]
from control import run
from livepair.protocol import binding

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--chip',required=True)
    p.add_argument('--organoid',required=True)
    p.add_argument('--started-at',required=True,help='Original sample start, ISO timestamp including timezone')
    p.add_argument('--replaces-session',help='Closed previous session when the new specimen reuses both identifiers')
    a=p.parse_args()
    session='bd-'+uuid.uuid4().hex[:16]
    b=dict(session_id=session,replicate=0,acquisition_host='192.0.2.21',device_hostname='researcher-biolab-1',
           device_model='MaxOne',chip_id=a.chip,organoid_id=a.organoid,started_at=a.started_at,
           record_root='/data/local/disk1/mea_live_pair',bad_channels=[],threshold_value=5,
           threshold_h5_dataset='settings/spike_threshold',device_flow='braindevice_5555_7215_v1',
           device_launch='/home/researcher/brainDesktop/mea-standby-20261118/session.py')
    if a.replaces_session:b['replaces_session']=a.replaces_session
    binding(b)
    servers=json.loads((ROOT/'output/mea_spatial_gate_20261118/servers.json').read_text(encoding='utf-8-sig'))
    run(b,servers,ROOT/'output'/session)

if __name__=='__main__':main()
