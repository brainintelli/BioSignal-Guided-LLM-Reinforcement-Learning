"""Verify locally readable source copies and the diagnostic plot's actual data."""
import datetime as dt
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
os.environ['HDF5_PLUGIN_PATH']=str(PROJECT/'.tools/mea_maxwell_hdf5_plugin')
import h5py
import numpy as np


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    receipts=[read(ROOT/name) for name in ['identity_backup_verification.json','baseline_backup_verification.json','zero_backup_verification.json']]
    traces=np.load(ROOT/'settling_diagnostic_traces.npz')
    files=[]
    for receipt in receipts:
        path=ROOT/receipt['file']
        assert receipt['verified'] and path.stat().st_size==receipt['bytes']
        with h5py.File(path,'r') as f:
            d=f['data_store/data0000'];g=d['groups/routed'];frames=g['frame_nos'][:]
            m=d['settings/mapping'][:]
            assert f['wellplate/id'][0].decode()=='22021'
            assert len(m)==744 and int(m[m['electrode']==7587]['channel'][0])==691
            assert np.all(np.diff(frames)==1)
            which=None
            if 'connected_baseline' in receipt['file']:which='before'
            if 'zero_settle' in receipt['file']:which='after'
            if which:assert np.array_equal(g['raw'][296,:80000],traces[which])
            files.append({'file':receipt['file'],'bytes':receipt['bytes'],'sha256':receipt['sha256'],
                          'duration_s':len(frames)/float(d['settings/sampling'][0]),'frame_gaps':0,'chip_id':'22021','mapped_channels':744})
    assert read(ROOT/'connected_baseline_qc.json')['status']=='failed'
    assert read(ROOT/'zero_settle_analysis.json')['independent_clipped_samples_full_record']==0
    assert read(ROOT/'no_stimulation_audit.json')['sequence_sends']==['sequence_send photo_22021_event_only_7001']
    zero=read(ROOT/'zero_only_cleanup_audit.json')
    assert zero['nonzero_dac_commands']==[] and zero['all_32_units_off_ack'] and zero['final_dac_zero_ack']=='OK'
    meta=read(ROOT/'metadata_backup_verification.json')
    assert meta['all_metadata_verified'] and not meta['nonzero_send_claim_exists']
    proposal=read(ROOT/'short_option/registration.json')
    assert not proposal['short_protocol_authorized'] and proposal['status']=='expired_without_user_selection'
    result={'passed':True,'verified_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'raw_files':files,
            'raw_bytes_total':sum(x['bytes'] for x in files),'recorded_seconds_total':sum(x['duration_s'] for x in files),
            'raw_sha256_verified_by_transfer_receipts':True,'local_full_frame_continuity_rechecked':True,
            'diagnostic_traces_exactly_match_local_raw':True,'metadata_files_verified':len(meta['files']),
            'nonzero_stimulation_sent':False,'model_training_performed':False,'alpha':1,'r2_reward_enabled':False,'realtime_training':False}
    (ROOT/'backup_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
