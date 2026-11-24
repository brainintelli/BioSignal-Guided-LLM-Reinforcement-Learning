"""Audit the completed zero-only check, including every raw sample for clipping."""
import os
os.environ.setdefault('HDF5_PLUGIN_PATH','/pub/soft/MaxLab/so')
import json
from pathlib import Path
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parent
reports=list((ROOT/'zero_settle_results').glob('*/run.json'))
assert len(reports)==1
report=json.loads(reports[0].read_text())
assert report['status']=='completed' and not report['cleanup_errors']
file=reports[0].parent/'h5/stimulation.raw.h5'
with h5py.File(file,'r') as f:
    d=f['data_store/data0000'];g=d['groups/routed'];raw=g['raw'];frames=g['frame_nos'][:]
    fs=int(d['settings/sampling'][0]);m=d['settings/mapping'][:]
    channels=[int(x['channel']) for x in m if int(x['electrode'])!=7587]
    rail_counts=np.zeros(raw.shape[0],dtype=np.int64)
    rail_by_second=[]
    for start in range(0,raw.shape[1],fs*2):
        x=raw[:,start:min(start+fs*2,raw.shape[1])]
        rails=(x==0)|(x==1023);rail_counts+=np.count_nonzero(rails,axis=1)
        for a in range(0,rails.shape[1],fs):rail_by_second.append(int(np.count_nonzero(rails[channels,a:a+fs])))
    s=d['spikes'][:];valid=np.isin(s['channel'],channels)&(s['frameno']>=frames[0])&(s['frameno']<=frames[-1])
    errors=[]
    if f['wellplate/id'][0].decode()!='22021':errors.append('chip_identity')
    if len(m)!=744 or np.any(np.diff(frames)!=1):errors.append('mapping_or_frames')
    if np.any(rail_counts[channels]>0):errors.append('observed_clipping_remains')
    result={'status':'zero_only_check_passed' if not errors else 'zero_only_check_failed','errors':errors,
        'source':str(file),'chip_id':f['wellplate/id'][0].decode(),'duration_s':len(frames)/fs,
        'mapped_channels':len(m),'independent_channels':len(channels),'frame_gaps':int(np.count_nonzero(np.diff(frames)!=1)),
        'independent_clipped_samples_full_record':int(sum(rail_counts[channels])),
        'independent_channels_with_clipping_full_record':int(np.count_nonzero(rail_counts[channels])),
        'independent_rail_counts_per_second':rail_by_second,
        'active_independent_channels':len(np.unique(s['channel'][valid])),
        'independent_population_rate_hz':int(np.count_nonzero(valid))/(len(frames)/fs),
        'nonzero_stimulation_sent':False,'real_model_training_performed':False,
        'ten_minute_pre_stimulus_qualification_replaced':False,
        'limitations':['Short zero-only check supports a settling-time explanation; it does not prove a biological mechanism or safety.']}
(ROOT/'zero_settle_analysis.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='independent_rail_counts_per_second'},indent=2))
