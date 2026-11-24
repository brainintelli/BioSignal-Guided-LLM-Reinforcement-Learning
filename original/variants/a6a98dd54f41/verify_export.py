"""Independent vectorized reconstruction from the two original H5 files."""
import hashlib,json
from pathlib import Path
import h5py,numpy as np
r=Path(__file__).resolve().parent;source=r.parent/'mea_full_28080_20261117_1406/backup/recording_resume2';meta=json.loads((r/'data/metadata.json').read_text())
checks={}
for phase in ['before','after']:
    m=meta['phases'][phase];records=json.loads((r/'data'/f'{phase}.json').read_text())
    with h5py.File(source/m['source_raw_name'],'r') as f:
        d=f['data_store/data0000'];s=d['spikes'][:];lsb=float(d['settings/lsb'][0]);channels=np.array([c for c,e in meta['mapping']])
        selected=(s['frameno']>=m['start_frame'])&(s['frameno']<m['end_frame'])&np.isin(s['channel'],channels)
        s=s[selected];bins=(s['frameno']-m['start_frame'])//20000
        count=np.bincount(bins,minlength=600);summed=np.bincount(bins,weights=np.abs(s['amplitude'].astype('float64'))*lsb*1e6,minlength=600)
        rate=count/804.;amps=np.divide(summed,count,out=np.zeros(600),where=count>0)
        active=np.array([len(np.unique(s['channel'][bins==i]))/804 for i in range(600)])
        expected=np.stack([count,rate,active,amps],axis=1);actual=np.array([row['features'] for row in records])
        if not np.allclose(expected,actual,rtol=1e-12,atol=1e-12):raise RuntimeError('Independent features mismatch')
        exported=sorted(e for row in records for e in row['events'])
        original=sorted([int(x['frameno']),int(x['channel']),float(x['amplitude'])*lsb*1e6] for x in s)
        if exported!=original:raise RuntimeError('Independent events mismatch')
        checks[phase]=dict(events=len(s),windows=600,max_feature_error=float(np.max(np.abs(actual-expected))),passed=True)
result=dict(passed=True,checks=checks,manifest_sha256=hashlib.sha256((r/'data/manifest.json').read_bytes()).hexdigest(),hardware_commands_sent=0)
(r/'export_independent_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
