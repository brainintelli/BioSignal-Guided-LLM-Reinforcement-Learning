import sys,os,json,hashlib,math
from pathlib import Path
here=Path.cwd()
extra=[here/'.tools/mea_baseline_deps',here/'standalone/mea_live_pair',here/'standalone/mea_stimulation_v2']
sys.path[:0]=[str(p) for p in extra]
os.environ['HDF5_PLUGIN_PATH']=str(here/'.tools/mea_maxwell_hdf5_plugin')
from types import SimpleNamespace
from mea_stim_v2.recording import H5Recording
from select_sites import candidates
out=Path(sys.argv[1])
b=json.loads((out/'baseline-binding.json').read_text(encoding='utf-8-sig'))
for raw,meta,stored in [('baseline-original.raw.h5','baseline-audit.json','site-candidates.json'),('baseline-final-route.raw.h5','routeB-audit.json','routeB-site-candidates.json')]:
    audit=json.loads((out/meta).read_text(encoding='utf-8-sig'))
    expected=json.loads((out/stored).read_text(encoding='utf-8-sig'))
    path=out/raw
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
    assert h.hexdigest()==audit['raw_sha256'] and path.stat().st_size==audit['size_bytes']
    with H5Recording(path,SimpleNamespace(threshold_h5_dataset=b['threshold_h5_dataset'])) as r:
        cs=candidates(r,b['chip_id'],b['bad_channels'])
        assert cs['suggested']==expected['suggested'] and cs['distances_um']==expected['distances_um'] and cs['candidates']==expected['candidates']
        print(raw,round(r.duration,3),len(cs['candidates']),len(r.events),h.hexdigest())
