import sys,os, json, math, hashlib, collections
from pathlib import Path
from types import SimpleNamespace
root=Path.cwd()
base=root/'output/pair28073-20261118T070238Z'
sys.path[:0]=[str(root/'.tools/mea_baseline_deps'),str(base/'mea-readback-runtime/mea_stimulation_v2')]
os.environ['HDF5_PLUGIN_PATH']=str(root/'.tools/mea_maxwell_hdf5_plugin')
import numpy as np
from mea_stim_v2.recording import H5Recording
from mea_stim_v2.qc import stability
p=base/'readout-feasibility';p.mkdir(exist_ok=True)
def write(name,d):
 with (p/name).open('x',encoding='utf8') as f:json.dump(d,f,indent=2,allow_nan=False)
cfg=SimpleNamespace(threshold_h5_dataset='settings/spike_threshold')
sites={793,955,968}
def scores(r,roi):
 ref=r.sample(roi,3.); ss=[stability(ref,r.sample(roi,t)) for t in (20.,40.,55.)]
 return ss
def show(ss):
 return [{'stable':sum(s['channel_pass']),'n':len(s['channel_pass']),'pct':round(100*s['fraction_channels_passed'],2)} for s in ss]
with H5Recording(base/'baseline-final-route.raw.h5',cfg) as r:
 a=sorted(set(map(int,r.spike_channels))-sites)
 ss=scores(r,a)
 stable=[c for j,c in enumerate(a) if all(v['channel_pass'][j] for v in ss)]
 endpoints=dict(rule='Select native-spike active independent channels passing all three original stability windows (3s reference; 20,40,55s offsets) on first passive final-route file. Freeze before applying to connected baseline. Exclude three physical stimulation channels. Minimum 50 channels. Independent gate remains 95% per window.',source=r.source,roi=stable,n_selected=len(stable),n_active_source=len(a),source_windows=show(ss),minimum_roi=50)
 write('channel-selection.json',endpoints)
with H5Recording(base/'readback-run/connected-baseline.raw.h5',cfg) as r:
 ss=scores(r,stable) if stable else []
 active=sorted(set(map(int,r.spike_channels))-sites)
 allwin=scores(r,active)
 counts=[]
 for s in allwin:
  pairs=collections.Counter()
  for fail,old,new in zip(s['channel_pass'],s['before_mad_adc'],s['after_mad_adc']):
   if not fail:pairs[(round(old/1.4826,2),round(new/1.4826,2))]+=1
  counts.append([(str(k),n) for k,n in pairs.most_common(5)])
 hold={'connected_source':r.source,'n_independent_fixed':len(stable),'independent_test_windows':show(ss),'minimum_roi_met':len(stable)>=50,'all_stability_windows_met':bool(ss) and all(s['passed'] for s in ss),'all_active_mad_transition_failures_counts':counts,'test_type':'retrospective file-level replay; original connected outcomes were already reported','eligible_for_new_live_evaluation':len(stable)>=50 and bool(ss) and all(s['passed'] for s in ss)}
 write('separate-record-check.json',hold)
 print(json.dumps({'selection':len(stable),'total_active_source':len(a),**hold},ensure_ascii=True))
