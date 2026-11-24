"""Freeze a true-data-only 20-point plan from the already closed baseline."""
from pathlib import Path
import argparse,json,hashlib,sys,os
sys.path.insert(0,os.environ.get('MEA_V2_ROOT',str(Path(__file__).parent.parent/'mea_stimulation_v2')))
import numpy as np
from mea_stim_v2.config import Config
from mea_stim_v2.recording import H5Recording
from mea_stim_v2.qc import _stats,_common
from mea_stim_v2.journal import exclusive_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--session',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.session)
 c=Config.from_dict(json.loads((root/'bound.json').read_text()));selection=json.loads((root/'multipoint_selection.json').read_text());raw=Path(selection['source']['path'])
 with H5Recording(raw,c) as r:
  if r.source!=selection['source']:raise ValueError('Selection source changed')
  check=_common(r,c,'selection')
  if check['errors']:raise ValueError('Selection identity/metadata mismatch: '+str(check['errors']))
  markers=[e for e in r.events if e['id']==8002 and e['phase']=='connected' and e['session']==c.session_id]
  if len(markers)!=1:raise ValueError('Unique actual connected marker required')
  connected=markers[0]['time_s']
  if r.duration<connected+600:raise ValueError('Incomplete 480-second calibration/reference coverage')
  channels=r.channels.tolist();t=r.spike_times
  counts=np.array([np.count_nonzero((r.spike_channels==ch)&(t>=connected+120)&(t<connected+600)) for ch in channels]);valid=counts/480>=.05
  for offset in range(150,571,30):
   st=_stats(r.sample(channels,connected+offset));valid&=(~st['rails'])&(st['mad']>0)
  eligible=[ch for ch,ok in zip(channels,valid) if ok];mapping=dict(r.mapping)
  points=sorted([dict(channel=ch,electrode=mapping[ch],rate_hz=float(n/480)) for ch,n,ok in zip(channels,counts,valid) if ok],key=lambda d:(-d['rate_hz'],d['channel']))[:20]
  if points!=selection['selected']:raise ValueError('Frozen point selection differs')
  roi=sorted(set(eligible)-{x['channel'] for x in points});selected=t[np.isin(r.spike_channels,roi)]
  rates=[float(np.count_nonzero((selected>=connected+i)&(selected<connected+i+30))/30) for i in range(120,600,30)];reference=float(np.median(rates))
  if len(roi)<50 or reference<=0:raise ValueError('Insufficient readout ROI/reference')
  metadata=r.metadata
 h=hashlib.sha256()
 with raw.open('rb') as f:
  for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
 stat=raw.stat()
 if (stat.st_size,stat.st_mtime_ns)!=(selection['source']['size_bytes'],selection['source']['mtime_ns']):raise ValueError('Source changed during hashing')
 plan=dict(schema='mea.28073.multipoint.capability.v1',base_config=c.to_dict(),previous_run=str(root/'v2run_eventfix'),points=points,roi=roi,reference_rate=reference,
  reference_30s_rates=rates,selection_source=selection['source'],selection_source_sha256=h.hexdigest(),source_metadata=metadata,
  waveform=dict(target_mv=300,phase_samples=4,pulses_per_group=3,groups=5,pulse_period_samples=200,group_period_samples=4000,sham=False),
  user_instruction='20 active points, sequential; 3 pulses x 5 groups, 10 ms pulse onset interval, 0.20 s group onset interval; previous amplitude/phase retained; no sham',
  interval_definition='onset_to_onset; GUI convention',purpose='capability_only_not_efficacy',r2_enabled=False,training_enabled=False,
  prior_nonzero_pulses=0,maximum_true_pulses=300,hard_deadline_utc='2026-11-17T08:13:00Z',
  manufacturer_note='10ms departs from initial >=300ms recommendation; user explicitly selected finite trains; no biological safety/efficacy validation inferred')
 exclusive_json(a.output,plan);print(json.dumps({'points':len(points),'roi':len(roi),'reference_rate':reference,'source_sha256':h.hexdigest()}),flush=True)

if __name__=='__main__':main()
