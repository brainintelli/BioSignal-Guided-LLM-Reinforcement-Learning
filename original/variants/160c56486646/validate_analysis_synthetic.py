"""Offline known-answer analysis check. No MaxLab imports or network calls."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import h5py
import numpy as np

base=Path(__file__).resolve().parent
root=base/'synthetic_analysis_validation'
root.mkdir(exist_ok=False)
(root/'results/test/h5').mkdir(parents=True)
shutil.copyfile(base/'analyze_first_probe.py',root/'analyze_first_probe.py')
(root/'qualified_route.cfg').write_text('0(10900)0/0; payload\n')
fs=20000
duration=1240.1
event_s=np.arange(600,641,10)
spike_s=np.sort(np.concatenate([event_s-.1,event_s+.1,[100,110,120,800,810,820]]))
path=root/'results/test/h5/stimulation.raw.h5'
with h5py.File(path,'w') as f:
 d=f.create_group('data_store/data0000')
 settings=d.create_group('settings')
 for name,value in [('sampling',fs),('lsb',6.2942504e-6),('gain',512),('hpf',300)]:settings[name]=[value]
 settings['mapping']=np.array([(0,10900,0.,0.)],dtype=[('channel','i4'),('electrode','i4'),('x','f8'),('y','f8')])
 g=d.create_group('groups/routed')
 g.create_dataset('raw',shape=(1,int(duration*fs)),chunks=(1,20000),dtype='u2',fillvalue=512)
 g['frame_nos']=np.arange(int(duration*fs),dtype='u8')
 d['spikes']=np.array([(int(t*fs),0,-20.) for t in spike_s],dtype=[('frameno','i8'),('channel','i4'),('amplitude','f4')])
 d['events']=np.array([(int(t*fs),1,i+2) for i,t in enumerate(event_s)],dtype=[('frameno','i8'),('eventtype','u4'),('eventid','u4')])
 f['wellplate/id']=np.array([b'22554'])
report={'status':'completed','cleanup_errors':[],'h5_directory_on_server':str(path.parent),
 'parameters':{'stimulation_electrodes':[10900]},'actual_amplitude_schedule_mv':[298.1901]*5}
(root/'results/test/run.json').write_text(json.dumps(report))
subprocess.run([sys.executable,str(root/'analyze_first_probe.py')],check=True)
a=json.loads((root/'analysis/analysis.json').read_text())
assert a['technical_qc_passed']
assert a['event_count']==5 and a['event_intervals_seconds']==[10]*4
assert all(t['pre_480ms_population_spikes']==1 and t['post_20_to_500ms_population_spikes']==1 for t in a['trials'])
assert a['source_h5_sha256']
assert (root/'analysis/first_probe_comparison.png').stat().st_size>10000
print('SYNTHETIC_ONLY_KNOWN_ANSWER_PASS; NOT REAL EXPERIMENT DATA')
