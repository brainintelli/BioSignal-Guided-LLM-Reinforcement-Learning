"""Freeze stricter common-mode exclusion before any held-out game was evaluated."""
from register import out,rd,store
import copy
import hashlib
import json
import sys
sys.path.insert(0,str(out.parents[1]/'standalone/mea_state2_r2'))
from mea_state2_r2.features import digest

p=rd(out/'paired-protocol.json')
d=copy.deepcopy(rd(out/'paired-windows.json'))
s=rd(out/'common-mode-sensitivity.json')
p['common_mode_blanking']=dict(rule='Within the retained panel subtract each channel median, then remove native spikes within +/-40 frames of absolute across-channel median >=4 ADC codes. Apply to pre and post identically. Four features calculated over the original 1 second with no imputation.',
                               frame_rate=20000,source_sha256=hashlib.sha256((out/'common-mode-sensitivity.json').read_bytes()).hexdigest(),
                               supersedes='Paired-only exploratory inputs; amended before any held-out gameplay outcomes.')
for site,record in d.items():
    for dt,w in zip(p['offsets'],record['windows']):
        w['features']=s['results'][site][str(dt)]['features']
        w['id']=digest({k:v for k,v in w.items() if k!='id'})
    record['prime']=record['windows'][0].copy()
store(out/'clean-protocol.json',p)
store(out/'clean-windows.json',d)
