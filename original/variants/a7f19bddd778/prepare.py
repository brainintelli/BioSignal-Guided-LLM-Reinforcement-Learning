"""Prepare the one admissible historical probe as a fixed input for a game reward comparison."""
import hashlib
import json
from pathlib import Path
import random

import h5py
import numpy as np
from mea_state2_r2.features import NAMES, SCHEMA, digest, extract

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'mea_external_post_20261117_8ba00003'
prior=json.loads((SOURCE/'state2_provisional.json').read_text(encoding='utf8'))
with (SOURCE/'source.raw.h5').open('rb') as stream:
    source_sha=hashlib.file_digest(stream,'sha256').hexdigest()
assert source_sha==prior['source_sha256']=='8ba000037b053f47e70c9576808b9d6f2e3609bf6d85dd8e2ffa55496f465771'
with h5py.File(SOURCE/'source.raw.h5') as f:
    d=f['data_store/data0000']
    assert f['wellplate/id'][0].decode()=='22021'
    rate=int(d['settings/sampling'][0]);last=int(d['events']['frameno'].max())
    lo,hi=last+61*rate,last+62*rate
    mapping=d['settings/mapping'][:]
    selected=json.loads(f['assay/inputs/electrodes'][0].decode())['electrodes']['0']['stimulation']
    exc=np.unique(mapping['channel'][np.isin(mapping['electrode'],selected)])
    pairs=sorted([int(r['channel']),int(r['electrode'])] for r in mapping if r['channel'] not in exc)
    channels=[r[0] for r in pairs]
    assert pairs==prior['mapping'] and len(set(channels))==734
    frames=d['spikes']['frameno']
    assert int(frames.min())<=lo and int(frames.max())>=hi and rate==20000
    left,right=np.searchsorted(frames,[lo,hi])
    spikes=d['spikes'][left:right]
    scale=float(d['settings/lsb'][0])*1e6
    events=sorted([int(r['frameno']),int(r['channel']),float(r['amplitude'])*scale] for r in spikes if int(r['channel']) in channels)
assert len(events)==158 and np.allclose(events,prior['window']['events'],rtol=0,atol=1e-10)
features=list(extract(events,lo,hi,channels,rate))
norm=dict(rule='fixed min(1,feature), three decimals; no calibration fit',schema=SCHEMA,names=list(NAMES))
context=dict(chip='22021',source_sha256=source_sha,source_quality=prior['quality'],frame_start=lo,frame_end=hi,
             sampling_hz=rate,roi_sha=digest(pairs),events_sha=digest(events),count=len(events),
             features=features,normalized=[round(min(1.,x),3) for x in features],normalization=norm)
rng=random.Random(220211118)
seeds=rng.sample(range(91000000,91090000),8)
holdout=rng.sample(range(96000000,96100000),48)
episode_seeds=rng.sample(range(92000000,92100000),len(seeds)*3)
study=dict(chip='22021',context_sha=digest(context),study='single_fixed_spike_window_simulated_game_r2',starting_scene='supplied_95',
           starting_reserves=dict(health=100,thirst=95,hunger=95),arms=['r1','ordered'],
           training_seeds=seeds,episodes={str(s):[s,*episode_seeds[3*i:3*i+3]] for i,s in enumerate(seeds)},holdout_seeds=holdout,
           batch=12,updates=4,eval_decisions=6,alpha=1,beta=.001,feedback_stride=3,minimum_reserve=85,
           endpoints=['new_cells','r1_sum'],unit='paired differences crossed by training replicate and game world; biological n=1',
           interval=dict(method='crossed paired bootstrap on training seeds and holdout worlds',quantiles=[.0125,.9875],draws=30000,seed=220219),
           stability='Both simultaneous-interval lower bounds > 0 and at least 6 of 8 training seeds with strictly positive mean on both endpoints.',
           real_game_contingent_outputs=0,source_organism_count=1,source_window_count=1)
assert len(set(sum(study['episodes'].values(),[])+holdout))==80
for name,doc in [('context.json',context),('design.json',study)]:
    with (ROOT/name).open('x',encoding='utf8') as stream:
        json.dump(doc,stream,ensure_ascii=False,indent=2)
print('Verified context',context['normalized'],'seeds',len(seeds),len(holdout))
