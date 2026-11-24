"""Finite, sealed before/after recording protocol. No hardware dependencies."""
import json,math
from pathlib import Path
from statistics import mean
from agent.mea_protocol import digest
from agent.mea_recording import sha256
from agent.mea_replay import RecordingSnapshot
from agent.mea_snapshot import Normalization
from agent.mea_long_ablation import ORIGINAL_TENSORS,paired_stats

RAW_PINS={'before':'0c5bf149410f41670b4b54b94ff241b088b8c79d1c13e13c03b44bf72687a14c',
          'after':'139cb2cb35b8a89ce8f4019e81c54d2880c82fe9200d1b85d0f6cc7ef198eb3e'}
PROTOCOL=dict(schema='mea.stim.offline.v1',chip_id='28080',arms=['none','before','after'],blocks=8,
    rounds=3,train_decisions=384,decisions_per_round=128,eval_decisions=150,minimum_windows=600,train_current=[0,384],bootstrap_index=384,
    train=[0,385],gap=[385,445],heldout=[445,596],post_artifact_guard_s=60,
    original_tensors=ORIGINAL_TENSORS,initial_seed=6100000,analysis_seed=6800100,
    primary='final_mean_discounted_R1_across_common_before_after_heldout_contexts',
    alpha=1.,r2_enabled=False,realtime_training=False,device_commands_sent=0,
    biological_replications=1,normalization='identity_engineering_only')

def assignment(split):
    if split=='train':return list(range(385))
    if split=='eval':return list(range(445,596))
    raise ValueError('Unknown split')

def cell_config(block,arm):
    if type(block) is not int or not 0<=block<8 or arm not in PROTOCOL['arms']:raise ValueError('Unregistered cell')
    return dict(block=block,arm=arm,policy_arm='none' if arm=='none' else 'real',initial_seed=6100000,
        world_seed=7100100+1000*block,action_seed=7200100+10000*block,ppo_seed=7300100+100*block,
        eval_world_seed=7400100+block,eval_action_seed=7500100+10000*block)

def validate_budget(decisions,steps):
    if type(decisions) is not int or decisions!=384 or len(steps)!=3:raise ValueError('Train budget mismatch')
    for s in steps:
        if s!={'actor':2,'critic':16} or any(type(v) is not int for v in s.values()):raise ValueError('Optimizer budget mismatch')

def features(events,*,channels,frames,rate,lsb_volts=1e-6):
    if channels<=0 or frames<=0 or rate<=0 or not math.isfinite(lsb_volts) or lsb_volts<=0:raise ValueError('Feature scale')
    count=len(events)
    return [float(count),count*rate/(frames*channels),len({x[1] for x in events})/channels,
            sum(abs(x[2])*lsb_volts*1e6 for x in events)/count if count else 0.]

def validate_window(row,index,start,mapping):
    lo=start+index*20000;hi=lo+20000;channels={c for c,e in mapping}
    if row['start']!=lo or row['end']!=hi:raise ValueError('Window frame mismatch')
    events=row['events']
    if events!=sorted(events) or any(type(f) is not int or type(c) is not int or not lo<=f<hi or c not in channels or not math.isfinite(a) for f,c,a in events):
        raise ValueError('Invalid event frame/channel/amplitude')
    expected=features(events,channels=len(mapping),frames=20000,rate=20000)
    if len(row['features'])!=4 or any(not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12) for a,b in zip(row['features'],expected)):
        raise ValueError('Feature mismatch')

class StimRecording:
    def __init__(self,root,*,pin,phase,split=None):
        self.root=Path(root);self.pin=pin;self.phase=phase;self.split=split
        if phase not in RAW_PINS or split not in (None,'train','eval'):raise ValueError('Phase/split')
        if sha256(self.root/'manifest.json')!=pin:raise ValueError('Manifest changed')
        manifest=json.loads((self.root/'manifest.json').read_text())
        if manifest['schema']!='mea.stim.sealed.windows.v1':raise ValueError('Wrong source schema')
        for name in ['metadata.json',phase+'.json']:
            if sha256(self.root/name)!=manifest['files'][name]['sha256']:raise ValueError('Source changed')
        self.meta=json.loads((self.root/'metadata.json').read_text());self.rows=json.loads((self.root/(phase+'.json')).read_text())
        if self.meta['raw_sha256']!=RAW_PINS or self.meta['chip_id']!='28080' or self.meta['sampling_hz']!=20000 or self.meta['post_guard_seconds']!=60:
            raise ValueError('Wrong source identity')
        self.mapping=[tuple(x) for x in self.meta['mapping']]
        if len(self.mapping)!=804 or len({c for c,e in self.mapping})!=804 or any(c==640 for c,e in self.mapping):raise ValueError('Analysis mapping')
        if len(self.rows)!=600:raise ValueError('Need exactly600 complete windows')
        self.start=self.meta['phases'][phase]['start_frame'];self.window_count=600
        if self.meta['phases'][phase]['raw_quality']['status']!='passed':raise ValueError('Raw quality failed')
        for i,row in enumerate(self.rows):validate_window(row,i,self.start,self.mapping)
        self.recording_sha256=manifest['files'][phase+'.json']['sha256'];self.map_id=digest(self.mapping)
    def snapshot(self,index,context_id):
        if type(index) is not int or not 0<=index<600 or (self.split and index not in assignment(self.split)):raise ValueError('Outside finite split')
        row=self.rows[index]
        return RecordingSnapshot(self.recording_sha256,'stim28080_'+self.phase,context_id,self.map_id,index,0,804,20000,
            row['start'],row['end'],digest(row['events']),tuple(row['features']),Normalization.identity())
    def summary(self):
        return dict(schema='mea.stim.recording.v1',manifest_sha256=self.pin,source='recording',phase=self.phase,chip_id='28080',
            window_count=600,window_frames=20000,channels=804,split=self.split,raw_sha256=RAW_PINS[self.phase],
            historical_stimulation=self.phase=='after',historical_pulses=15 if self.phase=='after' else 0,
            stimulation_during_training=False,device_commands_sent=0,realtime_training=False,
            association='sequential_offline',normalization='identity_engineering_only')

def analyze(cells):
    indexed={}
    for c in cells:
        cell_config(c['block'],c['arm']);key=(c['block'],c['arm'])
        if key in indexed:raise ValueError('Duplicate cell')
        for p in ['pre','post']:
            if set(c[p])!={'before','after'} or any(not math.isfinite(x) for x in c[p].values()):raise ValueError('Invalid contexts/outcomes')
        indexed[key]={p:mean(c[p].values()) for p in ['pre','post']}
    missing=[[b,a] for b in range(8) for a in PROTOCOL['arms'] if (b,a) not in indexed]
    if missing:return dict(status='incomplete',positive_evidence=False,missing=missing)
    contrasts={}
    for a in ['before','none']:
        post=[indexed[b,'after']['post']-indexed[b,a]['post'] for b in range(8)]
        gain=[post[b]-(indexed[b,'after']['pre']-indexed[b,a]['pre']) for b in range(8)]
        contrasts['after_minus_'+a]=dict(post=paired_stats(post),gain=paired_stats(gain))
    positive=all(c['post']['mean_difference']>0 and c['post']['bootstrap_95'][0]>0 and c['post']['p_bonferroni_two_contrasts']<.05 for c in contrasts.values())
    return dict(status='complete',positive_evidence=positive,contrasts=contrasts,
        arms={a:{p:mean(indexed[b,a][p] for b in range(8)) for p in ['pre','post']} for a in PROTOCOL['arms']},
        biological_replications=1,inference_scope='digital_seeds_conditional_on_one_before_after_recording',biological_causality_verified=False)
