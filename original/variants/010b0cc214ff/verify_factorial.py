"""Reconstruct the registered crossover from independent device frames and the C++ stream."""
import csv
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from mea_stim_v2.recording import H5Recording

KINDS = ('anchor_regular','anchor_jitter','new_regular','new_jitter','sham')
CONTRASTS = ('position','rhythm','interaction')
read = lambda path: json.loads(path.read_bytes())

def contrasts(values):
    ar, aj, nr, nj, _ = (values[c] for c in KINDS)
    return dict(position=(nr+nj-ar-aj)/2, rhythm=(aj+nj-ar-nr)/2, interaction=(nj-nr)-(aj-ar))

def sign_test(x):
    good=sum(i>0 for i in x);bad=sum(i<0 for i in x);n=good+bad
    return min(1.,2*sum(math.comb(n,i) for i in range(min(good,bad)+1))/(2**n))

def audit(r):
    bind=read(r/'binding.json');p=read(r/'plan.json');d=read(r/'design.json');route=read(r/'route.json')
    end=read(r/'end.json')
    if not end['closed'] or end['cleanup'] or end.get('error') or end['cpp_error'] or end['completed']!=20:
        raise ValueError('Complete factorial analysis needs 20 valid trials and confirmed closure')
    assert bind['session_id']==p['session'] and bind['chip_id']==p['chip']=='28080'
    assert end['attempts_total']==28 and end['new_nonzero_attempts']==16 and d['trials']==p['trials']
    assert {x.name for x in r.glob('attempt-*.json')}=={f'attempt-{i:02d}.json' for i in range(13,29)}
    stream=[json.loads(line) for line in (r/'cpp-spikes.jsonl').read_bytes().splitlines()]
    assert len(stream)==end['cpp_rows'];roi=d['roi'];included=set(roi);assert len(roi)>=50
    obs=np.sort([row[0] for row in stream if row[1] in included])
    acks=[datetime.fromisoformat(a['utc']).timestamp() for line in (r/'api_audit.jsonl').read_bytes().splitlines()
          if (a:=json.loads(line))['kind']=='ack' and a['command'].startswith('sequence_send ') and a['reply']=='OK']
    assert len(acks)==20
    config=SimpleNamespace(threshold_h5_dataset=bind['threshold_h5_dataset'])
    ch=[route['channels'][site] for site in 'HWF'];assert not included&set(ch)
    rows=[];inventory={};mapping=None;previous=None
    for i in range(21):
        stem=f'trial_{i:02d}' if i else 'warmup';path=r/stem/(stem+'.raw.h5')
        with path.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
        inventory[path.relative_to(r).as_posix()]=dict(sha256=sha,bytes=path.stat().st_size)
        with H5Recording(path,config) as rec:
            m=rec.metadata;assert m['chip_id']==bind['chip_id'] and m['fs']==20000 and m['frame_gaps']==0
            assert m['gain']==512 and m['hpf']==300 and m['threshold_value']==bind['threshold_value']
            assert previous is None or previous<rec.first_frame
            previous=rec.last_frame
            if not i:
                assert not rec.events and rec.duration>=60
                mapping=sorted(rec.mapping)
                assert [dict(mapping)[k] for k in ch]==[p['sites'][x] for x in 'HWF']
                continue
            assert sorted(rec.mapping)==mapping
            plan=p['trials'][i-1];e=read(r/(stem+'-evidence.json'));stim=plan['events']
            assert e['plan']==plan and e['pattern_verified'] and e['intended_edges_present'] and not e['late_site_rail']
            first=rec.events[0];t=first['time_s'];assert len(rec.events)==(len(stim) or 1) and t>=6 and rec.duration>=t+13
            raw=rec.sample(ch,t-.004,.7);rails=(raw==0)|(raw==1023);assert not rails[:,-1000:].any()
            for j,((site,offset),mark) in enumerate(zip(stim or [('quiet',0)],rec.events)):
                assert mark['id']==4*i+j and mark['eventtype']==1 and mark['frame']==first['frame']+offset and mark['message']['site']==site
                assert int(mark['message']['trial'])==i and int(mark['message']['idx'])==j
                if stim:
                    at=80+offset
                    assert [s for k,s in enumerate('HWF') if not rails[k,at-8:at].any() and rails[k,at:at+40].any()]==[site]
                else:
                    assert (np.abs(raw.astype(float)-np.median(rec.sample(ch,t-3),axis=1)[:,None])*m['lsb_uv']).max()<1000
            ack=acks[i-1]; assert 0<=e['ack_end']-ack<1 and obs[0]<(ack-6)*1000 and obs[-1]>(ack+13)*1000
            times=rec.spike_times[np.isin(rec.spike_channels,roi)]
            h5=[int(((times>=t+lo)&(times<t+hi)).sum()) for lo,hi in d['windows']]
            cpp=[int(np.searchsorted(obs,(ack+hi)*1000)-np.searchsorted(obs,(ack+lo)*1000)) for lo,hi in d['windows']]
            assert h5==[v['spikes'] for v in e['windows']]
            rows.append(dict(index=i,block=plan['block'],condition=plan['condition'],pulses=len(stim),cpp_pre=cpp[0],cpp_post=cpp[1],
                             h5_pre=h5[0],h5_post=h5[1],cpp_delta=cpp[1]-cpp[0],h5_delta=h5[1]-h5[0],
                             cpp_late=cpp[2],h5_late=h5[2],recovery_fraction=e['recovery_fraction']))
    assert sum(x['pulses'] for x in rows)==48
    groups={k:{} for k in KINDS};effects={k:{} for k in CONTRASTS}
    for channel in ('cpp','h5'):
        blocks=[{v['condition']:v[channel+'_delta'] for v in rows if v['block']==b} for b in range(1,5)]
        for k in KINDS:
            pre=[v[channel+'_pre'] for v in rows if v['condition']==k];post=[v[channel+'_post'] for v in rows if v['condition']==k]
            diff=[block[k]-block['sham'] for block in blocks]
            groups[k][channel]=dict(pre=sum(pre)/4,post=sum(post)/4,delta=sum(post)/4-sum(pre)/4,vs_sham=diff,mean_vs_sham=sum(diff)/4)
        for k in CONTRASTS:
            x=[contrasts(b)[k] for b in blocks]
            effects[k][channel]=dict(blocks=x,mean=sum(x)/4,p_sign=sign_test(x))
        largest=0
        for j,k in enumerate(sorted(CONTRASTS,key=lambda k: effects[k][channel]['p_sign'])):
            item=effects[k][channel];largest=max(largest,min(1.,(3-j)*item['p_sign']));item['p_holm']=largest
    for k in CONTRASTS:
        both=effects[k]['cpp']['blocks']+effects[k]['h5']['blocks']
        effects[k]['consistent_direction']=all(v>0 for v in both) or all(v<0 for v in both)
    result=dict(session=p['session'],finished=end['finished'],cpp_rows=end['cpp_rows'],roi=len(roi),trials=len(rows),new_pulses=48,
                cumulative_pulses=84,raw_bytes=sum(x['bytes'] for x in inventory.values()),raw_count=len(inventory),groups=groups,effects=effects)
    for name,obj in [('raw-hashes.json',inventory),('factorial-results.json',result)]:
        (r/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf8')
    with (r/'response-by-trial.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(result),flush=True)
if __name__=='__main__':audit(Path(sys.argv[1]))
