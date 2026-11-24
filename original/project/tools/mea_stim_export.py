"""Export a fixed, audited pair of600s recorded segments. No acquisition access."""
import argparse,json,math
from pathlib import Path
import h5py,numpy as np
from agent.mea_stim_offline import RAW_PINS,features,StimRecording
from agent.mea_recording import sha256
from agent.mea_protocol import canonical

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    meta=dict(chip_id='28080',sampling_hz=20000,post_guard_seconds=60,raw_sha256=RAW_PINS,phases={})
    for phase,name in [('before','baseline.raw.h5'),('after','stimulus_post.raw.h5')]:
        path=a.source/name
        if sha256(path)!=RAW_PINS[phase]:raise ValueError('Raw source hash mismatch')
        with h5py.File(path,'r') as f:
            d=f['data_store/data0000'];g=d['groups/routed'];raw=g['raw'];fr=g['frame_nos']
            if f['wellplate/id'][0].decode()!='28080' or float(d['settings/sampling'][0])!=20000:raise ValueError('Identity/rate')
            mapping=sorted((int(x['channel']),int(x['electrode'])) for x in d['settings/mapping'][:])
            if len(mapping)!=805 or (640,8858) not in mapping:raise ValueError('Unexpected map')
            mapping=[x for x in mapping if x[0]!=640]
            if phase=='before':meta['mapping']=mapping;start=int(fr[0])
            else:
                if mapping!=meta['mapping']:raise ValueError('Changed mapping')
                events=d['events'][:]
                if len(events)!=15 or list(events['eventid'])!=list(range(2,17)):raise ValueError('Stimulus events')
                offsets=[b*4000+p*200 for b in range(5) for p in range(3)]
                if list(events['frameno']-events['frameno'][0])!=offsets:raise ValueError('Stimulus timing')
                tail=int(events[-1]['frameno'])+8;start=tail+60*20000
                meta['stimulus_tail_frame']=tail
            end=start+600*20000;base=int(fr[0]);lo=start-base
            if lo<0 or end>int(fr[-1])+1 or raw.shape!=(1024,len(fr)) or not np.array_equal(g['channels'][:].reshape(-1),np.arange(1024)):
                raise ValueError('Raw interval/row shape')
            lsb=float(d['settings/lsb'][0]);channels=np.array([c for c,e in mapping]);rail_max=0.
            for i in range(600):
                left=lo+i*20000;block=raw[:,left:left+20000][channels,:]
                fn=fr[left:left+20000]
                if not np.array_equal(fn,np.arange(start+i*20000,start+(i+1)*20000)):raise ValueError('Frame discontinuity')
                if np.any(block>1023):raise ValueError('ADC bounds')
                rail=float(((block==0)|(block==1023)).mean(axis=1).max());rail_max=max(rail_max,rail)
                if rail>.01:raise ValueError('Raw rail quality failed')
                if i%60==0:print(canonical(dict(phase=phase,raw_seconds_checked=i,rail_max=rail_max)),flush=True)
            s=d['spikes'][:];s=s[(s['frameno']>=start)&(s['frameno']<end)&np.isin(s['channel'],channels)]
            exported=[]
            for i in range(600):
                left=start+i*20000;right=left+20000;chosen=s[(s['frameno']>=left)&(s['frameno']<right)]
                events=sorted([int(x['frameno']),int(x['channel']),float(x['amplitude'])*lsb*1e6] for x in chosen)
                vals=features(events,channels=804,frames=20000,rate=20000)
                exported.append(dict(start=left,end=right,events=events,features=vals))
            meta['phases'][phase]=dict(start_frame=start,end_frame=end,lsb_volts=lsb,event_count=len(s),
                raw_quality=dict(status='passed',checked_seconds=600,frame_gaps=0,max_rail_fraction=rail_max),
                source_raw_name=name,history='after_single_site15pulses' if phase=='after' else 'before_single_site15pulses')
            (a.output/(phase+'.json')).write_text(canonical(exported)+'\n',encoding='utf-8')
    (a.output/'metadata.json').write_text(canonical(meta)+'\n',encoding='utf-8')
    manifest=dict(schema='mea.stim.sealed.windows.v1',files={p.name:dict(size=p.stat().st_size,sha256=sha256(p)) for p in sorted(a.output.glob('*.json'))})
    (a.output/'manifest.json').write_text(canonical(manifest)+'\n',encoding='utf-8');pin=sha256(a.output/'manifest.json')
    for phase in RAW_PINS:StimRecording(a.output,pin=pin,phase=phase)
    print(canonical(dict(status='exported_and_reconstructed',manifest_sha256=pin,phases=meta['phases'])),flush=True)

if __name__=='__main__':main()
