"""Pinned filtered H5 event view; explicit failed full-source QC provenance."""
import argparse,hashlib,json,math
from pathlib import Path
from sr_protocol import PROTOCOL,fit_profile,score,features

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:f.write(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def scoped_events(events,lo,hi,channels):
    mapped=set(channels);valid=[];excluded=[];seen=set();last=None
    for index,frame,ch,amp in events:
        if any(type(v) is not int for v in (index,frame,ch)) or not math.isfinite(amp):raise ValueError('event identity/amplitude invalid')
        if (frame,ch) in seen or last is not None and frame<last:raise ValueError('duplicate/unordered event')
        seen.add((frame,ch));last=frame
        if lo<=frame<hi and ch in mapped:valid.append([index,frame,ch,amp])
        else:excluded.append(index)
    return valid,excluded

def window_rows(valid,start,end,channels,lsb):
    rows=[];count=(end-start-40000)//20000
    for i in range(count):
        lo=start+20000+i*20000;hi=lo+20000;ee=[e for e in valid if lo<=e[1]<hi];bins={}
        for _,frame,ch,amp in ee:bins.setdefault((frame-lo)//200,set()).add(ch)
        co=sum(len(bins[(e[1]-lo)//200])>=2 for e in ee)
        rows.append(dict(index=i,start_frame=lo,end_frame=hi,event_indices=[e[0] for e in ee],events_sha256=digest(ee),
            spike_count=len(ee),rate_hz_per_channel=len(ee)/len(channels),active_channel_fraction=len({e[2] for e in ee})/len(channels),
            mean_abs_amplitude_uv=sum(abs(e[3]) for e in ee)/len(ee)*lsb*1e6 if ee else 0.,coincident_event_fraction=co/len(ee) if ee else 0.))
    return rows

def build(source,output):
    import h5py,numpy as np
    source=Path(source);assert source.stat().st_size==1469210547 and sha(source)==PROTOCOL['source_sha256']
    with h5py.File(source,'r') as f:
        assert list(f['data_store'])==['data0000'];s=f['data_store/data0000'];raw=s['groups/routed'];assert list(s['groups'])==['routed']
        for p in (raw['frame_nos'],raw['channels'],s['well_id']):assert p.dtype.kind in 'iu'
        for p,fields in [(s['spikes'],('frameno','channel')),(s['settings/mapping'],('channel','electrode'))]:
            for name in fields:assert p.dtype.fields[name][0].kind in 'iu'
        assert f['wellplate/id'][0]==b'22554' and int(s['well_id'][0])==0
        frames=raw['frame_nos'][:];assert len(frames)==6001600 and np.all(frames[1:]==frames[:-1]+1)
        start,end=int(frames[0]),int(frames[-1])+1;rate=float(s['settings/sampling'][0]);lsb=float(s['settings/lsb'][0]);assert rate==20000 and lsb>0
        mapping=[[int(v['channel']),int(v['electrode'])] for v in s['settings/mapping']];channels=[v[0] for v in mapping]
        assert len(channels)==len(set(channels))==len({v[1] for v in mapping})==850 and set(channels)==set(map(int,raw['channels'][:]))
        assert list(raw['raw'].shape)==[850,6001600]
        events=[[i,int(e['frameno']),int(e['channel']),float(e['amplitude'])] for i,e in enumerate(s['spikes'])]
    valid,excluded=scoped_events(events,start,end,channels);assert len(events)==57652 and len(valid)==57517 and len(excluded)==135
    assert sum(e[2] not in channels for e in events)==133 and sum(not start<=e[1]<end for e in events)==2
    windows=window_rows(valid,start,end,channels,lsb);assert len(windows)==298;profile=fit_profile(windows)
    value=dict(schema='mea.filtered22554.dataset.v1',source=dict(sha256=PROTOCOL['source_sha256'],bytes=source.stat().st_size,chip_id='22554',well_id=0,sample_rate_hz=rate,
        raw_interval=[start,end],raw_frames=6001600,frames_contiguous=True,raw_frames_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),mapping=mapping,
        channel_map_sha256=digest(mapping),lsb_volts=lsb,source_full_structural_passed=False,scope='mapped-channel raw-frame subset only',
        excluded_event_indices=excluded,unmapped_events=133,out_of_range_events=2,stimulation='user_confirmed_none',paired_sdk_stream=False,biological_recordings=1),
        events=events,windows=windows,profile=profile,scores=[score(w,profile) for w in windows],normalization=[features(w,profile) for w in windows],protocol=PROTOCOL)
    assert sha(source)==PROTOCOL['source_sha256'];write(output,value);print(canonical(dict(dataset_sha256=sha(output),windows=len(windows),excluded=len(excluded),valid_events=len(valid),source_full_qc_passed=False)))

class Data:
    def __init__(self,path,pin):
        if sha(path)!=pin:raise ValueError('dataset pin mismatch')
        self.pin=pin;self.data=read(path);d=self.data;s=d['source']
        if d['schema']!='mea.filtered22554.dataset.v1' or d['protocol']!=PROTOCOL or s['sha256']!=PROTOCOL['source_sha256'] or s['source_full_structural_passed'] is not False:raise ValueError('source/protocol mismatch')
        valid,excluded=scoped_events(d['events'],*s['raw_interval'],[p[0] for p in s['mapping']])
        if len(d['events'])!=57652 or len(valid)!=57517 or excluded!=s['excluded_event_indices'] or len(excluded)!=135:raise ValueError('scope changed')
        rows=window_rows(valid,*s['raw_interval'],[p[0] for p in s['mapping']],s['lsb_volts'])
        if rows!=d['windows'] or len(rows)!=298 or fit_profile(rows)!=d['profile']:raise ValueError('window/fit mismatch')
        if [score(w,d['profile']) for w in rows]!=d['scores'] or [features(w,d['profile']) for w in rows]!=d['normalization']:raise ValueError('derived input mismatch')
    def snapshot(self,index,context):
        if type(index) is not int or not 0<=index<298:raise ValueError('finite window index required')
        w=self.data['windows'][index];s=self.data['source'];v=dict(dataset_sha256=self.pin,source_sha256=s['sha256'],index=index,context_id=context,
            start_frame=w['start_frame'],end_frame=w['end_frame'],events_sha256=w['events_sha256'],channel_map_sha256=s['channel_map_sha256'],
            raw_features={n:w[n] for n in ('spike_count','rate_hz_per_channel','active_channel_fraction','mean_abs_amplitude_uv')},
            state_values=self.data['normalization'][index],score=self.data['scores'][index],association='prerecorded_action_independent',coverage_basis='filtered_finalized_h5_subset',source_full_qc_passed=False)
        return json.loads(canonical(v|dict(snapshot_id=digest(v))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args();build(a.source,a.output)
