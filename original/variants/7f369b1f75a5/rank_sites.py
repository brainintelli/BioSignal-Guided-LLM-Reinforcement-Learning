import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from mea_stim_v2.recording import H5Recording
from mea_stim_v2.qc import stability

def rank(path,binding):
    with H5Recording(path,SimpleNamespace(**binding)) as rec:
        if rec.metadata['chip_id']!=binding['chip_id'] or rec.events or rec.metadata['frame_gaps'] or rec.duration<60:
            raise ValueError('Invalid passive recording or chip identity')
        chs=sorted(rec.channels.tolist())
        counts=[np.bincount(rec.spike_channels[(rec.spike_times>=i)&(rec.spike_times<i+20)],minlength=1024) for i in (0,20,40)]
        rates=np.min(counts,axis=0)
        active=[c for c in chs if rates[c]>0]
        good=[]; trials=[];eligible=[]
        for start in range(0,len(active),64):
            batch=active[start:start+64];ref=rec.sample(batch,3)
            passed=[stability(ref,rec.sample(batch,t))['channel_pass'] for t in (23,43,57)]
            trials+=np.asarray(passed).sum(axis=0).tolist()
            for c,ok in zip(batch,np.all(passed,axis=0)):
                if ok:
                    row=dict(channel=c,electrode=dict(rec.mapping)[c],xy=rec.channel_xy_um[c],min_spikes=int(rates[c]))
                    good.append(row)
                    if row['electrode'] not in (7158,6026,3621):eligible.append(row)
        return dict(metadata=rec.metadata,mapped=len(chs),active=len(active),pass_hist=np.bincount(trials,minlength=4).tolist(),
                    stable=good,new=sorted(eligible,key=lambda x:(-x['min_spikes'],x['electrode'])))
if __name__=='__main__':
    folder=Path(sys.argv[1]);b=json.loads((folder/'baseline-binding.json').read_bytes())
    ans=rank(Path(sys.argv[2]),b)
    with (folder/(sys.argv[3]+'.json')).open('x') as f:json.dump(ans,f)
    print(json.dumps({k:ans[k] for k in ('mapped','active','pass_hist','new')}))
