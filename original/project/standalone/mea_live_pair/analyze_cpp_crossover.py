"""Analyze the prospective single-chip crossover and export leakage-separated replay windows."""
import argparse
from datetime import datetime
from itertools import combinations,permutations
import json
import math
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from livepair.figures import _plot_style
from mea_state2_r2.features import digest,extract,fit


LABELS=('regular','irregular','sham')
NAMES=('固定位点规律刺激','多点随机刺激','零输出对照')


def analyze(folder):
    def read(name):return json.loads((folder/name).read_text(encoding='utf-8'))
    design=read('design.json');end=read('end.json');roi=design['roi'];include=set(roi)
    trials=[read(p.name) for p in sorted(folder.glob('trial_*-evidence.json'))]
    chip=str(read('binding.json')['chip_id'])
    if str(read('warmup.json')['metadata']['chip_id'])!=chip or any(str(t['metadata']['chip_id'])!=chip for t in trials):
        raise ValueError('Warmup or trial chip identity differs from the registered chip')
    acks=[datetime.fromisoformat(r['utc']).timestamp() for l in (folder/'api_audit.jsonl').read_text().splitlines()
          if (r:=json.loads(l))['kind']=='ack' and r['command'].startswith('sequence_send ') and r['reply']=='OK']
    stream=[json.loads(l) for l in (folder/'cpp-spikes.jsonl').read_text().splitlines()]
    if end['cpp_rows']!=len(stream) or end['cpp_error']:raise ValueError('Incomplete C++ stream')
    if len(acks)!=len(trials) or len(trials)!=end['completed']:
        raise ValueError('Send acknowledgments and completed trial receipts differ')
    samples=[(ms,ch,amp) for ms,ch,amp,_ in stream if ch in include]
    anchors=np.asarray([ms for ms,_,_ in samples])
    def window(t,index,phase):
        start=math.ceil(t*1000)
        ev=sorted((ms,ch,amp) for ms,ch,amp in samples if start<=ms<start+1000)
        return dict(start_ms=start,index=index,phase=phase,features=list(extract(ev,start,start+1000,roi,1000)),
                    events_sha=digest(ev),count=len(ev))
    payload=[];rows=[];bootstrap=None
    for tr,t,registered in zip(trials,acks,design['trials']):
        if tr['plan']!=registered:raise ValueError('Retrospective change in schedule')
        # The stored coarse ack is sampled after returning from the 6 ms SDK wait.
        if not 0<=tr['ack_end']-t<1:raise ValueError('C++ anchor is not the matching SDK ACK')
        counts=[int(((anchors/1000>=t+lo)&(anchors/1000<t+hi)).sum()) for lo,hi in design['windows']]
        p=tr['plan'];block=p['block'];i=p['index'];h5=[r['spikes'] for r in tr['windows']]
        w=window(t+12,i,'after')|dict(block=block,condition=p['condition'])
        payload.append(w)
        if i==12:bootstrap=window(t+13,100,'bootstrap')
        rows.append(dict(block=block,index=i,condition=p['condition'],cpp=counts,h5=h5,
            delta=counts[1]-counts[0],h5_delta=h5[1]-h5[0],late_delta=counts[2]-counts[0],
            verified=tr['pattern_verified'],recovery=tr['recovery_fraction'],
            outside=tr['metadata']['outside_frame_coverage'],gaps=tr['metadata']['frame_gaps']))
    if not rows:raise ValueError('No completed trials')
    complete=all(end[k] for k in ['closed']) and not end['cleanup'] and len(rows)==18 and end['completed']==18
    report=dict(rows=rows,complete=complete,primary='C++ spike count change in equal 5s windows',units='one organoid, six randomized blocks')
    by_class={s:[r for r in rows if r['condition']==s] for s in LABELS}
    report['counts']={s:dict(n=len(rs),pre=sum(r['cpp'][0] for r in rs),post=sum(r['cpp'][1] for r in rs),
                               deltas=[r['delta'] for r in rs],h5_deltas=[r['h5_delta'] for r in rs]) for s,rs in by_class.items()}
    if complete:
        orders=[tuple(x['condition'] for x in rows[3*i:3*i+3]) for i in range(6)]
        if set(orders)!=set(permutations(LABELS)):raise ValueError('Unbalanced randomized order')
        pairs=list(combinations(LABELS,2))
        def contrasts(order):
            data={k:[] for k in LABELS}
            for block,perm in enumerate(order):
                for pos,k in enumerate(perm):data[k].append(rows[3*block+pos]['delta'])
            return np.array([np.asarray(data[a])-data[b] for a,b in pairs])
        obs=contrasts(orders)
        null=np.array([np.abs(contrasts(o).mean(axis=1)) for o in permutations(orders)])
        report['comparisons']=[]
        for idx,(a,b) in enumerate(pairs):
            diff=obs[idx];mean=float(diff.mean())
            report['comparisons'].append(dict(a=a,b=b,block_differences=diff.tolist(),mean=mean,
                positive=int((diff>0).sum()),negative=int((diff<0).sum()),ties=int((diff==0).sum()),
                exact_p=float((null[:,idx]>=abs(mean)-1e-12).mean()),
                maxT_adjusted_p=float((null.max(axis=1)>=abs(mean)-1e-12).mean())))
    (folder/'response-comparison.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    with _plot_style():
        fig=Figure(figsize=(8.2,4),dpi=150,layout='constrained');ax=fig.subplots()
        for s,title,color,marker in zip(LABELS,NAMES,('#12698c','#b25736','#777777'),('o','s','^')):
            series=by_class[s]
            ax.plot([r['block'] for r in series],[r['delta'] for r in series],color=color,marker=marker,label=title)
        ax.axhline(0,ls=':',color='#777');ax.set(xlabel='随机对照轮次',ylabel='5 秒内尖峰数变化（后减前）',
            title=f'芯片 {chip} · {len(roi)} 个非刺激记录通道的 C++ 返回信号',xticks=range(1,7))
        ax.legend(ncol=3,fontsize=8,loc='lower right')
        FigureCanvasAgg(fig).print_png(str(folder/'responses-by-round.png'))
    if complete and all(r['verified'] and r['gaps']==0 for r in rows):
        warm=[window(acks[0]-30+i,-i-1,'baseline')['features'] for i in range(24)]
        export=dict(chip=chip,trials=payload,bootstrap=bootstrap,normalization=fit(warm),roi_sha=digest(roi),
            source='C++ 5555 host clock, milliseconds',stream_sha256=__import__('hashlib').sha256((folder/'cpp-spikes.jsonl').read_bytes()).hexdigest(),
            real_actions_during_recording=0,split='blocks 1-4 train; 5-6 held out')
        (folder/'ppo-data.json').write_text(json.dumps(export,indent=2),encoding='utf8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path)
    result=analyze(p.parse_args().folder)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
