"""Validate coverage from the closed survey H5s and report only completed exposures."""
from pathlib import Path
import hashlib
import json
import os
import sys
from types import SimpleNamespace
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
os.environ['HDF5_PLUGIN_PATH']=str(ROOT/'.tools/mea_maxwell_hdf5_plugin')
sys.path.insert(0,str(ROOT/'standalone/brain_device_session'))
from report_recorded_pulses import Figure,_plot_style,_save,np
from mea_stim_v2.recording import H5Recording
from rank_sites import rank
from mea_stim_v2.qc import stability

def load(f):return json.loads(f.read_bytes())
def verified(path,manifest):
    key=path.relative_to(OUT.parent).as_posix()
    with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
    assert actual==manifest[key]['sha256']

def candidate_diagnostics(rec, electrodes):
    """Count overlapping exclusion reasons once per independently recorded candidate."""
    mapping=dict(rec.mapping)
    channels=[c for c,e in mapping.items() if e in electrodes]
    counts=[np.bincount(rec.spike_channels[(rec.spike_times>=s)&(rec.spike_times<s+20)],minlength=1024) for s in (0,20,40)]
    active=[c for c in channels if min(a[c] for a in counts)>0]
    out=dict(covered=len(channels),active=len(active),rail=0,shift=0,noise=0,passed=0)
    for j in range(0,len(active),64):
        cs=active[j:j+64];base=rec.sample(cs,3)
        tests=[stability(base,rec.sample(cs,t)) for t in (23,43,57)]
        for k in range(len(cs)):
            out['rail']+=int(any(t['before_saturated'][k] or t['after_saturated'][k] for t in tests))
            out['shift']+=int(any(t['median_shifts_adc'][k]>5 for t in tests))
            out['noise']+=int(any(t['mad_ratios'][k] is None or not .67<=t['mad_ratios'][k]<=1.5 for t in tests))
            out['passed']+=int(all(t['channel_pass'][k] for t in tests))
    return out

def analyze():
    manifest=load(OUT/'verified-recordings.json')
    base=OUT/'survey';b=load(OUT/'baseline-binding.json')
    summary=load(base/'overview.json')
    grid=np.zeros((120,220),dtype=int)
    candidate=set()
    fields=sorted(base.glob('field-*'))
    for f in fields:
        result=load(f/'screen.json');raw=f/'quiet.raw.h5';verified(raw,manifest)
        with H5Recording(raw,SimpleNamespace(**b)) as r:
            assert not r.events and r.metadata['chip_id']==b['chip_id'] and r.duration>=12 and r.metadata['frame_gaps']==0
            array={e for ch,e in r.mapping};assert sorted(array)==result['routed']
        for e in array:grid[e//220,e%220]+=1
        candidate.update(x['electrode'] for x in result['rank'])
    assert (len(fields)==summary['layouts'] and int((grid>0).sum())==summary['actual_electrodes']
            and len(candidate)==summary['screen_candidates'])
    table=[]; new_points=[];confirmed=set();diagnostics=[]
    layouts=[(OUT,'原区域'),(OUT/'near','活跃点邻域'),(OUT/'wide','阵列稀疏取样'),
             (OUT/'compact','6257 周边紧凑区域'),(OUT/'survey-confirm','初筛候选独立确认')]
    layouts.extend((d,'其余候选独立确认 '+d.name[-2:]) for d in sorted(OUT.glob('survey-confirm-more-*')))
    for directory,title in layouts:
        if not directory.exists():continue
        path=directory/(b['session_id']+'-site-selection')/'quiet.raw.h5'
        verified(path,manifest)
        result=rank(path,b)
        if directory.name.startswith('survey-confirm-more'):
            result['new']=[r for r in result['new'] if r['electrode'] in candidate-confirmed]
        if directory.name.startswith('survey-confirm'):
            with H5Recording(path,SimpleNamespace(**b)) as r:
                diagnostics.append(candidate_diagnostics(r,candidate-confirmed))
                confirmed.update(e for _,e in r.mapping if e in candidate)
            new_points.extend(result['new'])
        cache=load(directory/('rank-initial.json' if directory==OUT else 'rank.json'))
        for k in ('mapped','active','stable','pass_hist','new'):assert json.loads(json.dumps(result[k]))==cache[k]
        table.append([title,cache['mapped'],cache['active'],len(cache['stable']),len(cache['new'])])
    totals={key:sum(row[key] for row in diagnostics) for key in ('covered','active','rail','shift','noise','passed')}
    assert totals['covered']==len(confirmed) and totals['passed']==len(new_points)
    with _plot_style():
        fig=Figure(figsize=(11.5,4.8),dpi=150,layout='constrained')
        ax,bar=fig.subplots(1,2,gridspec_kw={'width_ratios':[1.6,1]})
        ax.imshow(grid>0,cmap='Blues',vmin=0,vmax=3,aspect='equal',interpolation='nearest')
        if candidate:ax.scatter([e%220 for e in candidate],[e//220 for e in candidate],s=13,c='#a26929',alpha=.7,label='短记录候选')
        if new_points:ax.scatter([r['electrode']%220 for r in new_points],[r['electrode']//220 for r in new_points],s=52,c='#147955',label='独立确认通过')
        ax.scatter([e%220 for e in [7158,6026,3621]],[e//220 for e in [7158,6026,3621]],marker='x',s=52,c='#c13d3d',label='上轮三个位点')
        fig.suptitle('27701 新刺激位点筛查',fontsize=17)
        ax.set(xlabel='阵列列号',ylabel='阵列行号',title=f"短记录覆盖 {summary['actual_electrodes']:,} 个电极")
        ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.14),frameon=False)
        bars=bar.barh(['三段均有尖峰','电压基线漂移超限','波动幅度比超限','原始电压触边','通过全部条件'],
                      [totals[k] for k in ('active','shift','noise','rail','passed')],
                      color=['#487186','#ad7648','#ad7648','#ad7648','#267251'])
        bar.bar_label(bars,padding=4,fmt='%d')
        bar.invert_yaxis();bar.set_xlim(0,max(1,totals['covered']*1.1))
        bar.set(title=f"独立长记录复核 {totals['covered']} 个候选",xlabel='电极数（同一电极可有多项超限）')
        bar.spines[['top','right']].set_visible(False)
        _save(fig,OUT/'全阵列点位筛查.png')
    result=dict(fields=len(fields),electrodes=summary['actual_electrodes'],short_candidates=len(candidate),screen_two_checks=summary['candidates_two_checks'],
                confirmation=table,independently_confirmed=len(confirmed),candidate_diagnostics=diagnostics,candidate_totals=totals,eligible_new=new_points,raw_files=sum(k.endswith('.h5') for k in manifest),
                bytes=sum(v['bytes'] for k,v in manifest.items() if k.endswith('.h5')),final=load(OUT/'device-final.json'))
    (OUT/'复测核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    return result
if __name__=='__main__':print(json.dumps(analyze(),ensure_ascii=False))
