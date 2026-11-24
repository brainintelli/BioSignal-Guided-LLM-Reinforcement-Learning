from pathlib import Path
import json,sys,csv,numpy as np,hashlib
from docx import Document
from docx.oxml.ns import qn
w=Path(__file__).parent.resolve();root=Path.cwd();out=root/'output'
r=out/'mea_predictability_22021_20261118T093720Z/offline-mea-stim-ablation';cap=r.parent/'cpp-crossover-v2';ex=out/'mea_six_arm_r1_off_20261120'
sys.path[:0]=[str(root/'standalone/mea_state2_r2'),str(r),str(root/'.tools/mea_baseline_deps')]
from build_inputs import read_record,window
from mea_state2_r2.features import transform
js=lambda p:json.loads(p.read_text(encoding='utf8'))
roi=js(cap/'design.json')['roi'];hs=js(cap/'raw-hashes.json')
a,b,sp,marks=read_record('warmup',hs,roi);assert b-a+1==1281600 and len(roi)==322 and not marks
ctx=js(r/'pre-stim-baseline/baseline.json');old=ctx['series']['holdout'][0]
n=window(sp,a,b,a+58*20000,1058,0,58,roi)
assert n['events_sha']==old['events_sha'] and n['features']==old['features']
values=transform(n['features'],ctx['normalization']);assert np.allclose(values,old['normalized'])
e=sp[(sp['frameno']>=n['frame_start'])&(sp['frameno']<n['frame_end'])]
order={v:i+1 for i,v in enumerate(sorted(roi))}
ts=[(int(v['frameno'])-n['frame_start'])/20000 for v in e]
ch=[order[int(v['channel'])] for v in e];assert len(ts)==n['count']
feats={}
for group in ['masked','baseline','sham']:
 subfolder='baseline' if group=='baseline' else 'main'
 p=ex/subfolder/'models/102600001'/group/'holdout.jsonl'
 row=json.loads(next(p.open(encoding='utf8')));assert row['world']==172640420 and row['rows'][0]['action']['verb']=='move'
 feats[group]=row['rows'][0]['consumed_features']
 if group=='baseline':assert row['rows'][0]['wetware_events_sha']==old['events_sha'] and np.allclose(values,feats[group],atol=.00051)
print(len(e),len(set(ch)),n['features'],feats)
asset=out/'论文_湿件训练有效性_20261120'
p=asset/'类器官神经信号对语言模型探索行为的影响（1）.docx'
d=Document(p);sec=d.sections[0];print('dimensions',[round(getattr(sec,k).cm,2) for k in ('page_width','page_height','left_margin','right_margin','top_margin','bottom_margin')]);print('figure sizes',[(round(z.width.cm,2),round(z.height.cm,2)) for z in d.inline_shapes]);print('paragraphs',[(q.style.name,q.text[:120]) for q in d.paragraphs[17:26]])
brows=list(csv.DictReader((asset/'22021_逐局配对实测.csv').open(encoding='utf-8-sig')));assert len(brows)==768
assert [sum(int(k[x+'组新增格']) for k in brows) for x in ['屏蔽','自发','零输出']]==[31017,32543,32674]
rows=list(csv.DictReader((ex/'effects.csv').open(encoding='utf-8-sig')))
ests=[[(float(s[k]) if k!='positive_seeds' else int(s[k])) for k in ('mean_difference','ci_low','ci_high','positive_seeds')] for arm in ['baseline','sham'] for s in rows if s['chip']=='22021' and s['metric']=='exploration' and s['arm_a']==arm and s['arm_b']=='masked']
assert len(ests)==2
(w/'data.json').write_text(json.dumps(dict(ts=ts,ch=ch,active=len(set(ch)),n=len(ch),state2=feats,est=ests,raw=n['features'])),encoding='utf8')
