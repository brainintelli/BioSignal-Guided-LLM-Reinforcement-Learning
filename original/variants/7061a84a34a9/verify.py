from pathlib import Path
import csv,json,hashlib,zipfile,statistics as st
from collections import defaultdict
import xml.etree.ElementTree as ET
base=Path('tmp/nurture_report_1120'); m=json.loads((base/'meta.json').read_text());out=Path('output/PPT_湿件训练有效性_20261120')
def read(p):
 with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def pairs(rows,arm1,arm2,se,wo,met):
 data=defaultdict(dict)
 for r in rows:data[(r[se],r[wo])][r['arm']]=float(r[met])
 ds=defaultdict(list)
 for (s,w),v in data.items():ds[s].append(v[arm1]-v[arm2])
 u=[st.mean(x) for x in ds.values()]
 return st.mean(u),sum(x>0 for x in u),len(u)
a=read(m['A']+'metrics_by_world.csv'); assert len(a)==6*16*48
for x in read(m['A']+'effects.csv'):
 d,k,n=pairs([r for r in a if r['chip']==x['chip']],x['arm_a'],x['arm_b'],'train_seed','world_seed',x['metric'])
 assert abs(d-float(x['mean_difference']))<1e-10
 if x['metric']=='exploration':assert k==int(x['positive_seeds'])
b=read(m['B']+'episodes.csv');assert len(b)==18432 and sum(int(r['decisions']) for r in b)==669715
for x in json.loads(Path(m['B']+'verification/local-verification.json').read_text())['all32']:
 if x['family']!='primary_endpoints':continue
 d,k,n=pairs([r for r in b if r['scene']==x['scene']],'ordered',x['control'],'seed','world',x['metric'])
 assert abs(d-x['delta'])<1e-10 and k==x['favorable'] and n==32
with next(out.glob('*.pptx')).open('rb') as f:assert hashlib.sha256(f.read()).hexdigest()==json.loads((base/'validation.json').read_text())['finalSha256']
table=out/'数据总表.csv';data=read(table);header=list(data[0])
assert all(Path(r['源文件_项目相对路径']).exists() for r in data)
if not any(r['实验与口径']=='B 技术质控' for r in data):
 for name in ['source-verification','integrity-summary','checkpoint-verification','policy-reload']:
  src=m['B']+'verification/'+name+'.json';d=json.loads(Path(src).read_text())
  for k,v in d.items():
   if isinstance(v,(int,float)) and not isinstance(v,bool):
    data.append(dict(zip(header,['B 技术质控',name,k,v,'','','','','','','原报告单位',src,'描述性计数或核验值；未估计置信区间'])))
 with table.open('w',encoding='utf-8-sig',newline='') as f:
  q=csv.DictWriter(f,fieldnames=header);q.writeheader();q.writerows(data)
print('VERIFIED',len(a),'six-arm episodes;',len(b),'confirmatory episodes;',len(data),'CSV rows;',len(list((out/'图表').glob('*.png'))),'figures.')

