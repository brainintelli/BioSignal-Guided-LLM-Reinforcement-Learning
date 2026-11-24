import csv,json
from pathlib import Path
w=Path(__file__).parent; d=Path('output/mea_r2_beta_sweep_20261120')
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
rows=read(w/'paired.csv');q=read(d/'r1_reference_worlds.csv')+read(d/'fixed_tick_worlds.csv')
q=[r for r in q if r['chip']=='22021' and float(r['beta'])==0 and r['arm'] in ('masked','baseline','sham')]
vals={(r['train_seed'],r['world_seed'],r['arm']):r for r in q}
assert len(q)==len(vals)==2304
means=[]
for k,cn in [('masked','屏蔽'),('baseline','自发'),('sham','零输出')]:
 for row in rows:
  r=vals[row['训练种子'],row['地图种子'],k]
  assert r['tick']=='6' and int(r['decision6_cells'])==int(row[cn+'组新增格']) and int(r['end_ticks'])==int(row[cn+'组结束刻'])
  row[cn+'组第6刻新增格']=int(r['cells'])
 means.append(sum(row[cn+'组第6刻新增格'] for row in rows)/768)
print(means)
with (w/'paired_complete.csv').open('w',encoding='utf-8-sig',newline='') as f:
 wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
(w/'clock_means.json').write_text(json.dumps(means))
