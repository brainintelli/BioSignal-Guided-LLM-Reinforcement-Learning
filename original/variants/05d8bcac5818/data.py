from pathlib import Path
import csv,json,statistics as st,hashlib
from collections import defaultdict
from PIL import Image,ImageDraw,ImageFont
W=Path(__file__).parent
O=Path('output/论文_湿件训练有效性_20261120')
Figs=O/'figures';Figs.mkdir(exist_ok=True,parents=True)
A='output/mea_six_arm_r1_off_20261120/'
A1='output/mea_predictability_22021_20261118T093720Z/'
A2='output/mea_predictability_22021_20261119T042416Z/'
F=A1+'offline-mea-stim-ablation/'
G=A2+'offline-five-arm-replication/'
B=A2+'offline-confirmatory-results/'
Reg=A2+'offline-confirmatory-registration/'
R='output/mea_r2_beta_sweep_20261120/'
D='output/22021_弱阳性复验分析_20261119.md'
K='output/22554_offline_hwf_20261119/实验结果.md'
H='output/mea_28080_followup_20261117/RESULTS.md'
Q='output/mea_28073_live_20261117_1413/28073_MULTIPOINT_REPORT.md'
J='output/mea_predictability_27701_20261119T072625Z/frequency-retest/复测报告.md'
def js(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def read(p):
 with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
effects=read(A+'effects.csv'); groups=js(A+'summary.json')['groups']; b=js(B+'verification/local-verification.json')
assert len(groups)==6 and all(g['n_seeds']==16 and g['n_worlds']==768 for g in groups)
def verify(raw,a1,a2,seed,world,metric):
 by=defaultdict(dict)
 for r in raw:by[(r[seed],r[world])][r['arm']]=float(r[metric])
 dif=defaultdict(list)
 for (k,m),v in by.items():dif[k].append(v[a1]-v[a2])
 d=[st.mean(v) for v in dif.values()];return st.mean(d),sum(i>0 for i in d)
ra=read(A+'metrics_by_world.csv');rb=read(B+'episodes.csv')
assert len(ra)==4608 and len(rb)==18432 and sum(int(r['decisions']) for r in rb)==669715
for r in effects:
 d,n=verify([v for v in ra if v['chip']==r['chip']],r['arm_a'],r['arm_b'],'train_seed','world_seed',r['metric']);assert abs(d-float(r['mean_difference']))<1e-9
 if r['metric']=='exploration':assert n==int(r['positive_seeds'])
for r in b['all32']:
 if r['family']=='primary_endpoints':
  d,n=verify([v for v in rb if v['scene']==r['scene']],'ordered',r['control'],'seed','world',r['metric'])
  assert abs(d-r['delta'])<1e-10 and n==r['favorable']
records=[]
def add(exp,cond,metric,est,src,unit='',lo='',hi='',level='',seeds='',maps='',pos='',note=''):
 assert Path(src).is_file(),src
 records.append([exp,cond,metric,est,lo,hi,level,seeds,maps,pos,unit,src,note])
for x in effects:
 add('A 六决策',x['contrast'],x['metric'],x['mean_difference'],A+'effects.csv',{'exploration':'格','r1':'游戏分'}.get(x['metric'],'储备点'),x['ci_low'],x['ci_high'],95,16,48,x['positive_seeds'],'实验组减对照，种子比例统计原指标增加')
for g in groups:
 for m in ['exploration','r1','health','thirst','hunger']:
  v=g[m];add('A 六决策组均值',g['label'],m,v['mean'],A+'summary.json',{'exploration':'格','r1':'游戏分'}.get(m,'储备点'),*v['seed_ci95'],95,16,48)
 for m,v in g['action_probabilities'].items():
  add('A 行为',g['label'],m,v*100,A+'summary.json','%',seeds=16,maps=48,note='22021 为实选频率，22554 为策略概率均值')
for tag,path in [('11/18',F),('11/19',G)]:
 jj=js(path+'comparison.json')
 for c,v in jj['contrasts'].items():
  for m,x in v['metrics'].items():add('A 五组 '+tag,c,m,x['difference'],path+'comparison.json','格' if m=='new_cells' else '游戏分',*x['interval'],100*v['confidence'],8,48,sum(q>0 for q in x['by_seed']))
for r in read(W/'source_snapshot/paired_effects.csv'):
 for level,lo,hi in [(95,'ci95_low','ci95_high'),(100*(1-.05/3),'ci_three_low','ci_three_high')]:
  add('A 同刻与 β 扫描',f"{r['chip']} β={r['beta']} {r['contrast']} {r['detail']}",r['metric'],r['delta'],R+'paired_effects.csv',{'cells':'格','r1':'游戏分'}.get(r['metric'],'原指标单位'),r[lo],r[hi],level,r['n_seeds'],r['n_worlds_per_seed'],r['positive_seeds'],'第 6 游戏刻；只纳入该对比完成的种子与地图')
decomp=[('刺激前自发',.004017,.001292,.008037,.003385,84.3,.000632,-.000217,.002742,97.5),('当轮零输出',.004073,.000866,.008420,.003905,95.9,.000168,-.000382,.001124,100*(1-.05/6))]
for c,t,lo,hi,tm,p,res,rl,rh,cf in decomp:
 for m,v,l,h,cov,ps in [('总差',t,lo,hi,cf,8),('计时项',tm,'','','',''),('计时占比',p,'','','',''),('其余得分差',res,rl,rh,97.5,'')]:
  add('A 11/18 R1 分解',c+' − 屏蔽',m,v,D,'%' if m=='计时占比' else '游戏分',l,h,cov,8,48,ps,'分量按原报告六位小数')
for key,N in [('all32',32),('additional24_posthoc',24)]:
 for r in b[key]:
  fac=100 if r['metric']=='survived' else 1
  add('B 全样本' if N==32 else 'B 新增种子事后',r['scene']+' ordered vs '+r['control'],r['metric'],fac*r['delta'],B+'verification/local-verification.json',{'cells':'格','survived':'百分点'}.get(r['metric'],'游戏刻'),r['lo']*fac,r['hi']*fac,r['interval_level']*100,N,64,r['favorable'],'等待为对照减顺序，其余为顺序减对照')
for r in js(B+'verification/paired-power.json')['results']:
 for m in ['masked','shuffled','joint']:
  add('B 配对功效',f"{r['scene']} +{r['target_gain']}（原始尺度）",m,100*r[m],B+'verification/paired-power.json','%',seeds=32,maps=64,note='每方案 4000 次外层模拟，基于 8 种子先导数据')
for k,p in [('原始记录',B+'verification/source-verification.json'),('行为重放',B+'verification/integrity-summary.json'),('模型重载',B+'verification/policy-reload.json')]:
 for m,v in js(p).items():
  if type(v) in [int,float]:add('B 技术核验',k,m,v,p,'原指标单位')
for scene in sorted({r['scene'] for r in rb}):
 for arm in ['ordered','masked','shuffled']:
  selected=[v for v in rb if v['arm']==arm and v['scene']==scene]
  assert len(selected)==2048
  for metric in ['cells','survived','drink_wait','eat_wait']:
   if metric=='drink_wait' and scene!='thirst' or metric=='eat_wait' and scene!='hunger':continue
   value=st.mean(float(v[metric]) for v in selected)
   add('B 各组原始均值',scene+' '+arm,metric,value*(100 if metric=='survived' else 1),B+'episodes.csv',{'cells':'格','survived':'%'}.get(metric,'刻'),seeds=32,maps=64,note='每组每场景 2048 局')
for c,d,l,u in [('刺激后 − 刺激前',-.006883,-.074936,.059320),('刺激后 − 无信号',-.009394,-.098805,.079071)]:
 add('28080 单点独立训练',c,'R1',d,H,'游戏分',l,u,95,8,note='共同前后上下文评估，区别于六决策实验')
for r in read(W/'source_snapshot/coverage.csv'):
 add('A β 扫描快照',f"{r['chip']} {r['arm']} β={r['beta']}",'完整种子',r['complete_seeds'],R+'coverage.csv','个',seeds=r['expected_seeds'],maps=48,note='该条件已核验局数 '+r['complete_worlds'])
with (O/'data_table.csv').open('w',encoding='utf-8-sig',newline='') as f:
 c=csv.writer(f);c.writerow(['实验','组别或配对方向','指标','估计值','区间下界','区间上界','置信水平百分数','训练种子数','每种子地图数','正向种子数','单位','源文件_项目相对路径','判读说明']);c.writerows(records)
