from pathlib import Path
import csv,json,math
from PIL import Image,ImageDraw,ImageFont
root=Path.cwd()
w=root/'tmp/nurture_report_1120'
out=root/'output/PPT_湿件训练有效性_20261120'
fig=out/'图表'
A='output/mea_six_arm_r1_off_20261120/'
R='output/mea_r2_beta_sweep_20261120/'
F='output/mea_predictability_22021_20261118T093720Z/offline-mea-stim-ablation/'
G='output/mea_predictability_22021_20261119T042416Z/offline-five-arm-replication/'
B='output/mea_predictability_22021_20261119T042416Z/offline-confirmatory-results/'
D='output/22021_弱阳性复验分析_20261119.md'
def js(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
records=[]
def add(study,cond,metric,val,lo='',hi='',conf='',n='',maps='',pos='',unit='',path='',note=''):
 records.append([study,cond,metric,val,lo,hi,conf,n,maps,pos,unit,path,note])
effects=rows(A+'effects.csv'); ba=js(B+'verification/local-verification.json')
for r in effects:
 add('A 六决策',r['contrast'],r['metric'],r['mean_difference'],r['ci_low'],r['ci_high'],95,16,48,r['positive_seeds'],'原指标单位',A+'effects.csv','按训练种子均值配对重采样')
for g in js(A+'summary.json')['groups']:
 for m in ['exploration','r1','health','hunger','thirst']:
  d=g[m];add('A 六决策组均值',g['label'],m,d['mean'],*d['seed_ci95'],95,16,48,path=A+'summary.json')
 for m,z in g['action_probabilities'].items():
  add('A 动作',g['label'],m,z*100,n=16,maps=48,unit='%',path=A+'summary.json',note='22021 经验动作占比；22554 策略概率均值')
for dir_,round_ in [(F,'A 11/18 五组'),(G,'A 11/19 五组')]:
 for contrast,r in js(dir_+'comparison.json')['contrasts'].items():
  for name,d in r['metrics'].items():
   add(round_,contrast,name,d['difference'],*d['interval'],100*r['confidence'],8,48,sum(x>0 for x in d['by_seed']),path=dir_+'comparison.json',note='探索与 R1 两项同升种子 '+str(r['jointly_positive'])+'/8')
beta=rows(R+'paired_effects.csv')
for r in beta:
 for conf,lo,hi in [(95,'ci95_low','ci95_high'),(98.333333333,'ci_three_low','ci_three_high')]:
  add('A β 扫描·第6刻',f"{r['chip']} β={r['beta']} {r['contrast']} {r['detail']}",r['metric'],r['delta'],r[lo],r[hi],conf,r['n_seeds'],r['n_worlds_per_seed'],r['positive_seeds'],path=R+'paired_effects.csv')
for key,n in [('all32',32),('additional24_posthoc',24)]:
 for r in ba[key]:
  mult=100 if r['metric']=='survived' else 1
  add('B 确认性' if n==32 else 'B 新增种子·事后',r['scene']+' 顺序 vs '+r['control'],r['metric'],*[r[k]*mult for k in ('delta','lo','hi')],r['interval_level']*100,n,64,r['favorable'],{'cells':'格','survived':'百分点'}.get(r['metric'],'刻'),B+'verification/local-verification.json','补给等待为对照减顺序；其余为顺序减对照')
for cond,total,lo,hi,pct,res,rl,rh,con in [('自发',.004017,.001292,.008037,84.3,.000632,-.000217,.002742,97.5),('零输出',.004073,.000866,.008420,95.9,.000168,-.000382,.001124,99.1666667)]:
 for m,v,l,h,c,p in [('总R1差',total,lo,hi,con,8),('计时项差',round(total-res,6),'','','',''),('计时占比',pct,'','','',''),('扣计时残差',res,rl,rh,97.5,'')]:
  add('A 11/18 五组分解',cond+' − 屏蔽',m,v,l,h,c,8,48,p,'%' if m=='计时占比' else 'R1',D)
for r in rows(R+'clock_sensitivity.csv'):
 add('A 双时钟',f"{r['chip']} {r['arm']} − {r['reference']} β={r['beta']}",r['metric'],r['delta'],n=16,maps=48,path=R+'clock_sensitivity.csv',note=r['horizon'])
for path,study in [(B+'verification/paired-power.json','B 设计功效')]:
 d=js(path)
 for r in d['results']:
  for k,v in r.items():
   if isinstance(v,(int,float)):add(study,str(r.get('endpoint',r.get('metric','')))+' '+str(r.get('effect',r.get('alternative',''))),k,v,n=32,maps=64,path=path,note=json.dumps(r,ensure_ascii=False))
# Export traceable descriptive denominators and study results, including signed effects.
for p,study in [(B+'实验结果与完整性核验.md','B 核验与终点'),(D,'A 分解与等时复核'),(F+'离线对照结果.md','A 样本1'),(G+'五组训练复验结果.md','A 样本2'),(R+'report.md','A β阶段快照'),('output/22554_offline_hwf_20261119/实验结果.md','A H/W/F')]:
 head=''
 for l in Path(p).read_text(encoding='utf-8-sig').splitlines():
  if l.startswith('|'):
   cols=[x.strip() for x in l.strip('|').split('|')]
   if all(set(x)<=set(': -') for x in cols):continue
   if not head:head=' / '.join(cols);continue
   add(study,cols[0],head,' | '.join(cols[1:]),path=p,note='原表逐行，区间、分母及单位见各列名')
  else:head=''
def writecsv():
 with (out/'数据总表.csv').open('w',encoding='utf-8-sig',newline='') as f:
  q=csv.writer(f);q.writerow(['实验与口径','组别或配对方向','指标','数值','区间下界','区间上界','区间水平%','训练种子数','地图数','有利种子数','单位','源文件_项目相对路径','备注']);q.writerows(records)
ink='#193446';mut='#496471';green='#147559';amber='#A35D13';blue='#27619A';gray='#D8E2E7'
S=2
def canvas(W,H):
 global im,dr
 im=Image.new('RGB',(W*S,H*S),'white');dr=ImageDraw.Draw(im)
def xy(a):return tuple(round(s*S) for s in a)
def tx(x,y,t,size=18,col=ink,bold=False,anchor='lm'):
 f=ImageFont.truetype('C:/Windows/Fonts/msyh'+('bd' if bold else '')+'.ttc',size*S)
 dr.text(xy((x,y)),str(t),font=f,fill=col,anchor=anchor)
def line(x,y,X,Y,c=gray,k=1):dr.line(xy((x,y,X,Y)),fill=c,width=k*S)
def rect(x,y,X,Y,c):dr.rectangle(xy((x,y,X,Y)),fill=c)
def dot(x,y,col,r=6):dr.ellipse(xy((x-r,y-r,x+r,y+r)),fill=col)
def save(name):im.save(fig/(name+'.png'))
fx=[(r['contrast'].split(' ',1)[1].replace(' MEA',''),float(r['mean_difference']),float(r['ci_low']),float(r['ci_high']),int(r['positive_seeds']),16) for r in effects if r['metric']=='exploration']
def forest(compact):
 W=1160; data=[('自发 − 屏蔽',*fx[0][1:]),('零输出 − 屏蔽',*fx[1][1:])]
 if not compact:data.extend([('22554  有序 − 纯 R1',*fx[2][1:]),('22554  打乱 − 纯 R1',*fx[3][1:])])
 data.append(('低生命  顺序 − 屏蔽',5.591796875,-.18896484375,11.255706787,25,32))
 H=55+49*len(data)+77;canvas(W,H)
 tx(5,19,'配对条件',18,mut,True);tx(796,19,'平均差  [双侧区间]',18,mut,True);tx(1065,19,'正向种子',17,mut,True)
 left,right=285,760; X=lambda x:left+(x+3)/15*(right-left)
 for t in [-2,0,2,4,6,8,10,12]:
  line(X(t),40,X(t),H-65,'#78919C' if t==0 else '#E7ECEE',2 if t==0 else 1);tx(X(t),H-46,str(t),15,mut,anchor='mm')
 for i,(lab,d,l,h,n,N) in enumerate(data):
  y=61+i*49; c=green if i<2 else amber;tx(5,y,('22021  ' if i<2 else '')+lab,19)
  line(X(l),y,X(h),y,c,3);line(X(l),y-6,X(l),y+6,c,2);line(X(h),y-6,X(h),y+6,c,2);dot(X(d),y,c)
  tx(796,y,f'{d:+.2f}  [{l:+.2f}, {h:+.2f}]',19,c,True);tx(1085,y,f'{n}/{N}',18)
 tx(515,H-18,'新增揭示格数的配对差（格）',15,mut,anchor='mm')
 save('01_结论森林图' if compact else '02_跨条件森林图')
forest(True);forest(False)
canvas(1160,270)
for i,(name,total,timer,res,pct) in enumerate([('刺激前自发',.004017,.003385,.000632,84.3),('当轮零输出',.004073,.003905,.000168,95.9)]):
 start=24+i*583; tx(start,17,name+' − 屏蔽',22,ink,True);tx(start+532,17,f'计时占 {pct:.1f}%',18,amber,True,'rm')
 Y=lambda v:202-v*30000
 line(start,202,start+535,202)
 for j,(a,b,lab,num,color) in enumerate([(0,total,'总 R1 差',total,blue),(res,total,'减去计时项',-timer,amber),(0,res,'剩余 R1 差',res,green)]):
  x=start+30+j*180;rect(x,Y(b),x+120,Y(a),color);tx(x+60,Y(b)-14,f'{num:+.6f}',18,color,True,'mm');tx(x+60,228,lab,17,mut,anchor='mm')
  if j<2:line(x+120,Y(total if j==0 else res),x+180,Y(total if j==0 else res),'#79909A')
save('03_R1计时分解')
# first registered-map pair with different action sequences in the smallest training seed
tracebase=F+'models/102083199/'
readtrace=lambda a:[json.loads(l) for l in Path(tracebase+a+'/holdout.jsonl').read_text().splitlines()]
M,Z=readtrace('masked'),readtrace('sham')
m,z=next((m,z) for m,z in zip(M,Z) if [r['action'] for r in m['rows']]!=[r['action'] for r in z['rows']]);assert m['world']==z['world']
canvas(530,310); colors=[blue,green]
for n,rec in enumerate([m,z]):
 pts=[r['after']['player'] for r in rec['rows']]
 tx(10+n*263,22,['屏蔽','零输出 MEA'][n],19,colors[n],True)
 allpts=[r['after']['player'] for rec1 in [m,z] for r in rec1['rows']]
 xs=[p['x'] for p in allpts];ys=[p['y'] for p in allpts]; step=min(32,170/max(1,max(xs)-min(xs)),145/max(1,max(ys)-min(ys)))
 X=lambda p:35+n*263+(p-min(xs))*step;Y=lambda p:97+(p-min(ys))*step
 for u in range(min(xs),max(xs)+1):line(X(u),Y(min(ys))-10,X(u),Y(max(ys))+10)
 for u in range(min(ys),max(ys)+1):line(X(min(xs))-10,Y(u),X(max(xs))+10,Y(u))
 for p,q in zip(pts,pts[1:]):line(X(p['x']),Y(p['y']),X(q['x']),Y(q['y']),colors[n],3)
 for p in pts:dot(X(p['x']),Y(p['y']),colors[n],4)
 last=pts[-1];tx(X(last['x']),Y(last['y'])-16,'终',14,colors[n],True,'mm')
 tx(10+n*263,258,'终点已揭示 '+str(rec['rows'][-1]['after']['visited_count'])+' 格',17,ink)
 for i,r in enumerate(rec['rows']):add('A 路径示例',rec['world'],f"{['屏蔽','零输出'][n]} 动作{i+1}",r['action']['verb'],path=tracebase+['masked','sham'][n]+'/holdout.jsonl',note=json.dumps(r['after']['player']))
tx(10,291,'点为每次决策终点；连线连接相邻终点。',15,mut);save('04_配对探索路径')
canvas(570,310)
grp=js(A+'summary.json')['groups'][:3]
for i,(name,v) in enumerate([('移动','move'),('导航','navigate'),('采集','gather'),('等待','wait')]):
 y=74+i*55;tx(8,y,name,20,ink,True)
 for k,g in enumerate(grp):
  tx(171+k*150,y,f"{100*g['action_probabilities'][v]:.2f}%",24,[blue,green,green][k],False,'mm')
 line(5,y+24,566,y+24)
for x,l in [(170,'屏蔽'),(320,'自发'),(470,'零输出')]:tx(x,24,l,17,mut,True,'mm')
save('05_动作占比')
# Standalone figure legends and citations, also visible on slides.
for name,src,desc in [
 ('01_结论森林图',A+'effects.csv  |  '+B+'verification/local-verification.json','A  16 种子 × 48 图，6 决策，95% CI；B  32 × 64，60 刻，校正 99.167% CI。'),
 ('02_跨条件森林图',A+'effects.csv  |  '+B+'verification/local-verification.json','A  16 种子 × 48 图，6 决策，95% CI；B  32 × 64，60 刻，校正 99.167% CI。'),
 ('03_R1计时分解',D,'11/18 样本；两组各减屏蔽；8 种子 × 48 图。总差 CI 自发 97.5%、零输出 99.17%；分量为点估计。'),
 ('04_配对探索路径',tracebase+'{masked,sham}/holdout.jsonl',f"同一训练种子 102083199、地图 {m['world']} 的首对分歧轨迹；每臂 6 决策，个例不估计区间。"),
 ('05_动作占比',A+'summary.json','22021，16 种子 × 48 图，每局 6 决策；组内经验动作频率，描述值无区间。')]:
 p=fig/(name+'.png');old=Image.open(p).copy();W=old.width//2;H=old.height//2
 import textwrap
 lines=[desc]+[s for ss in src.split('  |  ') for s in textwrap.wrap(ss,width=max(40,int(W/7.0)),break_long_words=True,break_on_hyphens=False)]
 canvas(W,H+13+len(lines)*18);im.paste(old,(0,0))
 for i,t in enumerate(lines):tx(6,H+10+18*i,t,12,mut)
 save(name)
(w/'meta.json').write_text(json.dumps(dict(A=A,R=R,F=F,G=G,B=B,D=D,world=m['world'],trace=tracebase)),encoding='utf-8')
writecsv()

