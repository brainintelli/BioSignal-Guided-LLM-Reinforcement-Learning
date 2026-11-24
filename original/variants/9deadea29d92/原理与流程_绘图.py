from pathlib import Path
from collections import defaultdict
from math import atan2, cos, sin
from statistics import mean
import argparse
import csv, json, re
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pypdfium2 as pdfium

root=Path.cwd()
work=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(description='在项目根目录运行，使用实测记录重绘论文的原理图和流程图。')
ap.add_argument('--output',type=Path,default=work if work.name=='figures' else work/'figures')
output=ap.parse_args().output
output.mkdir(parents=True,exist_ok=True)
paper=root/'output/论文_湿件训练有效性_20261120'
study=root/'output/mea_predictability_22021_20261118T093720Z/offline-mea-stim-ablation'
verify=root/'output/mea_predictability_22021_20261119T042416Z/offline-confirmatory-results/verification'
read=lambda p: json.loads(p.read_text(encoding='utf8'))
data=read(root/'tmp/figure1_deep/data.json')
pairs=list(csv.DictReader((paper/'22021_逐局配对实测.csv').open(encoding='utf-8-sig')))
assert len(pairs)==len({(r['训练种子'], r['地图种子']) for r in pairs})==768
results=[]
for idx,a in enumerate(['自发','零输出']):
    diffs=defaultdict(list)
    for r in pairs:diffs[r['训练种子']].append(int(r[a+'组新增格'])-int(r['屏蔽组新增格']))
    vals=[mean(v) for v in diffs.values()]
    point,lo,hi,n=data['est'][idx]
    assert all(len(v)==48 for v in diffs.values()) and len(vals)==16 and abs(mean(vals)-point)<1e-9 and sum(v>0 for v in vals)==n
    results.append((point,lo,hi,n))
assert len(data['ts'])==len(data['ch'])==44 and len(set(data['ch']))==32
row=json.loads((paper/'实验复现材料/确认性实验_直接阅读/日志示例/train-1.jsonl').open(encoding='utf8').readline())
assert row['prepared']['wetware_features']==row['prepared']['critic_input'][-4:]==row['snapshot']['normalized']
assert row['action']['verb']=='move' and row['duration']==1 and row['contribution']==0 and abs(row['value']+row['advantage']-row['return'])<1e-11
reg=read(paper/'实验复现材料/确认性实验_直接阅读/日志示例/registration.json')
reloaded=read(verify/'policy-reload.json')
replay=read(verify/'native-replay.json')
assert reg['training_seed']==351855818 and reg['arm']=='ordered'
assert (reloaded['episodes'],reloaded['decisions'],reloaded['action_mismatches'])==(288,9351,0)
assert (replay['episodes'],replay['decisions'],replay['primary_and_wait_mismatches'],replay['r1_differences_over_1e_10'])==(18432,669715,0,64)
example=pairs[0]
def track(d,arm):
    f=d/'models'/example['训练种子']/arm/'holdout.jsonl'
    g=next(g for s in f.open(encoding='utf8') if (g:=json.loads(s))['world']==int(example['地图种子']))
    return [(r['after']['tick'],r['action']['verb']) for r in g['rows']]
tracks=[track(study,'masked'),track(study/'pre-stim-baseline','baseline')]
assert [len(t) for t in tracks]==[6,6] and [t[-1][0] for t in tracks]==[int(example[k]) for k in ['屏蔽组结束刻','自发组结束刻']]==[10,12]
spec=read(paper/'实验复现材料/确认性实验_直接阅读/原实验程序/stress/design.json')
assert (len(spec['training_seeds']),len(spec['arms']),len(spec['worlds']),len(spec['scenes']))==(32,3,64,3)
names=['图1_从类器官放电到游戏决策','图2_神经输入与策略更新','图3_配对训练与留存评估','图4_参数保存与重载核验']
for key,name in [('plain','msyh.ttc'),('bold','msyhbd.ttc'),('mono','consola.ttf')]:
    pdfmetrics.registerFont(TTFont(key,'C:/Windows/Fonts/'+name,subfontIndex=0))
ink='#152E35';gray='#42585D';teal='#006B63';light='#E9F3F0';line='#BDCDCA';grid='#DCE5E3';bg='#F3F6F5';blue='#315470';pblue='#EAF0F5';white='#FFFFFF'
W=560
def start(i,h):
    global c,H,stem
    H=h;stem=names[i-1];c=Canvas(str(output/(stem+'.pdf')),pagesize=(W,H),pageCompression=1);c.setTitle(stem)
    rect(0,0,W,H,white)
def end():
    c.showPage();c.save()
    with pdfium.PdfDocument(output/(stem+'.pdf')) as d:
        p=d[0];b=p.render(scale=4.15);b.to_pil().save(output/(stem+'.png'),dpi=(420,420));b.close();p.close()
def tx(x,y,s,sz=13,col=ink,face='plain',align='l',limit=None):
    s=str(s);wid=pdfmetrics.stringWidth(s,face,sz)
    if limit is not None:assert wid<=limit+.1,(stem,s,wid,limit)
    left=x if align=='l' else x-wid if align=='r' else x-wid/2
    assert left>=-.5 and left+wid<=W+.5 and 0<y<=H,(stem,s,left,y)
    c.setFillColor(HexColor(col));c.setFont(face,sz)
    {'l':c.drawString,'r':c.drawRightString,'c':c.drawCentredString}[align](x,H-y,s)
def rect(x,y,w,h,fill,stroke=None,r=0):
    c.setFillColor(HexColor(fill))
    if stroke:c.setStrokeColor(HexColor(stroke));c.setLineWidth(.85)
    c.roundRect(x,H-y-h,w,h,r,fill=1,stroke=bool(stroke))
def path(points,col=line,width=1,dash=None):
    c.saveState();c.setStrokeColor(HexColor(col));c.setLineWidth(width);c.setLineCap(1);c.setLineJoin(1)
    if dash:c.setDash(dash)
    p=c.beginPath();p.moveTo(points[0][0],H-points[0][1])
    for x,y in points[1:]:p.lineTo(x,H-y)
    c.drawPath(p);c.restoreState()
def arrow(pts,col=teal,w=1.45):
    path(pts,col,w);(a,b),(x,y)=pts[-2:];ang=atan2(y-b,x-a)
    path([(x-5*cos(ang-.53),y-5*sin(ang-.53)),(x,y),(x-5*cos(ang+.53),y-5*sin(ang+.53))],col,w)
def dot(x,y,r,color):
    c.setFillColor(HexColor(color));c.circle(x,H-y,r,fill=1,stroke=0)
def heading(letter,title,y,meta=''):
    tx(7,y,letter,15,teal,'bold');tx(30,y,title,17,ink,'bold')
    if meta:tx(W-7,y,meta,12,gray,align='r')

# Figure 1. Real recording, the corresponding four numbers, and a paired estimate.
start(1,349)
heading('A','从电极记录到一次游戏决策',23,'芯片 22021')
path([(7,34),(553,34)])
tx(7,56,'电极记录类器官放电',14,ink,'bold');tx(181,56,'压成四个读数',14,ink,'bold');tx(368,56,'与游戏状态一起读入',14,ink,'bold')
x0,x1,y0,y1=34,149,77,129
for z in (0,.5,1):path([(x0+z*(x1-x0),y0),(x0+z*(x1-x0),y1)],grid,.65)
path([(x0,y0),(x0,y1),(x1,y1)],gray,.8)
for t,ch in zip(data['ts'],data['ch']):
    x=x0+t*(x1-x0);y=y1-(ch-1)/321*(y1-y0);path([(x,y-2),(x,y+2)],teal,1.35)
tx(34,73,'电极通道',10.2,gray);tx(29,83,'322',10.4,gray,align='r');tx(29,130,'1',10.4,gray,align='r');tx(34,144,'0',11,gray)
tx(149,144,'1 秒',11,gray,align='r');tx(7,169,'44 次放电 · 32 个活跃通道',11.6,gray)
arrow([(153,105),(174,105)])
for i,(label,value) in enumerate(zip(['放电率','活跃通道占比','时间集中度','空间分散度'],data['state2']['baseline'])):
    y=79+i*20;tx(181,y,label,12.5);tx(325,y,f'{value:.3f}'.replace('-','−'),12.7,teal,'bold',align='r')
tx(181,169,'同一秒的标准化读数',11.6,gray)
arrow([(334,105),(361,105)])
rect(368,81,68,39,light,stroke=line,r=4);tx(402,106,'模型',15,teal,'bold','c')
rect(488,81,64,39,bg,stroke=line,r=4);tx(520,106,'游戏',15,ink,'bold','c')
arrow([(437,100),(484,100)]);tx(461,88,'动作',11.4,gray,align='c')
arrow([(519,120),(519,150),(401,150),(401,124)],gray);tx(460,144,'下一局面',11.4,gray,align='c')
tx(368,169,'历史记录按窗口逐步回放',11.6,gray)
path([(7,186),(553,186)],line,1.15)
heading('B','相同决策次数下多探索了多少格',210,'探索性比较')
tx(7,231,'共同对照为神经输入全零组。每组 16 个种子 × 相同的 48 张地图，每局 6 次决策。',12,gray)
tx(7,256,'输入条件',12.2,gray,'bold');tx(234,256,'每局新增格数差',12.2,gray,'bold','c')
tx(452,256,'均值差 [95% 区间]',12.2,gray,'bold','r');tx(551,256,'同向种子',12.2,gray,'bold','r')
xf=lambda v:148+v*55
for t in range(4):
    x=xf(t);path([(x,267),(x,321)],gray if t==0 else grid,1 if t==0 else .7);tx(x,339,t,11.4,gray,align='c')
for index,(est,label) in enumerate(zip(results,['刺激前自发','零输出时仍记录'])):
    v,lo,hi,n=est;y=279+index*31;tx(7,y+5,label,13.3,ink)
    path([(xf(lo),y),(xf(hi),y)],teal,2.2)
    for z in (lo,hi):path([(xf(z),y-3.8),(xf(z),y+3.8)],teal,1.6)
    dot(xf(v),y,4,teal);tx(452,y+5,f'+{v:.2f}  [{lo:.2f}, {hi:.2f}]',13.1,teal,'bold','r');tx(551,y+5,f'{n}/16',13.4,ink,'bold','r')
end()

# Figure 2. A single recorded decision follows two different training targets.
start(2,226)
tx(7,22,'一次决策怎样用于学习',17,ink,'bold');tx(553,22,'确认性实验 · 原样组',12,gray,align='r')
path([(7,32),(553,32)])
tx(7,53,'这一步读到什么',13.4,gray,'bold');tx(199,53,'两个网络各做什么',13.4,gray,'bold');tx(386,53,'按整批轨迹更新',13.4,gray,'bold')
rect(7,65,156,111,bg,r=4);tx(17,83,'游戏状态  state1',13.5,ink,'bold')
tx(17,102,'神经读数  state2',13.5,teal,'bold')
for j,(label,v) in enumerate(zip(['放电率','活跃通道占比','时间集中度','空间分散度'],row['prepared']['wetware_features'])):
    y=119+j*16;tx(17,y,label,11.8,gray);tx(151,y,f'{v:.3f}',12.5,teal,'mono','r')
path([(163,118),(179,118),(179,83)],gray,1.25);path([(179,118),(179,146)],gray,1.25)
arrow([(179,83),(194,83)],gray);arrow([(179,146),(194,146)],gray)
tx(199,83,'策略网络选动作',15,ink,'bold');tx(199,103,'移动一格，耗时 1 步',12.9,gray)
tx(199,125,f'游戏得分  +{row["reward"]:.6f}',12,teal)
tx(199,153,'价值网络估后续得分',15,ink,'bold');tx(199,174,f'当步估值  {row["value"]:.5f}',12.9,gray)
arrow([(341,84),(378,84)]);arrow([(341,147),(378,147)])
tx(386,83,'策略增量  LoRA',15,teal,'bold');tx(386,104,f'用优势 +{row["advantage"]:.5f} 更新',12.8,gray)
tx(386,153,'价值网络参数',15,teal,'bold');tx(386,174,f'拟合目标 {row["return"]:.5f}',12.8,gray)
path([(7,187),(553,187)]);tx(7,210,'全零组在两个入口都置零。三组仅用 R1 训练，附加反馈 R2 的权重 β = 0。',13.0,ink)
end()

# Figure 3. Bar widths are real game tick durations on the identical held-out map.
start(3,267)
heading('A','固定决策次数',22,'六组探索性实验 · 48 张留存地图')
tx(7,45,'同一种子和地图的两条实测轨迹。每段代表一次决策，段宽表示游戏用时。',12.3,gray)
origin,step=102,28
for i,(data_track,lbl) in enumerate(zip(tracks,['输入全零','刺激前自发'])):
    y=62+i*31;tx(7,y+16,lbl,13.1,ink,'bold');prev=0
    for k,(tick,verb) in enumerate(data_track):
        width=(tick-prev)*step
        rect(origin+prev*step,y,width-1.7,23,pblue if i==0 else light)
        tx(origin+(tick+prev)*step/2,y+16,{'move':'移动','navigate':'导航','wait':'等待','gather':'采集'}[verb],11.6,blue if i==0 else teal,'bold','c')
        prev=tick
    tx(origin+prev*step+9,y+16,f'第 {prev} 步',12.6,blue if i==0 else teal,'bold')
for tick in [0,2,4,6,8,10,12]:
    x=origin+tick*step;path([(x,118),(x,122)],gray,.7);tx(x,137,tick,11.2,gray,align='c')
path([(origin,118),(origin+12*step,118)],gray,.85);tx(548,137,'游戏时间步',11.3,gray,align='r')
path([(7,150),(553,150)])
heading('B','固定游戏时长',174,'确认性实验 · 同场景、同种子、同地图配对')
for x,a,b in [(7,'32 个种子 × 3 组','每个模型训练 4 批 × 12 次'),(204,'冻结 96 个模型','每个模型测 3 种开局 × 64 图'),(410,'18,432 局','分别做两项配对比较')]:
    tx(x,201,a,15.2,ink,'bold');tx(x,221,b,12.2,gray)
arrow([(158,197),(193,197)],teal);arrow([(362,197),(397,197)],teal)
rect(7,235,546,28,light,r=3);tx(17,254,'到第 60 个游戏时间步读数，提前死亡取终局值。',13.4,teal,'bold')
end()

# Figure 4. The train/freeze records and the two distinct verification procedures.
start(4,250)
heading('A','从训练记录找到最终参数',22,'按训练种子和输入组定位')
path([(7,32),(553,32)])
tx(7,55,'登记与逐批日志',15,ink,'bold');tx(279,55,'第 4 批后的参数快照',15,ink,'bold')
arrow([(210,50),(267,50)],teal)
for x,y,file,meaning in [(7,80,'registration.json','种子与组别'),(7,101,'train-1…4.jsonl','逐步轨迹'),(7,122,'update.json','优化步数'),(279,80,'adapter/  critic.pt','策略、价值参数'),(279,101,'optimizers.pt','优化器与随机状态'),(279,122,'policy.json  manifest.json','身份、校验值')]:
    tx(x,y,file,12.7,teal,'mono')
    tx(254 if x==7 else 553,y,meaning,12.1,gray,align='r')
path([(7,137),(553,137)])
heading('B','怎样核对保存的结果',161)
path([(273,177),(273,241)],line,.85)
tx(7,186,'重新载入模型，重新选动作',13.9,ink,'bold')
tx(7,209,'288 局 · 9,351 次决策',16,teal,'bold')
tx(7,230,'抽查原登记地图，动作差异为 0。',12,gray)
tx(289,186,'交给游戏重放已存动作',13.9,ink,'bold')
tx(289,209,'18,432 局 · 669,715 个动作',15.4,ink,'bold')
tx(289,230,'探索、存活与补给终点均一致。',12,gray)
end()
