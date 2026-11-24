from pathlib import Path
from math import atan2,cos,sin
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pypdfium2
out=Path('output/论文_湿件训练有效性_20261120/figures')
for key,font in [('r','msyh.ttc'),('b','msyhbd.ttc')]:
    pdfmetrics.registerFont(TTFont(key,'C:/Windows/Fonts/'+font,subfontIndex=0))
ink='#172E38'; mute='#435A62'; teal='#08675F'; pale='#EAF4F1'; line='#BDCBCD'; gray='#F2F5F5'; white='#FFFFFF'; amber='#835023'
W=660
def start(name,h):
    global c,H,fn
    H=h;fn=out/name;c=Canvas(str(fn.with_suffix('.pdf')),pagesize=(W,H));c.setTitle(name);c.setAuthor('芽野实验项目')
def end():
    c.showPage();c.save()
    d=pypdfium2.PdfDocument(str(fn.with_suffix('.pdf')));p=d[0];b=p.render(scale=3);b.to_pil().save(fn.with_suffix('.png'));b.close();p.close();d.close()
def txt(x,y,s,size=15,color=ink,b=False):
    assert pdfmetrics.stringWidth(s,'b' if b else 'r',size)+x<=W-4,(s,x,size)
    c.setFillColor(HexColor(color));c.setFont('b' if b else 'r',size);c.drawString(x,H-y,s)
def lines(points,col=line,lw=1.1):
    p=c.beginPath();p.moveTo(points[0][0],H-points[0][1])
    for x,y in points[1:]:p.lineTo(x,H-y)
    c.setStrokeColor(HexColor(col));c.setLineWidth(lw);c.setLineJoin(1);c.setLineCap(1);c.drawPath(p)
def ar(points,col=teal):
    lines(points,col,1.9)
    (a,b),(x,y)=points[-2:];ang=atan2(y-b,x-a)
    lines([(x-6*cos(ang-.52),y-6*sin(ang-.52)),(x,y),(x-6*cos(ang+.52),y-6*sin(ang+.52))],col,1.9)
def box(x,y,w,h,col):
    c.setFillColor(HexColor(col));c.roundRect(x,H-y-h,w,h,6,fill=1,stroke=0)
def node(x,y,w,h,a,b,col=gray):
    box(x,y,w,h,col);txt(x+12,y+24,a,16,teal if col==pale else ink,True);txt(x+12,y+45,b,13,mute)
start('图2_神经输入与策略更新',230)
txt(10,23,'同一秒的电极读数，同时进入两个网络',20,ink,True)
lines([(10,36),(650,36)])
node(10,54,167,54,'游戏状态  state1','生命、水分、地图与背包')
node(10,130,167,54,'神经读数  state2','每秒四个数，按组回放',pale)
lines([(177,81),(191,81),(191,157),(177,157)],teal,1.8)
ar([(191,119),(209,119)])
box(212,91,96,56,pale);txt(227,116,'合并输入',16,teal,True);txt(226,135,'同一决策步',13,mute)
lines([(308,119),(329,119),(329,81)],teal,1.8);lines([(329,119),(329,157)],teal,1.8)
ar([(329,81),(347,81)]);ar([(329,157),(347,157)])
node(351,54,139,54,'策略网络  Actor','选择动作和目标',pale)
node(351,130,139,54,'价值网络  Critic','估计未来得分',pale)
ar([(490,81),(506,81)]);ar([(490,157),(506,157)])
node(510,54,140,54,'游戏执行动作','得到 R1 与新局面')
node(510,130,140,54,'估计动作优势','用于这批策略更新')
ar([(579,108),(579,127)],mute)
txt(10,213,'确认性实验只用游戏得分更新策略（R = R1）。附加反馈 R2 的权重 β = 0。',15,ink)
end()
start('图3_配对训练与留存评估',273)
txt(10,23,'从一次训练登记到一次配对比较',20,ink,True)
lines([(10,36),(650,36)])
for x,title,sub1,sub2 in [(10,'登记','32 个训练种子','三组共用初始参数'),(176,'训练','每个种子训练三组','每组 4 批 × 12 次决策'),(342,'冻结','保存 96 个模型','评估期间固定权重'),(508,'配对','同一场景、种子和地图','逐局相减，再估区间')]:
    box(x,53,142,83,pale if x in (176,342) else gray)
    txt(x+9,78,title,17,teal if x in (176,342) else ink,True);txt(x+9,103,sub1,12.7,mute);txt(x+9,123,sub2,12.2,mute)
    if x<508:ar([(x+143,95),(x+161,95)])
txt(10,167,'评估时，在哪一步读出探索格数？',17,ink,True)
lines([(10,180),(650,180)])
txt(12,205,'六组探索性实验',15,ink,True);txt(170,205,'每局至多 6 次决策；每个训练种子对照同一组 48 张地图。',14,mute)
txt(12,237,'确认性实验',15,teal,True);txt(170,237,'3 种开局 × 64 张地图；到第 60 个游戏时间步或死亡时读数。',14,mute)
end()
start('图4_参数保存与重载核验',268)
txt(10,23,'训练结束后，怎样找回同一个策略？',20,ink,True)
lines([(10,36),(650,36)])
for x,label in [(10,'训练记录'),(230,'最终参数目录'),(463,'重载与核验')]:txt(x,66,label,18,teal,True)
ar([(135,61),(209,61)]);ar([(378,61),(443,61)])
for x in (210,443):lines([(x,79),(x,212)])
for x,y,a,b in [(10,97,'registration.json','种子、组别、初始参数摘要'),(10,144,'train-1…4.jsonl','每步动作、回报和优势'),(10,191,'update.json','每批损失、步数与训练前后摘要'),(230,97,'adapter/  与  critic.pt','策略增量与价值网络参数'),(230,144,'optimizers.pt','优化器和随机数状态'),(230,191,'policy.json  与  manifest.json','模型身份与逐文件校验值'),(463,97,'读取清单','核对文件摘要与模型身份'),(463,144,'装入参数','恢复策略和价值网络'),(463,191,'固定条件重测','核对重新产生的动作序列')]:
    txt(x,y,a,14.3,ink,True);txt(x,y+21,b,13,mute)
box(10,230,640,34,pale);txt(22,253,'96 个模型的独立重载抽查覆盖 288 局、9,351 次决策，动作差异为 0。',15,teal,True)
end()
