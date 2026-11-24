from pathlib import Path
import json,math
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor as HC
from reportlab.pdfbase import pdfmetrics as pm
from reportlab.pdfbase.ttfonts import TTFont
import pypdfium2 as pdfium
w=Path(__file__).parent;data=json.loads((w/'data.json').read_text())
for name,fn in [('R','msyh.ttc'),('B','msyhbd.ttc')]: pm.registerFont(TTFont(name,'C:/Windows/Fonts/'+fn,subfontIndex=0))
W,H=760,1054
c=canvas.Canvas(str(w/'plate.pdf'),pagesize=(W,H),pageCompression=1)
c.setTitle('芯片 22021｜类器官信号怎样进入语言模型训练');c.setAuthor('芽野实验项目')
ink='#172E37';mut='#425963';gray='#809198';rule='#CAD6DA'
blue='#265D8F';green='#126859';och='#85521A';pur='#64558B'
pb='#F1F6FB';pg='#EFF7F3';po='#FCF6EA';pp='#F4F1F8'
def rect(x,y,x2,y2,fill,edge=None,r=0):
 c.setFillColor(HC(fill));c.setStrokeColor(HC(edge or fill));c.setLineWidth(.8)
 c.roundRect(x,H-y2,x2-x,y2-y,r,fill=1,stroke=bool(edge))
def T(x,y,s,size=13,color=ink,b=False,anchor='l',limit=None):
 font='B' if b else 'R';s=str(s)
 sw=pm.stringWidth(s,font,size)
 if limit and sw>limit+.05:
  print('Fit',s,round(size*limit/sw,1));size=size*limit/sw
 c.setFont(font,size);c.setFillColor(HC(color))
 getattr(c,{'l':'drawString','c':'drawCentredString','r':'drawRightString'}[anchor])(x,H-y,s)
def line(pts,col=rule,lw=1,dash=False):
 c.setLineWidth(lw);c.setStrokeColor(HC(col));c.setLineJoin(1);c.setLineCap(1);c.setDash([4,4] if dash else [])
 p=c.beginPath();p.moveTo(pts[0][0],H-pts[0][1])
 for x,y in pts[1:]:p.lineTo(x,H-y)
 c.drawPath(p);c.setDash([])
def arrow(pts,col=mut,lw=1.65,dash=False):
 line(pts,col,lw,dash);x,y=pts[-1];a,b=pts[-2];ang=math.atan2(y-b,x-a);al=6
 line([(x-al*math.cos(ang+.53),y-al*math.sin(ang+.53)),(x,y),(x-al*math.cos(ang-.53),y-al*math.sin(ang-.53))],col,lw)
def dot(x,y,r,col):
 c.setFillColor(HC(col));c.circle(x,H-y,r,stroke=0,fill=1)
def header(y,letter,title,col):
 rect(32,y-17,55,y+6,col,r=3);T(43.5,y+.6,letter,15,'#FFFFFF',True,'c')
 T(65,y+1,title,19,ink,True)
def titlebox(x,y,X,Y,title,lines,col,bg):
 rect(x,y,X,Y,bg,r=6);rect(x,y,x+3.5,Y,col,r=1)
 T(x+12,y+22,title,17,col,True,limit=X-x-19)
 for i,s in enumerate(lines):T(x+12,y+42+17*i,s,12.6,ink,limit=X-x-19)
def note(x,y,s,size=12):T(x,y,s,size,mut)
def value(x,y,s,color=ink):T(x,y,s,14,color,False,'r')
# Heading and legend
rect(0,0,W,H,'#FFFFFF');T(32,32,'芯片 22021    /    2026.11.18—11.20',13,mut,True)
T(32,70,'类器官的神经信号怎样进入语言模型训练',30,ink,True)
line([(37,92),(66,92)],blue,1.8);T(75,97,'实线为已执行的数据流',13,mut)
line([(255,92),(286,92)],gray,1.65,True);T(295,97,'虚线为拟议的实时刺激路径',13,mut)
line([(32,108),(728,108)],rule)
# A
header(135,'A','游戏提供局面，模型选择下一步动作',blue)
arrow([(614,167),(614,148),(131,148),(131,167)],blue)
rect(255,139,429,155,'#FFFFFF');T(342,152,'下一步的地图与角色状态',11.7,blue,False,'c')
titlebox(37,168,226,240,'游戏状态  state1',['Godot 提供视野、背包、储备','生命 100，水和饱食各 95'],blue,pb)
titlebox(267,168,491,240,'模型读入  state1 + state2',['Qwen3.5-0.8B 的策略选动作','价值网络估计未来的游戏得分'],blue,pb)
titlebox(533,168,723,240,'执行动作',['选择动作及其目标','移动走 1 格，导航最多走 4 格'],blue,pb)
arrow([(226,202),(266,202)],blue);arrow([(492,202),(532,202)],blue)
# B receives a proposed input at the physical chip and sends real state2 to the policy.
arrow([(88,240),(88,253),(20,253),(20,365),(39,365)],gray,1.25,True)
T(98,261,'拟将游戏状态编码成刺激',11.8,mut)
arrow([(623,319),(623,273),(398,273),(398,241)],green)
T(492,267,'按决策步回放 state2',12.3,green,True)
header(302,'B','从电极记录还原模型实际读到的四个数',green)
rect(39,320,225,479,'#FFFFFF','#B9D6CB',5)
T(50,344,'类器官与电极阵列',16,green,True)
c.drawImage(str(w/'inset.png'),49,H-444,width=161,height=98,mask='auto',preserveAspectRatio=True,anchor='c')
T(132,454,'芯片与组织示意',11.7,mut,False,'c')
T(132,470,'预排刺激或零输出，MEA 均记录',11.5,ink,False,'c')
rect(252,320,470,479,pg,r=5)
T(264,344,'刺激前的 1 秒实测放电',16,green,True)
T(264,361,'第 58–59 秒，纵轴为通道序号',12.4,mut)
x0,x1,y0,y1=292,452,374,431
for val in (0,0.5,1):
 x=x0+val*(x1-x0);line([(x,y0),(x,y1)],'#C5D7D0',.55)
line([(x0,y0),(x0,y1),(x1,y1)],green,1)
for tm,rank in zip(data['ts'],data['ch']):
 x=x0+tm*(x1-x0);y=y1-(rank-1)/321*(y1-y0)
 line([(x,y-1.9),(x,y+1.9)],green,1.3)
T(x0-4,y0+3,'322',10.2,mut,False,'r');T(x0-4,y1+2,'1',10.2,mut,False,'r')
T(x0,y1+14,'0',10.3,mut,False,'c');T((x0+x1)/2,y1+14,'0.5',10.3,mut,False,'c');T(x1,y1+14,'1 秒',10.3,mut,False,'c')
T(264,465,'每条短线为一次放电，共 44 次。',12.5,ink)
rect(499,320,723,479,pg,r=5);T(510,344,'state2 如何取值',16,green,True)
T(510,366,'特征',12,mut,True);T(643,366,'原始值',12,mut,True,'r');T(711,366,'校准后',12,green,True,'r')
for i,(label,r,v) in enumerate(zip(['对数放电率','活跃通道占比','时间集中度','空间熵'],data['raw'],data['state2']['baseline'])):
 y=388+20*i;T(510,y,label,13,ink);value(643,y,f'{r:.3f}');value(711,y,f'{v:.3f}'.replace('-','−'),green)
T(511,469,'右列同时进入策略和价值网络。',12,green)
arrow([(226,397),(250,397)],green);arrow([(471,397),(497,397)],green)
T(40,501,'现场如何采集',14,green,True);T(191,501,'每轮含规律、随机、零输出，顺序预先抽签，共 6 轮。首次刺激前另录 64.08 秒。',12.8,ink)
c.translate(0,-22)
T(40,501,'怎样得到这个窗口',14,green,True);T(191,501,'20 kHz 采集，核对标记和回执。选定 322 个非刺激通道，此秒有 32 个活跃。',13,ink,limit=530)
T(40,522,'怎样隔开训练和测试',14,green,True);T(191,522,'自发记录用第 12–36 秒校准。第 2–54 秒供训练及估值，第 58–64 秒留存。',13,ink)
T(40,543,'怎样比较三组输入',14,green,True);T(191,543,'同一开局首步，屏蔽组四项为 0。零输出组仍读到 0.495、0.502、0.077、0.230。',12.6,ink)
line([(32,560),(728,560)],rule)
# C
header(588,'C','游戏得分怎样改变下一轮策略',och)
titlebox(38,608,248,701,'执行后的游戏回报',['R1 来自存活计时和资源变化','另记模拟动作反馈 R2','R2 权重 β = 0，因此 R = R1'],och,po)
titlebox(273,608,477,701,'按步保存游戏轨迹',['记录状态、动作、奖励和下一状态','GAE 将实际回报与估值比较','估计这一步比原先预期好多少'],och,po)
titlebox(501,608,722,701,'PPO 更新模型',['调整策略的 LoRA 和价值网络','基础模型权重冻结','每个种子训练 4 批，每批 12 步'],pur,pp)
arrow([(249,653),(271,653)],och);arrow([(478,653),(500,653)],pur)
arrow([(723,189),(738,189),(738,598),(144,598),(144,607)],och,1.05)
# paired ports identify the model update without a page-spanning return arrow
T(502,720,'更新后返回 A，评估时冻结参数。',12.5,pur,True)
# D
line([(32,736),(728,736)],rule);header(764,'D','在相同训练种子和地图上检验探索差异',pur)
T(39,790,'探索格数取局末已揭示格数减去开局格数；每局给 6 次决策，先配对再求平均。',14,ink)
line([(352,810),(352,975)],rule,1)
T(39,823,'11 月 18 日  ·  五组探索性试验',17,ink,True)
T(39,847,'8 种子 × 48 地图，每组 384 局。',14,mut)
T(39,871,'自发、零输出相对屏蔽的 R1 差分别为',13,ink)
T(39,892,'+0.004017、+0.004073 游戏分。',14,och,True)
T(39,918,'相对零输出，规律刺激后少 0.81 格，',13,ink)
T(39,938,'随机刺激后多 0.20 格。两项校正后',13,ink)
T(39,958,'99.17% 区间均跨零。',13,ink)
T(373,823,'11 月 20 日  ·  扩至 16 个训练种子',17,ink,True)
T(373,847,'沿用 48 张地图与首批 8 个种子，每组 768 局。',13.4,mut)
T(373,874,'输入组减屏蔽组',12.8,mut,True);T(716,874,'配对差 [95% 区间]  ·  正向种子',12.8,mut,True,'r')
for i,(label,v) in enumerate(zip(['刺激前自发','当轮零输出'],data['est'])):
 y=901+i*31;T(373,y,label,14,ink);T(716,y,f'+{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}] 格    {v[3]}/16',15,green,True,'r')
T(373,959,'屏蔽 40.387 格，自发 42.374 格，零输出 42.544 格。',12.5,mut)
line([(32,977),(728,977)],rule)
T(39,1000,'同一游戏时长的复核',14,blue,True);T(195,1000,'固定在第 6 个游戏刻，自发和零输出相对屏蔽仍多探索 1.21 格和 1.53 格。',13,ink)
T(39,1020,'实测窗口对应训练种子 102600001、地图 172640420 的首次留存决策。三组该步均选择单格移动。',11.5,mut)
c.showPage();c.save()
p=pdfium.PdfDocument(w/'plate.pdf');im=p[0].render(scale=3.5).to_pil();im.save(w/'plate.png',dpi=(252,252));im.thumbnail((1100,1500));im.save(w/'preview.png')
