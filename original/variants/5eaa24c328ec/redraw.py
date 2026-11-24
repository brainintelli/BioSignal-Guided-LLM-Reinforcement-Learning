from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
import csv,json,math
import pypdfium2 as pdfium
work=Path(__file__).resolve().parent
prev=work.parent/'figure1_deep'
data=json.loads((prev/'data.json').read_text())
project=Path.cwd()
asset=project/'output/论文_湿件训练有效性_20261120'
with (asset/'22021_逐局配对实测.csv').open(encoding='utf-8-sig') as f: pairs=list(csv.DictReader(f))
assert len(pairs)==768 and len({(r['训练种子'],r['地图种子']) for r in pairs})==768
arm=['屏蔽','自发','零输出']
means=[sum(int(r[n+'组新增格']) for r in pairs)/len(pairs) for n in arm]
assert all(abs(means[i+1]-means[0]-est[0])<1e-10 for i,est in enumerate(data['est']))
assert (len(data['ch']),len(set(data['ch'])))==(44,32)
for name,f in [('regular','msyh.ttc'),('strong','msyhbd.ttc')]:
 pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+f,subfontIndex=0))
W,H=780,487
c=canvas.Canvas(str(work/'figure.pdf'),pagesize=(W,H),pageCompression=1)
c.setTitle('从类器官放电到游戏决策');c.setAuthor('芽野实验项目')
ink='#172F34';mid='#435A60';light='#D5DEDE';accent='#086D64';soft='#EFF7F5';white='#FFFFFF'
def text(x,y,s,size=15,color=ink,b=False,align='left'):
 c.setFillColor(HexColor(color));c.setFont('strong' if b else 'regular',size)
 getattr(c,{'left':'drawString','right':'drawRightString','center':'drawCentredString'}[align])(x,H-y,str(s))
def path(points,color=ink,width=1.4):
 c.setStrokeColor(HexColor(color));c.setLineWidth(width);c.setLineCap(1);c.setLineJoin(1)
 p=c.beginPath();p.moveTo(points[0][0],H-points[0][1])
 for x,y in points[1:]:p.lineTo(x,H-y)
 c.drawPath(p)
def arrow(points,color=accent,width=1.7):
 path(points,color,width);(x0,y0),(x,y)=points[-2:];d=math.atan2(y-y0,x-x0)
 path([(x-6*math.cos(d-.5),y-6*math.sin(d-.5)),(x,y),(x-6*math.cos(d+.5),y-6*math.sin(d+.5))],color,width)
def box(x,y,w,h,color,rad=0):
 c.setFillColor(HexColor(color));c.roundRect(x,H-(y+h),w,h,rad,fill=1,stroke=0)
def point(x,y,r,col=accent):
 c.setFillColor(HexColor(col));c.circle(x,H-y,r,fill=1,stroke=0)
def image_viewport(file,x,y,ww,hh,region):
 img=ImageReader(file);iw,ih=img.getSize()
 l,t,r,b=region;fac=min(ww/(r-l),hh/(b-t))
 xx=x+(ww-(r-l)*fac)/2;yy=y+(hh-(b-t)*fac)/2
 c.saveState();clip=c.beginPath();clip.rect(xx,H-(yy+(b-t)*fac),(r-l)*fac,(b-t)*fac);c.clipPath(clip,stroke=0)
 c.drawImage(img,xx-l*fac,H-(yy-t*fac)-ih*fac,width=iw*fac,height=ih*fac,mask='auto')
 c.restoreState()
# A single reading line: biological source -> observed data -> policy and game.
box(0,0,W,H,white)
text(27,45,'从类器官放电到游戏决策',31,ink,True)
text(752,43,'芯片 22021  ·  离线回放',15,mid,align='right')
path([(28,62),(752,62)],light,.9)
text(30,97,'电极记录放电',21,ink,True)
text(236,97,'每秒提取四个读数',21,ink,True)
text(501,97,'模型选择游戏动作',21,ink,True)
image_viewport(str(prev/'inset.png'),30,120,161,90,(2,257,1533,896))
arrow([(194,161),(225,161)])
# Every spike mark is drawn directly from the original H5 window.
x0,x1=258,454;y0,y1=122,183
for t in [0,.5,1]:path([(x0+t*(x1-x0),y0),(x0+t*(x1-x0),y1)],light,.65)
path([(x0,y0),(x0,y1),(x1,y1)],mid,.9)
for tm,ch in zip(data['ts'],data['ch']):
 x=x0+tm*(x1-x0);y=y1-(ch-1)/321*(y1-y0)
 path([(x,y-2.4),(x,y+2.4)],accent,1.7)
text(251,y0+4,'322',10.6,mid,align='right');text(251,y1+1,'1',10.6,mid,align='right')
text(257,118,'通道序号',9.7,mid)
text(x0,201,'0',11.5,mid,align='center');text(x1,201,'1 秒',11.5,mid,align='right')
arrow([(465,161),(493,161)])
# The only returning arrow belongs to the actual model/game loop.
box(501,134,101,51,soft,7)
text(551.5,165,'语言模型',18,accent,True,align='center')
arrow([(607,153),(679,153)])
text(645,140,'执行动作',13,mid,align='center')
image_viewport(str(project/'assets/art/hero_directions_v1.png'),683,108,69,78,(0,0,313,313))
text(719,202,'游戏角色',14.4,ink,True,align='center')
arrow([(687,178),(667,178),(667,201),(551,201),(551,186)],mid,1.25)
text(612,195,'新局面',12.5,mid,align='center')
text(30,229,'微电极阵列记录类器官活动。',14.4,mid)
text(30,250,'所示 1 秒内共有 44 次放电。',14.1,mid)
text(236,229,'放电频率、活跃通道占比、',14.4,mid)
text(236,250,'时间集中度和空间分散程度。',14.4,mid)
text(501,229,'模型同时读取神经信号与游戏状态。',14.2,mid)
text(501,250,'训练时用游戏得分 R1 更新策略。',14.4,mid)
# Evidence: one comparison, one metric and a shared baseline.
box(0,277,W,H-277,'#F3F8F6')
text(29,311,'同样做 6 次决策，接入记录后平均多探索约 2 格',21,ink,True)
text(30,338,f'探索性比较以信号屏蔽组为对照（{means[0]:.2f} 格），每组 16 个训练种子各测相同的 48 张地图。',14.2,mid)
text(30,369,'输入条件',13,mid,True)
text(347,369,'相对屏蔽组多探索的格数',13,mid,True,align='center')
text(642,369,'平均差 [95% 区间]',13,mid,True,align='right')
text(750,369,'正向种子',13,mid,True,align='right')
x0,x1=205,483
def pos(v):return x0+(x1-x0)*v/3
for tick in range(4):
 x=pos(tick);path([(x,381),(x,448)],'#829997' if tick==0 else '#D4E1DD',1.3 if tick==0 else .75)
 text(x,467,str(tick),12,mid,align='center')
for i,(label,est) in enumerate(zip(['刺激前自发记录','当轮不刺激，仍记录'],data['est'])):
 y=396+i*38;v,lo,hi,n=est
 text(30,y+6,label,16,ink)
 xL,xR=pos(lo),pos(hi);path([(xL,y),(xR,y)],accent,2.1)
 for x in (xL,xR):path([(x,y-4),(x,y+4)],accent,1.75)
 point(pos(v),y,4.5)
 text(643,y+6,f'+{v:.2f}  [{lo:.2f}, {hi:.2f}]',16,accent,True,align='right')
 text(750,y+6,f'{n}/16',16,ink,align='right')
c.showPage();c.save()
page=pdfium.PdfDocument(work/'figure.pdf')
export=page[0].render(scale=4).to_pil();export.save(work/'figure.png',dpi=(288,288))
preview=page[0].render(scale=1.45).to_pil();preview.save(work/'preview.png')
