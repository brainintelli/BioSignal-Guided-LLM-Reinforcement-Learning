from data import *
import math
from functools import lru_cache
ink='#172E38';mut='#52656E';green='#08675F';violet='#574783';orange='#A85520';line='#CED8DA';blue='#30678C'
S=2
@lru_cache(None)
def font(sz,bold=False):return ImageFont.truetype('C:/Windows/Fonts/msyh'+('bd' if bold else '')+'.ttc',int(sz*S))
def new(h):
 global im,dr
 im=Image.new('RGB',(1000*S,h*S),'white');dr=ImageDraw.Draw(im)
def pos(x):return tuple(round(v*S) for v in x)
def t(x,y,s,size=23,c=ink,b=False,anc='lm'):dr.text(pos((x,y)),str(s),font=font(size,b),fill=c,anchor=anc)
def ln(x,y,X,Y,col=line,w=2):dr.line(pos((x,y,X,Y)),fill=col,width=w*S)
def box(x,y,X,Y,fill):dr.rounded_rectangle(pos((x,y,X,Y)),radius=7*S,fill=fill)
def point(x,y,c,r=5):dr.ellipse(pos((x-r,y-r,x+r,y+r)),fill=c)
def arrow(x,y,X,Y,c):
 ln(x,y,X,Y,c,2);ang=math.atan2(Y-y,X-x)
 for a in [ang+2.58,ang-2.58]:ln(X,Y,X+11*math.cos(a),Y+11*math.sin(a),c,2)
def save(n,desc):im.save(Figs/n,dpi=(300,300));return n
new(350)
box(10,10,320,108,'#EDF3F8');box(350,10,655,108,'#EAF5F1');box(685,10,990,108,'#F2F0F7')
for x,h,d in [(27,'电极阵列','采集类器官的电活动'),(368,'历史记录','每秒整理成四个数'),(701,'神经输入  state2','原顺序、打乱或屏蔽')]:
 t(x,38,h,25,ink,True);t(x,78,d,21,mut)
arrow(320,59,349,59,mut);arrow(655,59,684,59,mut)
box(10,202,328,321,'#EEF2F4');box(674,202,990,321,'#EAF5F1')
t(27,235,'生存游戏',27,ink,True);t(27,280,'地图、生命、水和食物',22,mut)
t(693,235,'语言模型',27,ink,True);t(693,280,'选动作，估计未来得分',22,mut)
arrow(838,109,838,200,green);t(850,152,'回放',20,green)
arrow(329,226,672,226,blue);t(500,208,'游戏状态 state1 和得分 R1',20,blue,False,'mm')
arrow(673,292,329,292,green);t(500,270,'动作',20,green,False,'mm')
t(23,154,'附加反馈 R2 仅用于另行的权重实验，主比较权重为 0。',21,mut)
save('fig1_system.png','')
def pick(a,control):
 r=next(x for x in effects if x['arm_a']==a and x['arm_b']==control and x['metric']=='exploration')
 return tuple(float(r[k]) for k in ['mean_difference','ci_low','ci_high'])+(int(r['positive_seeds']),16)
def bp(c):
 r=next(x for x in b['all32'] if x['family']=='primary_endpoints' and x['scene']=='low_health' and x['control']==c)
 return r['delta'],r['lo'],r['hi'],r['favorable'],32
new(360);t(8,17,'实验组减对照组',20,mut,True);t(730,17,'平均差和区间（格）',18,mut,True);t(992,17,'正向数',16,mut,True,'rm')
xx=lambda v:303+(v+3)*26
for v in [-2,0,2,4,6,8,10,12]:
 ln(xx(v),40,xx(v),282,'#74878D' if v==0 else '#E7EBEC',2 if v==0 else 1);t(xx(v),305,str(v),18,mut,anc='mm')
entries=[('22021  自发 − 屏蔽',pick('baseline','masked')),('22021  零输出 − 屏蔽',pick('sham','masked')),('22554  顺序 − 纯 R1',pick('ordered','pure')),('低生命  顺序 − 屏蔽',bp('masked')),('低生命  顺序 − 打乱',bp('shuffled'))]
for i,(title,(v,l,h,k,n)) in enumerate(entries):
 y=61+i*48;co=green if i<2 else blue if i==2 else violet
 t(8,y,title,21,ink);ln(xx(l),y,xx(h),y,co,3);ln(xx(l),y-6,xx(l),y+6,co);ln(xx(h),y-6,xx(h),y+6,co);point(xx(v),y,co,6)
 t(730,y,f'{v:+.2f} [{l:+.2f}, {h:+.2f}]',19,co,True);t(990,y,f'{k}/{n}',18,ink,anc='rm')
ln(5,182,995,182,line,1);t(520,341,'与对照相比，每局平均多发现的格数',20,mut,anc='mm')
save('fig2_forest.png','')
new(285)
for i,(name,total,low,high,timer,pct,res,*_) in enumerate(decomp):
 off=i*500;t(off+7,21,name+' − 屏蔽',23,ink,True);t(off+488,21,f'计时占 {pct:.1f}%',22,orange,True,'rm')
 yy=lambda v:221-v*35800
 ln(off+4,221,off+492,221,line)
 for j,(top,bot,val,lab,color) in enumerate([(total,0,total,'总得分差',blue),(total,res,-timer,'扣除计时',orange),(res,0,res,'剩余分差',green)]):
  x=off+30+157*j;dr.rectangle(pos((x,yy(top),x+94,yy(bot))),fill=color)
  t(x+47,yy(top)-16,f'{val:+.6f}',18,color,True,'mm');t(x+47,252,lab,20,mut,anc='mm')
  if j<2:ln(x+94,yy(top if j==0 else res),x+157,yy(top if j==0 else res),mut,1)
save('fig3_reward.png','')
new(256)
t(5,20,'动作',19,mut,True);t(305,20,'占全部动作的比例',19,mut,True);t(738,20,'屏蔽',19,ink,True,'mm');t(833,20,'自发',19,green,True,'mm');t(944,20,'零输出',19,violet,True,'mm')
arms=[next(g for g in groups if g['chip']=='22021' and g['arm']==k) for k in ('masked','baseline','sham')]
xx=lambda v:95+v*11.2
for v in [0,10,20,30,40,50]:
 ln(xx(v),45,xx(v),213,'#E2E8EA',1);t(xx(v),237,str(v)+'%',16,mut,anc='mm')
for i,(label,verb) in enumerate([('单格移动','move'),('导航','navigate'),('采集','gather'),('等待','wait')]):
 y=64+i*44;t(5,y,label,18,ink,True)
 vals=[g['action_probabilities'][verb]*100 for g in arms]
 for k,(v,co) in enumerate(zip(vals,[ink,green,violet])):
  Y=y+(k-1)*7;ln(xx(0),Y,xx(v),Y,co,2);point(xx(v),Y,co,3);t([738,833,944][k],y,f'{v:.2f}%',20,co,False,'mm')
save('fig4_actions.png','')
