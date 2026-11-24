import csv,json,hashlib,shutil
from pathlib import Path
from copy import deepcopy
from zipfile import ZipFile
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm,Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH as A
from docx.enum.table import WD_TABLE_ALIGNMENT as TA, WD_CELL_VERTICAL_ALIGNMENT
from docx.table import Table
from lxml import etree as E
w=Path(__file__).parent;out=Path('output/论文_湿件训练有效性_20261120');orig=out/'湿件训练有效性文档.docx'
m=json.loads((w/'numbers.json').read_text(encoding='utf8')); clocks=json.loads((w/'clock_means.json').read_text())
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
pairs=read(w/'paired_complete.csv');tb=read(out/'data_table.csv');doc=Document(orig)
(w/'before.sha256').write_text(hashlib.sha256(orig.read_bytes()).hexdigest())
shutil.copyfile(orig,w/'original.docx')
def find(start):return next(p for p in doc.paragraphs if p.text.startswith(start))
def change(p,s):
 for item in list(p._p):
  if item.tag!=qn('w:pPr'):p._p.remove(item)
 p.add_run(s);return p
def caption(p,n,s):
 change(p,'');p.add_run(f'表 {n}　').bold=True;p.add_run(s);p.paragraph_format.keep_with_next=False
def table(widths,rows,style_seed,aligns,foot=False):
 t=doc.add_table(rows=0,cols=len(widths));t.autofit=False;t.alignment=TA.CENTER
 old=t._tbl.tblPr;t._tbl.replace(old,deepcopy(style_seed._tbl.tblPr))
 for col,sz in zip(t.columns,widths):col.width=Cm(sz)
 for i,items in enumerate(rows):
  row=t.add_row();pr=row._tr.get_or_add_trPr();pr.append(OxmlElement('w:cantSplit'))
  if i==0:pr.append(OxmlElement('w:tblHeader'))
  for j,(c,v,sz) in enumerate(zip(row.cells,items,widths)):
   c.width=Cm(sz);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   p=c.paragraphs[0];p.style='TableBody';p.paragraph_format.space_after=Pt(0);p.paragraph_format.keep_with_next=False;p.paragraph_format.line_spacing=1.12
   p.alignment=aligns[j]
   r=p.add_run(str(v));r.bold=i==0 or (foot and i==len(rows)-1);r.font.size=Pt(9.4)
   if i==0 or (foot and i==len(rows)-1):
    bg=OxmlElement('w:shd');bg.set(qn('w:fill'),'E4EEF0' if i==0 else 'F0F5F5');c._tc.get_or_add_tcPr().append(bg)
 return t
def signed(n,d=2):
 return f'{n:+.{d}f}'.replace('-','−') if n else f'{n:.{d}f}'
def bound(v):
 return f'{float(v):.2f}'.replace('-','−')
def rowvals(k,scene,arm):
 if scene is None:
  sel=next(z for z in tb if z['实验']=='A 六决策' and z['指标']=='exploration' and ('自发' if arm=='baseline' else '零输出') in z['组别或配对方向'] and '纯 R1' in z['组别或配对方向'])
  num=[m['tot'][1 if arm=='baseline' else 2]/768,m['tot'][0]/768]
  label=('自发' if arm=='baseline' else '零输出')+'／屏蔽'
  obs='／'.join(f'{v:.3f}' for v in num)
 else:
  metric='cells' if scene=='low_health' else 'survived'
  sel=next(z for z in tb if z['实验']=='B 全样本' and z['指标']==metric and z['组别或配对方向']==f'{scene} ordered vs {arm}')
  label='顺序／'+('屏蔽' if arm=='masked' else '打乱')
  cv=m['confirm'][scene];n=0 if metric=='cells' else 1
  num=[cv[a][n] for a in ['ordered',arm]]
  obs='／'.join(f'{v/2048:.3f}' if metric=='cells' else f'{v:,}' for v in num)
  assert abs((num[0]-num[1])/2048*(1 if n==0 else 100)-float(sel['估计值']))<1e-9
 return [k,label,obs,signed(float(sel['估计值']))+' ['+bound(sel['区间下界'])+', '+bound(sel['区间上界'])+']',sel['正向种子数']+'/'+sel['训练种子数']]
p=find('在 22021 的 48 张测试地图上')
channel={}
for group in ['masked','baseline','sham']:
 root=Path('output/mea_six_arm_r1_off_20261120')/('baseline' if group=='baseline' else 'main')
 with (root/'models/102600001'/group/'holdout.jsonl').open(encoding='utf8') as f:episode=json.loads(next(f))
 assert episode['world']==172640420
 step=episode['rows'][0];assert step['action']['verb']=='move'
 channel[group]=tuple(round(float(v),3) for v in step['consumed_features'])
assert channel=={'masked':(0,0,0,0),'baseline':(-.147,-.131,-.315,-.134),'sham':(.495,.502,.077,.230)}
find('游戏状态（state1）').add_run('第 4 节列出同一开局下三组的实际输入。')
np=find('配对自举（paired bootstrap）').insert_paragraph_before('训练种子 102600001 的三个模型从地图 172640420 的同一位置出发。第一次决策时，按放电频率、活跃通道占比、时间集中程度和通道分散程度的顺序，屏蔽组读到（0，0，0，0），自发组读到（−0.147，−0.131，−0.315，−0.134），零输出组读到（0.495，0.502，0.077，0.230）。这些是校准后的实际输入值。三组这一步都选择了单格移动。')
np.alignment=A.JUSTIFY
change(p,'芯片 22021 上，两种实测神经输入都提高了平均探索格数。每组用 16 个训练种子各玩同样的 48 张地图，每局作 6 次决策。屏蔽组在 768 局中累计新揭示 31,017 格，自发组为 32,543 格，零输出组为 32,674 格。分别减去屏蔽组，再除以 768，每局多出 1.99 格和 2.16 格（图 2、表 2）。')
old=doc.tables[1]
rows=[['开局与指标','实验组／对照组','两组实测值¹','配对差 [置信区间]','正向种子']]
rows+=[rowvals('六决策·探索',None,a) for a in ['baseline','sham']]
for sc,lab in [('low_health','低生命·探索'),('thirst','濒渴·存活'),('hunger','濒饿·存活')]:
 rows.extend(rowvals(lab,sc,a) for a in ['masked','shuffled'])
new=table([2.65,2.9,3.55,5.35,1.85],rows,old,[A.LEFT,A.LEFT,A.CENTER,A.CENTER,A.CENTER]);old._tbl.addprevious(new._tbl);old._tbl.getparent().remove(old._tbl)
caption(find('表 2'),2,'芯片 22021 同种子、同地图的实测对照。¹ 左值为实验组，右值为对照组。探索列为每局新增格数，存活列为活到第 60 刻的局数。前两行为探索性实验，每组 768 局（16×48），95% 区间。后六行为确认性实验，每组每种开局 2,048 局（32×64），校正后 99.167% 区间。存活差以百分点计，所有差值均用未取整的原始数据计算。')
ac=m['actions'];p=find('读到神经记录后')
change(p,f"每组 4,608 次实际决策中，屏蔽组有 {ac['masked']['move']:,} 次单格移动，零输出组有 {ac['sham']['move']:,} 次，占比从 47.89% 降至 45.20%。两组的地标导航从 {ac['masked']['navigate']} 次增至 {ac['sham']['navigate']} 次，采集从 {ac['masked']['gather']} 次增至 {ac['sham']['gather']} 次（图 4）。导航一次最多可走四格，这种动作变化与探索增加一致。动作计数本身无法判断某次移动是否经过已走过的区域。")
p=find('固定第 6 个游戏刻后')
q=find('换到独立芯片 22554')
raw22554=[x for x in read(Path('output/mea_six_arm_r1_off_20261120/metrics_by_world.csv')) if x['chip']=='22554']
units={r['arm'] for r in raw22554};assert {'pure','ordered'}<=units
def arm_mean(a):
 rr=[int(y['exploration']) for y in raw22554 if y['arm']==a];assert len(rr)==768;return sum(rr)/len(rr)
q.text=q.text.replace('顺序组对只用 R1 的模型为',f'顺序组每局新揭示 {arm_mean("ordered"):.3f} 格，只用 R1 的模型为 {arm_mean("pure"):.3f} 格，配对差为')
p.text=p.text.replace('在跨越边界的动作中读取当刻地图。',f'在跨越边界的动作中读取当刻地图。此时屏蔽、自发和零输出组每局分别新揭示 {clocks[0]:.3f}、{clocks[1]:.3f} 和 {clocks[2]:.3f} 格。')
p=find('随文数据总表列出')
change(p,'短程实验证据可从单局复算到整组。我们把芯片 22021 三组的六步动作交回游戏引擎，对照各步结束时的状态和游戏得分。相同地图和动作序列共 146 种，重放结果与全部 2,304 局记录吻合。附录 D 列出完整的种子对照，随文逐局表保留全部 768 对地图结果。')
head=doc.add_paragraph('附录 D　从逐局记录复算探索差值','Heading 1');head.paragraph_format.page_break_before=True
p=doc.add_paragraph('同一训练种子在相同的 48 张地图上各运行三种模型。表 7 列出每个种子的原始格数合计。沿列相加、再除以 768，就能从整数还原正文表 2 的每局均值；在每行内比较，则能数出哪些种子受益。')
p.alignment=A.JUSTIFY
rows=[['训练种子','屏蔽组','自发组','零输出组','自发减屏蔽','零输出减屏蔽']]
for s,a,b,c in m['seeds']:rows.append([str(s),f'{a:,}',f'{b:,}',f'{c:,}',signed(b-a,0),signed(c-a,0)])
a,b,c=m['tot'];rows.append(['全部 16 个',f'{a:,}',f'{b:,}',f'{c:,}',signed(b-a,0),signed(c-a,0)])
table([3.1,2.35,2.35,2.35,3.0,3.15],rows,doc.tables[1],[A.LEFT,A.RIGHT,A.RIGHT,A.RIGHT,A.RIGHT,A.RIGHT],True)
p=doc.add_paragraph(style='FigCaption');caption(p,7,'芯片 22021，每格数值为该组在 48 局中新揭示的格数之和。两个差值列均以同一行屏蔽组为对照。自发组有 14 个种子增加、2 个减少；零输出组有 15 个增加、1 个持平。三组共享地图，累计格数允许不同局揭示同一个地图位置。')
p=doc.add_paragraph('单局的结果有升有降。','Heading 2');p.text='同一张地图上的实测读数'
ex=pairs[0]
assert [int(ex[k]) for k in ('训练种子','地图种子','屏蔽组新增格','自发组新增格','零输出组新增格','开局已揭示格数')]==[102083199,172012841,86,71,71,149]
p=doc.add_paragraph('按训练种子、地图编号从小到大排列，首对为种子 102083199、地图 172012841。开局已揭示 149 格。六次决策后，屏蔽组共看见 235 格，自发组和零输出组均看见 220 格。三组各自减去开局的 149 格，得到 86、71、71 格。这一局中两种输入都比屏蔽少探索 15 格。');p.alignment=A.JUSTIFY
cnt=[]
for k in ['自发减屏蔽_格','零输出减屏蔽_格']:
 vals=[int(r[k]) for r in pairs];cnt.append((sum(v>0 for v in vals),sum(v==0 for v in vals),sum(v<0 for v in vals)))
assert cnt==[(101,615,52),(109,602,57)]
p=doc.add_paragraph('全部 768 对中，自发组有 101 对多探索、615 对持平、52 对更少；零输出组相应为 109、602、57 对。多数对局揭示的格数相同，平均增益来自其余对局。种子层面的正向数则先在 48 张图上取平均，再判断方向。');p.alignment=A.JUSTIFY
doc.save(w/'generated.docx')
with ZipFile(w/'original.docx') as a,ZipFile(w/'generated.docx') as b,ZipFile(w/'draft.docx','w') as z:
 for pack in a.infolist(): z.writestr(pack,b.read(pack.filename) if pack.filename=='word/document.xml' else a.read(pack.filename))
print('Written draft',len(pairs),'pairs')
