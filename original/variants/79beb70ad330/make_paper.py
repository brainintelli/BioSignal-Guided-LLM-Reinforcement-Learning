from pathlib import Path
from zipfile import ZipFile
from copy import deepcopy
from lxml import etree
from docx import Document
from docx.shared import Cm,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
import re,tarfile,json
w=Path(__file__).resolve().parent
out=Path('output/论文_湿件训练有效性_20261120')
src=next(out.glob('*流程核订版）.docx'))
dest=out/'类器官神经信号对语言模型探索行为的影响（算法与复现版）.docx'
d=Document(src);orig=d.paragraphs;anchor=orig[33]._p
def num(m):return '图'+m[1]+str(int(m[2])+3)
for par in orig:
    for r in par.runs:
        s=re.sub(r'图(\s*)([234])(?=\D|$)',num,r.text)
        if s!=r.text:r.text=s
# Preserve the source's WPS layout and fields, replacing only changed document parts.
def place(p):
    anchor.addprevious(p._p)
    return p
def p0(text='',style=None):
    p=place(d.add_paragraph(style=style))
    if text:p.add_run(text)
    return p
def putfont(run,size,bold=False,color='172E38',east='宋体'):
    run.font.name='Cambria';run.font.size=Pt(size);run.bold=bold;run.font.color.rgb=RGBColor.from_string(color)
    run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),east)
def newheading(s,fresh=False):
    p=p0(style='Heading 2');p._p.remove(p._p.pPr);p._p.insert(0,deepcopy(orig[25]._p.pPr));p.paragraph_format.page_break_before=fresh
    r=p.add_run(s);putfont(r,15,True,'172E38','微软雅黑');return p
def prose(s):
    p=p0();f=p.paragraph_format;f.first_line_indent=Pt(22);f.line_spacing=1.4;f.space_before=Pt(0);f.space_after=Pt(5);f.widow_control=True;f.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
    putfont(p.add_run(s),11);return p
def figure(stem,caption):
    p=p0();f=p.paragraph_format;f.alignment=WD_ALIGN_PARAGRAPH.CENTER;f.space_before=Pt(3);f.space_after=Pt(2);f.line_spacing=1;f.keep_with_next=True
    shape=p.add_run().add_picture(str(out/'figures'/(stem+'.png')),width=Cm(14.15));shape._inline.docPr.set('descr',caption)
    p=p0();f=p.paragraph_format;f.line_spacing=1.25;f.space_before=Pt(1);f.space_after=Pt(7);f.keep_together=True
    putfont(p.add_run(caption),9.2,False,'354A51')
def rule(p,where):
    borders=OxmlElement('w:pBdr');bar=OxmlElement('w:'+where)
    for k,v in [('val','single'),('sz','7'),('space','4'),('color','72A39B')]:bar.set(qn('w:'+k),v)
    borders.append(bar);p._p.get_or_add_pPr().append(borders)
def algo(title,rows):
    p=p0();f=p.paragraph_format;f.space_before=Pt(7);f.space_after=Pt(4);f.keep_with_next=True;rule(p,'top');putfont(p.add_run(title),11,True,'08675F','微软雅黑')
    for i,(level,text) in enumerate(rows,1):
        p=p0();f=p.paragraph_format;f.line_spacing=1.23;f.space_before=Pt(0);f.space_after=Pt(1.6)
        f.left_indent=Pt(23+level*13);f.first_line_indent=Pt(-22);f.tab_stops.add_tab_stop(Pt(23+level*13));f.keep_with_next=i<len(rows)
        putfont(p.add_run(f'{i:02d}  '),9,False,'53696C');putfont(p.add_run(text),10.2)
        if i==len(rows):rule(p,'bottom');f.space_after=Pt(8)
source=next(Path('output').rglob('offline-confirmatory-results/verification/evaluated-evidence.tar.gz'))
with tarfile.open(source) as f:
    mm={m.name.removeprefix('./'):m for m in f.getmembers()}
    pref=next(k[:-len('train-1.jsonl')] for k in mm if k.endswith('351855818/ordered/train-1.jsonl'))
    row=json.loads(f.extractfile(mm[pref+'train-1.jsonl']).readline());up=json.load(f.extractfile(mm[pref+'update.json']))
assert row['action']['verb']=='move' and row['duration']==1 and row['contribution']==0 and up['rounds'][0]['decisions']==12
newheading('3.5　一次决策怎样变成一次参数更新',True)
prose('同一个神经读数有两个去处。策略网络用它选动作，价值网络用它估计从当前局面起还能得到多少分。图 2 标出两处输入。若只在一处屏蔽读数，另一处仍能利用信号，组间比较就回答不了原来的问题。')
figure('图2_神经输入与策略更新','图 2　神经信号进入策略与价值估计的两条路径。全零组在两个入口同时置零。游戏执行动作后返回新局面与得分 R1，同批实际回报用于校正价值估计。')
algo('算法 1　用一批游戏决策更新策略',[
(0,'FOR 每个训练种子与输入组：载入共同的初始参数。'),
(1,'FOR 每批（共 4 批）：重置为登记的训练地图，收集 12 次决策。'),
(2,'按组别取四个神经读数，与游戏状态一同送入策略和价值网络。'),
(2,'采样动作及目标，记录实际用时 d、游戏得分 r、价值估计 V 和终止标记。'),
(2,'反向遍历本批。死亡时后继价值 V′ 取 0；批末仍存活则用后继局面估值。'),
(2,'令 δ = r + γᵈV′ − V。批末或终止时 A = δ，其余步 A = δ + (γλ)ᵈ A′，A′ 为下一步优势。'),
(2,'写出轨迹，用优势 A 更新策略，用目标 V + A 拟合价值网络。'),
(1,'每批策略更新 2 步、价值网络更新 16 步，共 8 步与 64 步。')])
prose(f'用真实日志读算法。种子 351855818 的原样组在首批第 1 步移动一格，经过 1 个游戏时间步，得到 {row["reward"]:.6f} 分。价值网络原先估计为 {row["value"]:.5f}，按整批后续行动算出的优势为 {row["advantage"]:+.5f}，对应的拟合目标是 {row["return"]:.5f}。式中 γ = 0.99985、λ = 0.999，两者都按游戏时间步折算。')
newheading('3.6　冻结模型后在相同地图上比较',True)
prose('确认性实验把训练和测量分开。每个种子训练三份策略，各自保留训练时的信号读法；评估时锁定参数，让三者在同一开局、同一张地图行动。地图容易或困难，对同一对策略都是共同条件。')
figure('图3_配对训练与留存评估','图 3　芯片 22021 的确认性实验流程。三组分别按原样、全零和打乱方式读取记录。下方列出两轮实验各自的读数时点。一次导航可以跨越多个游戏时间步。')
algo('算法 2　在留存地图上计算探索差值',[
(0,'FOR 每个场景、训练种子与留存地图：取三组各自冻结的模型。'),
(1,'各组从相同局面起步，依登记顺序或打乱次序回放留存窗口；全零组置零。'),
(1,'逐步执行动作，到第 60 个游戏时间步或死亡即读数。跨过第 60 步的动作按轨迹截取。'),
(1,'统计本局新揭示的格子，每格只计一次；同步保存结局及饮水、进食事件。'),
(0,'逐对计算“原样 − 全零”和“原样 − 打乱”，并在各场景分别汇总。'),
(0,'独立重抽种子与地图 40,000 次。逐项报告配对差、校正区间和同向种子数。')])
prose('这套流程产生 18,432 局记录，即 96 个冻结模型各跑 3 种开局、每种 64 张地图。探索的六项主要比较各用 99.167% 区间；同一场景中两项对照均须区间下界高于零、且各有至少 24/32 个种子同向，才达到预登记的稳定改善判据。')
newheading('3.7　由训练日志追溯到模型参数',True)
prose('每个模型留下一套可重载的参数快照（checkpoint）。训练日志记录角色做过什么、得了几分；参数快照保存最终的策略和价值网络。两者由训练种子、输入组和登记摘要关联，可以从一条动作记录找到产生它的那次训练。')
figure('图4_参数保存与重载核验','图 4　确认性实验的训练记录、最终参数和重载抽查。每个模型训练 4 批，批末分别写日志；第 4 批后保存一份最终参数。独立重载抽查为每个模型在三种开局各跑首张登记地图。')
algo('算法 3　保存参数并检查重载结果',[
(0,'在临时目录保存策略增量、价值网络、两个优化器和随机数状态。'),
(0,'写入组别、模型身份及登记摘要，计算组件的 SHA-256 文件指纹。'),
(0,'写清单后把临时目录改为最终名称；已存在的快照保留。'),
(0,'重载时重算组件指纹，核对模型身份，再装入策略增量和价值网络。'),
(0,'清空策略缓存，核对参数及同批动作概率。冻结模型，在登记地图复测动作。')])
prose('完整性、动作和得分分别核验。原实验的外部清单覆盖 96 × 6 = 576 个组件文件，每个模型还带一份内部清单。独立重载抽查的 9,351 次决策全部一致。另把全量保存动作交给游戏重放，18,432 局的终点一致，其中 64 局的 R1 差异超过 10⁻¹⁰，最大绝对差为 0.01945。')
# Use the specified 11 pt / 1.4 body setting while retaining all source text and headings.
for par in orig:
    if par.style.name!='Normal' or not par.text or not par._p.xpath('w:pPr/w:ind[@w:firstLine]'):
        continue
    par.paragraph_format.line_spacing=1.4
    for run in par.runs:
        if run.text:run.font.size=Pt(11)
for i,par in enumerate(orig):
    if par._p.xpath('.//w:drawing') and i+1<len(orig) and re.match(r'^图\s*\d+',orig[i+1].text):
        par.paragraph_format.keep_with_next=True
    if re.match(r'^表\s*\d+[：　]',par.text):
        par.paragraph_format.keep_with_next=True
for table in d.tables:
    for cell in table.rows[0].cells:
        for pp in cell.paragraphs:pp.paragraph_format.keep_with_next=True
tail=d._element.body[-2]
if tail.tag==qn('w:p') and not ''.join(tail.itertext()).strip():
    d._element.body.remove(tail)
# Preserve original DOCX parts that python-docx may not round-trip (WPS, custom metadata, etc.).
inter=w/'constructed.docx';d.save(inter);final=w/'draft.docx'
with ZipFile(src) as a,ZipFile(inter) as b,ZipFile(final,'w') as target:
    note={'word/document.xml','word/_rels/document.xml.rels'}
    old=set(a.namelist());new=set(b.namelist())
    for info in a.infolist():target.writestr(info,b.read(info.filename) if info.filename in note else a.read(info.filename))
    for info in b.infolist():
        if info.filename not in old:target.writestr(info,b.read(info.filename))
check=Document(final);assert len(check.inline_shapes)==len(Document(src).inline_shapes)+3 and len(check.tables)==len(Document(src).tables)
assert [int(m[1]) for q in check.paragraphs if (m:=re.match(r'图\s*(\d+)[：　]',q.text))]==list(range(1,8))
assert len([q for q in check.paragraphs if q.text.startswith('算法 ')])==3
print(final,len(check.paragraphs),len(check.inline_shapes))
