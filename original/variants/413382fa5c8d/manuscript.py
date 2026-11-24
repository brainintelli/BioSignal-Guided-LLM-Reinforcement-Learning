from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from docx import Document
import hashlib,posixpath
w=Path(__file__).parent
out=Path('output/论文_湿件训练有效性_20261120')
src=out/'类器官神经信号对语言模型探索行为的影响（原理图深化版）.docx'
caption='图 1　芯片 22021 的离线训练与配对评估。中间每条短线代表一个电极通道的一次放电。在首次刺激前记录的第 58 至 59 秒，322 个通道中有 32 个活跃。模型在训练和测试时逐步读取历史记录，实时刺激是后续拟验证的路径。附加反馈 R2 的训练权重为零。下方的点是每局新增格数的配对均值差，横线是 95% 置信区间。正向种子指在 48 张地图上平均多探索的训练种子。'
ns=dict(w='http://schemas.openxmlformats.org/wordprocessingml/2006/main',r='http://schemas.openxmlformats.org/officeDocument/2006/relationships',wp='http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',a='http://schemas.openxmlformats.org/drawingml/2006/main',pic='http://schemas.openxmlformats.org/drawingml/2006/picture')
def xp(e,query):return e.xpath(query,namespaces=ns)
def key(nsname,k):return '{'+ns[nsname]+'}'+k
with ZipFile(src) as z,ZipFile(w/'draft.docx','w') as target:
 root=E.fromstring(z.read('word/document.xml'))
 cap=next(p for p in xp(root,'/w:document/w:body/w:p') if ''.join(xp(p,'.//w:t/text()')).startswith('图 1　芯片 22021'))
 pic=cap.getprevious()
 rel=xp(pic,'.//a:blip')[0].get(key('r','embed'))
 media=next(n.get('Target') for n in E.fromstring(z.read('word/_rels/document.xml.rels')) if n.get('Id')==rel)
 media=posixpath.normpath(posixpath.join('word',media));assert media=='word/media/image1.png'
 for e in xp(pic,'./w:pPr/w:pageBreakBefore'):e.getparent().remove(e)
 for node in xp(pic,'.//wp:extent|.//pic:spPr/a:xfrm/a:ext'):
  node.set('cy',str(round(int(node.get('cx'))*487/780)))
 xp(pic,'.//wp:docPr')[0].set('descr','芯片离线放电记录经四项读数输入游戏模型；相对信号屏蔽，两组平均多探索1.99格和2.16格。')
 ts=xp(cap,'.//w:t');ts[0].text=caption
 for t in ts[1:]:t.text=''
 for i in z.infolist():
  blob=(w/'figure.png').read_bytes() if i.filename==media else E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True) if i.filename=='word/document.xml' else z.read(i.filename)
  target.writestr(i,blob)
doc=Document(w/'draft.docx');old=Document(src)
assert len(doc.paragraphs)==len(old.paragraphs)
assert len(doc.inline_shapes)==len(old.inline_shapes) and len(doc.tables)==len(old.tables)
assert [p.text for p,q in zip(doc.paragraphs,old.paragraphs) if p.text!=q.text]==[caption]
(w/'base.sha256').write_text(hashlib.sha256(src.read_bytes()).hexdigest())
