from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
import posixpath,hashlib
w=Path(__file__).parent
out=Path('output/论文_湿件训练有效性_20261120')
src=out/'类器官神经信号对语言模型探索行为的影响（1）.docx'
cap='芯片 22021 的离线训练与配对评估。B 中的芯片为示意图，放电点与四项输入值来自同一秒的记录。时间集中度描述放电在这一秒内是否成簇，空间熵描述放电分散在多少通道。D 中两轮结果分别统计，正向种子数按每个种子在 48 张地图上的平均差判断。'
ns=dict(w='http://schemas.openxmlformats.org/wordprocessingml/2006/main',r='http://schemas.openxmlformats.org/officeDocument/2006/relationships',wp='http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',a='http://schemas.openxmlformats.org/drawingml/2006/main',pic='http://schemas.openxmlformats.org/drawingml/2006/picture')
def xp(p,s):return p.xpath(s,namespaces=ns)
def q(k):pre,n=k.split(':');return '{%s}%s'%(ns[pre],n)
def get(p,name):
 n=p.find(q('w:'+name))
 if n is None:n=E.SubElement(p,q('w:'+name))
 return n
def tx(p):return ''.join(xp(p,'.//w:t/text()'))
with ZipFile(src) as z,ZipFile(w/'draft.docx','w') as f:
 tree=E.fromstring(z.read('word/document.xml'))
 paras=xp(tree,'/w:document/w:body/w:p');p=next(v for v in paras if tx(v).startswith('图 1：游戏、语言模型'))
 image=p.getprevious()
 b=xp(image,'.//a:blip');assert len(b)==1
 rid=b[0].get(q('r:embed'));rels=E.fromstring(z.read('word/_rels/document.xml.rels'))
 media=posixpath.normpath(posixpath.join('word',next(n.get('Target') for n in rels if n.get('Id')==rid)));assert media.endswith('.png')
 cx=round(15.5*360000);cy=round(cx*1054/760)
 for node in xp(image,'.//wp:extent|.//pic:spPr/a:xfrm/a:ext'):node.set('cx',str(cx));node.set('cy',str(cy))
 xp(image,'.//wp:docPr')[0].set('descr','芯片 22021 的游戏决策与训练闭环、实测放电和四项读数、两轮配对探索结果')
 ir=get(image,'pPr');get(ir,'pageBreakBefore').set(q('w:val'),'1')
 get(ir,'keepNext').set(q('w:val'),'1');get(ir,'jc').set(q('w:val'),'center')
 indent=get(ir,'ind')
 for k in ['firstLine','firstLineChars','hanging','leftChars','rightChars']:
  indent.attrib.pop(q('w:'+k),None)
 for k in ['left','right']:indent.set(q('w:'+k),'-369')
 space=get(ir,'spacing')
 for k,v in [('before','0'),('after','0'),('line','240'),('lineRule','auto')]:space.set(q('w:'+k),v)
 # Preserve character/style properties of the caption and all other manuscript content.
 texts=xp(p,'.//w:t');texts[0].text='图 1　'+cap
 for t in texts[1:]:t.text=''
 pr=get(p,'pPr');get(pr,'keepNext').set(q('w:val'),'0');get(pr,'keepLines').set(q('w:val'),'1')
 for k,v in [('before','40'),('after','40'),('line','240'),('lineRule','auto')]:get(pr,'spacing').set(q('w:'+k),v)
 for name in ['firstLine','firstLineChars']:get(pr,'ind').attrib.pop(q('w:'+name),None)
 for r in xp(p,'./w:r'):
  rp=get(r,'rPr');get(rp,'sz').set(q('w:val'),'18')
 # Keep existing figures with their legends and the two appendix data tables with their sections.
 for other in paras:
  if xp(other,'.//wp:inline') and other is not image:
   get(get(other,'pPr'),'keepNext').set(q('w:val'),'1')
  if tx(other).startswith(('附录 A　','附录 B　')):
   get(get(other,'pPr'),'pageBreakBefore').set(q('w:val'),'1')
 for item in z.infolist():
  data=(w/'plate.png').read_bytes() if item.filename==media else E.tostring(tree,encoding='UTF-8',xml_declaration=True,standalone=True) if item.filename=='word/document.xml' else z.read(item.filename)
  f.writestr(item,data)
(w/'original.sha').write_text(hashlib.sha256(src.read_bytes()).hexdigest())
print(media,cx,cy)
