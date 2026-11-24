from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
import re,shutil,hashlib
w=Path(__file__).parent
orig=Path('output/论文_湿件训练有效性_20261120/main.docx');stage=w/'main.docx'
wr='http://schemas.openxmlformats.org/wordprocessingml/2006/main';n={'w':wr}
def ps(root):return root.xpath('/w:document/w:body/w:p',namespaces=n)
def txt(p):return ''.join(p.xpath('.//w:t/text()',namespaces=n))
def update(p,body):
 t=p.xpath('.//w:t',namespaces=n);t[0].text=body
 for v in t[1:]:v.text=''
with ZipFile(orig) as z,ZipFile(stage,'w') as out:
 root=E.fromstring(z.read('word/document.xml'))
 orig_caps=[txt(p) for p in ps(root) if re.match(r'(图|表) \d+',txt(p))]
 removed=0;updated=0
 for p in ps(root):
  s=txt(p)
  if p.xpath('string(w:pPr/w:pStyle/@w:val)',namespaces=n)=='SourcePath':
   p.getparent().remove(p);removed+=1
  elif s.startswith('数据总表 data_table.csv 的每行'):
   update(p,'随文数据总表列出各项实验的对照方向、单位、样本数和置信区间。附录给出训练参数，以及模型作出一次决策时实际收到的完整提示词。');updated+=1
  elif s.startswith('随文 data_table.csv'):
   update(p,s.replace('随文 data_table.csv 为可筛选的完整数值表。','随文数据总表可按实验、对照和指标筛选。').replace('所有文件路径均以项目根目录为起点。',''));updated+=1
 assert removed==14 and updated==2,(removed,updated)
 assert [txt(p) for p in ps(root) if re.match(r'(图|表) \d+',txt(p))]==orig_caps and len(orig_caps)==10
 assert len(root.xpath('//w:tbl',namespaces=n))==6
 for i in z.infolist():out.writestr(i,E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True) if i.filename=='word/document.xml' else z.read(i))
with ZipFile(stage) as z,ZipFile(orig) as inp:
 assert z.testzip() is None
 assert sorted(z.namelist())==sorted(inp.namelist())
 assert [k for k in z.namelist() if z.read(k)!=inp.read(k)]==['word/document.xml']
 texts=' '.join(z.read(k).decode('utf8') for k in z.namelist() if k.endswith('.xml') and k.startswith('word/') and '/styles' not in k)
 for word in ['output/','standalone/','docs/MEA','数据来源','所有文件路径','项目相对路径','线程 A','01a1e3f0-','01a1f6d9-']:assert word not in texts,word
 assert len(z.read('word/_rels/document.xml.rels').split(b'TargetMode="External"'))-1==5
(w/'original_sha256.txt').write_text(hashlib.sha256(orig.read_bytes()).hexdigest())
print('Preserved 10 captions, 6 tables, 5 academic references and original media. Removed',removed,'source lines.')
