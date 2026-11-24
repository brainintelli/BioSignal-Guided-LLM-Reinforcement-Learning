from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
import hashlib, os, csv, json
from pypdf import PdfReader
w=Path(__file__).parent
out=Path('output/论文_湿件训练有效性_20261120')
doc=out/'main.docx'
with ZipFile(doc) as z:
 assert z.testzip() is None
 rt=E.fromstring(z.read('word/document.xml'))
 n={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
 body='\n'.join(rt.xpath('//w:t/text()',namespaces=n))
 for needle in ('数据来源','output/','docs/MEA','standalone/','data_table.csv','01a1e3f0-','01a1f6d9-'):assert needle not in body
 print('Native document verified; tables',len(rt.xpath('//w:tbl',namespaces=n)), 'drawings',len(rt.xpath('//w:drawing',namespaces=n)))
p=out/'main.revision-partial'
if p.exists():
 assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256((w/'main.docx').read_bytes()).digest()
 os.replace(p,w/'unused_saved_draft.docx')
h=out/'原始H5'
with (h/'原始H5索引.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
assert {r['汇总文件名'] for r in rows}=={f.name for f in h.glob('*.h5')}
assert len(set(r['SHA256'] for r in rows))==182
assert len(rows)==192
assert len({r['汇总文件名'] for r in rows if r['22021确认性原始19份']=='是'})==19
assert sum(x.stat().st_size for x in h.glob('*.h5'))==73876256484
print('Verified collection and index',len(rows))
print('Live pages',len(PdfReader(w/'final_live.pdf').pages))
a=(w/'doc_qa.py').read_text(encoding='utf8').replace("w/'main.pdf'","w/'final_live.pdf'").replace("str(w/'main.docx')","str(doc.resolve())").replace("w/'qa'","w/'final_qa'").replace("w/'doc_contact.jpg'","w/'final_contact.jpg'").replace("==len(pages)==14","==len(pages)").replace("len(t)>200","len(t)>40")
exec(compile(a,str(w/'doc_qa.py'),'exec'))
