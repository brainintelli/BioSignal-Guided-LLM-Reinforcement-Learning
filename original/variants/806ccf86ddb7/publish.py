import os,csv,hashlib,shutil,re
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from pypdf import PdfReader
w=Path(__file__).parent; out=Path('output/论文_湿件训练有效性_20261120')
dest=out/'湿件训练有效性文档.docx';extra=out/'22021_逐局配对实测.csv'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def body(z):
 rt=E.fromstring(z.read('word/document.xml'));ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
 return '\n'.join(rt.xpath('//w:t/text()',namespaces=ns)),len(rt.xpath('//w:tbl',namespaces=ns)),len(rt.xpath('//w:drawing',namespaces=ns))
with ZipFile(w/'original.docx') as old,ZipFile(w/'draft.docx') as new:
 assert new.testzip() is None and old.namelist()==new.namelist()
 assert [k for k in old.namelist() if old.read(k)!=new.read(k)]==['word/document.xml']
 text,nt,nd=body(new);assert (nt,nd)==(7,4)
 for x in ['数据来源','output/','standalone/','docs/MEA','01a1e3f0-','01a1f6d9-']:assert x not in text
 for x in ['31,017','32,543','32,674','2,304','34.264','35.479','35.796','−0.147','615','附录 D','表 7','669,715']:assert x in text
pdf=PdfReader(w/'draft.pdf');assert len(pdf.pages)==15
for i,p in enumerate(pdf.pages):assert len(p.extract_text().strip())>500
assert len(list(csv.DictReader((w/'paired_complete.csv').open(encoding='utf-8-sig'))))==768
assert sha(dest)==(w/'before.sha256').read_text(), '稿件在校验期间被编辑，需保留最新编辑'
if extra.exists():assert sha(extra)==sha(w/'paired_complete.csv')
else:shutil.copyfile(w/'paired_complete.csv',extra)
tmp=w/'checked-for-publication.docx';shutil.copyfile(w/'draft.docx',tmp);os.replace(tmp,dest)
assert sha(dest)==sha(w/'draft.docx') and sha(extra)==sha(w/'paired_complete.csv')
print(dest.resolve(),extra.resolve(),sep='\n')
