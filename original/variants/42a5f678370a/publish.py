from pathlib import Path
from zipfile import ZipFile
from hashlib import sha256
from docx import Document
from PIL import Image
from pypdf import PdfReader
import shutil
w=Path(__file__).parent
out=Path('output/论文_湿件训练有效性_20261120')
src=out/'类器官神经信号对语言模型探索行为的影响（1）.docx'
assert sha256(src.read_bytes()).hexdigest()==(w/'original.sha').read_text()
draft=w/'draft.docx'
p=Document(src);d=Document(draft)
assert len(p.paragraphs)==len(d.paragraphs) and len(p.tables)==len(d.tables) and len(d.inline_shapes)==len(p.inline_shapes)==4
diff=[i for i,(a,b) in enumerate(zip(p.paragraphs,d.paragraphs)) if a.text!=b.text]
assert diff==[22] and d.paragraphs[22].text.startswith('图 1')
assert [[row._tr.xml for row in t.rows] for t in p.tables]==[[row._tr.xml for row in t.rows] for t in d.tables]
with ZipFile(draft) as z,ZipFile(src) as old:
 assert z.testzip() is None and z.namelist()==old.namelist()
 assert set(i for i in old.namelist() if z.read(i)!=old.read(i))=={'word/media/image1.png','word/document.xml'}
 assert z.read('word/media/image1.png')==(w/'plate.png').read_bytes()
r=PdfReader(w/'draft.pdf');assert len(r.pages)==16 and '芯片 22021' in r.pages[3].extract_text()
r2=PdfReader(w/'plate.pdf');assert len(r2.pages)==1
with Image.open(w/'plate.png') as im:assert im.width>2600 and im.height>3600
files=[(w/'plate.png',out/'figures/图1_芯片22021_训练原理深化.png'),(w/'plate.pdf',out/'figures/图1_芯片22021_训练原理深化.pdf'),(draft,out/'类器官神经信号对语言模型探索行为的影响（原理图深化版）.docx')]
for a,b in files:
 if b.exists(): assert a.read_bytes()==b.read_bytes()
 else:shutil.copyfile(a,b)
 assert sha256(b.read_bytes()).digest()==sha256(a.read_bytes()).digest()
 print(b.resolve())
