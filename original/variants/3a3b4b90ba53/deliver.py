from pathlib import Path
from zipfile import ZipFile
from hashlib import sha256
from pypdf import PdfReader
from PIL import Image
import shutil
w=Path(__file__).parent
folder=Path('output/论文_湿件训练有效性_20261120')
base=folder/'类器官神经信号对语言模型探索行为的影响（原理图深化版）.docx'
assert sha256(base.read_bytes()).hexdigest()==(w/'base.sha256').read_text()
with ZipFile(base) as a, ZipFile(w/'draft.docx') as b:
 assert b.testzip() is None and a.namelist()==b.namelist()
 assert {k for k in a.namelist() if a.read(k)!=b.read(k)}=={'word/document.xml','word/media/image1.png'}
 assert b.read('word/media/image1.png')==(w/'figure.png').read_bytes()
with Image.open(w/'figure.png') as img:assert img.size==(3120,1948)
p=PdfReader(w/'draft.pdf');assert len(p.pages)==15 and '芯片 22021' in p.pages[3].extract_text()
p=PdfReader(w/'figure.pdf');t=p.pages[0].extract_text();assert len(p.pages)==1 and all(x in t for x in ['1.99','2.16','14/16','15/16','1.18','2.75','44'])
for a,b in [(w/'figure.png',folder/'figures/图1_从类器官放电到游戏决策.png'),(w/'figure.pdf',folder/'figures/图1_从类器官放电到游戏决策.pdf'),(w/'draft.docx',folder/'类器官神经信号对语言模型探索行为的影响（图1重设计版）.docx')]:
 if not b.exists():shutil.copyfile(a,b)
 assert a.read_bytes()==b.read_bytes()
 print(b.resolve())
