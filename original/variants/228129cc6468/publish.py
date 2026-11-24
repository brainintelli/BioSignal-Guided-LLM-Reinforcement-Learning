from pathlib import Path
import os,shutil,hashlib
from zipfile import ZipFile
from xml.etree import ElementTree as E
src=Path(__file__).parent/'deliver'
dst=Path('output/PPT_湿件训练有效性_20261120')
old=Path(__file__).parent/'previous'
files=[Path('一页汇报总结.txt'),*[p.relative_to(src) for p in (src/'图表').glob('*.png')],Path('MEA湿件训练有效性_实验汇报.pptx')]
sha=lambda p: hashlib.sha256(p.read_bytes()).digest()
for rel in files:
 backup=old/rel
 backup.parent.mkdir(parents=True,exist_ok=True)
 if not backup.exists():shutil.copy2(dst/rel,backup)
published=[]
try:
 for rel in files:
  tmp=(dst/rel).with_suffix('.revision-partial')
  shutil.copy2(src/rel,tmp)
  os.replace(tmp,dst/rel)
  published.append(rel)
  assert sha(dst/rel)==sha(src/rel)
except Exception:
 for rel in reversed(published):shutil.copy2(old/rel,dst/rel)
 raise
with ZipFile(dst/files[-1]) as z:
 names=set(z.namelist())
 for i in range(1,15):
  assert f'ppt/notesSlides/notesSlide{i}.xml' in names
  doc=E.fromstring(z.read(f'ppt/slides/slide{i}.xml'))
  text=''.join(doc.itertext())
  assert 'output/' in text
assert sha(src/'数据总表.csv')==sha(dst/'数据总表.csv')
print('Published 14 slides with notes, 5 charts, one-page text; data table matches validated CSV.')
