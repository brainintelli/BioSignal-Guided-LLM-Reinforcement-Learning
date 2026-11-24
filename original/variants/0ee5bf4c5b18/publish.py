from pathlib import Path
import shutil,os,hashlib
from zipfile import ZipFile
w=Path(__file__).parent
out=Path('output/论文_湿件训练有效性_20261120/main.docx')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(out)==(w/'original_sha256.txt').read_text().strip(),'Document edited since start'
assert (w/'doc_contact.jpg').is_file()
old=w/'before_source_removal.docx'
if not old.exists():shutil.copy2(out,old)
tmp=out.with_suffix('.revision-partial')
shutil.copy2(w/'main.docx',tmp)
os.replace(tmp,out)
assert sha(out)==sha(w/'main.docx')
print('Published verified main.docx')
