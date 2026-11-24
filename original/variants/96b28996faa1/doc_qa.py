from pathlib import Path
import importlib.util
from PIL import Image,ImageDraw
from pypdf import PdfReader
w=Path(__file__).parent.resolve()
fp=Path('./.codex/plugins/cache/openai-primary-runtime/documents/26.1119.11507/skills/documents/render_docx.py')
spec=importlib.util.spec_from_file_location('renderer',fp);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.convert_to_pdf=lambda *a,**kw:(str(w/'main.pdf'),'Local Office export on Windows')
pages=m.rasterize(str(w/'main.docx'),str(w/'qa'),110,False,False)
reader=PdfReader(w/'main.pdf');assert len(reader.pages)==len(pages)==14
for i,page in enumerate(reader.pages):
 t=page.extract_text()
 assert len(t)>200 and '数据来源' not in t and 'output/' not in t and 'standalone/' not in t
img=Image.new('RGB',(4*313,4*461),'#b2bdc3')
for i,p in enumerate(pages):
 with Image.open(p) as v:
  v.thumbnail((300,429));x=i%4*313;y=i//4*461;img.paste(v,(x,y));ImageDraw.Draw(img).text((x+7,y+430),f'{i+1:02}',fill='black')
img.save(w/'doc_contact.jpg',quality=92)
print('Verified',len(pages),'rendered A4 pages.')
