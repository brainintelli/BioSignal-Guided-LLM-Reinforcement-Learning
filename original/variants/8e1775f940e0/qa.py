from pathlib import Path
from PIL import Image,ImageDraw
from pypdf import PdfReader
import importlib.util
w=Path(__file__).parent.resolve();pdf=w/'draft.pdf';r=PdfReader(pdf)
print('pages',len(r.pages))
for i,p in enumerate(r.pages):
 t=' '.join(p.extract_text().split())
 print(i+1,len(t),t[:43],t[-31:])
fp=Path('./.codex/plugins/cache/openai-primary-runtime/documents/26.1119.11507/skills/documents/render_docx.py')
s=importlib.util.spec_from_file_location('render',fp);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.convert_to_pdf=lambda *a,**kw:(str(pdf),'WPS PDF export')
imgs=m.rasterize(str(w/'draft.docx'),str(w/'pages'),120,False,False)
img=Image.new('RGB',(4*260,385*((len(imgs)+3)//4)),'#CAD2D6')
for i,p in enumerate(imgs):
 with Image.open(p) as im:
  im.thumbnail((250,355));x=(i%4)*260;y=(i//4)*385;img.paste(im,(x,y));ImageDraw.Draw(img).text((x+4,y+356),str(i+1),fill='black')
img.save(w/'contact.jpg',quality=94)
