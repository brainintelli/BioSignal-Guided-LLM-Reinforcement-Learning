from pathlib import Path
from PIL import Image,ImageDraw
from pypdf import PdfReader
import pypdfium2 as pdfium
w=Path(__file__).parent/'qa';pdf=w/'paper.pdf'
for i,p in enumerate(PdfReader(pdf).pages):
 t=p.extract_text(); (w/f'text-{i+1:02}.txt').write_text(t,encoding='utf8')
 print(i+1,len(t),t[:47].replace('\n',' '),t[-60:].replace('\n',' '))
pages=pdfium.PdfDocument(pdf)
out=Image.new('RGB',(1050,((len(pages)+3)//4)*395),'#a6b9bd')
for i,pg in enumerate(pages):
 im=pg.render(scale=1.6).to_pil();im.save(w/f'page-{i+1:02}.png');im.thumbnail((256,365));x=i%4*262;y=i//4*395;out.paste(im,(x,y));ImageDraw.Draw(out).text((x+6,y+368),str(i+1),fill='black')
out.save(w/'contact.jpg',quality=90)
