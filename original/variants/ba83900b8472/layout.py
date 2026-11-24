from pathlib import Path
from zipfile import ZipFile
from PIL import Image,ImageOps,ImageFont,ImageDraw
from xml.etree import ElementTree as E
w=Path(__file__).parent
p='{http://schemas.openxmlformats.org/presentationml/2006/main}'
a='{http://schemas.openxmlformats.org/drawingml/2006/main}'
fonts={}
def font(sz,b):
 k=(sz,b)
 if k not in fonts:fonts[k]=ImageFont.truetype('C:/Windows/Fonts/msyh'+('bd' if b else '')+'.ttc',sz*3)
 return fonts[k]
with ZipFile(w/'candidate.pptx') as z:
 for n in range(1,15):
  root=E.fromstring(z.read(f'ppt/slides/slide{n}.xml'))
  for sh in root.iter(p+'sp'):
   ext=sh.find('./'+p+'spPr/'+a+'xfrm/'+a+'ext')
   if ext is None:continue
   width=int(ext.get('cx'))/9525
   for para in sh.iter(a+'p'):
    t='';length=0
    for run in para.findall(a+'r'):
     rp=run.find(a+'rPr'); tx=run.findtext(a+'t','')
     if rp is None or not tx:continue
     size=round(float(rp.get('sz','1600'))/75);length+=font(size,rp.get('b')=='1').getlength(tx)/3;t+=tx
    if length>width-2:print(n,round(length),int(width),t)
imgs=list(sorted(w.glob('slide??.png')))
print('Rendered',len(imgs))
if len(imgs)==14:
 contact=Image.new('RGB',(1200,5*247),'#cbd5da');dr=ImageDraw.Draw(contact)
 for i,png in enumerate(imgs):
  img=Image.open(png);img.thumbnail((396,223));x=(i%3)*400;y=(i//3)*247
  contact.paste(img,(x,y));dr.text((x+8,y+225),str(i+1),fill='#203746')
 contact.save(w/'contact.jpg',quality=91)
