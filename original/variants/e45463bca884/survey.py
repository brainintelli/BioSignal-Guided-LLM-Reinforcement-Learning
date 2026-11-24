from pathlib import Path
from docx import Document
from docx.oxml.ns import qn
from PIL import Image
import json
O=Path('output/论文_湿件训练有效性_20261120')
for f in O.glob('*.docx'):
 if f.name.startswith('~'):continue
 d=Document(f);p=d.paragraphs
 print('\n',f.name,len(p),len(d.inline_shapes),[(k,q.text[:80]) for k,q in enumerate(p) if q.text.startswith(('图 1','图1','图一','图表 1'))][:5])
 for k,q in enumerate(p):
  if q._p.xpath('.//w:drawing') and 0<k<60:print('figure near',k,'next:',p[k+1].text[:70]);break
 print('first headings:',[q.text[:35] for q in p if q.style.name.startswith('Heading')][:9])
