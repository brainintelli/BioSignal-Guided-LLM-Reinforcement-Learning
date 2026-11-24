from pathlib import Path
import re, csv, hashlib,os,sys,importlib.util
from zipfile import ZipFile
from lxml import etree as E
from pypdf import PdfReader
from PIL import Image,ImageDraw
w=Path(__file__).parent;out=Path('output/论文_湿件训练有效性_20261120')
# Use the native Windows Office PDF export as the conversion backend to the bundled document rasterizer.
renderer=Path('./.codex/plugins/cache/openai-primary-runtime/documents/26.1119.11507/skills/documents/render_docx.py')
sp=importlib.util.spec_from_file_location('document_renderer',renderer);rr=importlib.util.module_from_spec(sp);sp.loader.exec_module(rr)
pdf=w/'qa/paper.pdf';assert pdf.stat().st_mtime>=(out/'main.docx').stat().st_mtime
rr.convert_to_pdf=lambda *a,**kw:(str(pdf.resolve()),'native Windows Office export')
pages=rr.rasterize(str(out/'main.docx'),str(w/'qa/print'),110,False,False);assert len(pages)==14
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
with ZipFile(out/'main.docx') as z:
 doc=E.fromstring(z.read('word/document.xml'));styles=E.fromstring(z.read('word/styles.xml'));rels=E.fromstring(z.read('word/_rels/document.xml.rels'))
 lines=[''.join(par.xpath('.//w:t/text()',namespaces=ns)) for par in doc.xpath('/w:document/w:body/w:p',namespaces=ns)]
 assert len(doc.xpath('//w:tbl',namespaces=ns))==6
 assert sum(1 for k in z.namelist() if k.startswith('word/media/'))==4
 assert len([r for r in rels if r.get('TargetMode')=='External' and r.get('Target','').startswith('https://')])==5
 assert [int(re.match('图 ([0-9]+)',s).group(1)) for s in lines if re.match('图 ([0-9]+)',s)]==[1,2,3,4]
 assert [int(re.match('表 ([0-9]+)',s).group(1)) for s in lines if re.match('表 ([0-9]+)',s)]==list(range(1,7))
 assert (out/'abstract.txt').read_text(encoding='utf-8-sig').strip() in lines
 assert styles.xpath('string(//w:style[@w:styleId="Normal"]/w:rPr/w:sz/@w:val)',namespaces=ns)=='22'
 assert styles.xpath('string(//w:style[@w:styleId="Normal"]/w:pPr/w:spacing/@w:line)',namespaces=ns)=='336'
 assert doc.xpath('string(//w:sectPr/w:pgSz/@w:w)',namespaces=ns)=='11906'
 assert 'PAGE' in z.read('word/footer1.xml').decode()
 text=''.join(lines)
 assert not re.search(r'(?i)password=|api[_-]key\\s*=|BEGIN PRIVATE KEY|/home/|[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+',text)
with (out/'data_table.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
assert len(rows)==645 and all(Path(r['源文件_项目相对路径']).is_file() for r in rows)
assert len(list((out/'figures').glob('*.png')))==4
assert (out/'refs.bib').read_text(encoding='utf8').count('@')==5
with ZipFile(out/'精简汇报.pptx') as z:
 assert len([n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',n)])==8
 imgs={hashlib.sha256(z.read(n)).digest() for n in z.namelist() if n.startswith('ppt/media/')}
 assert all(hashlib.sha256(path.read_bytes()).digest() in imgs for path in (out/'figures').glob('*.png'))
thumb=Image.new('RGB',(1040,4*157),'#c6d2d5')
for i,path in enumerate(sorted((w/'brief').glob('slide??.png'))):
 img=Image.open(path);img.thumbnail((254,143));thumb.paste(img,((i%4)*260,(i//4)*157))
thumb.crop((0,0,1040,314)).save(w/'brief/contact.jpg')
print('14 Word pages; 4 embedded figures; 6 tables; 645 numeric rows; 5 reference IDs; 8 slides; all four PNGs identical in slides.')
