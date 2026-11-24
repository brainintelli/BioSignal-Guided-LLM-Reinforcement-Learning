from data import *
from refs import items
from docx import Document
from docx.shared import Pt,Cm,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH as Align
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.enum.style import WD_STYLE_TYPE
doc=Document();sec=doc.sections[0]
sec.page_width=Cm(21);sec.page_height=Cm(29.7);sec.left_margin=Cm(2.35);sec.right_margin=Cm(2.35);sec.top_margin=Cm(2.15);sec.bottom_margin=Cm(2.12);sec.footer_distance=Cm(.95)
grid=sec._sectPr.find(qn('w:docGrid'))
if grid is not None:sec._sectPr.remove(grid)
def sty(name,size,ea='宋体',latin='Times New Roman',bold=False,after=4,before=0,lead=1.4,col='161C20'):
 try:s=doc.styles[name]
 except KeyError:s=doc.styles.add_style(name,WD_STYLE_TYPE.PARAGRAPH)
 s.font.name=latin;s.font.size=Pt(size);s.font.bold=bold;s.font.italic=False;s.font.color.rgb=RGBColor.from_string(col)
 el=s._element.get_or_add_rPr().rFonts
 for k in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme']:
  if qn('w:'+k) in el.attrib:del el.attrib[qn('w:'+k)]
 for k,v in [('ascii',latin),('hAnsi',latin),('eastAsia',ea),('cs',latin)]:el.set(qn('w:'+k),v)
 fmt=s.paragraph_format;fmt.line_spacing=lead;fmt.space_after=Pt(after);fmt.space_before=Pt(before);fmt.widow_control=True
 return s
sty('Normal',11,after=6)
sty('Title',20,'微软雅黑','Arial',True,after=6,before=0,lead=1.15,col='000000')
sty('EnglishTitle',13,latin='Arial',after=7,lead=1.15)
sty('Byline',10,'宋体',after=12,lead=1.15)
sty('Heading 1',16,'微软雅黑','Arial',True,after=8,before=6,lead=1.14,col='000000')
sty('Heading 2',12,'微软雅黑','Arial',True,after=4,before=6,lead=1.16,col='000000')
sty('FigCaption',9.3,bold=False,after=2,before=4,lead=1.15)
sty('SourcePath',7.9,ea='宋体',latin='Arial',after=6,lead=1.12,col='465761')
sty('TableBody',9.5,after=0,lead=1.18)
sty('BodyQuote',10,ea='仿宋',after=4,lead=1.26)
sty('RefText',10.4,after=6,lead=1.2)
sty('PageNumberText',9,after=0,lead=1)
for n in ('Title','Heading 1','Heading 2'):
 props=doc.styles[n]._element.get_or_add_pPr()
 for e in list(props):
  if e.tag==qn('w:pBdr'):props.remove(e)
def para(s,style='Normal'):
 p=doc.add_paragraph(s,style=style)
 if style=='Normal':p.alignment=Align.JUSTIFY
 if style=='BodyQuote':p.alignment=Align.LEFT
 return p
def h(s,break_=False,small=False):
 p=para(s,'Heading 2' if small else 'Heading 1')
 if break_:p.paragraph_format.page_break_before=True
 return p
def fig(n,file,text,paths):
 p=doc.add_paragraph();p.paragraph_format.line_spacing=1;p.paragraph_format.space_before=Pt(1);p.paragraph_format.space_after=Pt(0);p.paragraph_format.keep_with_next=True;p.alignment=Align.CENTER
 pic=p.add_run().add_picture(str(Figs/file),width=Cm(16.1));pic._inline.docPr.set('descr',text)
 cap('图',n,text,paths)
def cap(typ,n,desc,paths):
 p=doc.add_paragraph(style='FigCaption');p.add_run(f'{typ} {n}　').bold=True;p.add_run(desc);p.paragraph_format.keep_with_next=True
 pp=doc.add_paragraph(style='SourcePath');pp.add_run('数据来源　')
 for i,path in enumerate(dict.fromkeys(paths)):
  assert Path(path).exists(),path
  if i:pp.add_run('　·　')
  pp.add_run(path)
def table(widths,rows):
 tb=doc.add_table(rows=0,cols=len(widths));tb.autofit=False;tb.alignment=WD_TABLE_ALIGNMENT.CENTER
 pr=tb._tbl.tblPr
 bd=OxmlElement('w:tblBorders')
 for side in ['top','bottom','left','right','insideH','insideV']:
  e=OxmlElement('w:'+side);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');bd.append(e)
 pr.append(bd)
 mar=OxmlElement('w:tblCellMar')
 for k,v in [('top','44'),('bottom','44'),('start','75'),('end','75')]:
  e=OxmlElement('w:'+k);e.set(qn('w:w'),v);e.set(qn('w:type'),'dxa');mar.append(e)
 pr.append(mar)
 for c,x in zip(tb.columns,widths):c.width=Cm(x)
 for i,vals in enumerate(rows):
  row=tb.add_row()
  trpr=row._tr.get_or_add_trPr();trpr.append(OxmlElement('w:cantSplit'))
  if i==0:trpr.append(OxmlElement('w:tblHeader'))
  for j,(c,v) in enumerate(zip(row.cells,vals)):
   c.width=Cm(widths[j]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   pp=c.paragraphs[0];pp.style='TableBody';r=pp.add_run(str(v));r.bold=i==0
   if i==0:
    bg=OxmlElement('w:shd');bg.set(qn('w:fill'),'E4EEF0');c._tc.get_or_add_tcPr().append(bg)
 return tb
def ref(i,r):
 key,a,t,j,y,doi,arx=r
 names=a.split(' and '); au=', '.join(names[:3])+(', et al.' if len(names)>3 else '.')
 p=doc.add_paragraph(style='RefText');p.add_run(f'[{i}]  {au} ({y}). {t}. ');p.add_run(j+'. ').italic=True
 uri='https://doi.org/'+doi if doi else 'https://arxiv.org/abs/'+arx
 link=OxmlElement('w:hyperlink');link.set(qn('r:id'),p.part.relate_to(uri,RT.HYPERLINK,is_external=True));rr=OxmlElement('w:r');tt=OxmlElement('w:t');tt.text='DOI '+doi if doi else 'arXiv '+arx;rr.append(tt);link.append(rr);p._p.append(link)
footer=sec.footer.paragraphs[0];footer.style='PageNumberText';footer.alignment=Align.CENTER
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
doc.core_properties.author='作者待填写';doc.core_properties.title='类器官神经信号对语言模型游戏探索的影响'
doc.core_properties.subject='相同决策预算下的离线配对实验'
