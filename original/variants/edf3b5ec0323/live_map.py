from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
import json
w=Path(__file__).parent
n={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
def plist(file):
 with ZipFile(file) as z:
  rt=E.fromstring(z.read('word/document.xml'))
  return [(p.xpath('string(w:pPr/w:pStyle/@w:val)',namespaces=n),''.join(p.xpath('.//w:t/text()',namespaces=n))) for p in rt.xpath('/w:document/w:body/w:p',namespaces=n)]
orig=plist(w/'before_source_removal.docx')
stage=plist(w/'main.docx')
records=json.loads((w/'live_before.json').read_text(encoding='utf-8-sig'))
sources=[t for sty,t in orig if sty=='SourcePath']
repl={t:next(u for _,u in stage if u.startswith(begin)) for _,t in orig for pref,begin in [('数据总表 data_table.csv','随文数据总表列出'),('随文 data_table.csv','随文数据总表可按')] if t.startswith(pref)}
edits=[]
future=[]
for row in records:
 v=row['t'].rstrip('\r\x07')
 if v in sources or v in repl:
  new=repl.get(v,'')
  edits.append(dict(s=row['s'],e=row['e'],old=row['t'],mode='replace' if new else 'remove',new=new))
  if new:future.append(new+row['t'][len(v):])
 else:
  future.append(row['t'])
  if any(x in v for x in ('output/','docs/MEA','来源','data_table.csv','01a1e3f0-','01a1f6d9-','standalone/')):print('other',row['s'],repr(v[:120]))
print('source matches',sum(x['mode']=='remove' for x in edits),'replace matches',sum(x['mode']=='replace' for x in edits),'live',len(records),'absent source',[(t[:45],len(t)) for t in sources if not any(r['t'].rstrip('\r\x07')==t for r in records)])
for x in edits:print(x['s'],x['e'],repr(x['old'][:18]),repr(x['old'][-4:]))
(w/'edits.json').write_text(json.dumps(edits,ensure_ascii=False),encoding='utf8')
(w/'future.json').write_text(json.dumps(future,ensure_ascii=False),encoding='utf8')
