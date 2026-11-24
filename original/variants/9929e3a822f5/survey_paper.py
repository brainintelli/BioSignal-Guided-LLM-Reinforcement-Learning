from pathlib import Path
import csv,json
from docx import Document
from lxml import etree
o=Path('output');root=o/'mea_six_arm_r1_off_20261120';b=o/'mea_predictability_22021_20261119T042416Z/offline-confirmatory-results'
docs=list((o/'论文_湿件训练有效性_20261120').glob('*.docx'));d=Document(docs[0])
for i,p in enumerate(d.paragraphs):
 print(i,p.style.name,p.text[:57])
print('tables',[(len(t.rows),len(t.columns),[c.text for c in t.rows[0].cells]) for t in d.tables])
e=json.loads(next((root/'main/models/102600001/masked/holdout.jsonl').open()))
print('holdout',e.keys(),[(k,str(v)[:100]) for k,v in e.items() if k!='rows'])
for f in [b/'episodes.csv',b/'verification/native-replay.csv']:
 print(f.name,next(csv.DictReader(f.open())))
c=list(csv.DictReader((root/'metrics_by_world.csv').open(encoding='utf-8-sig')))
print('seeds',sorted({int(x['train_seed']) for x in c if x['chip']=='22021'}))
print('paths for first seed baseline', [str(x) for x in (o/'mea_predictability_22021_20261118T093720Z/offline-mea-stim-ablation').glob('**/102122616/baseline/holdout.jsonl')][:4])
print('sample',e['rows'][-1]['after'])
