from pathlib import Path
code=Path(__file__).with_name('verify_pairs.py').read_text(encoding='utf8')
before,after=code.split('v=measures();assert len(v)==2304')
exec(before)
paired=read(w/'paired.csv')
assert len(paired)==768
v={}
for r in paired:
 for a,label in zip(arms,['屏蔽','自发','零输出']):
  s=int(r['训练种子']);k=int(r['地图种子']);n=int(r[label+'组新增格']);start=int(r['开局已揭示格数'])
  v[s,k,a]=(n,start,start+n,int(r[label+'组结束刻']))
paired=[{a:int(v) for a,v in r.items()} for r in paired]
exec("byseed="+after.split("byseed=",1)[1])
