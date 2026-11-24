from scan import ROOT,W,TARGET,is_h5
from pathlib import Path,PurePosixPath
from contextlib import contextmanager
from collections import defaultdict,Counter
from datetime import datetime,timedelta,timezone
import os,hashlib,re,json,csv,tarfile,zipfile,gzip,shutil,sys,time
sys.path.insert(0,str(ROOT/'.tools/mea_baseline_deps'))
import h5py
RAW=ROOT/'output/mea_predictability_22021_20261119T042416Z/cpp-crossover-v1'
CONF={v['sha256']:v['bytes'] for v in json.loads((RAW/'raw-hashes.json').read_text(encoding='utf8')).values()}
BUF=4*1024**2
def assess(p,origin):
 d=dict(chip='',identity='',start='',mapped='',adc='',frames='',fs='',spikes='',status='',detail='')
 try:
  with h5py.File(p,'r') as f:
   if 'wellplate' not in f or 'data_store' not in f:return None
   chip=f.get('wellplate/id')
   if chip is not None:
    c=chip[0];d['chip']=(c.decode() if isinstance(c,bytes) else str(c)).strip()
    d['identity']='H5 内 wellplate/id'
   stores=f['data_store']
   if not len(stores):
    if 'ackprobe' in origin.lower():return None
    d['status']='无采集数据集';return d
   if len(stores)!=1:d['detail']=f'{len(stores)} 个数据段；数值以首段为准'
   g=stores[sorted(stores.keys())[0]]
   if 'start_time' in g:
    ts=int(g['start_time'][0])
    if ts>0:d['start']=datetime.fromtimestamp(ts/1000,timezone(timedelta(hours=8))).isoformat(timespec='milliseconds')
   if 'settings/mapping' in g:d['mapped']=g['settings/mapping'].shape[0]
   if 'settings/sampling' in g:d['fs']=float(g['settings/sampling'][0])
   if 'spikes' in g:d['spikes']=g['spikes'].shape[0]
   waves=[x['raw'] for x in g.get('groups',{}).values() if 'raw' in x]
   if waves:
    w=max(waves,key=lambda a:a.size)
    d['adc'],d['frames']=w.shape
    d['status']='连续波形' if w.shape[-1] else '波形零帧'
   elif d['spikes'] not in ('',0):d['status']='尖峰事件记录'
   else:d['status']='无波形帧'
 except Exception as e:
  d['status']='结构异常，保留原件';d['detail']=type(e).__name__+' '+str(e).splitlines()[0]
 if not d['chip']:
  m=re.search(r'(?<!\d)(22021|22554|28080|28073|27701)(?!\d)',origin)
  if m:d.update(chip=m[1],identity='按原目录标识，文件头未核实')
 return d
def stream_hash(src,out=None):
 h=hashlib.sha256();n=0
 while True:
  buf=src.read(BUF)
  if not buf:break
  h.update(buf);n+=len(buf)
  if out is not None:out.write(buf)
 return h.hexdigest(),n
def digest(p):
 with p.open('rb') as s:return stream_hash(s)
@contextmanager
def reader(ref,mem):
 if not mem:
  with Path(ref).open('rb') as f:yield f
 elif ref.lower().endswith('.zip'):
  with zipfile.ZipFile(ref) as z,z.open(mem) as f:yield f
 elif ref.lower().endswith(('.h5.gz','.hdf5.gz')):
  with gzip.open(ref,'rb') as f:yield f
 else:
  with tarfile.open(ref,'r:*') as t,t.extractfile(mem) as f:yield f
def sink_name(meta,ref,dig):
 date=meta['start'][:19].replace('-','').replace(':','').replace('T','_') or '时间未读'
 stem=Path(ref).name
 stem=re.sub(r'(?i)(?:\.raw)?\.(?:hdf5|h5|hdf)$','',stem)
 stem=re.sub(r'[^\w.-]+','_',stem)[:65]
 status={'波形零帧':'_零帧','无波形帧':'_无波形帧','尖峰事件记录':'_尖峰','结构异常，保留原件':'_结构异常'}.get(meta['status'],'')
 return f"{meta['chip'] or '芯片未读'}_{date}_{stem}{status}_{dig[:12]}.h5"
fields=['汇总文件名','芯片编号','芯片标识依据','记录起始时间_北京时间','记录类型','映射通道数','波形行数','波形帧数','采样率_Hz','尖峰事件数','字节数','SHA256','22021确认性原始19份','项目内来源','压缩包内成员','备注']
def main():
 TARGET.mkdir(parents=True,exist_ok=True)
 already={};by_size=defaultdict(set); rows=[];excluded=[];bytes_written=0
 catalog=json.loads((W/'loose.json').read_text(encoding='utf8'))
 members=json.loads((W/'archives.json').read_text(encoding='utf8'))['candidates']
 todo=[(s,'',(ROOT/s).stat().st_size) for s in catalog]+[tuple(v) for v in members]
 log=W/'copy_progress.jsonl'
 def record(sha,meta,name,size,ref,mem):
  rows.append([name,meta['chip'],meta['identity'],meta['start'],meta['status'],meta['mapped'],meta['adc'],meta['frames'],meta['fs'],meta['spikes'],size,sha,'是' if sha in CONF else '否',ref,mem,meta['detail']])
 for idx,(ref,mem,size) in enumerate(todo,1):
  src=ROOT/ref;origin=ref+'::'+mem if mem else ref
  # Loose files are inspected without rewriting their HDF5 metadata.
  if not mem:
   meta=assess(src,origin)
   if meta is None:excluded.append(origin);continue
  stamp=(src.stat().st_size,src.stat().st_mtime_ns)
  tmp=W/'record.partial';sha='';has_temp=False
  if size in by_size:
   with reader(src.as_posix(),mem) as f:sha,n=stream_hash(f)
   assert n==size
  if sha not in already:
   with reader(src.as_posix(),mem) as f,tmp.open('wb') as o:actual,n=stream_hash(f,o)
   assert n==size and (not sha or sha==actual),origin
   has_temp=True;sha=actual
   if mem:
    meta=assess(tmp,origin)
    if meta is None:excluded.append(origin);tmp.unlink();continue
   name=sink_name(meta,mem or ref,sha);dest=TARGET/name
   if dest.exists():
    assert digest(dest)==(sha,size),'Conflicting output '+name
    tmp.unlink()
   else:
    os.replace(tmp,dest)
   already[sha]=(meta,name,size);by_size[size].add(sha);bytes_written+=n
  meta,name,size=already[sha];record(sha,meta,name,size,ref,mem)
  assert stamp==(src.stat().st_size,src.stat().st_mtime_ns),'Source changed '+origin
  if idx%4==0 or size>2*1024**3 or idx==len(todo):
   with log.open('a',encoding='utf8') as f:f.write(json.dumps([idx,len(todo),len(already),bytes_written,ref],ensure_ascii=False)+'\n')
 assert CONF.keys()<=already.keys()
 for k,z in CONF.items():assert z==already[k][2]
 # Independently reread destination copies, including partial and event-only acquisitions.
 for n,(sha,(meta,name,size)) in enumerate(already.items(),1):
  assert digest(TARGET/name)==(sha,size),name
  if n%8==0 or n==len(already):(W/'verify_progress.txt').write_text(f'{n}/{len(already)}',encoding='utf8')
 pt=W/'manifest.csv'
 with pt.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(fields);w.writerows(sorted(rows,key=lambda r:(r[0],r[13],r[14])))
 os.replace(pt,TARGET/'原始H5索引.csv')
 assert len(list(TARGET.glob('*.h5')))==len(already)
 stats=dict(unique=len(already),bytes=bytes_written,source_occurrences=len(rows),loose_sources=len(set(r[13] for r in rows if not r[14])),archive_sources=sum(bool(r[14]) for r in rows),excluded=excluded,confirm=len(CONF),confirm_bytes=sum(CONF.values()),chips=dict(Counter(q[0]['chip'] for q in already.values())),statuses=dict(Counter(q[0]['status'] for q in already.values())))
 (W/'verified.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(stats,ensure_ascii=False))
if __name__=='__main__':main()
