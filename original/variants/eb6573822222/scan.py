from pathlib import Path,PurePosixPath
import json,subprocess,os,zipfile,tarfile,re,sys
ROOT=Path.cwd(); W=ROOT/'tmp/paper_h5_revision'; TARGET=ROOT/'output/论文_湿件训练有效性_20261120/原始H5'
def enumerate_files():
 patterns=['*.h5','*.hdf5','*.hdf','*.tar','*.tar.gz','*.tgz','*.zip','*.7z','*.tar.xz','*.tar.bz2','*.h5.gz','*.hdf5.gz']
 args=['rg','--files','--hidden','--no-ignore','-i','-0']
 for g in patterns: args+=['-g',g]
 args+=['-g','!**/.git/**','-g','!**/node_modules/**','-g','!**/原始H5/**','-g','!tmp/paper_h5_revision/**']
 out=subprocess.run(args,capture_output=True,check=True)
 return sorted([Path(p.decode('utf-8')) for p in out.stdout.split(b'\0') if p],key=lambda p:p.as_posix().casefold())
def is_h5(name):return name.lower().endswith(('.h5','.hdf5','.hdf'))
def fixture(name):
 s='/'+name.replace('\\','/').lower().lstrip('/')
 return '/h5py/tests/' in s or '/__macosx/' in s or PurePosixPath(s).name.startswith('._')
if __name__=='__main__':
 allfiles=enumerate_files()
 loose=[p.as_posix() for p in allfiles if is_h5(p.name) and not fixture(p.as_posix())]
 (W/'loose.json').write_text(json.dumps(loose,ensure_ascii=False),encoding='utf8')
 arcs=[p for p in allfiles if not is_h5(p.name)]
 found=[];errors=[];nested=[]
 for p in arcs:
  loc=p.as_posix()
  def add(name,size):
   if is_h5(name) and not fixture(name): found.append([loc,name,size])
   elif name.lower().endswith(('.tar','.tgz','.tar.gz','.zip','.7z','.h5.gz')):nested.append([loc,name,size])
  try:
   if p.suffix.lower()=='.zip':
    with zipfile.ZipFile(p) as z:
     for i in z.infolist():
      if not i.is_dir():add(i.filename,i.file_size)
   elif loc.lower().endswith(('.tar','.tgz','.tar.gz','.tar.bz2','.tar.xz')):
    with tarfile.open(p,'r:*') as z:
     for i in z:
      if i.isfile():add(i.name,i.size)
   elif loc.lower().endswith(('.h5.gz','.hdf5.gz')):found.append([loc,p.stem,-1])
   else:errors.append([loc,'unhandled archive'])
  except Exception as e:errors.append([loc,type(e).__name__+' '+str(e)])
  if len(found)>0:(W/'archive_progress.txt').write_text(f'{loc}: {len(found)}',encoding='utf8')
 (W/'archives.json').write_text(json.dumps(dict(candidates=found,errors=errors,nested=nested),ensure_ascii=False),encoding='utf8')
 print(f'Loose {len(loose)}; inspected archives {len(arcs)}; candidate members {len(found)}; errors {len(errors)}; nested {len(nested)}')
