from scan import W,ROOT,is_h5,fixture
import tarfile,zipfile,io,json,hashlib
j=json.loads((W/'archives.json').read_text(encoding='utf8'));seen=set();found=[]
def edges(id,raw):
 sha=hashlib.sha256(raw).hexdigest()
 if sha in seen:return
 seen.add(sha);buf=io.BytesIO(raw)
 if zipfile.is_zipfile(buf):
  with zipfile.ZipFile(buf) as z:
   for i in z.infolist():
    if i.is_dir():continue
    if is_h5(i.filename) and not fixture(i.filename):found.append((id,i.filename,i.file_size))
    elif i.filename.lower().endswith(('.zip','.tgz','.tar','.tar.gz')):edges(id+'::'+i.filename,z.read(i))
 else:
  buf.seek(0)
  with tarfile.open(fileobj=buf,mode='r:*') as t:
   for i in t:
    if i.isfile() and is_h5(i.name) and not fixture(i.name):found.append((id,i.name,i.size))
    elif i.isfile() and i.name.lower().endswith(('.zip','.tgz','.tar','.tar.gz')):edges(id+'::'+i.name,t.extractfile(i).read())
for p,name,_ in j['nested']:
 if p.endswith('.zip'):
  with zipfile.ZipFile(ROOT/p) as z:raw=z.read(name)
 else:
  with tarfile.open(ROOT/p,'r:*') as t:raw=t.extractfile(name).read()
 edges(p+'::'+name,raw)
print('Nested archives scanned',len(seen),'additional H5',found)
(W/'nested-raw.json').write_text(json.dumps(found,ensure_ascii=False),encoding='utf8')
