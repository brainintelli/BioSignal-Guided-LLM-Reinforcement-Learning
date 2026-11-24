"""Stage the offline run, read progress, and retrieve only verified per-seed evidence."""
import concurrent.futures
import hashlib
import io
import json
import os
import shlex
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / ".tools/mea_ssh_deps"))
import paramiko

REMOTE = "/home/researcher/mea-offline-single-pulse/22554-registered-ratio02-20261120"
FILES = ("registration.json", "training-trace.json", "eval-trace.json", "result.json", "updates.json")


def call(client, code, *args, binary=False):
    _, out, err = client.exec_command(shlex.join(["python3", "-c", code, REMOTE, *args]), timeout=100)
    value = out.read()
    error = err.read()
    if out.channel.recv_exit_status():
        raise RuntimeError(error.decode(errors="replace")[-1200:])
    return value if binary else value.decode().strip()


def stage(client, host):
    call(client, "import sys,pathlib;pathlib.Path(sys.argv[1]).mkdir(parents=True,exist_ok=True)")
    with client.open_sftp() as session:
        for p in (ROOT / "tools/run_r2_ratio_pair.py", HERE / "protocol.json"):
            dest = REMOTE + "/" + p.name
            data = p.read_bytes()
            try:
                with session.open(dest) as stream:
                    if stream.read() != data:
                        raise ValueError(f"Previous registration differs for {dest}")
            except FileNotFoundError:
                with session.open(dest + ".part", "wb") as stream:
                    stream.write(data)
                session.rename(dest + ".part", dest)
    return call(client, """from pathlib import Path
import subprocess,sys,json
r=Path(sys.argv[1]); pid=r/'worker.pid'
if pid.exists():
 f=Path('/proc')/pid.read_text()/'cmdline'
 if f.exists() and str(r/'run_r2_ratio_pair.py').encode() in f.read_bytes():print('running');sys.exit()
status=r/'status.json'
if status.exists() and json.loads(status.read_bytes()).get('finished'):print('complete');sys.exit()
with (r/'batch.log').open('ab') as f:
 p=subprocess.Popen([sys.executable,'-u',str(r/'run_r2_ratio_pair.py'),'batch',str(r),sys.argv[2]],stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
pid.write_text(str(p.pid));print('started',p.pid)""", host)


def status(client):
    return json.loads(call(client, """from pathlib import Path
import json,sys
r=Path(sys.argv[1]);p=r/'status.json';v=json.loads(p.read_bytes()) if p.exists() else {}
v['done']=len(list(r.glob('runs/*/*/done.json')))
if 'current' in v and 'finished' not in v:
 tag,s=v['current'];run=r/'runs'/tag/str(s); progress=run/'training'/str(s)/'progress.json'
 if progress.exists():v['progress']=json.loads(progress.read_bytes())
 if (r/'error.json').exists():v['last_error']=(run/'process.log').read_text(errors='replace')[-800:]
print(json.dumps(v))"""))


def collect(client, host):
    done = call(client, """import pathlib,sys,json
r=pathlib.Path(sys.argv[1]);print(json.dumps([str(p.parent.relative_to(r/'runs')) for p in sorted(r.glob('runs/*/*/done.json'))]))""")
    missing = [item for item in json.loads(done) if not (HERE / "runs" / item / "done.json").exists()]
    if not missing:
        return 0
    data = call(client, """from pathlib import Path
import sys,json,tarfile,io
r=Path(sys.argv[1]);keys=json.loads(sys.argv[2]);files=json.loads(sys.argv[3]);buf=io.BytesIO()
with tarfile.open(mode='w:gz',fileobj=buf,compresslevel=2) as t:
 for key in keys:
  src=r/'runs'/key
  for n in [*files,'done.json']:
   t.add(src/n if n=='done.json' else src/'training'/src.name/n,arcname=key+'/'+n,recursive=False)
sys.stdout.buffer.write(buf.getvalue())""", json.dumps(missing), json.dumps(FILES), binary=True)
    expected = {key + "/" + n for key in missing for n in (*FILES, "done.json")}
    with tarfile.open(fileobj=io.BytesIO(data)) as bundle:
        if {p.name for p in bundle.getmembers() if p.isfile()} != expected:
            raise ValueError("Evidence archive did not match requested files")
        for key in missing:
            dest = HERE / "runs" / key
            reg = (HERE / "runs").resolve()
            if not dest.resolve().is_relative_to(reg):
                raise ValueError(key)
            dest.mkdir(parents=True, exist_ok=True)
            marker = json.load(bundle.extractfile(key + "/done.json"))
            if marker["host"] != host or str(marker["seed"]) != dest.name:
                raise ValueError("Incorrect result identity")
            for name in FILES:
                blob = bundle.extractfile(key + "/" + name).read()
                if name=="registration.json" and hashlib.sha256(blob).hexdigest()!=marker["registration"]:
                    raise ValueError("Registration transfer mismatch")
                temp = dest / (name + ".part")
                temp.write_bytes(blob)
                temp.replace(dest / name)
            (dest / "done.json").write_text(json.dumps(marker), encoding="utf8")
    return len(missing)


def run(host, operation):
    address = "192.168.77." + host
    with paramiko.SSHClient() as client:
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        client.connect(address, username="researcher", password=os.getenv("MEA_SSH_" + address.replace(".", "_")),
                       timeout=8, banner_timeout=12, auth_timeout=12)
        client.get_transport().set_keepalive(25)
        if operation == "start":
            return stage(client, host)
        if operation == "collect":
            return collect(client, host)
        if operation == "status":
            return status(client)
        raise ValueError(operation)


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        tasks = {h:pool.submit(run,h,sys.argv[1]) for h in ("16","23")}
        for host, task in tasks.items():
            print(host, task.result(), flush=True)
