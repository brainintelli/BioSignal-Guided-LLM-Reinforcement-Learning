"""Read-only GPU and bounded model-directory inventory."""
import datetime
import json
import os
from pathlib import Path
import subprocess

report = dict(host='192.0.2.16', hostname=os.uname().nodename,
              checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), training_started=False)
for name, args in [('gpu', ['nvidia-smi', '--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu', '--format=csv,noheader']),
                   ('compute_processes', ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'])]:
    p = subprocess.run(args, capture_output=True, text=True, timeout=8)
    report[name] = dict(exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)
report['previous_model_root'] = dict(path='/home/researcher/meadowmind-gpu', exists=Path('/home/researcher/meadowmind-gpu').exists())
report['storage_roots'] = []
report['possible_model_paths'] = []
for base in ('/home/researcher', '/data', '/mnt', '/models', '/workspace', '/pub', '/opt'):
    root = Path(base)
    if not root.is_dir():
        continue
    report['storage_roots'].append(dict(path=base, entries=sorted(p.name for p in root.iterdir() if not p.name.startswith('.'))))
    for current, dirs, files in os.walk(root):
        depth = len(Path(current).relative_to(root).parts)
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', 'site-packages', 'lib', 'include', 'locale', 'share')]
        if depth >= 2:
            dirs[:] = []
        if any(word in Path(current).name.lower() for word in ('qwen', 'meadow', 'checkpoint', 'm9-run')):
            report['possible_model_paths'].append(current)
print(json.dumps(report, ensure_ascii=False, indent=2))
