"""Read-only host inventory; no MaxLab commands or signal-stream connections."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=8)
        return dict(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(error=str(exc))


report = dict(
    schema='mea.r2.host_preflight.v1',
    checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    hostname=os.uname().nodename,
    user_reported_host='192.0.2.21',
    user_reported_chip_id=22554,
    chip_id_verified_from_device=False,
    device_commands_sent=False,
    recording_started=False,
    stimulation_started=False,
    training_started=False,
    os_release=Path('/etc/os-release').read_text(),
    cpu_count=os.cpu_count(),
)
report['memory'] = [row for row in Path('/proc/meminfo').read_text().splitlines()
                    if row.startswith(('MemTotal:', 'MemAvailable:'))]
root = Path('/home/researcher/brainDesktop')
report['program_directory'] = str(root)
report['program_files'] = []
if root.is_dir():
    for current, dirs, files in os.walk(root):
        depth = len(Path(current).relative_to(root).parts)
        dirs[:] = sorted(d for d in dirs if d not in ('node_modules', '.git'))
        if depth >= 2:
            dirs[:] = []
        report['program_files'].append(dict(path=current, directories=dirs[:], files=sorted(files)[:60]))
report['processes'] = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name) == os.getpid():
        continue
    try:
        name = (proc / 'comm').read_text().strip()
        if not any(word in name.lower() for word in
                   ('maxlab', 'mxw', 'scope', 'brain', 'tcp_server', 'godot', 'python')):
            continue
        executable = os.readlink(proc / 'exe')
        row = dict(pid=int(proc.name), name=name, executable=executable,
                   cwd=os.readlink(proc / 'cwd'), open_h5=[])
        for fd in (proc / 'fd').iterdir():
            try:
                target = os.readlink(fd)
                if target.endswith(('.h5', '.h5 (deleted)')):
                    row['open_h5'].append(target)
            except OSError:
                pass
        if 'tcp_server' in name.lower():
            row['executable_sha256'] = hashlib.sha256(Path(executable).read_bytes()).hexdigest()
        report['processes'].append(row)
    except OSError:
        continue
report['ports'] = command(['ss', '-ltnp'])
report['connections_5555'] = command(['ss', '-Htnp', 'state', 'established', '( sport = :5555 or dport = :5555 )'])
report['pci_devices'] = command(['lspci'])
report['gpu'] = command(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader'])
report['vendor_python_exists'] = Path('/pub/soft/MaxLab/python/bin/python3').exists()
report['sdk_directory_exists'] = Path('/pub/soft/MaxLab/include').exists()
print(json.dumps(report, ensure_ascii=False, indent=2))
