"""Password-safe remote GPU staging, launch, monitoring, and SHA-verified result collection.

Supports running the 2026-11-23 afternoon 22021 clean 233-ROI training funnel on any LAN
Linux GPU host (1-GPU or 8x RTX 4090) without ever storing or logging SSH credentials:
  - `probe`   : Inspect remote GPUs, Python/CUDA runtime, and `/home/researcher/meadowmind-gpu` paths.
  - `stage`   : Upload SHA-verified code + `output/existing_wetware_libraries/` + `m9_model_paths.json`.
  - `run`     : Launch detached `library_smoke`, `library_stage0`, `library_stage3`, `library_stage4`,
                or gated `funnel` (`library_smoke` -> `library_stage0` -> `library_stage3`).
  - `status`  : Inspect running PID, completed job counts, GPU utilization, and log tail.
  - `collect` : Retrieve SHA-verified summaries & logs and render `REPORT.md` + 3 comparison figures.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT, ROOT / '.tools' / 'mea_ssh_deps'):
    s = str(extra)
    if s not in sys.path:
        sys.path.insert(0, s)

DEFAULT_REMOTE_DIR = '/home/researcher/wetware-afternoon-22021-20261123'
STAGE_ITEMS = (
    'project.godot',
    'agent',
    'configs',
    'godot_tests',
    'scenes',
    'scripts',
    'standalone/mea_oddball',
    'tools',
    'output/existing_wetware_libraries',
)
DEFAULT_MODEL_PATHS = {
    'model': '/home/researcher/meadowmind-gpu/models/Qwen3.5-0.8B',
    'initial_checkpoint': '/home/researcher/meadowmind-gpu/m8b-runs/exploration_attempt1/checkpoint_0032',
    'parent_checkpoint': '/home/researcher/meadowmind-gpu/m6-runs/small_v2_stable/checkpoint_0016',
    'warm_checkpoint': '/home/researcher/meadowmind-gpu/m8s3-runs/survival_supply_attempt3/checkpoint_0048',
    'kernel_overlay': '/home/researcher/meadowmind-gpu/m8b-runtime-candidates/causal_conv1d_1_7_0_sourcebuild/site',
}
DEFAULT_REMOTE_GODOT = '/home/researcher/meadowmind-gpu/godot/Godot_v4.7.2-stable_linux.x86_64'


def build_stage_tarball(project_root: Path = ROOT) -> tuple[bytes, str, int]:
    """Pack only essential source code, Godot assets, and purified wetware libraries into a tar.gz."""
    buf = io.BytesIO()
    n_files = 0
    with tarfile.open(mode='w:gz', fileobj=buf, compresslevel=4) as tar:
        for rel in STAGE_ITEMS:
            p = project_root / rel
            if not p.exists():
                raise FileNotFoundError(f'Missing required staging path: {p}')
            if p.is_file():
                tar.add(p, arcname=rel, recursive=False)
                n_files += 1
            else:
                for child in sorted(p.rglob('*')):
                    if not child.is_file():
                        continue
                    parts = set(child.parts)
                    if '__pycache__' in parts or child.suffix in ('.pyc', '.pyo'):
                        continue
                    arc = child.relative_to(project_root).as_posix()
                    tar.add(child, arcname=arc, recursive=False)
                    n_files += 1
    blob = buf.getvalue()
    return blob, hashlib.sha256(blob).hexdigest(), n_files


def _connect(host: str, user: str, port: int, password: str | None, key_path: Path | None):
    import paramiko
    client = paramiko.SSHClient()
    known = Path.home() / '.ssh' / 'known_hosts'
    if known.is_file():
        client.load_host_keys(str(known))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    kw = dict(
        hostname=host,
        port=int(port),
        username=user,
        timeout=12,
        banner_timeout=18,
        auth_timeout=20,
    )
    if password:
        kw.update(password=password, allow_agent=False, look_for_keys=False)
    elif key_path:
        kw.update(key_filename=str(key_path), allow_agent=False, look_for_keys=False)
    else:
        kw.update(allow_agent=True, look_for_keys=True)
    client.connect(**kw)
    client.get_transport().set_keepalive(25)
    return client


def _exec(client, cmd: str, *, timeout: int = 120) -> str:
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    rc = stdout.channel.recv_exit_status()
    if rc != 0:
        raise RuntimeError(f'Remote command exited {rc}:\nCMD: {cmd}\nSTDERR:\n{err[-1500:]}\nSTDOUT:\n{out[-1500:]}')
    return out.strip()


def cmd_probe(client, remote_dir: str, remote_python: str) -> dict:
    probe_py = f"""
import json, os, shutil, subprocess
from pathlib import Path
out = {{'hostname': os.uname().nodename, 'remote_dir_exists': Path({remote_dir!r}).exists()}}
try:
    smi = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu', '--format=csv,noheader'],
        text=True
    ).strip().splitlines()
    out['gpus'] = [line.strip() for line in smi if line.strip()]
except Exception as e:
    out['gpus_error'] = str(e)
paths = {json.dumps(DEFAULT_MODEL_PATHS)}
out['model_paths_exists'] = {{k: Path(v).exists() for k, v in paths.items()}}
out['godot_exists'] = Path({DEFAULT_REMOTE_GODOT!r}).is_file()
for cand in ({remote_python!r}, '/home/researcher/meadowmind-gpu/.venv/bin/python', 'python3'):
    p = shutil.which(cand) if not os.path.isabs(cand) else (cand if os.path.isfile(cand) else None)
    if p:
        try:
            info = subprocess.check_output([p, '-c', 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'], text=True).strip()
            out['python_cuda'] = {{'executable': p, 'torch': info}}
            break
        except Exception:
            pass
print(json.dumps(out))
"""
    raw = _exec(client, 'python3 -c ' + shlex.quote(probe_py))
    return json.loads(raw)


def cmd_stage(client, remote_dir: str, model_paths_override: Path | None) -> dict:
    blob, sha_hex, n_files = build_stage_tarball(ROOT)
    _exec(client, 'mkdir -p ' + shlex.quote(remote_dir))
    remote_tgz = remote_dir.rstrip('/') + '/stage_bundle.tar.gz'
    with client.open_sftp() as sftp:
        with sftp.open(remote_tgz + '.part', 'wb') as f:
            f.write(blob)
        sftp.posix_rename(remote_tgz + '.part', remote_tgz)
    remote_sha = _exec(client, 'sha256sum ' + shlex.quote(remote_tgz)).split()[0]
    if remote_sha != sha_hex:
        raise RuntimeError(f'Uploaded tarball SHA-256 mismatch: local={sha_hex}, remote={remote_sha}')
    _exec(client, f'tar -xzf {shlex.quote(remote_tgz)} -C {shlex.quote(remote_dir)} && rm -f {shlex.quote(remote_tgz)}')

    mpaths = (
        json.loads(model_paths_override.read_text(encoding='utf-8'))
        if model_paths_override else dict(DEFAULT_MODEL_PATHS)
    )
    remote_cfg = remote_dir.rstrip('/') + '/m9_model_paths.json'
    with client.open_sftp() as sftp:
        with sftp.open(remote_cfg, 'w') as f:
            f.write(json.dumps(mpaths, indent=2))
    return dict(remote_dir=remote_dir, n_files=n_files, bundle_bytes=len(blob), bundle_sha256=sha_hex, model_config=remote_cfg)


def build_remote_run_script(
    *,
    remote_dir: str,
    remote_python: str,
    remote_godot: str,
    phase: str,
    gpus: str,
    seeds: int,
    confirm_seeds: int,
    eval_worlds: int,
    n_boot: int,
    tier: int,
    ladder_tiers: str,
    r2_mode: str,
    include_stage4: bool,
    extra_pythonpath: str = '',
) -> str:
    rdir = shlex.quote(remote_dir)
    py = shlex.quote(remote_python)
    gd = shlex.quote(remote_godot)
    mcfg = shlex.quote(remote_dir.rstrip('/') + '/m9_model_paths.json')
    libs = shlex.quote(remote_dir.rstrip('/') + '/output/existing_wetware_libraries')
    runner = 'tools/gpu_existing_data_runner.py'
    out_root = remote_dir.rstrip('/') + '/results'

    lines = [
        '#!/usr/bin/env bash',
        'set -euo pipefail',
        f'cd {rdir}',
        'export PYTHONUNBUFFERED=1',
        'export HF_HUB_OFFLINE=1',
        'export TRANSFORMERS_OFFLINE=1',
        'export TOKENIZERS_PARALLELISM=false',
        'export OMP_NUM_THREADS=1',
        'export MKL_NUM_THREADS=1',
        'export PYTHONDONTWRITEBYTECODE=1',
        'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',
    ]
    if extra_pythonpath:
        lines.append(f'export PYTHONPATH={shlex.quote(extra_pythonpath)}:${{PYTHONPATH:-}}')
    lines.append(f'mkdir -p {shlex.quote(out_root)}')

    def _cmd(orch: str, out_sub: str, extra: str = '') -> str:
        return (
            f'{py} -u {runner} --orchestrate {orch} '
            f'--model-config {mcfg} --godot {gd} --project {rdir} '
            f'--libraries-dir {libs} --r2-mode {shlex.quote(r2_mode)} '
            f'--gpus {shlex.quote(gpus)} --microbatch 1 --eval-worlds {int(eval_worlds)} --n-boot {int(n_boot)} '
            f'--out {shlex.quote(out_root + "/" + out_sub)} {extra}'
        ).strip()

    if phase == 'library_smoke':
        lines.append(_cmd('library_smoke', 'smoke'))
    elif phase == 'library_stage0':
        lines.append(_cmd('library_stage0', 'stage0',
                          f'--seeds {int(seeds)} --confirm-seeds {int(confirm_seeds)} --ladder-tiers {shlex.quote(ladder_tiers)}'))
    elif phase == 'library_stage3':
        st0 = shlex.quote(out_root + '/stage0/stage0_summary.json')
        lines.append(_cmd('library_stage3', 'stage3', f'--stage0-report {st0} --tier {int(tier)} --seeds {int(seeds)}'))
    elif phase == 'library_stage4':
        st0 = shlex.quote(out_root + '/stage0/stage0_summary.json')
        lines.append(_cmd('library_stage4', 'stage4', f'--stage0-report {st0} --tier {int(tier)} --seeds {int(seeds)}'))
    elif phase == 'funnel':
        st0 = shlex.quote(out_root + '/stage0/stage0_summary.json')
        lines.append('echo "=== [1/3] Running library_smoke ==="')
        lines.append(_cmd('library_smoke', 'smoke'))
        lines.append('echo "=== [2/3] Running library_stage0 ==="')
        lines.append(_cmd('library_stage0', 'stage0',
                          f'--seeds {int(seeds)} --confirm-seeds {int(confirm_seeds)} --ladder-tiers {shlex.quote(ladder_tiers)}'))
        lines.append('echo "=== [3/3] Running library_stage3 (22021 clean 233-ROI) ==="')
        lines.append(_cmd('library_stage3', 'stage3', f'--stage0-report {st0} --tier {int(tier)} --seeds {int(seeds)}'))
        if include_stage4:
            lines.append(
                f'{py} -c "import json,sys; r=json.load(open({out_root + "/stage3/stage3_summary.json"!r})); '
                f'sys.exit(0 if r[\'outcome\'][\'verdict\'] in (\'weak_positive\',\'dual_weak_positive\',\'stable_positive\') else 99)" '
                f'&& echo "=== [4/4] Stage 3 positive -> Running library_stage4 (28080 replication) ===" '
                f'&& {_cmd("library_stage4", "stage4", f"--stage0-report {st0} --tier {int(tier)} --seeds {int(seeds)}")} '
                f'|| echo "=== Stage 3 did not reach weak_positive; skipping Stage 4 ==="'
            )
    else:
        raise ValueError(f'Unknown phase: {phase}')

    lines.append('echo "=== ALL REQUESTED PHASES FINISHED ==="')
    return '\n'.join(lines) + '\n'


def cmd_run(client, args) -> dict:
    script = build_remote_run_script(
        remote_dir=args.remote_dir,
        remote_python=args.remote_python,
        remote_godot=args.remote_godot,
        phase=args.phase,
        gpus=args.gpus,
        seeds=args.seeds,
        confirm_seeds=args.confirm_seeds or args.seeds,
        eval_worlds=args.eval_worlds,
        n_boot=args.n_boot,
        tier=args.tier,
        ladder_tiers=args.ladder_tiers,
        r2_mode=args.r2_mode,
        include_stage4=args.include_stage4,
        extra_pythonpath=args.extra_pythonpath,
    )
    script_path = args.remote_dir.rstrip('/') + '/run_campaign.sh'
    log_path = args.remote_dir.rstrip('/') + '/run.log'
    pid_path = args.remote_dir.rstrip('/') + '/runner.pid'
    with client.open_sftp() as sftp:
        with sftp.open(script_path, 'w') as f:
            f.write(script)
    _exec(client, f'chmod +x {shlex.quote(script_path)}')
    pid = _exec(
        client,
        f'nohup bash {shlex.quote(script_path)} > {shlex.quote(log_path)} 2>&1 < /dev/null & echo $! | tee {shlex.quote(pid_path)}',
    )
    return dict(started=True, pid=pid.strip(), phase=args.phase, log=log_path, script=script_path)


def cmd_status(client, remote_dir: str) -> dict:
    st_py = f"""
import json, os, subprocess
from pathlib import Path
r = Path({remote_dir!r})
res = r / 'results'
pid_f = r / 'runner.pid'
pid = pid_f.read_text().strip() if pid_f.is_file() else None
alive = bool(pid and Path('/proc', pid).exists())
counts = {{}}
if res.is_dir():
    for p in sorted(res.rglob('jobs')):
        rel = str(p.relative_to(res))
        counts[rel] = len(list(p.glob('*.json')))
summaries = [str(p.relative_to(res)) for p in sorted(res.rglob('*_summary.json'))] if res.is_dir() else []
log_tail = (r / 'run.log').read_text(errors='replace').splitlines()[-25:] if (r / 'run.log').is_file() else []
try:
    gpus = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader'], text=True).strip().splitlines()
except Exception:
    gpus = []
print(json.dumps(dict(pid=pid, alive=alive, job_counts=counts, summaries=summaries, gpus=gpus, log_tail=log_tail)))
"""
    return json.loads(_exec(client, 'python3 -c ' + shlex.quote(st_py)))


def cmd_collect(client, remote_dir: str, local_out: Path, *, include_jobs: bool = False) -> dict:
    local_out = Path(local_out)
    local_out.mkdir(parents=True, exist_ok=True)
    pack_py = f"""
import hashlib, json, tarfile
from pathlib import Path
r = Path({remote_dir!r})
res = r / 'results'
arc = r / 'collect_bundle.tar.gz'
added = []
with tarfile.open(arc, 'w:gz') as t:
    for f in (r / 'run.log', r / 'run_campaign.sh', r / 'm9_model_paths.json'):
        if f.is_file():
            t.add(f, arcname=f.name)
            added.append(f.name)
    if res.is_dir():
        for p in sorted(res.rglob('*.json')):
            if (not {bool(include_jobs)!r}) and ('jobs' in p.parts):
                continue
            name = 'results/' + p.relative_to(res).as_posix()
            t.add(p, arcname=name)
            added.append(name)
sha = hashlib.sha256(arc.read_bytes()).hexdigest()
print(json.dumps(dict(archive=str(arc), sha256=sha, n_files=len(added))))
"""
    meta = json.loads(_exec(client, 'python3 -c ' + shlex.quote(pack_py), timeout=300))
    local_tgz = local_out / 'collect_bundle.tar.gz'
    with client.open_sftp() as sftp:
        sftp.get(meta['archive'], str(local_tgz))
    actual_sha = hashlib.sha256(local_tgz.read_bytes()).hexdigest()
    if actual_sha != meta['sha256']:
        raise RuntimeError(f"Collected archive SHA-256 mismatch: {actual_sha} != {meta['sha256']}")
    with tarfile.open(local_tgz, 'r:gz') as t:
        for member in t.getmembers():
            dest = (local_out / member.name).resolve()
            if not dest.is_relative_to(local_out.resolve()):
                raise ValueError(f'Unsafe archive member path: {member.name}')
        t.extractall(local_out, filter='data')
    local_tgz.unlink(missing_ok=True)

    rep_info = None
    st3 = local_out / 'results/stage3/stage3_summary.json'
    st0 = local_out / 'results/stage0/stage0_summary.json'
    st4 = local_out / 'results/stage4/stage4_summary.json'
    if st3.is_file():
        from tools import report_wetware_comparison as rwc
        rep_info = rwc.generate_report(
            manifest_path=ROOT / 'output/existing_wetware_libraries/gate_manifest.json',
            stage0_path=st0 if st0.is_file() else None,
            stage3_path=st3,
            stage4_path=st4 if st4.is_file() else None,
            out_dir=local_out / 'report',
        )
    elif st0.is_file():
        from tools import report_wetware_comparison as rwc
        diag_p = local_out / 'results/stage0/stage0_diagnostic.json'
        smoke_p = local_out / 'results/smoke/smoke_summary.json'
        rep_info = rwc.generate_stage0_report(
            manifest_path=ROOT / 'output/existing_wetware_libraries/gate_manifest.json',
            stage0_path=st0,
            stage0_diag_path=diag_p if diag_p.is_file() else None,
            smoke_path=smoke_p if smoke_p.is_file() else None,
            out_dir=local_out / 'report',
        )
    return dict(local_out=str(local_out), n_files=meta['n_files'], archive_sha256=actual_sha, report=rep_info)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('probe', 'stage', 'run', 'status', 'collect'))
    p.add_argument('--host', required=True)
    p.add_argument('--user', default='researcher')
    p.add_argument('--port', type=int, default=22)
    p.add_argument('--key', type=Path)
    p.add_argument('--password-stdin', action='store_true')
    p.add_argument('--remote-dir', default=DEFAULT_REMOTE_DIR)
    p.add_argument('--remote-python', default='/home/researcher/meadowmind-gpu/.venv/bin/python')
    p.add_argument('--remote-godot', default=DEFAULT_REMOTE_GODOT)
    p.add_argument('--extra-pythonpath', default='')
    p.add_argument('--model-paths-json', type=Path)
    p.add_argument('--phase', choices=('library_smoke', 'library_stage0', 'library_stage3', 'library_stage4', 'funnel'), default='funnel')
    p.add_argument('--gpus', default='0')
    p.add_argument('--tier', type=int, default=1)
    p.add_argument('--ladder-tiers', default='1,2,3,4')
    p.add_argument('--seeds', type=int, default=16)
    p.add_argument('--confirm-seeds', type=int)
    p.add_argument('--eval-worlds', type=int, default=48)
    p.add_argument('--n-boot', type=int, default=10000)
    p.add_argument('--r2-mode', default='dual')
    p.add_argument('--include-stage4', action='store_true')
    p.add_argument('--local-out', type=Path, default=ROOT / 'output/wetware_afternoon_22021_gpu')
    p.add_argument('--include-jobs', action='store_true')
    return p


def main():
    args = build_parser().parse_args()
    password = None
    if args.password_stdin:
        password = sys.stdin.readline().rstrip('\r\n')
    elif 'WETWARE_GPU_PASSWORD' in os.environ:
        password = os.environ.pop('WETWARE_GPU_PASSWORD')

    client = _connect(args.host, args.user, args.port, password, args.key)
    try:
        if args.action == 'probe':
            res = cmd_probe(client, args.remote_dir, args.remote_python)
        elif args.action == 'stage':
            res = cmd_stage(client, args.remote_dir, args.model_paths_json)
        elif args.action == 'run':
            res = cmd_run(client, args)
        elif args.action == 'status':
            res = cmd_status(client, args.remote_dir)
        elif args.action == 'collect':
            res = cmd_collect(client, args.remote_dir, args.local_out, include_jobs=args.include_jobs)
        else:
            raise AssertionError('unreachable')
    finally:
        client.close()
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
