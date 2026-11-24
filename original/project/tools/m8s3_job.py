"""Detached finite worker supervisor: own process groups only, cleanup on every exit."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.m8s3_training import sha, write, verify_files
from agent.m8s3_diag_contract import verify_registration_digest


def group_members(pgid):
    result = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            stat = (proc / 'stat').read_text()
            fields = stat[stat.rfind(')')+2:].split()
            if fields[0] != 'Z' and int(fields[2]) == pgid and int(fields[3]) == pgid:
                result.append(int(proc.name))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return result


def cleanup_group(process):
    # The supervisor creates a new session for each worker. Its children share
    # that session; unrelated jobs are never selected by executable or username.
    members = group_members(process.pid)
    if members:
        os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + 10
        while group_members(process.pid) and time.monotonic() < deadline:
            time.sleep(.1)
        if group_members(process.pid):
            os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=15)
    if group_members(process.pid):
        raise RuntimeError('Owned worker descendants remain after cleanup')


def worker(job):
    launch = json.loads((job / 'launch.json').read_text())
    registration = Path(launch['registration'])
    if sha(registration) != launch['registration_file_sha256']:
        raise ValueError('Registration changed after submission')
    reg = json.loads(registration.read_text())
    verify_registration_digest(reg)
    c = reg['config']
    verify_files(c['project'], reg)
    output = Path(c['output'])
    env = dict(os.environ, PYTHONPATH=c['kernel_overlay'] + ':' + c['project'],
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0')
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    began = time.monotonic()
    stages = []

    def run_stage(name, extra):
        if stopping:
            raise InterruptedError('Stop requested')
        command = [sys.executable, '-u', *extra]
        with (job / (name + '.log')).open('xb', buffering=0) as stream:
            p = subprocess.Popen(command, cwd=c['project'], env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=stream, start_new_session=True)
            write(job / 'active_child.json', dict(pid=p.pid, process_group=p.pid, stage=name,
                  command=command, launched_utc=datetime.now(timezone.utc).isoformat()))
            try:
                while p.poll() is None:
                    if stopping or time.monotonic()-began > c['hard_wall_seconds']:
                        raise TimeoutError('Stop requested or registered hard wall reached')
                    time.sleep(1)
            finally:
                cleanup_group(p)
        stages.append(dict(stage=name, exit_code=p.returncode))
        write(job / 'stages.json', stages)
        if p.returncode:
            raise RuntimeError(f'{name} exited with code {p.returncode}')

    status = 'failed'
    error = None
    try:
        run_stage('training', ['-m', 'agent.m8s3_run', '--registration', str(registration)])
        selection = output / 'selection.json'
        selected = (json.loads(selection.read_text())['checkpoint'] if selection.exists()
                    else str(output / f"checkpoint_{c['updates']:04d}"))
        run_stage('reload', ['-m', 'agent.m8s3_run', '--registration', str(registration), '--reload-checkpoint', selected])
        run_stage('replay', ['tools/audit_m8s3_training.py', '--registration', str(registration)])
        if c['mode'] == 'formal' and selection.exists() and json.loads(selection.read_text())['eligible_for_test']:
            run_stage('test', ['-m', 'agent.m8s3_run', '--registration', str(registration), '--test-stage'])
            run_stage('test_replay', ['tools/audit_m8s3_training.py', '--registration', str(registration), '--evaluation-only'])
            run_stage('legacy_test', ['-m', 'agent.m8s3_run', '--registration', str(registration), '--legacy-test'])
        status = json.loads((output / 'status.json').read_text())['status']
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                                       '--format=csv,noheader'], text=True).strip()
        write(job / 'finished.json', dict(status=status, error=error, stages=stages,
            seconds=time.monotonic()-began, finished_utc=datetime.now(timezone.utc).isoformat(),
            owned_process_groups_cleaned=True, remaining_gpu_compute_processes=gpu,
            target_completed=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['start', '_run'])
    ap.add_argument('--registration', required=True)
    ap.add_argument('--job', required=True)
    a = ap.parse_args()
    job = Path(a.job).resolve()
    if a.action == '_run':
        worker(job)
        return
    registration = Path(a.registration).resolve()
    reg = json.loads(registration.read_text())
    verify_registration_digest(reg)
    verify_files(reg['config']['project'], reg)
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                      '--format=csv,noheader'], text=True).strip()
    if active:
        raise RuntimeError('GPU is busy: ' + active)
    job.mkdir(parents=True, exist_ok=False)
    write(job / 'launch.json', dict(registration=str(registration), registration_file_sha256=sha(registration),
          created_utc=datetime.now(timezone.utc).isoformat(), config=reg['config']))
    with (job / 'manager.log').open('xb', buffering=0) as stream:
        p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '_run',
            '--registration', str(registration), '--job', str(job)], cwd=reg['config']['project'],
            stdin=subprocess.DEVNULL, stdout=stream, stderr=stream, start_new_session=True)
    write(job / 'manager.json', dict(pid=p.pid, registration_sha256=reg['registration_sha256']))
    print(json.dumps(dict(pid=p.pid, job=str(job), registration_sha256=reg['registration_sha256'])))


if __name__ == '__main__':
    main()
