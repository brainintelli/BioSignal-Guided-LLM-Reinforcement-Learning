"""Own only the M8X service. Never stop M6, training, or another GPU workload."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import tempfile
import urllib.request

ROOT = Path.home() / "meadowmind-gpu"
RECORD = ROOT / "m8x_service.json"
MODEL_ID = "meadowmind-qwen3.5-0.8b-m8x"


class LifecycleLock:
    """OS-owned cross-process lock; persistent inode, automatically released on exit."""
    def __init__(self, path): self.path, self.stream = Path(path), None

    def __enter__(self):
        self.stream = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.stream.write(b"0"); self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close(); self.stream = None
            raise RuntimeError("Another M8X lifecycle operation holds the exclusive lock") from exc
        return self

    def __exit__(self, *_):
        if self.stream is not None:
            self.stream.close(); self.stream = None


def atomic_record(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(directory)
            finally: os.close(directory)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def reclaim_created(process, created):
    """Only the Popen child whose exact live identity was captured may be reclaimed."""
    if process.poll() is not None: return
    if created is None or identity(process.pid) != created:
        raise RuntimeError("Cannot reclaim unverified/reused child identity; no signal sent")
    if os.getpgid(process.pid) != process.pid:
        raise RuntimeError("New child no longer owns its process group; no signal sent")
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if identity(process.pid) != created or os.getpgid(process.pid) != process.pid:
            raise RuntimeError("Child identity changed during cleanup; no further signal sent")
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def spawn_owned(command, project, env, log_path, expected_record):
    # Caller holds LifecycleLock across the preflight, spawn, publish and health check.
    with Path(log_path).open("ab", buffering=0) as log:
        process = subprocess.Popen(command, cwd=project, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=log, start_new_session=True)
    created = None
    try:
        created = identity(process.pid)
        if created["command"] != command:
            raise RuntimeError("New child command differs from requested M8X service")
        saved = dict(created, **expected_record, created_utc=datetime.now(timezone.utc).isoformat())
        atomic_record(RECORD, saved)
        return saved
    except BaseException:
        reclaim_created(process, created)
        raise


def validate_health(health, report, checkpoint, verification_sha):
    from agent.m8x_server import decision_identity
    from agent.m8x_contract import VERSION, REWARD_CONFIG
    expected = dict(decision_identity(report["selected_actor_sha256"], verification_sha),
        status="ok", decision_ready=True, model=MODEL_ID, checkpoint=str(checkpoint),
        release_authorized=True, evidence_complete=True, release_phase="test", survival_gate_passed=True,
        environment_profile="open3", registration_sha256=report["registration_sha256"],
        task_goals=False, scripted_fallback=False, transport_context_required=True)
    expected["source_sha256"] = report["source_sha256"]
    metadata=health.get("policy",{})
    bound=dict(contract=VERSION,physical_forward_shape=[4,1536],reward_config=REWARD_CONFIG,
               kernel_identity=report["execution_identity"]["runtime"]["kernel_identity"])
    if any(metadata.get(k)!=v for k,v in bound.items()):raise ValueError("Loaded health policy/runtime differs")
    if any(health.get(k) != value for k, value in expected.items()):
        raise ValueError("M8X health does not match complete release identity")


def identity(pid):
    proc = Path(f"/proc/{pid}")
    stat = (proc / "stat").read_text()
    fields = stat[stat.rfind(")") + 2:].split()
    if fields[0] == "Z": raise ProcessLookupError(pid)
    command = (proc / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
    return dict(pid=pid, start_ticks=fields[19], executable=str((proc / "exe").resolve()), command=command)


def owned():
    if not RECORD.exists(): return None
    saved = json.loads(RECORD.read_text())
    if (type(saved.get("pid")) is not int or saved["pid"] <= 1
            or "agent.m8x_server" not in saved.get("command", [])):
        raise RuntimeError("Invalid M8X ownership record; nothing stopped")
    try: current = identity(saved["pid"])
    except (FileNotFoundError, ProcessLookupError): return None
    if any(saved.get(k) != value for k, value in current.items()):
        raise RuntimeError("Process identity/command mismatch; nothing stopped")
    return saved


def stop(saved):
    if saved is None: return
    if owned() != saved or os.getpgid(saved["pid"]) != saved["pid"]:
        raise RuntimeError("Owned process group changed; nothing stopped")
    os.killpg(saved["pid"], signal.SIGTERM)
    for _ in range(100):
        try:
            current = identity(saved["pid"])
            if any(current[k] != saved[k] for k in current):
                raise RuntimeError("PID changed after termination; replacement left untouched")
        except (FileNotFoundError, ProcessLookupError): return
        time.sleep(.1)
    raise RuntimeError("M8X did not stop within 10 seconds; no forced kill")


def isolated_environment(config, project):
    overlay=Path(config['kernel_overlay']).resolve();project=Path(project).resolve()
    if (not overlay.is_relative_to((ROOT/'m8b-runtime-candidates').resolve()) or not overlay.is_dir()
            or project!=(ROOT/'m8b-project').resolve()):
        raise ValueError('Unexpected isolated serving runtime/project')
    return dict(os.environ,CUDA_VISIBLE_DEVICES='0',TOKENIZERS_PARALLELISM='false',
                HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH=str(overlay)+':'+str(project))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "stop", "status"])
    parser.add_argument("--verification")
    args = parser.parse_args()
    with LifecycleLock(ROOT / "m8x_service.lock"):
        run_locked(args)


def run_locked(args):
    saved = owned()
    if args.action == "status":
        print(json.dumps(dict(running=saved is not None, service=saved), indent=2)); return
    if args.action == "stop":
        stop(saved); print("OWNED_M8X_SERVICE_STOPPED"); return
    if not args.verification: raise ValueError("Full accepted test verification required")
    verification = Path(args.verification).resolve()
    if not any(verification.is_relative_to((ROOT / name).resolve()) for name in ("m8b-runs", "m8b-evidence")):
        raise ValueError("Verification must be inside project M8X evidence")
    report = json.loads(verification.read_text())
    project = Path(report["config"]["project"]).resolve()
    if project != (ROOT / "m8b-project").resolve(): raise ValueError("Wrong registered M8X project")
    sys.path.insert(0, str(project))
    from agent.m8x_identity import sha
    from agent.m8x_release import rebuild_release
    report = rebuild_release(report, project)
    checkpoint = (Path(report["config"]["output"]) / report["selected_checkpoint"]).resolve()
    model = Path(report["config"]["model"]).resolve()
    if not checkpoint.is_relative_to((ROOT / "m8b-runs").resolve()) or not model.is_relative_to((ROOT / "models").resolve()):
        raise ValueError("Model/checkpoint outside owned project paths")
    verification_sha = sha(verification)
    expected_record = dict(checkpoint=str(checkpoint), verification=str(verification), verification_sha256=verification_sha,
                           expected_actor_sha256=report["selected_actor_sha256"], model_path=str(model))
    if saved and any(saved.get(k) != v for k, v in expected_record.items()):
        raise RuntimeError("A different owned M8X service is running; explicit stop required")
    if not saved:
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"], text=True).strip()
        if active: raise RuntimeError("GPU compute work left untouched: " + active)
        with socket.socket() as sock: sock.bind(("127.0.0.1", 18080))
        logdir = ROOT / "logs"
        logdir.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-u", "-m", "agent.m8x_server", "--model", str(model), "--checkpoint", str(checkpoint),
                   "--verification", str(verification), "--project", str(project), "--port", "18080",
                   "--log", str(logdir / "m8x_live_decisions.jsonl")]
        env = isolated_environment(report["config"], project)
        saved = spawn_owned(command, project, env, logdir / "m8x_server.log", expected_record)
    for _ in range(300):
        if owned() is None: raise RuntimeError("M8X service exited; inspect logs/m8x_server.log")
        try:
            with urllib.request.urlopen("http://127.0.0.1:18080/health", timeout=2) as response:
                health = json.load(response)
        except (OSError, ValueError):
            time.sleep(1); continue
        validate_health(health, report, str(checkpoint), verification_sha)
        print(json.dumps(dict(service=saved, health=health), indent=2)); return
    raise RuntimeError("Health deadline exceeded; M8X process remains recorded for explicit inspection/stop")


if __name__ == "__main__": main()
