"""Run the audited 22021 spontaneous-MEA policy in a local Godot M9 game."""

import argparse
from contextlib import ExitStack, redirect_stdout
import getpass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
STUDY = "mea-22021-20261118-mea-stim-ablation/pre-stim-baseline"
LOCAL_STUDY = "output/mea_predictability_22021_20261118T093720Z/offline-mea-stim-ablation/pre-stim-baseline"
TRAINING_SEED = 102122616
ARM = "baseline"
BLOCK = f"models/{TRAINING_SEED}/{ARM}"
MODEL = f"22021 pre-stim baseline / {TRAINING_SEED} / PPO 4"
STATE2_MODE = "historical_holdout_cycle"
REPLY = "MEADOW_MODEL_JSON="
HOST = "192.0.2.16"
MIN_GPU_FREE_MIB = 12 * 1024
GPU_ALLOCATOR_LIMIT_MIB = 6 * 1024


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_identity(folder):
    manifest = read(folder / "checkpoint/manifest.json")
    policy = read(folder / "checkpoint/policy.json")
    audit = read(folder / "replay-audit.json")
    if (audit["seed"], audit["arm"], audit["worlds"], audit["decisions"]) != (TRAINING_SEED, ARM, 48, 288):
        raise ValueError("Unexpected model audit")
    if policy["comparison"] != ARM or policy["r2_enabled"] or manifest["update"] != 4:
        raise ValueError("Unexpected model checkpoint")
    return {
        "model": MODEL,
        "profile": "m9",
        "world_version": policy["world_version"],
        "environment_version": policy["environment_version"],
        "balance_sha256": policy["balance_sha256"],
        "map_mode": policy["map_mode"],
        "navigation_steps": policy["navigation_steps"],
        "actor_sha256": manifest["actor_sha256"],
        "registration_sha256": manifest["registration_sha256"],
        "state2_mode": STATE2_MODE,
    }


def run_worker():
    home = Path.home()
    study = home / STUDY
    folder = study / BLOCK
    run = home / "mea-state2-r2-20261118/run02"
    packages = home / "mea-predictability-check-369c6af196e0"
    environment = read(run / "environment.json")
    os.environ.update(environment)
    sys.path[:0] = [str(study), str(packages), str(packages / "mea_binary_r2"),
                    str(packages / "mea_state2_r2"), *environment["PYTHONPATH"].split(":")]

    from agent.mea_contract import observation_key
    from agent.m9_contract import validate
    from agent.m9_env import GodotEnv
    from agent.m9_policy import load
    from agent.mea_policy import RecordingPolicy
    from train import FixedContextPolicy
    from mea_state2_r2.features import digest
    import torch

    total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(GPU_ALLOCATOR_LIMIT_MIB * 1024**2 / total_gpu_memory, 0)

    context = read(study / f"{ARM}.json")
    windows = context["series"]["holdout"]
    spec = read(study / "design.json")
    registration = read(folder / "registration.json")
    identity = checkpoint_identity(folder)
    if (len(windows) != spec["eval_decisions"] or spec["eval_decisions"] != 6
            or digest(context) != spec["contexts"][ARM]
            or digest(context["series"]) != registration["windows_digest"]
            or digest(spec) != registration["study_digest"]
            or digest(registration) != identity["registration_sha256"]):
        raise ValueError("Study or recorded input changed")

    class ServingPolicy(FixedContextPolicy):
        def __init__(self, config):
            # Reuse the registered encoder and checkpoint metadata without allocating PPO's reference model.
            self.comparison = ARM
            self.context = context
            self.arm = "C"
            self.normalization_id = digest(context["normalization"])
            RecordingPolicy.__init__(self, config, reference=False)

    def choose(observation, world_seed, decision_index):
        window = windows[decision_index % len(windows)]
        torch.manual_seed(world_seed * 10 + decision_index)
        prepared, snapshot = policy.replay_input(observation, window)
        sample = policy.act([prepared])
        action = sample["actions"][0]
        if action not in observation["allowed_actions"]:
            raise ValueError("Actor returned an illegal action")
        return action, snapshot["window_index"]

    def emit(payload):
        print(REPLY + json.dumps(payload, ensure_ascii=True, allow_nan=False), flush=True)

    with redirect_stdout(sys.stderr):
        torch.manual_seed(6100000)
        policy = ServingPolicy(SimpleNamespace(**registration["config"],
            **read(study / "model_config.json"), rng_seed=6100000))
        load(policy, folder / "checkpoint")
        policy.eval()
        if policy.digest() != identity["actor_sha256"]:
            raise ValueError("Loaded actor does not match the audited checkpoint")
        assets = home / "mea-r2-22554/runtime_20261116_01/assets"
        godot = assets / "godot/Godot_v4.7.2-stable_linux.x86_64"
        project = home / "cpp28080-crossover-20261118-replay/start-scene-paired/game"
        with (folder / "holdout.jsonl").open(encoding="utf-8") as trace:
            first = json.loads(trace.readline())
        with GodotEnv(str(godot), str(project), map_mode="infinite_freeze", navigation_steps=4) as game:
            if len(first["rows"]) != len(windows):
                raise ValueError("Unexpected audited trajectory length")
            observation = game.reset(first["world"])["observation"]
            for index, reference in enumerate(first["rows"]):
                action, window_index = choose(observation, first["world"], index)
                if (observation_key(observation) != reference["context"]
                        or action != reference["action"] or window_index != reference["wetware_window"]):
                    raise ValueError(f"Checkpoint did not reproduce audited game decision {index + 1}")
                observation = game.step(action)["observation"]
    emit(dict(identity, decision_ready=True, probe="holdout_trajectory_reproduced",
              gpu=os.environ["CUDA_VISIBLE_DEVICES"],
              startup_reserved_mib=round(torch.cuda.max_memory_reserved() / 1024**2)))

    decisions = {}
    for line in sys.stdin:
        tick = None
        try:
            request = json.loads(line)
            observation = request["observation"]
            agent_context = request["agent_context"]
            validate(observation)
            if any(observation[key] != identity[key] for key in ("map_mode", "navigation_steps")):
                raise ValueError("M9 rules do not match the checkpoint")
            tick = observation["tick"]
            if type(tick) is not int or tick < 0:
                raise ValueError("Expected a nonnegative game tick")
            journey = (agent_context["session_id"], agent_context["episode"], agent_context["world_seed"])
            last_tick, index = decisions.get(journey, (-1, -1))
            if tick < last_tick:
                raise ValueError("Game tick precedes the last model decision")
            if tick > last_tick:
                index += 1
            start_time = time.monotonic()
            with redirect_stdout(sys.stderr):
                action, window_index = choose(observation, agent_context["world_seed"], index)
            decisions[journey] = (tick, index)
            response = dict(identity, tick=tick, agent_context=agent_context, action=action,
                            source="model", raw_valid=True, raw_output=json.dumps(action, ensure_ascii=False),
                            decoding="stochastic_by_world_seed_and_decision", state2_window=window_index,
                            latency_ms=round((time.monotonic() - start_time) * 1000, 1),
                            reason="22021 自发 MEA 检查点 · 历史信号窗口")
        except torch.cuda.OutOfMemoryError as error:
            print(f"Model inference failed at tick {tick}: {error}", file=sys.stderr, flush=True)
            response = {"error": "GPU 推理显存不足"}
        except Exception as error:
            print(f"Model inference failed at tick {tick}: {type(error).__name__}: {error}",
                  file=sys.stderr, flush=True)
            response = {"error": str(error)}
        emit(response)


def worker_reply(stream, log):
    for line in stream:
        if line.startswith(REPLY):
            return json.loads(line[len(REPLY):])
        print(line.rstrip(), file=log, flush=True)
    raise RuntimeError("GPU model worker disconnected")


def handler_for(health, remote_in, remote_out, log):
    lock = threading.Lock()
    ready = True

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, payload):
            body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                return self.reply(404, {"error": "unknown endpoint"})
            self.reply(200 if ready else 503, dict(health, decision_ready=ready))

        def do_POST(self):
            nonlocal ready
            if self.path != "/decide":
                return self.reply(404, {"error": "unknown endpoint"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                session = self.headers.get("X-Agent-Session", "")
                if not 0 < length <= 65536 or not 0 < len(session) <= 128:
                    raise ValueError("Invalid request size or session")
                context = dict(session_id=session, episode=int(self.headers["X-Agent-Episode"]),
                               world_seed=int(self.headers["X-World-Seed"]))
                observation = json.loads(self.rfile.read(length))
            except (TypeError, ValueError, KeyError) as error:
                return self.reply(400, {"error": str(error)})

            with lock:
                if not ready:
                    return self.reply(503, {"error": "worker unavailable"})
                try:
                    remote_in.write(json.dumps(dict(observation=observation, agent_context=context)) + "\n")
                    remote_in.flush()
                    result = worker_reply(remote_out, log)
                except (OSError, RuntimeError, ValueError) as error:
                    ready = False
                    print(f"GPU worker connection failed: {error}", file=log, flush=True)
                    return self.reply(503, {"error": "GPU 推理服务已断开"})
            self.reply(422 if "error" in result else 200, result)

        def log_message(self, *_):
            pass

    return Handler


def launch_game(seed, smoke_seconds, password):
    sys.path.insert(0, str(ROOT / ".tools/mea_ssh_deps"))
    import paramiko

    local_folder = ROOT / LOCAL_STUDY / BLOCK
    expected = checkpoint_identity(local_folder)
    logs = ROOT / "output/runtime_logs"
    logs.mkdir(exist_ok=True)
    engines = ROOT / ".tools/godot"
    engine = engines / ("Godot_v4.7.2-stable_win64_console.exe" if smoke_seconds else
                        "Godot_v4.7.2-stable_win64.exe")
    if not engine.is_file():
        raise FileNotFoundError(engine)

    with ExitStack() as stack:
        ssh = stack.enter_context(paramiko.SSHClient())
        ssh.load_system_host_keys()
        try:
            ssh.connect(HOST, username="researcher", password=password, timeout=10, auth_timeout=10)
        except paramiko.AuthenticationException as error:
            raise RuntimeError(f"GPU SSH login failed for researcher@{HOST}. Check the password or SSH key.") from error
        except paramiko.SSHException as error:
            raise RuntimeError(f"GPU SSH connection failed for researcher@{HOST}: {error}") from error
        ssh.get_transport().set_keepalive(20)
        _, folder_out, _ = ssh.exec_command("mktemp -d", timeout=10)
        temporary = folder_out.read().decode().strip()
        if folder_out.channel.recv_exit_status() or not temporary.startswith("/tmp/"):
            raise RuntimeError("Could not create a temporary GPU worker directory")
        sftp = stack.enter_context(ssh.open_sftp())
        script = temporary + "/mea_model_game.py"
        stack.callback(sftp.rmdir, temporary)
        sftp.put(__file__, script)
        stack.callback(sftp.remove, script)
        command = (
            "gpu=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits "
            f"| awk -F, '$2 >= {MIN_GPU_FREE_MIB} && $3 < 10 && $2 > best "
            "{ best=$2; gpu=$1 } END { if (best) print gpu }'); "
            f'test -n "$gpu" || {{ echo "No GPU below 10% utilization with {MIN_GPU_FREE_MIB} MiB free"; exit 1; }}; '
            'export CUDA_VISIBLE_DEVICES="$gpu"; '
            "exec ~/mea-r2-22554/runtime_20261116_01/assets/python/bin/python -u "
            + shlex.quote(script) + " --worker"
        )
        channel = ssh.get_transport().open_session(timeout=10)
        stack.callback(channel.close)
        channel.settimeout(90)
        channel.set_combine_stderr(True)
        channel.exec_command(command)
        remote_in = channel.makefile_stdin("w", -1)
        remote_out = channel.makefile("r").__iter__()
        # Paramiko may return bytes even for its text file wrapper.
        remote_lines = (line.decode("utf-8", "replace") if isinstance(line, bytes) else line for line in remote_out)
        log = stack.enter_context((logs / f"mea-22021-{os.getpid()}.log").open("x", encoding="utf-8"))
        print("Loading the audited 22021 model and checking the first game action...", flush=True)
        try:
            health = worker_reply(remote_lines, log)
        except (OSError, RuntimeError, ValueError) as error:
            raise RuntimeError(f"GPU model did not become ready. See {log.name}") from error
        if (any(health.get(k) != v for k, v in expected.items())
                or health.get("probe") != "holdout_trajectory_reproduced"
                or health.get("decision_ready") is not True):
            raise ValueError("GPU model does not match the local audited checkpoint")
        print(f"GPU {health['gpu']} ready. Startup peak {health['startup_reserved_mib']} MiB.", flush=True)

        with ThreadingHTTPServer(("127.0.0.1", 0), handler_for(health, remote_in, remote_lines, log)) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            game_log = logs / f"godot-22021-{os.getpid()}.log"
            base = [str(engine), "--path", str(ROOT), "--log-file", str(game_log)]
            flags = ["--", "--difficulty=m9", "--model", "--require-model-context", "--observer",
                     f"--seed={seed}", f"--broker-url=http://127.0.0.1:{server.server_port}",
                     "--expected-actor=" + expected["actor_sha256"],
                     "--expected-registration=" + expected["registration_sha256"]]
            if smoke_seconds:
                base += ["--headless"]
                flags += ["--smoke-test", f"--smoke-duration={smoke_seconds}"]
            try:
                with subprocess.Popen(base + flags, cwd=ROOT,
                                      stdout=subprocess.PIPE if smoke_seconds else subprocess.DEVNULL,
                                      stderr=subprocess.STDOUT if smoke_seconds else log,
                                      text=True, encoding="utf-8", errors="replace") as game:
                    print(f"MODEL_GAME pid={game.pid} model={MODEL}", flush=True)
                    output, _ = game.communicate()
                    print(f"MODEL_GAME_EXIT pid={game.pid} code={game.returncode}", flush=True)
                    if game.returncode:
                        raise RuntimeError(f"Godot exited with code {game.returncode}: {output or game_log}")
                    if smoke_seconds:
                        print(output, end="", flush=True)
                        marker = "SMOKE_SESSION_LOG "
                        log_path = next(line[len(marker):] for line in output.splitlines()
                                        if line.startswith(marker))
                        records = [json.loads(line) for line in Path(log_path).read_text(encoding="utf-8").splitlines()]
                        decisions = [row for row in records if row.get("source") == "model"]
                        if (len(decisions) < 3 or any(row.get("event") == "model_error" for row in records)
                                or any(row["model_decision"]["actor_sha256"] != expected["actor_sha256"]
                                       or row["model_decision"]["registration_sha256"] != expected["registration_sha256"]
                                       or row["world_version"] != 7 or row["tick_after"] <= row["tick_before"]
                                       for row in decisions)):
                            raise RuntimeError("The game did not sustain decisions from the specified model")
                        print(f"MODEL_SMOKE_PASS decisions={len(decisions)} last_tick={decisions[-1]['tick_after']}")
            finally:
                server.shutdown()
                thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seed", type=int, default=2304)
    parser.add_argument("--smoke-seconds", type=int, default=0)
    parser.add_argument("--password-stdin", action="store_true")
    arguments = parser.parse_args()
    if arguments.worker:
        run_worker()
    else:
        try:
            secret = (sys.stdin.readline().rstrip("\r\n")
                      if arguments.password_stdin or not sys.stdin.isatty() else
                      getpass.getpass(f"GPU researcher@{HOST} password (blank uses SSH key): "))
            launch_game(arguments.seed, arguments.smoke_seconds, secret or None)
        except (OSError, RuntimeError, ValueError, EOFError) as error:
            parser.exit(1, f"{error}\n")
