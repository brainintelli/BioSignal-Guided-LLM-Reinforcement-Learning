"""Persistent JSON-lines client for the real headless Godot environment."""
from __future__ import annotations
import json
import os
from pathlib import Path
import queue
import subprocess
import threading


class GodotEnv:
    def __init__(self, executable: str, project: str, timeout: float = 45):
        self.timeout = timeout
        self.counter = 0
        self._last_step_result = None
        self.lines = queue.Queue()
        self.diagnostics = []
        self.process = subprocess.Popen([executable, "--headless", "--path", str(Path(project).resolve()),
                                         "--script", "godot_tests/rl_environment.gd"],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, encoding="utf-8", bufsize=1,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                                        env={**os.environ, "GODOT_SILENCE_ROOT_WARNING": "1"})
        def read():
            for line in self.process.stdout:
                if line.startswith("MM_RL_JSON="):
                    self.lines.put(json.loads(line[len("MM_RL_JSON="):]))
                else:
                    self.diagnostics.append(line.rstrip())
            self.lines.put(None)
        threading.Thread(target=read, daemon=True).start()

    def call(self, op: str, **kwargs) -> dict:
        self.counter += 1
        requested_op = op
        wire_op = "observe" if op == "landmark" else op
        request = dict(id=self.counter, op=wire_op, **kwargs)
        self.process.stdin.write(json.dumps(request, ensure_ascii=True, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        try:
            result = self.lines.get(timeout=self.timeout)
        except queue.Empty as exc:
            raise TimeoutError("Godot protocol timed out: " + repr(self.diagnostics[-8:])) from exc
        if result is None:
            raise RuntimeError("Godot exited: " + repr(self.diagnostics[-8:]))
        if result.get("id") != self.counter or not result.get("ok"):
            raise RuntimeError("Godot protocol failure: " + repr(result))
        if requested_op == "step":
            self._last_step_result = result
        elif requested_op == "landmark":
            result = dict(result)
            trace = ((self._last_step_result or {}).get("transition") or {}).get("trace") or []
            candidates = [entry for entry in trace if int(entry.get("tick", -1)) <= 60]
            if candidates:
                entry = candidates[-1]
                player = entry.get("player", {})
                current = result.get("observation") or {}
                landmark = dict(current)
                landmark["player"] = dict(current.get("player", {}))
                landmark["tick"] = int(entry["tick"])
                landmark["world_tick"] = int(entry["tick"])
                landmark["visited_count"] = int(entry["visited_count"])
                landmark["player"].update({
                    "health": entry["health"],
                    "hunger": entry["hunger"],
                    "thirst": entry["thirst"],
                    "x": player.get("x"),
                    "y": player.get("y"),
                })
                if "inventory" in entry:
                    landmark["inventory"] = entry["inventory"]
                if "camp" in entry:
                    landmark["camp"] = entry["camp"]
                result["landmark"] = landmark
            else:
                result["landmark"] = result.get("observation")
        return result

    def reset(self, seed: int, profile: str = "open1") -> dict:
        return self.call("reset", seed=seed, profile=profile)

    def step(self, action: dict) -> dict:
        return self.call("step", action=action)

    def close(self):
        if self.process.poll() is None:
            try:
                self.call("close")
                self.process.wait(timeout=5)
            except (OSError, RuntimeError, TimeoutError, subprocess.TimeoutExpired):
                self.process.terminate()
                self.process.wait(timeout=5)
        for handle in (self.process.stdin, self.process.stdout):
            if handle:
                handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
