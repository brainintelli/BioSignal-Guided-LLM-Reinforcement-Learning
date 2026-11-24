"""Native M9 research transport; unchanged lifetime and owned process cleanup."""
import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading

from agent.rl_env import GodotEnv as BaseEnv


class GodotEnv(BaseEnv):
    def __init__(self, executable, project, timeout=45, *, map_mode="finite", navigation_steps=1):
        self.map_mode, self.navigation_steps = map_mode, navigation_steps
        self.timeout, self.counter = timeout, 0
        self.lines, self.diagnostics = queue.Queue(), []
        self._log_root = Path(project).resolve() / "output/runtime_logs"
        self._log_root.mkdir(parents=True, exist_ok=True)
        self._logs = tempfile.TemporaryDirectory(prefix="m8s3-godot-", dir=self._log_root)
        self.process = subprocess.Popen(
            [executable, "--headless", "--path", str(Path(project).resolve()),
             "--script", "godot_tests/m9_environment.gd", "--log-file", str(Path(self._logs.name) / "godot.log")],
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

    def reset(self, seed, profile="m9"):
        return self.call("reset", seed=seed, profile=profile, map_mode=self.map_mode, navigation_steps=self.navigation_steps)

    def close(self):
        try:
            super().close()
        finally:
            if not Path(self._logs.name).resolve().is_relative_to(self._log_root):
                raise RuntimeError("Cleanup escaped owned log directory")
            self._logs.cleanup()
