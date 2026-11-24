"""Real persistent Godot RPC checks; no Python substitute for world transitions."""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
GODOT = Path(os.environ.get("MEADOW_GODOT", ROOT / ".tools/godot/Godot_v4.7.2-stable_win64_console.exe"))


@unittest.skipUnless(GODOT.is_file(), "Set MEADOW_GODOT to run actual engine bridge tests")
class PersistentBridgeTests(unittest.TestCase):
    def setUp(self):
        self.process = subprocess.Popen(
            [str(GODOT), "--headless", "--path", str(ROOT), "--script", "godot_tests/rl_environment.gd"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.lines = queue.Queue()
        self.raw_lines = []

        def read_lines():
            for line in self.process.stdout:
                self.raw_lines.append(line)
                if line.startswith("MM_RL_JSON="):
                    self.lines.put(json.loads(line[len("MM_RL_JSON="):]))
            self.lines.put(None)

        self.reader = threading.Thread(target=read_lines, daemon=True)
        self.reader.start()

    def request(self, payload):
        self.process.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self.process.stdin.flush()
        try:
            reply = self.lines.get(timeout=15)
        except queue.Empty:
            self.fail("Persistent RPC timed out: " + "".join(self.raw_lines))
        self.assertIsNotNone(reply, "Engine exited without replying: " + "".join(self.raw_lines))
        return reply

    def tearDown(self):
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait(timeout=10)
        self.reader.join(timeout=2)
        self.process.stdin.close()
        self.process.stdout.close()
        self.assertNotIn("SCRIPT ERROR", "".join(self.raw_lines))

    def test_real_world_persists_and_resets(self):
        reset = self.request({"id": 1, "op": "reset", "seed": 2305, "profile": "open1"})
        self.assertTrue(reset["ok"])
        self.assertEqual(reset["observation"]["tick"], 0)
        self.assertEqual(reset["observation"]["profile"], "open1")
        self.assertNotIn("task", reset["observation"])
        original_pid = self.process.pid
        bad = self.request({"id": 2, "op": "step", "action": {"verb": "teleport"}})
        self.assertFalse(bad["ok"])
        rest = self.request({"id": 3, "op": "step", "action": {"verb": "rest"}})
        self.assertEqual(rest["transition"]["elapsed_ticks"], 3)
        self.assertEqual([r["tick"] for r in rest["transition"]["trace"]], [1, 2, 3])
        self.assertAlmostEqual(rest["transition"]["after"]["thirst"], 78 - 3 * .45)
        self.assertEqual(rest["id"], 3)
        again = self.request({"id": 4, "op": "reset", "seed": 2305, "profile": "open1"})
        self.assertEqual(reset["observation"], again["observation"])
        self.assertEqual(original_pid, self.process.pid)
        self.assertTrue(self.request({"id": 5, "op": "close"})["closed"])
        self.process.wait(timeout=10)
        self.assertEqual(self.process.returncode, 0)

    def test_multiple_framed_requests_and_eof(self):
        self.process.stdin.write('{"id":1,"op":"reset"}\n{"id":2,"op":"observe"}\n')
        self.process.stdin.flush()
        a = self.lines.get(timeout=15)
        b = self.lines.get(timeout=15)
        self.assertEqual((a["id"], b["id"]), (1, 2))
        self.assertEqual(a["observation"], b["observation"])
        self.process.stdin.close()
        self.process.wait(timeout=10)
        self.assertEqual(self.process.returncode, 0)


if __name__ == "__main__":
    unittest.main()
