import json
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.files import read_json, sha256

REPO_ROOT = Path(__file__).resolve().parents[1]


class SmokeCliTests(unittest.TestCase):
    def run_smoke(self, output_root):
        process = subprocess.run(
            [sys.executable, "-m", "src.reproduce", "smoke", "--output-dir", str(output_root)],
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
        self.assertEqual(process.returncode, 0, process.stderr or process.stdout)
        report = json.loads(process.stdout)
        self.assertEqual(report["command"], "smoke")
        run_directory = Path(report["run_directory"])
        self.assertEqual(run_directory.parent, output_root)
        self.assertTrue(run_directory.is_dir())
        return run_directory, report["summary"]

    def check_run(self, run_directory, summary, input_hashes, config_hash, expected):
        manifest = read_json(run_directory / "run_manifest.json")
        self.assertEqual(manifest["command"], "smoke")
        self.assertEqual(manifest["input_sha256"], input_hashes)
        self.assertEqual(manifest["config_sha256"], config_hash)
        self.assertEqual(set(manifest["outputs"]), {"result.json", "study.yaml"})
        for name, digest in manifest["outputs"].items():
            with self.subTest(output=name):
                self.assertEqual(sha256(run_directory / name), digest)
        self.assertEqual(manifest["outputs"]["study.yaml"], config_hash)

        result = read_json(run_directory / "result.json")
        self.assertEqual(summary, result)
        for field in ("pairs", "wins", "ties", "losses"):
            with self.subTest(field=field):
                self.assertIs(type(result[field]), int)
                self.assertEqual(result[field], expected[field])
        actual_mean = result["mean_difference"]
        expected_mean = expected["mean_difference"]
        self.assertIn(type(actual_mean), (int, float))
        self.assertTrue(
            math.isclose(actual_mean, expected_mean, rel_tol=1e-11, abs_tol=1e-11),
            f"mean_difference: measured {actual_mean!r}, published {expected_mean!r}",
        )
        return manifest["outputs"]

    def test_smoke_verifies_inputs_and_creates_two_immutable_runs(self):
        pins_path = REPO_ROOT / "data" / "input_hashes.json"
        config_path = REPO_ROOT / "configs" / "study.yaml"
        reference_path = REPO_ROOT / "results" / "exploration.json"
        pins = read_json(pins_path)
        expected = read_json(reference_path)["d6"]["intact"]
        release_files = {REPO_ROOT / name for name in pins}
        release_files.update((pins_path, config_path, reference_path))
        release_before = {path: sha256(path) for path in release_files}
        input_hashes = {name: release_before[REPO_ROOT / name] for name in pins}
        self.assertEqual(input_hashes, pins)
        config_hash = release_before[config_path]

        with TemporaryDirectory() as directory:
            output_root = Path(directory).resolve() / "runs"
            first, first_summary = self.run_smoke(output_root)
            first_outputs = self.check_run(
                first, first_summary, input_hashes, config_hash, expected
            )
            first_files = {path.name: path.read_bytes() for path in first.iterdir()}
            self.assertEqual(set(first_files), set(first_outputs) | {"run_manifest.json"})

            second, second_summary = self.run_smoke(output_root)
            self.assertNotEqual(first, second)
            self.assertEqual(set(output_root.iterdir()), {first, second})
            self.assertEqual(
                {path.name: path.read_bytes() for path in first.iterdir()}, first_files
            )
            second_outputs = self.check_run(
                second, second_summary, input_hashes, config_hash, expected
            )
            self.assertEqual(second_outputs, first_outputs)
            self.assertEqual({path.name for path in second.iterdir()}, set(first_files))

        self.assertEqual({path: sha256(path) for path in release_files}, release_before)


if __name__ == "__main__":
    unittest.main()
