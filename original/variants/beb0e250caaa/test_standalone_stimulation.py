"""Exercise the exported program without importing the desktop application."""

import importlib
import json
from pathlib import Path
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import socket
import signal
from dataclasses import replace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


class FakeMaxLab(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self):
        self.commands = []
        self.fail_prefix = None
        self.empty_route = False
        self.same_unit = False
        self.lsb = "2.9"
        self.transport_behavior = None
        self.drop_response_prefix = None
        super().__init__(("127.0.0.1", 0), FakeHandler)
        self.worker = threading.Thread(target=self.serve_forever, daemon=True)
        self.worker.start()

    def reply(self, command):
        self.commands.append(command)
        if self.fail_prefix and command.startswith(self.fail_prefix):
            return "Error: injected failure"
        if command == "system_query_dac_lsb_mv":
            return self.lsb
        if command.startswith("sequence_new"):
            return "test-sequence"
        if "mea_array_query_stimulation_at_electrode" in command:
            if self.empty_route:
                return ""
            return "3" if self.same_unit or command.endswith("3580") else "4"
        if "mea_array_get_config" in command:
            return "test routing config"
        return "Ok"

    def close(self):
        self.shutdown()
        self.server_close()
        self.worker.join()


class FakeHandler(socketserver.BaseRequestHandler):
    def handle(self):
        data = b""
        while True:
            chunk = self.request.recv(4096)
            if not chunk:
                return
            data += chunk
            while b"\r\n###" in data:
                command, data = data.split(b"\r\n###", 1)
                if self.server.transport_behavior == "disconnect":
                    return
                if self.server.transport_behavior == "timeout":
                    time.sleep(0.3)
                    return
                response = self.server.reply(command.decode("latin1"))
                if self.server.drop_response_prefix and command.decode("latin1").startswith(self.server.drop_response_prefix):
                    return
                reply = response.encode("latin1") + b"\r\n###"
                if self.server.transport_behavior == "fragment":
                    for byte in reply:
                        self.request.sendall(bytes([byte]))
                else:
                    self.request.sendall(reply)


class StandaloneExportTest(unittest.TestCase):
    def test_bundle_runs_outside_repository_without_site_packages(self):
        with tempfile.TemporaryDirectory() as tmp:
            build = subprocess.run(
                [sys.executable, str(ROOT / "scripts/release/package_stimulation.py"),
                 "--output-dir", tmp], capture_output=True, text=True,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            bundle = Path(tmp) / "BrainStimulation"
            result = subprocess.run(
                [sys.executable, "-I", "-S", str(bundle / "run_stimulation.py"),
                 str(bundle / "example.json"), "--dry-run"],
                cwd=tmp, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["amplitude_schedule_mv"], [100, 100, 100])
            self.assertAlmostEqual(plan["stimulation_seconds"], 2.0504)
            self.assertTrue((Path(tmp) / "BrainStimulation-ubuntu22.04-python.tar.gz").is_file())


class StandaloneRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.export = tempfile.TemporaryDirectory()
        subprocess.run([sys.executable, str(ROOT / "scripts/release/package_stimulation.py"),
                        "--output-dir", cls.export.name], check=True, capture_output=True)
        cls.bundle = Path(cls.export.name) / "BrainStimulation"
        sys.path.insert(0, str(cls.bundle))
        cls.config = importlib.import_module("brain_stimulation.config")
        cls.runner = importlib.import_module("brain_stimulation.runner")
        cls.transport = importlib.import_module("brain_stimulation.transport")

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(cls.bundle))
        for name in list(sys.modules):
            if name == "brain_stimulation" or name.startswith("brain_stimulation."):
                del sys.modules[name]
        cls.export.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name)
        self.server = FakeMaxLab()
        self.addCleanup(self.server.close)
        self.endpoint = self.transport.Endpoint("127.0.0.1", self.server.server_address[1], 1)
        self.params = self.config.load_config(self.bundle / "example.json")

    def run_experiment(self, params=None, wait=lambda seconds: None):
        result = self.runner.run_experiment(params or self.params, self.endpoint,
                                             self.output, wait=wait)
        reports = list(self.output.glob("*/run.json"))
        self.assertEqual(len(reports), 1)
        return result, json.loads(reports[0].read_text())

    def test_check_is_readonly_and_uses_explicit_port_even_with_base_port_env(self):
        with patch.dict("os.environ", {"MXW_BASE_PORT": "9876"}):
            result = self.runner.check_connection(self.params, self.endpoint)
        self.assertEqual(result["dac_lsb_mv"], 2.9)
        self.assertEqual(self.server.commands, ["system_query_dac_lsb_mv"])

    def test_recording_and_biphasic_sequence_match_preview(self):
        waits = []
        code, report = self.run_experiment(wait=waits.append)
        self.assertEqual(code, 0, report)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["plan"]["amplitude_schedule_mv"], [100, 100, 100])
        self.assertEqual(waits, [5, 6.0, 2.0504, 6.0])
        commands = self.server.commands
        self.assertTrue(any(c.startswith("saving_group_define 0 routed 0,1,2") for c in commands))
        appended = [c.split("\n", 1)[1] for c in commands if c.startswith("sequence_append")]
        dac = [c for c in appended if c.startswith("mea_set_dac")]
        self.assertEqual(dac[:3], ["mea_set_dac 0 478 512", "mea_set_dac 0 546 512", "mea_set_dac 0 512 512"])
        self.assertEqual(len(dac), 90)
        samples = sum(int(c.rsplit(" ", 1)[1]) for c in appended if c.startswith("system_delay_samples"))
        self.assertAlmostEqual(samples * 0.00005, report["plan"]["stimulation_seconds"])
        self.assertIn("saving_use_old_file_format 0", commands)
        self.assertLess(commands.index("saving_start_recording"), commands.index("sequence_send test-sequence"))
        self.assertIn("saving_stop_file", commands)
        self.assertIn("sequence_delete test-sequence", commands)

    def test_spikes_only_never_defines_raw_group(self):
        code, report = self.run_experiment(replace(self.params, record_only_spikes=True))
        self.assertEqual(code, 0, report)
        self.assertFalse(any(c.startswith("saving_group_define") for c in self.server.commands))

    def test_server_error_after_partial_start_still_closes_recording(self):
        self.server.fail_prefix = "saving_start_recording"
        code, report = self.run_experiment()
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "failed")
        self.assertIn("saving_stop_recording", self.server.commands)
        self.assertIn("saving_stop_file", self.server.commands)
        self.assertFalse(any(c.startswith("sequence_send") for c in self.server.commands))

    def test_initial_sequence_upload_failure_still_deletes_sequence(self):
        self.server.fail_prefix = "sequence_append"
        code, report = self.run_experiment()
        self.assertEqual(code, 1)
        self.assertIn("sequence_delete test-sequence", self.server.commands)

    def test_creation_response_loss_still_cleans_named_resources(self):
        for create, delete in (("mea_array_new", "mea_array_delete"), ("sequence_new", "sequence_delete")):
            with self.subTest(create=create), tempfile.TemporaryDirectory() as output:
                self.server.commands.clear()
                self.server.drop_response_prefix = create
                code = self.runner.run_experiment(self.params, self.endpoint, Path(output), wait=lambda _: None)
                self.assertEqual(code, 1)
                request = next(c for c in self.server.commands if c.startswith(create))
                self.assertEqual(len(request.split()), 2, "resource must have a client-generated token")
                self.assertIn(delete + " " + request.split()[1], self.server.commands)
                report = json.loads(next(Path(output).glob("*/run.json")).read_text())
                self.assertEqual(report["cleanup_errors"], [])

    def test_interrupt_during_stimulation_powers_down_and_closes_file(self):
        def interrupt(seconds):
            if seconds == self.params.stimulation_seconds():
                raise KeyboardInterrupt()
        code, report = self.run_experiment(wait=interrupt)
        self.assertEqual(code, 130)
        self.assertEqual(report["status"], "interrupted")
        self.assertIn("saving_stop_recording", self.server.commands)
        self.assertIn("saving_stop_file", self.server.commands)
        for unit in range(32):
            self.assertIn(f"mea_set_stimulation_unit {unit} 0 0 0 0 0 0", self.server.commands)

    def test_cleanup_failure_is_not_reported_as_completed(self):
        self.server.fail_prefix = "saving_stop_file"
        code, report = self.run_experiment()
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["cleanup_errors"])

    def test_invalid_dac_amplitude_fails_before_initializing_hardware(self):
        self.server.lsb = "0.1"
        code, report = self.run_experiment()
        self.assertEqual(code, 1)
        self.assertEqual(self.server.commands, ["system_query_dac_lsb_mv"])

    def test_nonzero_amplitude_quantizing_to_zero_is_not_api_ready(self):
        self.server.lsb = "2.981901"
        for control in (False, True):
            with self.subTest(control=control):
                self.server.commands.clear()
                params = replace(self.params, amplitude_range=(0.07006989305, 0.07006989305, 1),
                                 control=control)
                with self.assertRaisesRegex(ValueError, "quantizes to zero"):
                    self.runner.check_connection(params, self.endpoint)
                self.assertEqual(self.server.commands, ["system_query_dac_lsb_mv"])

    def test_nonzero_amplitude_quantizing_to_zero_fails_before_hardware_writes(self):
        self.server.lsb = "2.981901"
        params = replace(self.params, amplitude_range=(0.07006989305, 0.07006989305, 1))
        code, report = self.run_experiment(params)
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "failed")
        self.assertIn("quantizes to zero", report["error"])
        self.assertEqual(self.server.commands, ["system_query_dac_lsb_mv"])

    def test_quantization_keeps_explicit_zero_and_one_step_distinct(self):
        self.server.lsb = "2.981901"
        for amplitude in (0, 2.981901):
            with self.subTest(amplitude=amplitude), tempfile.TemporaryDirectory() as output:
                self.server.commands.clear()
                params = replace(self.params, amplitude_range=(amplitude, amplitude, 1))
                code = self.runner.run_experiment(params, self.endpoint, Path(output), wait=lambda _: None)
                self.assertEqual(code, 0)
                report = json.loads(next(Path(output).glob("*/run.json")).read_text())
                self.assertEqual(report["actual_amplitude_schedule_mv"], [amplitude] * 3)
                appended = [c.split("\n", 1)[1] for c in self.server.commands if c.startswith("sequence_append")]
                dac = [c for c in appended if c.startswith("mea_set_dac")]
                bits = 0 if amplitude == 0 else 1
                self.assertEqual(dac[:3], [f"mea_set_dac 0 {512-bits} 512",
                                          f"mea_set_dac 0 {512+bits} 512", "mea_set_dac 0 512 512"])

    def test_duplicate_or_unroutable_stim_units_prevent_sending(self):
        for field in ("same_unit", "empty_route"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as output:
                setattr(self.server, field, True)
                code = self.runner.run_experiment(self.params, self.endpoint, Path(output), wait=lambda _: None)
                self.assertEqual(code, 1)
                self.assertFalse(any(c.startswith("sequence_send") for c in self.server.commands))
                setattr(self.server, field, False)

    def test_invalid_config_rejected_without_default_substitution(self):
        base = json.loads((self.bundle / "example.json").read_text())
        for key, value in (("pulses", 0), ("pulses", "ten"), ("ipi", float("nan")),
                           ("phase", -1), ("control", "false"),
                           ("amplitude_range", [100, 200, 0]),
                           ("amplitude_range", [1e20, 1e20, 1])):
            with self.subTest(key=key, value=value):
                path = self.output / "invalid.json"
                path.write_text(json.dumps(dict(base, **{key: value})))
                with self.assertRaises(ValueError):
                    self.config.load_config(path)

    def test_cloud_params_wrapper_and_config_relative_assay_path(self):
        source = json.loads((self.bundle / "example.json").read_text())
        source["assay_path"] = "routing.cfg"
        (self.output / "routing.cfg").write_text("test cfg")
        path = self.output / "wrapped.json"
        path.write_text(json.dumps({"command": 12, "experiment_category": 4, "params": source}))
        params = self.config.load_config(path)
        self.assertEqual(params.assay_path, str(self.output / "routing.cfg"))
        self.assertEqual(params.stimulation_seconds(), self.params.stimulation_seconds())

    def test_remote_recording_path_required_before_api_calls(self):
        remote = self.transport.Endpoint("192.0.2.1", 7215, 0.1)
        with self.assertRaisesRegex(ValueError, "recording-dir"):
            self.runner.run_experiment(self.params, remote, self.output)

    def test_fragmented_responses_and_transport_failures(self):
        self.server.transport_behavior = "fragment"
        self.assertEqual(self.runner.check_connection(self.params, self.endpoint)["status"], "api_ready")
        for behavior in ("disconnect", "timeout"):
            self.server.transport_behavior = behavior
            with self.subTest(behavior=behavior), self.assertRaises(OSError):
                self.runner.check_connection(self.params, replace(self.endpoint, timeout=0.1))

    def test_failed_connection_does_not_swallow_oserror(self):
        with socket.socket() as unused:
            unused.bind(("127.0.0.1", 0))
            endpoint = replace(self.endpoint, port=unused.getsockname()[1])
            with self.assertRaises(OSError):
                self.runner.check_connection(self.params, endpoint)

    def test_random_order_is_generated_once_and_saved_exactly(self):
        params = replace(self.params, randomize=True, control=True, amplitude_range=(50, 100, 50))
        with patch("brain_stimulation.plan.random.shuffle", side_effect=lambda xs: xs.reverse()) as shuffle:
            code, report = self.run_experiment(params)
        self.assertEqual(code, 0)
        self.assertEqual(shuffle.call_count, 1)
        self.assertEqual(report["plan"]["amplitude_schedule_mv"], [100] * 3 + [50] * 3 + [0] * 3)

    @unittest.skipIf(sys.platform == "win32", "POSIX SIGTERM behavior")
    def test_cli_sigterm_closes_recording_and_returns_130(self):
        config = json.loads((self.bundle / "example.json").read_text())
        config["pre_stim_period"] = 1
        path = self.output / "long.json"
        path.write_text(json.dumps(config))
        process = subprocess.Popen([
            sys.executable, "-I", "-S", str(self.bundle / "run_stimulation.py"), str(path),
            "--run", "--port", str(self.endpoint.port), "--output-dir", str(self.output),
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 15
            while "saving_start_recording" not in self.server.commands and process.poll() is None:
                if time.monotonic() > deadline:
                    self.fail("CLI did not reach recording")
                time.sleep(0.02)
            process.send_signal(signal.SIGTERM)
            _, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 130, stderr)
            self.assertIn("saving_stop_recording", self.server.commands)
            self.assertIn("saving_stop_file", self.server.commands)
            self.assertFalse(any(c.startswith("sequence_send") for c in self.server.commands))
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()


if __name__ == "__main__":
    unittest.main()
