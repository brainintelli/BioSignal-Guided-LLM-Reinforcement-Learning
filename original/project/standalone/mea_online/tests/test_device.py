"""Local-only device broker and native fake SDK behavior tests."""
import datetime
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from test_adapter import binding, digest, packet


class DeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.device = importlib.import_module("mea_online.device")
        except ImportError:
            cls.device = None

    def setUp(self):
        self.assertIsNotNone(self.device, "device broker implementation is missing")

    def test_original_deadline_rollback_and_finish_reserve(self):
        wall, mono = [10000.0], [20.0]
        start = datetime.datetime.fromtimestamp(10000-7000, datetime.timezone.utc).isoformat()
        clock = self.device.DeviceDeadline(start, wall=lambda: wall[0], monotonic=lambda: mono[0])
        self.assertEqual(clock.remaining(), 80)
        wall[0] -= 5000
        mono[0] += 81
        with self.assertRaises(TimeoutError):
            clock.check()

    def test_monotonic_regression_is_a_clock_fault(self):
        mono = [20.0]
        start = datetime.datetime.fromtimestamp(10000, datetime.timezone.utc).isoformat()
        clock = self.device.DeviceDeadline(start, wall=lambda:10000, monotonic=lambda:mono[0])
        mono[0] = 19.0
        with self.assertRaises(TimeoutError):
            clock.check()

    def test_registry_identifies_original_organism_not_changed_session(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.device.DeviceLease(directory, binding())
            changed = binding()
            changed["session_id"] = "another-session"
            with self.assertRaises((ValueError, FileExistsError)):
                self.device.DeviceLease(directory, changed)
            first.close()
            with self.assertRaises((ValueError, FileExistsError)):
                self.device.DeviceLease(directory, changed)

    def test_subprocess_timeout_and_failure_preserve_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.device.SubprocessSource([sys.executable, "-c", "import time; time.sleep(10)"],
                binding(), directory, live=False, timeout=.05)
            with self.assertRaises((TimeoutError, ValueError)):
                source.capture("req")
            source.close()
            self.assertTrue(list(Path(directory).glob("*.stderr")))
            with self.assertRaises((RuntimeError, ValueError)):
                source.capture("another")

    def test_simulation_binary_cannot_pass_live_preflight(self):
        code = "import json; print(json.dumps(dict(schema='mea.online.producer.v1',source='simulation',filtered_frame_info=True,headers='stub',library='stub')))"
        with tempfile.TemporaryDirectory() as directory:
            source = self.device.SubprocessSource([sys.executable, "-c", code], binding(), directory, live=True)
            with self.assertRaises(ValueError):
                source.preflight()

    def test_request_validation_rejects_commands_identity_and_duplicates(self):
        request = dict(schema="mea.online.request.v1", request_id="request-1", binding_sha256=digest(binding()))
        self.assertEqual(self.device.validate_request(request, binding(), set()), "request-1")
        for change in ({"command": "stimulate"}, {"binding_sha256": "0"*64}, {"request_id": ""}):
            bad = dict(request, **change)
            with self.assertRaises(ValueError):
                self.device.validate_request(bad, binding(), set())
        with self.assertRaises(ValueError):
            self.device.validate_request(request, binding(), {"request-1"})

    def test_broker_socket_request_response_failure_closes_resources(self):
        device = self.device
        class SimulationSource:
            closed = False
            def preflight(self): return {"source": "simulation"}
            def capture(self, request): return packet(request)
            def close(self): self.closed = True
        source = SimulationSource()
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        broker = device.WindowBroker(source, binding(), device.DeviceDeadline(now), port=0)
        errors = []
        def serve():
            try: broker.serve()
            except ValueError as error: errors.append(error)
        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        self.addCleanup(broker.close)
        until = time.monotonic()+2
        while (broker._listener is None or not broker._listener.getsockname()[1]) and time.monotonic()<until:
            time.sleep(.01)
        with socket.create_connection(broker._listener.getsockname(), timeout=2) as connection:
            request = dict(schema="mea.online.request.v1", request_id="request-1", binding_sha256=digest(binding()))
            connection.sendall((json.dumps(request)+"\n").encode())
            result = json.loads(device.receive_line(connection, time.monotonic()+2))
            self.assertEqual(result["request_id"], "request-1")
            self.assertEqual(result["source"], "simulation")
            connection.sendall((json.dumps(request)+"\n").encode())
            self.assertEqual(connection.recv(1), b"")
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(source.closed)
        self.assertTrue(errors)

    def lifecycle(self, idle_seconds):
        """Real TCP; inject one receive timeout to advance idle time without waiting."""
        device = self.device
        adapter = importlib.import_module("mea_online.adapter")
        elapsed = [0.0]
        advanced = threading.Event()
        class IdleSocket:
            def __init__(self, connection): self.connection = connection
            def __getattr__(self, key): return getattr(self.connection, key)
            def recv(self, size):
                if not advanced.is_set():
                    elapsed[0] += idle_seconds
                    advanced.set()
                    raise socket.timeout("simulated long model/update idle interval")
                return self.connection.recv(size)
        class ProtocolFixture:
            """Synthetic protocol peer; no binary provenance or hardware claim."""
            closed = False
            calls = 0
            broker = None
            def preflight(self): return {"source": "protocol-test-fixture"}
            def capture(self, request):
                self.calls += 1
                if self.calls == 1:
                    self.broker._connection = IdleSocket(self.broker._connection)
                return packet(request, source="live", start=50000+40000*(self.calls-1))
            def close(self): self.closed = True
        source = ProtocolFixture()
        start = datetime.datetime.fromtimestamp(10000, datetime.timezone.utc).isoformat()
        deadline = device.DeviceDeadline(start, wall=lambda:10000+elapsed[0], monotonic=lambda:elapsed[0])
        broker = device.WindowBroker(source, binding(), deadline, port=0)
        source.broker = broker
        errors = []
        def serve():
            try: broker.serve()
            except Exception as error: errors.append(error)
        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        self.addCleanup(broker.close)
        until = time.monotonic()+2
        while (broker._listener is None or not broker._listener.getsockname()[1]) and time.monotonic()<until:
            time.sleep(.01)
        client = adapter.WindowClient("127.0.0.1", broker._listener.getsockname()[1], binding(), timeout=1)
        self.addCleanup(client.close)
        self.assertEqual(client.capture("before-model-load")["start_frame"], 50000)
        self.assertTrue(advanced.wait(2))
        return client, thread, source, errors

    def test_persistent_client_survives_long_idle_and_normal_close(self):
        client, thread, source, errors = self.lifecycle(180)
        self.assertEqual(client.capture("after-model-load")["start_frame"], 90000)
        client.close()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(source.calls, 2)
        self.assertTrue(source.closed)

    def test_idle_still_closes_at_original_deadline(self):
        client, thread, source, errors = self.lifecycle(7081)
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(source.calls, 1)
        self.assertTrue(source.closed)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], TimeoutError)
        self.assertIn("original biological deadline", str(errors[0]))

    def test_partial_request_eof_is_still_an_error(self):
        adapter = importlib.import_module("mea_online.adapter")
        left, right = socket.socketpair()
        with left, right:
            left.sendall(b'{"schema":')
            left.shutdown(socket.SHUT_WR)
            with self.assertRaisesRegex(ValueError, "truncated"):
                adapter.receive_line(right, time.monotonic()+1, allow_eof=True)

    def test_subprocess_oversize_stdout_and_stderr_are_bounded(self):
        for target, size in (("stdout", 9*1024*1024), ("stderr", 2*1024*1024)):
            with tempfile.TemporaryDirectory() as directory:
                code = f"import sys; sys.{target}.write('x'*{size})"
                source = self.device.SubprocessSource([sys.executable, "-c", code], binding(), directory, live=False)
                with self.subTest(target=target), self.assertRaises(ValueError):
                    source.preflight()
                self.assertLessEqual(sum(path.stat().st_size for path in Path(directory).iterdir()), 9*1024*1024)
                self.assertIsNone(source._process)

    def test_future_and_expired_session_fail_before_device_open(self):
        for start in (10001, 2800):
            value = datetime.datetime.fromtimestamp(start, datetime.timezone.utc).isoformat()
            with self.assertRaises((ValueError, TimeoutError)):
                self.device.DeviceDeadline(value, wall=lambda:10000, monotonic=lambda:0)


@unittest.skipUnless(os.environ.get("MEA_ONLINE_TEST_COLLECTOR"), "set MEA_ONLINE_TEST_COLLECTOR to compiled simulation collector")
class NativeCollectorTests(unittest.TestCase):
    def run_native(self, mode, extra=()):
        command = [os.environ["MEA_ONLINE_TEST_COLLECTOR"], "--capture", "native-request", digest(binding()), "sim-session", "0,2"] + list(extra)
        return subprocess.run(command, capture_output=True, text=True, timeout=5, env=dict(os.environ, MEA_STUB_MODE=mode))

    def test_contiguous_transport_late_peaks_and_quiet(self):
        for mode in ("normal", "quiet"):
            result = self.run_native(mode)
            self.assertEqual(result.returncode, 0, result.stderr)
            value = json.loads(result.stdout)
            self.assertEqual(value["source"], "simulation")
            self.assertEqual([value["start_frame"], value["end_frame"], value["frame_count"]], [21000, 41000, 20000])
            self.assertGreaterEqual(value["capture_elapsed_s"], .75)
            if mode == "quiet":
                self.assertEqual(value["events"], [])
            else:
                self.assertEqual(value["events"][0], [21000, 20990, 0, -2.5])

    def test_bad_native_streams_fail_without_success_json(self):
        for mode in ("gap", "reorder", "corrupt", "well", "event_well", "future_peak", "duplicate",
                     "unknown_channel", "open_error", "close_error", "sdk_error", "backlog", "timeout", "too_fast"):
            result = self.run_native(mode)
            with self.subTest(mode=mode):
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertTrue(result.stderr)

    def test_preflight_does_not_open_stream(self):
        result = subprocess.run([os.environ["MEA_ONLINE_TEST_COLLECTOR"], "--describe-json"],
            capture_output=True, text=True, timeout=3, env=dict(os.environ, MEA_STUB_MODE="open_error"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["source"], "simulation")

    def test_actual_native_output_passes_python_and_archives_raw_events(self):
        device = importlib.import_module("mea_online.device")
        with tempfile.TemporaryDirectory() as directory:
            source = device.SubprocessSource([os.environ["MEA_ONLINE_TEST_COLLECTOR"]], binding(), directory, live=False)
            result = source.capture("integration")
            self.assertEqual(result["events"], [[21000, 20990, 0, -2.5]])
            self.assertEqual(json.loads((Path(directory)/"integration.window.json").read_text())["window_sha256"], result["window_sha256"])
            self.assertEqual(json.loads((Path(directory)/"integration.stdout.json").read_text())["events"], result["events"])
            self.assertIn("sdk_close_status=0", (Path(directory)/"integration.stderr").read_text())
            source.close()

    def test_preflight_can_precede_live_launch_in_same_evidence_directory(self):
        device = importlib.import_module("mea_online.device")
        with tempfile.TemporaryDirectory() as directory:
            first = device.SubprocessSource([os.environ["MEA_ONLINE_TEST_COLLECTOR"]], binding(), directory, live=False)
            first.preflight()
            first.close()
            second = device.SubprocessSource([os.environ["MEA_ONLINE_TEST_COLLECTOR"]], binding(), directory, live=False)
            self.assertEqual(second.preflight()["source"], "simulation")
            second.close()

    def test_even_pinned_simulation_binary_fails_live_admission(self):
        device = importlib.import_module("mea_online.device")
        binary = os.environ["MEA_ONLINE_TEST_COLLECTOR"]
        with tempfile.TemporaryDirectory() as directory:
            source = device.SubprocessSource([binary], binding(), directory, live=True,
                provenance={"binary_sha256": device.file_digest(binary)})
            with self.assertRaises(ValueError):
                source.preflight()


if __name__ == "__main__":
    unittest.main()
