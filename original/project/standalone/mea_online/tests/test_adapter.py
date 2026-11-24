"""Protocol failures must never manufacture quiet or reusable windows."""
import copy
import hashlib
import importlib
import json
import socket
import threading
import time
import unittest


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def binding():
    return dict(session_id="sim-session", chip_id="sim-chip", organoid_id="sim-organoid",
                started_utc="2026-11-17T00:00:00Z", channel_map=[[0, 10], [2, 20]],
                sample_rate_hz=20000, window_frames=20000, well_id=0, amplitude_scale_uv=0.5)


def packet(request="request-1", source="simulation", start=50000):
    b = binding()
    value = dict(schema="mea.online.window.v1", source=source, request_id=request,
                 binding_sha256=digest(b), session_id=b["session_id"], start_frame=start,
                 end_frame=start+20000, frame_count=20000, complete=True, dropped=0,
                 corrupted=0, stream_opened_after_request=True, warmup_frames=20000, warmup_elapsed_s=1.0, capture_elapsed_s=1.0,
                 events=[[start+1, start-10, 0, -3.0], [start+3, start, 2, 4.0]])
    value["window_sha256"] = digest(value)
    return value


def reseal(value):
    value.pop("window_sha256", None)
    value["window_sha256"] = digest(value)
    return value


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.adapter = importlib.import_module("mea_online.adapter")
        except ImportError:
            cls.adapter = None

    def setUp(self):
        self.assertIsNotNone(self.adapter, "acquisition adapter implementation is missing")

    def test_late_peak_preserved_and_result_detached(self):
        source = packet()
        result = self.adapter.validate_window(source, binding(), "request-1", live=False)
        self.assertEqual(result["events"][0], [50001, 49990, 0, -3.0])
        source["events"][0][3] = 100
        self.assertEqual(result["events"][0][3], -3.0)

    def test_complete_silence_requires_positive_frame_evidence(self):
        quiet = packet()
        quiet["events"] = []
        self.assertEqual(self.adapter.validate_window(reseal(quiet), binding(), "request-1", live=False)["events"], [])
        for key, value in (("frame_count", 0), ("complete", False), ("stream_opened_after_request", False)):
            bad = copy.deepcopy(quiet)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.adapter.validate_window(reseal(bad), binding(), "request-1", live=False)

    def test_identity_source_coverage_age_and_types_fail_closed(self):
        changes = dict(session_id="other", request_id="old", binding_sha256="0"*64,
                       end_frame=70001, frame_count=True, dropped=True, corrupted=1,
                       complete=1, source="live", capture_elapsed_s=0.1, extra=3)
        for key, value in changes.items():
            bad = packet()
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.adapter.validate_window(reseal(bad), binding(), "request-1", live=False)
        with self.assertRaises(ValueError):
            self.adapter.validate_window(packet(), binding(), "request-1", live=True)
        with self.assertRaises(ValueError):
            self.adapter.validate_window(packet(), binding(), "request-1", previous_end=50001, live=False)

    def test_events_reject_unknown_duplicate_out_of_range_and_reordered(self):
        invalid = [ [[50000, 49999, 7, 1]], [[49999, 49998, 0, 1]],
                    [[50000, 50001, 0, 1]], [[50000, 49999, 0, 1], [50001, 49999, 0, 2]],
                    [[50001, 50000, 0, 1], [50000, 49999, 0, 2]], [[50000, -1, 0, 1]],
                    [[50000, 49999, True, 1]], [[50000, 49999, 0, True]] ]
        for events in invalid:
            bad = packet()
            bad["events"] = events
            with self.subTest(events=events), self.assertRaises(ValueError):
                self.adapter.validate_window(reseal(bad), binding(), "request-1", live=False)

    def test_strict_json_hash_and_binding_validation(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}', '[1,2]'):
            with self.assertRaises(ValueError):
                self.adapter.strict_json(raw)
        bad = packet()
        bad["events"] = []
        with self.assertRaises(ValueError):
            self.adapter.validate_window(bad, binding(), "request-1", live=False)
        for key, value in (("chip_id", ""), ("chip_id", "placeholder"), ("amplitude_scale_uv", 0),
                           ("channel_map", [[0, 26400]]), ("channel_map", [[2, 20], [0, 10]]),
                           ("started_utc", "tomorrow"), ("well_id", True), ("sample_rate_hz", 19999)):
            bad_binding = binding()
            bad_binding[key] = value
            with self.assertRaises(ValueError):
                self.adapter.binding_digest(bad_binding)
        self.assertEqual(self.adapter.binding_digest(binding()), digest(binding()))

    def test_warmup_coverage_and_real_time_evidence_required(self):
        for key, value in (("warmup_frames", 19999), ("warmup_frames", True), ("warmup_elapsed_s", .1),
                           ("warmup_elapsed_s", 2.6), ("capture_elapsed_s", 2.6)):
            bad = packet()
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.adapter.validate_window(reseal(bad), binding(), "request-1", live=False)

    def serve(self, callback):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        self.addCleanup(listener.close)
        errors = []
        def worker():
            try:
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(2)
                    callback(connection)
            except (OSError, ValueError) as exc:
                errors.append(exc)
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        self.addCleanup(lambda: thread.join(2))
        return listener.getsockname()[1]

    def test_client_lazy_single_request_nonce_binding_and_reuse_block(self):
        observed = []
        def source(connection):
            raw = bytearray()
            while not raw.endswith(b"\n"):
                raw.extend(connection.recv(4096))
            observed.append(json.loads(raw))
            connection.sendall((json.dumps(packet(source="live"))+"\n").encode())
        port = self.serve(source)
        client = self.adapter.WindowClient("127.0.0.1", port, binding())
        self.addCleanup(client.close)
        self.assertEqual(client.capture("request-1")["start_frame"], 50000)
        self.assertEqual(observed, [dict(schema="mea.online.request.v1", request_id="request-1", binding_sha256=digest(binding()))])
        with self.assertRaises(ValueError):
            client.capture("request-1")
        client.close()
        client.close()

    def test_client_timeout_or_unsolicited_tail_poison_connection(self):
        for mode in ("timeout", "tail", "oversize", "bad_nonce"):
            def source(connection):
                connection.recv(4096)
                if mode == "timeout":
                    time.sleep(.15)
                elif mode == "tail":
                    connection.sendall((json.dumps(packet(source="live"))+"\n{}\n").encode())
                elif mode == "oversize":
                    connection.sendall(b"x" * (8*1024*1024+1))
                else:
                    connection.sendall((json.dumps(packet("wrong", "live"))+"\n").encode())
            port = self.serve(source)
            client = self.adapter.WindowClient("127.0.0.1", port, binding(), timeout=.08)
            self.addCleanup(client.close)
            with self.subTest(mode=mode), self.assertRaises((ValueError, OSError, TimeoutError)):
                client.capture("request-1")
            with self.assertRaises((ValueError, RuntimeError)):
                client.capture("request-2")


if __name__ == "__main__":
    unittest.main()
