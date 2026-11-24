"""Request-scoped acquisition broker. Import/preflight never opens MaxLab.

Only the explicit ``live`` CLI mode listens, and only a validated request starts
the pinned producer. The broker exposes no SDK control or stimulation command.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import uuid

from .adapter import (MAX_LINE, binding_digest, canonical, digest, finite, identifier,
                      receive_line, strict_json, utc_timestamp, validate_binding, validate_window)


class DeviceDeadline:
    """Original biological UTC start plus monotonic elapsed; 120s cleanup reserve."""
    def __init__(self, started_utc, wall=time.time, monotonic=time.monotonic):
        self.wall, self.monotonic = wall, monotonic
        self.started = utc_timestamp(started_utc)
        self.initial_wall, self.initial_mono = wall(), monotonic()
        self.previous_mono = self.initial_mono
        if self.started > self.initial_wall:
            raise ValueError("biological start cannot be in future")
        self.high_water = self.initial_wall
        self.check()

    def remaining(self):
        now = self.monotonic()
        if now < self.previous_mono:
            raise TimeoutError("monotonic clock regressed")
        self.previous_mono = now
        self.high_water = max(self.high_water, self.wall(), self.initial_wall + now-self.initial_mono)
        return self.started+7200-120-self.high_water

    def check(self, required=0):
        if self.remaining() <= required:
            raise TimeoutError("original biological deadline / finish reserve")


class DeviceLease:
    """Durable, fail-closed claim; completed/crashed sessions cannot reset budget."""
    def __init__(self, directory, binding):
        binding = validate_binding(binding)
        key = digest(dict(organoid_id=binding["organoid_id"], started_epoch=utc_timestamp(binding["started_utc"])))
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory/(key+".device-lease.json")
        self.value = dict(schema="mea.online.device-lease.v1", state="claimed", binding_sha256=binding_digest(binding),
                          session_id=binding["session_id"], pid=os.getpid(), original_started_utc=binding["started_utc"])
        with self.path.open("x", encoding="utf-8") as stream:
            stream.write(canonical(self.value)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.closed = False

    def close(self):
        if not self.closed:
            self.closed = True
            self.value["state"] = "closed_no_automatic_resume"
            with self.path.open("w", encoding="utf-8") as stream:
                stream.write(canonical(self.value)+"\n")
                stream.flush()
                os.fsync(stream.fileno())


def validate_request(request, binding, seen):
    if not isinstance(request, dict) or request.keys() != {"schema", "request_id", "binding_sha256"}:
        raise ValueError("request fields; device commands forbidden")
    if request["schema"] != "mea.online.request.v1" or request["binding_sha256"] != binding_digest(binding):
        raise ValueError("request schema/binding")
    request_id = identifier(request["request_id"])
    if request_id in seen or len(seen) >= 10000:
        raise ValueError("reused request/capacity")
    return request_id


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


class SubprocessSource:
    def __init__(self, command, binding, evidence_dir, live=True, timeout=7.0, provenance=None):
        self.binding = validate_binding(binding)
        if not isinstance(command, list) or not command or any(not isinstance(item, str) or not item for item in command):
            raise ValueError("producer argv required")
        if not 0 < finite(timeout, "source timeout") <= 7:
            raise ValueError("source timeout must be <=7 seconds")
        self.command, self.live, self.timeout = list(command), bool(live), timeout
        self.provenance = provenance
        self.evidence = Path(evidence_dir)
        self.evidence.mkdir(parents=True, exist_ok=True)
        self._closed, self._preflight, self._process = False, None, None
        self._seen, self._previous_end = set(), None

    def _stop(self):
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)

    def _execute(self, arguments, label):
        """Bound BOTH pipes while draining; preserve bounded raw stdout/stderr."""
        paths = [self.evidence/(label+suffix) for suffix in (".stdout.json", ".stderr")]
        buffers = [bytearray(), bytearray()]
        exceeded = threading.Event()
        threads = []
        started = time.monotonic()
        with paths[0].open("xb") as out, paths[1].open("xb") as err:
            try:
                self._process = subprocess.Popen(self.command+arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                                 stdin=subprocess.DEVNULL, shell=False)
                def drain(pipe, target, buffer, limit):
                    try:
                        while True:
                            block = pipe.read(65536)
                            if not block:
                                break
                            available = max(0, limit-len(buffer))
                            part = block[:available]
                            buffer.extend(part)
                            target.write(part)
                            target.flush()
                            if len(block) > available:
                                exceeded.set()
                                break
                    except (OSError, ValueError):
                        exceeded.set()
                    finally:
                        pipe.close()
                for pipe, target, buffer, limit in ((self._process.stdout, out, buffers[0], MAX_LINE),
                                                     (self._process.stderr, err, buffers[1], 1024*1024)):
                    thread = threading.Thread(target=drain, args=(pipe, target, buffer, limit), daemon=True)
                    threads.append(thread)
                    thread.start()
                while self._process.poll() is None:
                    if exceeded.is_set():
                        raise ValueError("producer output capacity/read failure")
                    if time.monotonic()-started > self.timeout:
                        raise TimeoutError("producer subprocess timeout")
                    time.sleep(.01)
                for thread in threads:
                    thread.join(1)
                if any(thread.is_alive() for thread in threads) or exceeded.is_set():
                    raise ValueError("producer pipes failed to close or overflowed")
                if time.monotonic()-started > self.timeout:
                    raise TimeoutError("producer subprocess timeout")
                if self._process.returncode != 0:
                    raise ValueError("producer failed (see archived stderr)")
                return bytes(buffers[0])
            finally:
                self._stop()
                for thread in threads:
                    thread.join(1)
                self._process = None

    def preflight(self):
        if self._closed:
            raise ValueError("source closed/failed")
        if self._preflight is not None:
            return dict(self._preflight)
        try:
            if self.live:
                if len(self.command) != 1 or not Path(self.command[0]).is_absolute() or not isinstance(self.provenance, dict):
                    raise ValueError("live requires absolute pinned native executable/provenance")
                if file_digest(self.command[0]) != self.provenance.get("binary_sha256"):
                    raise ValueError("producer binary digest mismatch")
            result = strict_json(self._execute(["--describe-json"], "producer-preflight-"+uuid.uuid4().hex))
            expected = "live" if self.live else "simulation"
            if result.get("schema") != "mea.online.producer.v1" or result.get("source") != expected or result.get("filtered_frame_info") is not True:
                raise ValueError("producer provenance/source incompatible")
            if not result.get("headers") or result.get("headers") != result.get("library"):
                raise ValueError("SDK header/library version mismatch")
            if self.live:
                for key in ("header_sha256", "library_sha256"):
                    if result.get(key) != self.provenance.get(key) or not isinstance(result.get(key), str) or len(result[key]) != 64:
                        raise ValueError("SDK digest provenance mismatch")
            self._preflight = result
            return dict(result)
        except BaseException:
            self.close()
            raise

    def capture(self, request_id):
        if self._closed:
            raise ValueError("source closed/failed; retry forbidden")
        try:
            identifier(request_id)
            if request_id in self._seen:
                raise ValueError("reused producer request")
            self._seen.add(request_id)
            self.preflight()
            channels = ",".join(str(item[0]) for item in self.binding["channel_map"])
            raw = self._execute(["--capture", request_id, binding_digest(self.binding), self.binding["session_id"], channels], request_id)
            if raw.count(b"\n") != 1 or not raw.endswith(b"\n"):
                raise ValueError("producer must emit exactly one complete JSON line")
            value = strict_json(raw)
            if "window_sha256" in value:
                raise ValueError("native source must not impersonate sealed broker window")
            value["window_sha256"] = digest(value)
            value = validate_window(value, self.binding, request_id, self._previous_end, self.live)
            self._previous_end = value["end_frame"]
            with (self.evidence/(request_id+".window.json")).open("x", encoding="utf-8") as stream:
                stream.write(canonical(value)+"\n")
            return value
        except BaseException:
            self.close()
            raise

    def close(self):
        self._closed = True
        self._stop()


class WindowBroker:
    """One controller connection for the entire session; never reconnect/retry."""
    def __init__(self, source, binding, deadline, host="127.0.0.1", port=5560):
        self.binding = validate_binding(binding)
        if host not in ("127.0.0.1", "::1") or type(port) is not int or not 0 <= port < 65536:
            raise ValueError("broker requires loopback endpoint")
        self.source, self.deadline = source, deadline
        self.host, self.port = host, port
        self._listener, self._connection = None, None
        self._seen = set()

    def serve(self):
        try:
            self.deadline.check(8)
            self.source.preflight()
            family = socket.AF_INET6 if self.host == "::1" else socket.AF_INET
            self._listener = socket.socket(family, socket.SOCK_STREAM)
            self._listener.bind((self.host, self.port))
            self._listener.listen(1)
            while self._connection is None:
                self.deadline.check()
                self._listener.settimeout(min(1, self.deadline.remaining()))
                try:
                    self._connection, _ = self._listener.accept()
                except socket.timeout:
                    continue
            while True:
                self.deadline.check()
                raw = receive_line(self._connection, time.monotonic()+self.deadline.remaining(), 4096,
                                   allow_eof=True, budget=self.deadline)
                if raw is None:
                    return
                request = strict_json(raw)
                request_id = validate_request(request, self.binding, self._seen)
                self._seen.add(request_id)
                self.deadline.check(8)
                window = self.source.capture(request_id)
                self.deadline.check()
                self._connection.settimeout(min(1, self.deadline.remaining()))
                self._connection.sendall((canonical(window)+"\n").encode("utf-8"))
        finally:
            self.close()

    def close(self):
        for connection in (self._connection, self._listener):
            if connection is not None:
                connection.close()
        self._connection = self._listener = None
        self.source.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "live"))
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = strict_json(Path(args.config).read_bytes())
    if config.get("r2_enabled") is not False or config.get("stimulation_enabled") is not False:
        raise ValueError("R2 and stimulation must remain false")
    binding = validate_binding(config["binding"])
    deadline = DeviceDeadline(binding["started_utc"])
    producer = config["producer"]
    source = SubprocessSource([producer["path"]], binding, config["evidence_dir"], live=True, provenance=producer)
    lease = None
    try:
        metadata = source.preflight()
        if args.mode == "check":
            print(canonical(dict(status="configuration_and_binary_checked", producer=metadata,
                                 hardware_connected=False, identity_assurance="configured_identity_not_device_readback")))
            return 0
        lease = DeviceLease(config["registry_dir"], binding)
        receipt = dict(schema="mea.online.device-launch.v1", binding=binding, binding_sha256=binding_digest(binding),
                       producer=metadata, binary_sha256=file_digest(producer["path"]),
                       identity_assurance="configured_identity_not_device_readback",
                       coverage_basis="filtered_transport_frames", absolute_hardware_latency_verified=False,
                       r2_enabled=False, stimulation_enabled=False)
        with (source.evidence/"launch.json").open("x", encoding="utf-8") as stream:
            stream.write(canonical(receipt)+"\n")
        WindowBroker(source, binding, deadline, config.get("host", "127.0.0.1"), config.get("port", 5560)).serve()
        return 0
    finally:
        source.close()
        if lease is not None:
            lease.close()


if __name__ == "__main__":
    raise SystemExit(main())
