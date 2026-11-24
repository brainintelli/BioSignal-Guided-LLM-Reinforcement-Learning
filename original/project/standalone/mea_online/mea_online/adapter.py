"""Strict, bounded request/response transport; no MaxLab dependency.

Coverage describes filtered TRANSPORT frames, never spike-peak completeness.
The digest proves consistency; deployment provenance / SSH authenticates origin.
"""
import copy
import datetime as dt
import hashlib
import json
import math
import re
import socket
import time

MAX_LINE = 8 * 1024 * 1024
MAX_EVENTS = 100000
BINDING_FIELDS = frozenset(("session_id", "chip_id", "organoid_id", "started_utc", "channel_map",
                            "sample_rate_hz", "window_frames", "well_id", "amplitude_scale_uv"))
WINDOW_FIELDS = frozenset(("schema", "source", "request_id", "binding_sha256", "session_id", "start_frame",
    "end_frame", "frame_count", "complete", "dropped", "corrupted", "stream_opened_after_request", "warmup_frames", "warmup_elapsed_s",
    "capture_elapsed_s", "events", "window_sha256"))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("nonfinite JSON")
    def parse_float(raw):
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("nonfinite JSON number")
        return value
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant, parse_float=parse_float)
    except (TypeError, UnicodeError, RecursionError) as exc:
        raise ValueError("invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def integer(value, name, maximum=2**64-1):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("invalid " + name)
    return value


def finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("invalid " + name)
    return value


def identifier(value):
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value) is None:
        raise ValueError("invalid request/session identifier")
    return value


def utc_timestamp(value):
    try:
        stamp = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.utcoffset() != dt.timedelta(0):
            raise ValueError("UTC required")
        return stamp.timestamp()
    except (AttributeError, TypeError, OverflowError) as exc:
        raise ValueError("invalid started_utc") from exc


def validate_binding(binding):
    if not isinstance(binding, dict) or not BINDING_FIELDS <= binding.keys() or binding.keys() - BINDING_FIELDS - {"binding_sha256"}:
        raise ValueError("binding fields")
    for key in ("session_id", "chip_id", "organoid_id"):
        identifier(binding[key])
        if binding[key].lower() in ("unknown", "null", "none", "todo", "placeholder"):
            raise ValueError("unbound " + key)
    utc_timestamp(binding["started_utc"])
    for key, expected in (("sample_rate_hz", 20000), ("window_frames", 20000), ("well_id", 0)):
        if integer(binding[key], key) != expected:
            raise ValueError("unsupported " + key)
    if finite(binding["amplitude_scale_uv"], "amplitude units") <= 0:
        raise ValueError("measured native-to-uV scale required")
    mapping = binding["channel_map"]
    if not isinstance(mapping, list) or not 1 <= len(mapping) <= 1024:
        raise ValueError("nonempty channel map required")
    channels, electrodes = set(), set()
    for item in mapping:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("channel/electrode pair")
        channel, electrode = item
        integer(channel, "channel", 1023)
        integer(electrode, "electrode", 26399)
        if channel in channels or electrode in electrodes:
            raise ValueError("duplicate channel/electrode")
        channels.add(channel)
        electrodes.add(electrode)
    if mapping != sorted(mapping):
        raise ValueError("channel map must be sorted")
    detached = {key: copy.deepcopy(binding[key]) for key in BINDING_FIELDS}
    if "binding_sha256" in binding and binding["binding_sha256"] != digest(detached):
        raise ValueError("binding digest mismatch")
    return detached


def binding_digest(binding):
    return digest(validate_binding(binding))


def validate_window(window, binding, request_id, previous_end=None, live=True):
    """Return detached validated window or fail; quiet windows still need coverage."""
    binding = validate_binding(binding)
    identifier(request_id)
    if not isinstance(window, dict) or window.keys() != WINDOW_FIELDS:
        raise ValueError("window fields")
    if window["schema"] != "mea.online.window.v1" or window["source"] != ("live" if live else "simulation"):
        raise ValueError("schema/source mismatch")
    if window["binding_sha256"] != digest(binding) or window["session_id"] != binding["session_id"] or window["request_id"] != request_id:
        raise ValueError("window binding/request identity mismatch")
    start = integer(window["start_frame"], "start frame")
    end = integer(window["end_frame"], "end frame")
    count = integer(window["frame_count"], "frame count")
    if count != binding["window_frames"] or end-start != count:
        raise ValueError("incomplete transport frame window")
    if previous_end is not None and start < integer(previous_end, "previous end"):
        raise ValueError("overlapping/reused window")
    if window["complete"] is not True or window["stream_opened_after_request"] is not True:
        raise ValueError("coverage/drain barrier missing")
    if integer(window["dropped"], "dropped") or integer(window["corrupted"], "corrupted"):
        raise ValueError("invalid native coverage")
    if integer(window["warmup_frames"], "warmup frames") != 20000 or not .75 <= finite(window["warmup_elapsed_s"], "warmup elapsed") <= 2.5:
        raise ValueError("fresh stream warmup missing/accelerated")
    if not .75 <= finite(window["capture_elapsed_s"], "capture elapsed") <= 2.5:
        raise ValueError("stale/backlogged or timed-out capture")
    events = window["events"]
    if not isinstance(events, list) or len(events) > MAX_EVENTS:
        raise ValueError("event capacity")
    channels = {pair[0] for pair in binding["channel_map"]}
    seen, previous = set(), start
    for event in events:
        if not isinstance(event, list) or len(event) != 4:
            raise ValueError("event shape")
        transport, peak, channel, amplitude = event
        integer(transport, "transport frame")
        integer(peak, "peak frame")
        integer(channel, "event channel", 1023)
        finite(amplitude, "amplitude")
        if not start <= transport < end or transport < previous or peak > transport or channel not in channels:
            raise ValueError("event frame/order/channel")
        if (peak, channel) in seen:
            raise ValueError("duplicate event")
        seen.add((peak, channel))
        previous = transport
    content = {key: value for key, value in window.items() if key != "window_sha256"}
    if window["window_sha256"] != digest(content):
        raise ValueError("window digest mismatch")
    if len(canonical(window).encode("utf-8")) > MAX_LINE:
        raise ValueError("window byte capacity")
    return copy.deepcopy(window)


def receive_line(connection, deadline, maximum=MAX_LINE, allow_eof=False, budget=None):
    """Bounded line; optional session budget allows idle polling without frames.

    EOF is clean only before any bytes of a new broker request. Client responses
    retain strict EOF handling and their independent per-capture deadline.
    """
    data = bytearray()
    while True:
        remaining = deadline-time.monotonic()
        if budget is not None:
            budget.check()
            remaining = min(remaining, budget.remaining())
        if remaining <= 0:
            raise TimeoutError("response deadline")
        connection.settimeout(min(1, remaining) if budget is not None else remaining)
        try:
            chunk = connection.recv(min(65536, maximum+1-len(data)))
        except socket.timeout:
            if budget is None:
                raise
            continue
        if not chunk:
            if allow_eof and not data:
                return None
            raise ValueError("truncated response")
        data.extend(chunk)
        if len(data) > maximum:
            raise ValueError("response exceeds byte bound")
        if b"\n" in data:
            line, tail = bytes(data).split(b"\n", 1)
            if tail:
                raise ValueError("unsolicited trailing response")
            return line


class WindowClient:
    def __init__(self, host, port, binding, timeout=8.0):
        self.binding = validate_binding(binding)
        if not isinstance(host, str) or not host or type(port) is not int or not 0 < port < 65536:
            raise ValueError("pinned endpoint required")
        if not 0 < finite(timeout, "timeout") <= 8.0:
            raise ValueError("request timeout must be <=8 seconds")
        self.host, self.port, self.timeout = host, port, timeout
        self._socket = None
        self._closed = False
        self._seen = set()
        self._previous_end = None

    def capture(self, request_id):
        if self._closed:
            raise ValueError("client closed/failed; reconnect forbidden")
        try:
            identifier(request_id)
            if request_id in self._seen or len(self._seen) >= 10000:
                raise ValueError("request reused/capacity")
            self._seen.add(request_id)
            deadline = time.monotonic()+self.timeout
            if self._socket is None:
                self._socket = socket.create_connection((self.host, self.port), self.timeout)
            request = dict(schema="mea.online.request.v1", request_id=request_id, binding_sha256=digest(self.binding))
            self._socket.settimeout(max(.001, deadline-time.monotonic()))
            self._socket.sendall((canonical(request)+"\n").encode("utf-8"))
            result = validate_window(strict_json(receive_line(self._socket, deadline)), self.binding,
                                     request_id, self._previous_end, live=True)
            if time.monotonic() > deadline:
                raise TimeoutError("request exceeded deadline")
            self._previous_end = result["end_frame"]
            return result
        except BaseException:
            self.close()
            raise

    def close(self):
        self._closed = True
        if self._socket is not None:
            self._socket.close()
            self._socket = None
