"""Bounded MaxLab adapter. Importing and constructing it never contacts a device.

The controller owns dose reservation, biological budgets and QC admission. This
adapter owns API auditing, finite sequences, resources and independent cleanup.
Only the local acquisition host is supported; it never SSHs to another host.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import uuid

from vendor.brain_stimulation import maxlab as mx
from vendor.brain_stimulation import transport
from .protocol import pulse_codes


class ManagedSequence(mx.Sequence):
    # SDK Sequence.__del__ opens a socket even with persistent=True.
    # Explicit ownership, including failed constructors, replaces GC cleanup.
    def __del__(self):
        pass


def _server_snapshot(proc_root=Path('/proc')):
    """Fail closed on unreadable server descriptors; ignore vanished processes."""
    servers, opened = [], []
    for process in Path(proc_root).iterdir():
        if not process.name.isdigit():
            continue
        try:
            if (process / 'comm').read_text().strip() != 'mxwserver':
                continue
            servers.append(int(process.name))
            for descriptor in (process / 'fd').iterdir():
                try:
                    target = os.readlink(descriptor)
                except FileNotFoundError:
                    continue
                if '.h5' in target.lower():
                    opened.append(target)
        except FileNotFoundError:
            continue
    if len(servers) != 1:
        raise RuntimeError('Exactly one mxwserver process is required')
    return servers[0], opened


def _operation(fn):
    """Expose SDK time for the last public operation, including nested calls."""
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        if not self._depth:
            self.last_api_seconds = 0.
        self._depth += 1
        try:
            return fn(self, *args, **kwargs)
        finally:
            self._depth -= 1
    return wrapped


class MaxLabBackend:
    def __init__(self, config, run_dir, clock):
        self.config, self.run_dir, self.clock = config, Path(run_dir), clock
        self.last_api_seconds = 0.
        self._depth = 0
        self._authorized = False
        self._preflight_done = False
        self._prepare_attempted = False
        self.prepared = False
        self.touched = False
        self.server_pid = None
        self.stim_channel = self.stim_unit = None
        self.array_token = None
        self.array = None
        self.sequences = []
        self.saving = mx.Saving()
        self.file_open = self.recording = False
        self.connected = False
        self.record_path = None
        self._record_opened_utc = None
        self._cleanup_mode = False
        self._cleanup_errors = []
        self._closed = False
        self._events_attempted = set()
        self._faulted = False
        self.before_sequence_send = None
        self.sent_monotonic_actual = None
        self.sent_event = None
        self._pending_event = None
        self._pending_pulse = False

    def _utc(self):
        return datetime.fromtimestamp(self.clock.wall(), timezone.utc).isoformat()

    def _authorize(self):
        self.config.validate_live()
        if platform.system() != 'Linux' or socket.gethostname() != self.config.device_hostname:
            raise RuntimeError('Live execution requires the explicitly bound Linux acquisition hostname')
        self._authorized = True

    def _audit(self, kind, command, **details):
        row = {'utc': self._utc(), 'kind': kind, 'command': command,
               'command_sha256': hashlib.sha256(command.encode('latin1')).hexdigest(), **details}
        with (self.run_dir / 'api_audit.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())

    def _log(self, kind, command, **details):
        try:
            self._audit(kind, command, **details)
        except Exception as exc:
            if not self._cleanup_mode:
                self._faulted = True
                raise
            # Audit storage failure must not prevent any safety command.
            self._cleanup_errors.append('audit ' + kind + ': ' + repr(exc))

    @contextmanager
    def _sdk(self):
        if not self._authorized:
            raise RuntimeError('No network before live authorization and host verification')
        if self._faulted and not self._cleanup_mode:
            raise RuntimeError('Previous API/audit failure prohibits further normal requests')
        endpoint = transport.Endpoint(timeout=1 if self._cleanup_mode else 10)
        with transport.bind_sdk(mx, endpoint):
            raw_context = mx.comm.api_context
            backend = self
            class AuditedApi:
                def send(self, message, timeout=None):
                    if message.startswith('mea_set_dac ') and message != 'mea_set_dac 0 512 512':
                        raise RuntimeError('Direct nonzero DAC writes are forbidden')
                    backend._log('request', message)
                    began = backend.clock.mono()
                    accounted = 0.
                    gate_failed = False
                    try:
                        # Open after durable request audit; include connection time.
                        with raw_context() as api:
                            if message.startswith('sequence_send '):
                                accounted = max(0., backend.clock.mono() - began)
                                backend.last_api_seconds += accounted
                                if backend._pending_pulse and backend.before_sequence_send is not None:
                                    # Gate after building AND connecting, immediately before
                                    # sending bytes. The controller can account all API time.
                                    try:
                                        backend.before_sequence_send(dict(backend._pending_event))
                                    except BaseException:
                                        gate_failed = True
                                        raise
                                backend.sent_monotonic_actual = backend.clock.mono()
                                backend.sent_event = dict(backend._pending_event)
                            reply = api.send(message, timeout)
                    except BaseException as exc:
                        if not gate_failed:
                            backend._faulted = True
                        backend._log('error', message, error=repr(exc))
                        raise
                    finally:
                        backend.last_api_seconds += max(0., backend.clock.mono() - began - accounted)
                    backend._log('ack', message, reply=reply)
                    return reply
            @contextmanager
            def audited_context(*args, **kwargs):
                yield AuditedApi()
            mx.comm.api_context = audited_context
            try:
                yield
            finally:
                mx.comm.api_context = raw_context

    def free_bytes(self):
        return shutil.disk_usage(self.run_dir).free

    @_operation
    def preflight(self):
        if self.touched or self._closed:
            raise RuntimeError('Preflight must precede hardware preparation')
        self._authorize()
        pid, opened = _server_snapshot()
        if opened:
            raise RuntimeError('Device server has open H5 files: ' + repr(opened))
        with self._sdk():
            lsb = float(mx.query_DAC_lsb_mV())
        pulse_codes('true', lsb)
        self.server_pid = pid
        self._preflight_done = True
        return {'free_bytes': self.free_bytes(), 'lsb_mv': lsb, 'server_pid': pid}

    def _check_server(self, allow_owned_file=True):
        pid, opened = _server_snapshot()
        if pid != self.server_pid:
            raise RuntimeError('Acquisition server PID changed')
        allowed = str(self.record_path.resolve()) if allow_owned_file and self.file_open and self.record_path else None
        if any(target != allowed for target in opened):
            raise RuntimeError('Unowned/open H5 file detected: ' + repr(opened))

    def _check_mapping(self):
        routed = mx.send_raw('mea_array_command ' + self.array_token + '\nmea_array_get_config')
        first = routed.split()[0] if routed.split() else ''
        actual = sorted((int(c), int(e)) for c, e in re.findall(r'(\d+)\((\d+)\)', first))
        expected = self.config.validate_binding()
        if actual != expected:
            raise RuntimeError('Owned array route differs from the bound route')
        unit = self.array.query_stimulation_at_electrode(self.config.stim_electrode).strip()
        if unit != str(self.stim_unit):
            raise RuntimeError('Owned stimulation unit mapping changed')
        return routed

    @_operation
    def prepare(self):
        if self._prepare_attempted or not self._preflight_done or self._closed:
            raise RuntimeError('Preparation requires successful preflight and is attempted only once')
        self._prepare_attempted = True
        self._authorize()
        self._check_server(False)
        with self._sdk():
            self.touched = True  # before initialize, including an ambiguous acknowledgement
            mx.initialize()
            mx.send(mx.Core().enable_stimulation_power(True))
            for unit in range(32):
                mx.send(mx.StimulationUnit(unit).power_up(False).connect(False))
            mx.send(mx.DAC(0, 512))
            mx.set_gain(512)
            mx.hpf('300Hz')
            self.array_token = 'v2_array_' + uuid.uuid4().hex
            self.array = mx.Array(self.array_token, persistent=True)
            self.array.load_config(self.config.route_file)
            self.array.connect_electrode_to_stimulation(self.config.stim_electrode)
            value = self.array.query_stimulation_at_electrode(self.config.stim_electrode).strip()
            if not value.isdigit() or not 0 <= int(value) < 32:
                raise RuntimeError('No valid stimulation unit in bound route')
            self.stim_unit = int(value)
            self.stim_channel = dict((e, c) for c, e in self.config.validate_binding())[self.config.stim_electrode]
            mx.activate([0])
            self.array.download([0])
            self.clock.sleep(5)
            mx.offset()
            self.clock.sleep(5)
            mx.send(mx.DAC(0, 512))
            routed = self._check_mapping()
        with (self.run_dir / 'downloaded_route.cfg').open('x', encoding='utf-8') as handle:
            handle.write(routed)
            handle.flush()
            os.fsync(handle.fileno())
        self.prepared = True
        return {'stim_channel': self.stim_channel, 'stim_unit': self.stim_unit}

    @_operation
    def assert_ready(self):
        if not self.prepared or self._closed:
            raise RuntimeError('Device is not prepared or has been cleaned up')
        self._authorize()
        self._check_server()
        with self._sdk():
            self._check_mapping()

    @_operation
    def open_record(self, name):
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', name):
            raise ValueError('Invalid recording name')
        if self.file_open or self.recording or self.sequences:
            raise RuntimeError('Close and retire the previous recording before opening another')
        self.assert_ready()
        folder = self.run_dir / name
        folder.mkdir(exist_ok=False)
        self.record_path = folder / (name + '.raw.h5')
        self._record_opened_utc = self._utc()
        with self._sdk():
            # No Event object exists for this file until after registry clearing.
            mx.clear_events()
            self.saving.open_directory(str(folder.resolve()))
            self.saving.set_legacy_format(False)
            self.file_open = True
            self.saving.start_file(name)
            self.saving.group_delete_all()
            self.saving.group_define(0, 'routed', list(range(1024)))
            self.recording = True
            self.saving.start_recording([0])
        return {'path': str(self.record_path), 'opened_utc': self._record_opened_utc}

    @_operation
    def connect(self, enabled):
        if type(enabled) is not bool:
            raise ValueError('Connection state must be boolean')
        self.assert_ready()
        with self._sdk():
            mx.send(mx.DAC(0, 512))
            # Set uncertain state before send; cleanup always disables all 32.
            self.connected = False
            mx.send(mx.StimulationUnit(self.stim_unit).power_up(enabled).connect(enabled).set_voltage_mode().dac_source(0))
            self.connected = enabled

    def _retire_sequences(self):
        for token in list(self.sequences):
            mx.send_raw('sequence_delete ' + token)
            self.sequences.remove(token)

    def _sequence(self, event_id, properties, codes=()):
        if not self.file_open or not self.recording:
            raise RuntimeError('Events require an open active recording')
        if type(event_id) is not int or not 1 <= event_id < 2**20 or event_id in self._events_attempted:
            raise ValueError('Invalid or previously attempted event ID')
        if not isinstance(properties, dict):
            raise ValueError('Event properties must be a dictionary')
        for key, value in properties.items():
            if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', key):
                raise ValueError('Event property keys must be safe ASCII tokens')
            if type(value) is not int and not (isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.-]+', value)):
                raise ValueError('Event property values must be integers or safe ASCII tokens')
        properties = {**properties, 'session': self.config.session_id}
        # The wire API takes alternating key/value tokens. The server converts
        # them to JSON string values in H5; sending JSON here is rejected.
        encoded = ' '.join(f'{key} {value}' for key, value in sorted(properties.items()))
        if len(encoded.encode('ascii')) > 240:
            raise ValueError('Event properties exceed the recording metadata budget')
        # The previous finite sequence has completed before a replacement exists.
        self._retire_sequences()
        token = 'v2_seq_' + uuid.uuid4().hex
        self.sequences.append(token)  # own before constructor; ACK may be lost
        self._events_attempted.add(event_id)
        sequence = ManagedSequence(token=token, initial_delay=100, persistent=True)
        if sequence.token != token:
            raise RuntimeError('Server returned a different sequence token')
        sequence.append(mx.Event(0, 1, event_id, encoded))
        for kind, value in codes:
            sequence.append(mx.DAC(0, value) if kind == 'dac' else mx.DelaySamples(value))
        self._pending_event = {'id': event_id, **properties}
        self._pending_pulse = bool(codes)
        try:
            sequence.send()
        finally:
            self._pending_event = None
            self._pending_pulse = False
        self.clock.sleep(.006)  # 5 ms initial + 0.4 ms pulse; no future event queued
        return {'send_mono': self.sent_monotonic_actual, 'event': dict(self.sent_event)}

    @_operation
    def marker(self, event_id, properties):
        self.assert_ready()
        with self._sdk():
            return self._sequence(event_id, properties)

    @_operation
    def pulse(self, event, lsb_mv):
        if not isinstance(event, dict) or not {'id', 'arm', 'pair'} <= event.keys():
            raise ValueError('Pulse requires id, arm and pair')
        codes = pulse_codes(event['arm'], lsb_mv)
        if not self.connected:
            raise RuntimeError('Pulse requires the selected zero-DAC connected unit')
        self.assert_ready()
        with self._sdk():
            actual_lsb = float(mx.query_DAC_lsb_mV())
            pulse_codes(event['arm'], actual_lsb)
            if actual_lsb != lsb_mv:
                self._faulted = True
                raise RuntimeError('DAC step changed since its admitted value')
            return {**self._sequence(event['id'], event, codes), 'lsb_mv': actual_lsb}

    @_operation
    def close_record(self):
        if not self.file_open and not self.recording:
            raise RuntimeError('No recording is open')
        errors = []
        with self._sdk():
            for label, flag, method in [('recording', self.recording, self.saving.stop_recording),
                                        ('file', self.file_open, self.saving.stop_file)]:
                if flag:
                    try:
                        method()
                        if label == 'recording': self.recording = False
                        else: self.file_open = False
                    except Exception as exc:
                        errors.append(label + ': ' + repr(exc))
            try:
                self.saving.group_delete_all()
            except Exception as exc:
                errors.append('groups: ' + repr(exc))
            if not errors:
                self._retire_sequences()
        if errors:
            raise RuntimeError('Recording close failed: ' + '; '.join(errors))
        self._check_server(False)
        return {'path': str(self.record_path), 'opened_utc': self._record_opened_utc,
                'closed_utc': self._utc(), 'server_pid': self.server_pid}

    @_operation
    def cleanup(self):
        if self._closed:
            return list(self._cleanup_errors)
        self._closed = True
        if not self._authorized or not self.touched:
            return []
        self._cleanup_mode = True
        self._cleanup_errors = []
        def attempt(label, method):
            try:
                method()
            except BaseException as exc:
                self._cleanup_errors.append(label + ': ' + repr(exc))
        try:
            with self._sdk():
                # At most 39 requests: 32 off + zero + 3 file + <=2 seq + array.
                # Each has <=1 s connect and <=1 s reply, within 120 s reserve.
                for unit in range(32):
                    attempt('off ' + str(unit), lambda unit=unit: mx.send(mx.StimulationUnit(unit).power_up(False).connect(False)))
                attempt('zero DAC', lambda: mx.send(mx.DAC(0, 512)))
                if self.recording:
                    attempt('stop recording', self.saving.stop_recording)
                if self.file_open:
                    attempt('stop file', self.saving.stop_file)
                    attempt('clear groups', self.saving.group_delete_all)
                for token in list(self.sequences):
                    attempt('delete sequence ' + token, lambda token=token: mx.send_raw('sequence_delete ' + token))
                if self.array_token:
                    attempt('delete array', lambda: mx.send_raw('mea_array_delete ' + self.array_token))
            attempt('verify closed files', lambda: self._check_server(False))
        finally:
            self._cleanup_mode = False
            self.connected = False
            self.prepared = False
        return list(self._cleanup_errors)
