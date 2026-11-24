"""Bounded byte-stream decoders. Legacy host time is never device time.

Research watermarks assert complete coverage of [start_frame, end_frame).
Only a device producer that can attest to that coverage may send them; the
legacy spike-only CSV feed cannot be upgraded by guessing a watermark.
"""
from dataclasses import dataclass
import hashlib
import json
import math


def number(value, name):
    if type(value) not in (int, float):
        raise ValueError(f'{name}: finite number required')
    try:
        valid = math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f'{name}: finite number required')
    return value


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name}: integer >= {minimum} required')
    return value


def identifier(value, name):
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError(f'{name}: nonempty identifier of at most 256 characters required')
    return value


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f'Invalid JSON number: {value}')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def fields(row, names):
    if not isinstance(row, dict) or set(row) != set(names.split()):
        raise ValueError('Unexpected or missing protocol fields')


class _Lines:
    def __init__(self, max_line_bytes):
        self.max_line_bytes = integer(max_line_bytes, 'max_line_bytes', 1)
        self._pending = bytearray()
        self.failed = False
        self._closed = False

    def feed(self, data):
        if self.failed or self._closed:
            raise ValueError('Stream failed or closed; create a new decoder')
        try:
            if not isinstance(data, bytes):
                raise ValueError('feed requires bytes')
            result = []
            start = 0
            while start < len(data):
                end = data.find(b'\n', start)
                stop = len(data) if end < 0 else end
                if len(self._pending) + stop - start > self.max_line_bytes:
                    raise ValueError('Line exceeds buffer limit')
                self._pending.extend(data[start:stop])
                if end < 0:
                    break
                line = bytes(self._pending).removesuffix(b'\r').decode('utf-8')
                self._pending.clear()
                if not line:
                    raise ValueError('Empty protocol line')
                result.append(self._line(line))
                start = end + 1
            return result
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
            self.failed = True
            raise ValueError(f'Invalid stream: {exc}') from exc

    def finish(self):
        if self.failed or self._pending:
            self.failed = True
            raise ValueError('Failed stream or truncated final line')
        self._closed = True


@dataclass(frozen=True)
class LegacySpike:
    host_time_ms: float
    channel: int
    amplitude_uv: float
    device_frame: None = None


class LegacyCsvDecoder(_Lines):
    def __init__(self, time_unit='ms', max_line_bytes=4096):
        super().__init__(max_line_bytes)
        if time_unit not in ('ms', 's'):
            raise ValueError('Explicit legacy timestamp unit must be ms or s')
        self.time_unit = time_unit

    def _line(self, line):
        parts = line.split(',')
        if len(parts) != 3:
            raise ValueError('Expected timestamp,channel,amplitude_uv')
        timestamp = number(float(parts[0]), 'host time')
        if timestamp < 0:
            raise ValueError('Negative host time')
        channel = integer(int(parts[1]), 'channel')
        amplitude = number(float(parts[2]), 'amplitude_uv')
        milliseconds = number(timestamp * (1000 if self.time_unit == 's' else 1), 'host ms')
        return LegacySpike(milliseconds, channel, amplitude)


@dataclass(frozen=True)
class Session:
    session_id: str
    source: str
    sample_rate_hz: float
    well_id: int
    start_frame: int
    channel_map_id: str
    channels: tuple


@dataclass(frozen=True)
class Stimulus:
    event_id: str
    frame: int
    context_id: str
    pattern_id: str


class ResearchStream(_Lines):
    def __init__(self, max_line_bytes=1_048_576, max_spikes=100_000, max_stimuli=4096):
        super().__init__(max_line_bytes)
        self.max_spikes = integer(max_spikes, 'max_spikes', 1)
        self.max_stimuli = integer(max_stimuli, 'max_stimuli', 1)
        self._session = None
        self._seq = -1
        self._watermark = 0
        self._last_spike_frame = -1
        self._spikes = []
        self._spike_keys = set()
        self._stimuli = {}

    @property
    def session(self):
        return self._session

    @property
    def session_id(self):
        return self.session.session_id if self.session else None

    @property
    def watermark(self):
        return self._watermark

    @property
    def spikes(self):
        return tuple(self._spikes)

    @property
    def stimuli(self):
        return tuple(self._stimuli.values())

    def stimulus(self, event_id):
        try:
            return self._stimuli[event_id]
        except KeyError as exc:
            raise ValueError('Unknown stimulus event') from exc

    def _line(self, line):
        row = strict_json(line)
        if not isinstance(row, dict):
            raise ValueError('Protocol packet must be an object')
        if integer(row.get('seq'), 'seq') != self._seq + 1:
            raise ValueError('Sequence gap, duplicate or reorder')
        kind = row.get('type')
        if self.session is None:
            fields(row, 'type protocol session_id seq source sample_rate_hz well_id start_frame channel_map_id channels')
            if kind != 'session' or row['protocol'] != 'mea.spikes.v1':
                raise ValueError('Versioned session must be first')
            source = row['source']
            if source not in ('simulation', 'recording', 'live'):
                raise ValueError('Unknown source')
            rate = number(row['sample_rate_hz'], 'sample_rate_hz')
            if rate <= 0:
                raise ValueError('Sample rate must be positive')
            channels = row['channels']
            if not isinstance(channels, list) or not 1 <= len(channels) <= 65536:
                raise ValueError('Bounded nonempty channel map required')
            for pair in channels:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError('Expected channel/electrode pairs')
                integer(pair[0], 'channel'); integer(pair[1], 'electrode')
            if any(len({pair[i] for pair in channels}) != len(channels) for i in (0, 1)):
                raise ValueError('Channel map must be one-to-one')
            self._session = Session(identifier(row['session_id'], 'session_id'), source, rate,
                                    integer(row['well_id'], 'well_id'), integer(row['start_frame'], 'start_frame'),
                                    identifier(row['channel_map_id'], 'channel_map_id'),
                                    tuple(tuple(pair) for pair in channels))
            self._watermark = self.session.start_frame
        else:
            if row.get('session_id') != self.session_id:
                raise ValueError('Session changed: reconnect requires a new stream')
            if kind == 'stimulus':
                fields(row, 'type session_id seq event_id frame context_id pattern_id')
                event_id = identifier(row['event_id'], 'event_id')
                frame = integer(row['frame'], 'stimulus frame')
                if event_id in self._stimuli or len(self._stimuli) >= self.max_stimuli:
                    raise ValueError('Duplicate stimulus or stimulus capacity exceeded')
                if frame < max(self.watermark, self._last_spike_frame) or any(frame <= s.frame for s in self.stimuli):
                    raise ValueError('Stimulus clock regressed')
                self._stimuli[event_id] = Stimulus(event_id, frame, identifier(row['context_id'], 'context_id'),
                                                  identifier(row['pattern_id'], 'pattern_id'))
            elif kind == 'spikes':
                fields(row, 'type session_id seq events')
                events = row['events']
                if not isinstance(events, list) or len(events) + len(self._spikes) > self.max_spikes:
                    raise ValueError('Spike capacity exceeded or invalid events')
                mapped = {pair[0] for pair in self.session.channels}
                for event in events:
                    if not isinstance(event, list) or len(event) != 3:
                        raise ValueError('Expected frame/channel/amplitude triple')
                    frame, channel = integer(event[0], 'frame'), integer(event[1], 'channel')
                    amplitude = number(event[2], 'amplitude_uv')
                    if frame < max(self.watermark, self._last_spike_frame):
                        raise ValueError('Late or reordered device spike')
                    if channel not in mapped or (frame, channel) in self._spike_keys:
                        raise ValueError('Unmapped channel or duplicate device spike')
                    self._spike_keys.add((frame, channel))
                    self._spikes.append((frame, channel, amplitude))
                    self._last_spike_frame = frame
            elif kind == 'watermark':
                fields(row, 'type session_id seq end_frame complete dropped')
                end = integer(row['end_frame'], 'end_frame')
                if end < self.watermark or row['complete'] is not True or integer(row['dropped'], 'dropped') != 0:
                    raise ValueError('Incomplete coverage, loss or regressed watermark')
                self._watermark = end
            else:
                raise ValueError('Unknown packet or repeated session')
        self._seq = row['seq']
        return row
