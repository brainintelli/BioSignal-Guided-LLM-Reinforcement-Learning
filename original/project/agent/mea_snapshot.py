"""Immutable, content-addressed engineering features, not a comfort score."""
from dataclasses import asdict, dataclass, field
from agent.mea_protocol import digest, identifier, integer, number

FEATURE_NAMES = ('spike_count', 'rate_hz_per_channel', 'active_channel_fraction', 'mean_abs_amplitude_uv')
SCHEMA = 'mea.features.p0.v1'


def vector(values, name):
    if not isinstance(values, (tuple, list)) or len(values) != len(FEATURE_NAMES):
        raise ValueError(f'{name}: four numeric values required')
    return tuple(float(number(v, name)) for v in values)


@dataclass(frozen=True)
class Normalization:
    centers: tuple
    scales: tuple

    def __post_init__(self):
        object.__setattr__(self, 'centers', vector(self.centers, 'centers'))
        object.__setattr__(self, 'scales', vector(self.scales, 'scales'))
        if any(v <= 0 for v in self.scales):
            raise ValueError('Normalization scales must be positive')

    @classmethod
    def identity(cls):
        return cls((0.,) * 4, (1.,) * 4)

    @property
    def normalization_id(self):
        return digest(dict(schema=SCHEMA, centers=self.centers, scales=self.scales))

    def transform(self, values):
        return vector([(x - c) / s for x, c, s in zip(vector(values, 'features'), self.centers, self.scales)], 'normalized')


class IncompleteWindow(ValueError):
    pass


@dataclass(frozen=True)
class WetwareSnapshot:
    session_id: str
    source: str
    event_id: str
    context_id: str
    pattern_id: str
    channel_map_id: str
    well_id: int
    mapped_channels: int
    sample_rate_hz: float
    stimulus_frame: int
    window_start_frame: int
    window_end_frame: int
    coverage_end_frame: int
    window_spikes_digest: str
    feature_values: tuple
    normalization: Normalization
    schema: str = field(default=SCHEMA, init=False)
    normalized_values: tuple = field(init=False)
    normalization_id: str = field(init=False)
    snapshot_id: str = field(init=False)

    def __post_init__(self):
        for key in ('session_id', 'event_id', 'context_id', 'pattern_id', 'channel_map_id', 'window_spikes_digest'):
            identifier(getattr(self, key), key)
        if self.source not in ('simulation', 'recording', 'live') or not isinstance(self.normalization, Normalization):
            raise ValueError('Invalid source or normalization')
        for key in ('well_id', 'stimulus_frame', 'window_start_frame', 'window_end_frame', 'coverage_end_frame'):
            integer(getattr(self, key), key)
        integer(self.mapped_channels, 'mapped_channels', 1)
        rate = number(self.sample_rate_hz, 'sample_rate_hz')
        if rate <= 0 or not self.stimulus_frame <= self.window_start_frame < self.window_end_frame <= self.coverage_end_frame:
            raise ValueError('Invalid or incomplete window')
        values = vector(self.feature_values, 'features')
        count, frequency, fraction, amplitude = values
        expected_rate = count * rate / ((self.window_end_frame - self.window_start_frame) * self.mapped_channels)
        if (any(x < 0 for x in values) or count != int(count) or fraction > 1 or frequency != expected_rate
                or (count == 0 and values != (0., 0., 0., 0.)) or (count > 0 and fraction <= 0)):
            raise ValueError('Inconsistent neural features')
        object.__setattr__(self, 'feature_values', values)
        object.__setattr__(self, 'normalized_values', self.normalization.transform(values))
        object.__setattr__(self, 'normalization_id', self.normalization.normalization_id)
        object.__setattr__(self, 'snapshot_id', digest(self._payload()))

    def _payload(self):
        return {key: getattr(self, key) for key in self.__dataclass_fields__ if key not in ('snapshot_id', 'normalization')} | {
            'normalization': asdict(self.normalization)}

    def to_dict(self):
        # JSON-like detached values, with no references to caller-owned containers.
        import json
        from agent.mea_protocol import canonical
        return json.loads(canonical(self._payload() | {'snapshot_id': self.snapshot_id}))

    @classmethod
    def from_dict(cls, data):
        try:
            if not isinstance(data, dict) or set(data) != set(cls.__dataclass_fields__):
                raise ValueError('Snapshot fields do not match schema')
            kwargs = {k: v for k, v in data.items() if cls.__dataclass_fields__[k].init}
            kwargs['normalization'] = Normalization(**kwargs['normalization'])
            result = cls(**kwargs)
            if result.to_dict() != data:
                raise ValueError('Snapshot digest or derived values mismatch')
            return result
        except (TypeError, KeyError, OverflowError) as exc:
            raise ValueError('Invalid snapshot') from exc


def build_snapshot(stream, event_id, *, blank_frames, window_frames, normalization):
    integer(blank_frames, 'blank_frames')
    integer(window_frames, 'window_frames', 1)
    if stream.failed or stream.session is None:
        raise ValueError('Cannot snapshot failed or uninitialized stream')
    event = stream.stimulus(event_id)
    start = event.frame + blank_frames
    end = start + window_frames
    if stream.watermark < end:
        raise IncompleteWindow('Device coverage has not reached window end')
    if any(event.frame < other.frame < end for other in stream.stimuli):
        raise ValueError('Response window overlaps a subsequent stimulus')
    spikes = tuple(row for row in stream.spikes if start <= row[0] < end)
    session = stream.session
    channels = len(session.channels)
    count = len(spikes)
    values = (float(count), count * session.sample_rate_hz / (window_frames * channels),
              len({row[1] for row in spikes}) / channels,
              sum(abs(row[2]) for row in spikes) / count if count else 0.)
    return WetwareSnapshot(session.session_id, session.source, event.event_id, event.context_id, event.pattern_id,
                           session.channel_map_id, session.well_id, channels, session.sample_rate_hz,
                           event.frame, start, end, end, digest(spikes), values, normalization)
