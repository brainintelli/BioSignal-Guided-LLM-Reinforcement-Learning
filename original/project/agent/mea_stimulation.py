"""Immutable software-only stimulation plans. No device SDK or transport."""
from dataclasses import asdict, dataclass, field
from decimal import Decimal, ROUND_FLOOR

from agent.mea_contract import observation_key
from agent.mea_protocol import digest, identifier, integer, number


def decimal(value, name):
    return Decimal(str(number(value, name)))


@dataclass(frozen=True)
class StimulationParameters:
    """Registered voltage profile; changing a unit cannot silently change dose."""
    mode: str = 'voltage'
    polarity: str = 'positive_first'
    p90_multiplier: float = 1.2
    phase_us: int = 200
    inter_phase_us: int = 0
    time_step_us: int = 50
    dac_channel: int = 0
    initial_delay_samples: int = 100
    onset_interval_min_ms: int = 10000
    onset_interval_max_ms: int = 14000
    interval_seed: int = 20261117

    def __post_init__(self):
        fixed = dict(mode='voltage', polarity='positive_first', p90_multiplier=1.2,
                     phase_us=200, inter_phase_us=0, time_step_us=50, dac_channel=0,
                     initial_delay_samples=100)
        for key, expected in fixed.items():
            if type(getattr(self, key)) is not type(expected) or getattr(self, key) != expected:
                raise ValueError('Unregistered stimulation parameter: ' + key)
        integer(self.onset_interval_min_ms, 'onset_interval_min_ms', 10000)
        integer(self.onset_interval_max_ms, 'onset_interval_max_ms', self.onset_interval_min_ms)
        integer(self.interval_seed, 'interval_seed')
        if self.onset_interval_max_ms > 14000: raise ValueError('Onset interval exceeds 14 seconds')

    def to_dict(self): return asdict(self)

    def post_pulse_delay_samples(self, onset_interval_ms):
        """Onset-to-onset interval minus BOTH phases; not a frequency field."""
        integer(onset_interval_ms, 'onset_interval_ms', self.onset_interval_min_ms)
        if onset_interval_ms > self.onset_interval_max_ms: raise ValueError('Onset interval exceeds profile')
        return (onset_interval_ms * 1000 - 2 * self.phase_us - self.inter_phase_us) // self.time_step_us


DEFAULT_PARAMETERS = StimulationParameters()


@dataclass(frozen=True)
class Baseline:
    session_id: str
    chip_id: str
    well_id: int
    sample_rate_hz: int
    channels: tuple
    electrode_id: int
    start_frame: int
    end_frame: int
    events: tuple
    source: str = 'simulation'
    p90_uv: float = field(init=False)
    cap_mv: float = field(init=False)
    cap_mv_exact: str = field(init=False)
    channel_map_id: str = field(init=False)
    baseline_id: str = field(init=False)

    def __post_init__(self):
        identifier(self.session_id, 'session_id'); identifier(self.chip_id, 'chip_id')
        integer(self.well_id, 'well_id'); integer(self.sample_rate_hz, 'sample_rate_hz', 1)
        integer(self.electrode_id, 'electrode_id'); integer(self.start_frame, 'start_frame')
        integer(self.end_frame, 'end_frame', self.start_frame + 1)
        if self.source != 'simulation':
            raise ValueError('Only synthetic baselines admitted by this software runner')
        if not isinstance(self.channels, (list, tuple)) or not 2 <= len(self.channels) <= 1024:
            raise ValueError('Need stimulation and separate recording channels')
        channels = []
        for pair in self.channels:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError('Invalid channel/electrode pair')
            channel, electrode = pair
            integer(channel, 'channel'); integer(electrode, 'electrode')
            if electrode > 26399: raise ValueError('Physical electrode out of range')
            channels.append((channel, electrode))
        if any(len({p[i] for p in channels}) != len(channels) for i in (0, 1)):
            raise ValueError('Mapping must be one-to-one')
        lookup = dict(channels)
        if self.electrode_id not in lookup.values(): raise ValueError('Stimulating electrode absent')
        if not isinstance(self.events, (tuple, list)) or not 100 <= len(self.events) <= 100000:
            raise ValueError('Bounded baseline with at least 100 events required')
        rows, seen, own = [], set(), []
        previous = self.start_frame
        for row in self.events:
            if not isinstance(row, (list, tuple)) or len(row) != 3: raise ValueError('Invalid baseline event')
            frame, channel, amplitude = row
            integer(frame, 'frame'); integer(channel, 'channel'); number(amplitude, 'amplitude_uv')
            if not previous <= frame < self.end_frame or channel not in lookup or (frame, channel) in seen:
                raise ValueError('Baseline order, mapping, coverage or duplicate error')
            previous = frame; seen.add((frame, channel)); rows.append((frame, channel, float(amplitude)))
            if lookup[channel] == self.electrode_id: own.append(abs(decimal(amplitude, 'amplitude_uv')))
        if len(own) < 100: raise ValueError('At least 100 own-electrode spikes required')
        own.sort()
        at = Decimal('0.9') * (len(own) - 1)
        lower = int(at)
        p90 = own[lower] + (own[min(lower + 1, len(own) - 1)] - own[lower]) * (at - lower)
        if p90 <= 0: raise ValueError('Positive baseline P90 required')
        object.__setattr__(self, 'channels', tuple(channels))
        object.__setattr__(self, 'events', tuple(rows))
        object.__setattr__(self, 'p90_uv', float(p90))
        cap = decimal(DEFAULT_PARAMETERS.p90_multiplier, 'P90 multiplier') * p90 / 1000
        object.__setattr__(self, 'cap_mv', float(cap))
        object.__setattr__(self, 'cap_mv_exact', str(cap))
        object.__setattr__(self, 'channel_map_id', digest(self.channels))
        object.__setattr__(self, 'baseline_id', digest({k: getattr(self, k) for k in self.__dataclass_fields__ if k != 'baseline_id'}))

    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class OutputPath:
    path_id: str
    dac_lsb_mv: float
    uncertainty_mv: float
    calibration_id: str
    simulation_only: bool = True

    def __post_init__(self):
        identifier(self.path_id, 'path_id'); identifier(self.calibration_id, 'calibration_id')
        if decimal(self.dac_lsb_mv, 'DAC LSB') <= 0 or decimal(self.uncertainty_mv, 'uncertainty') < 0:
            raise ValueError('Invalid output resolution or uncertainty')
        if self.simulation_only is not True: raise ValueError('Hardware output paths are disabled')


@dataclass(frozen=True)
class StimulusPlan:
    session_id: str
    chip_id: str
    well_id: int
    channel_map_id: str
    baseline_id: str
    electrode_id: int
    event_id: str
    context_id: str
    path_id: str
    calibration_id: str
    condition: str
    reference_frame: int
    stimulus_frame: int
    code_delay_ms: int
    phase_us: int
    dac_offset: int
    dac_codes: tuple
    nominal_peak_mv: float
    peak_upper_mv: float
    cap_mv: float
    parameters: StimulationParameters = DEFAULT_PARAMETERS
    plan_id: str = field(init=False)

    def __post_init__(self):
        if not isinstance(self.parameters, StimulationParameters): raise ValueError('Frozen parameter profile required')
        values = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != 'plan_id'}
        values['parameters'] = self.parameters.to_dict()
        object.__setattr__(self, 'plan_id', digest(values))

    def to_dict(self): return asdict(self)


def make_plan(observation, baseline, path, event_id, reference_frame, condition='stimulus', *,
              parameters=DEFAULT_PARAMETERS, previous_stimulus_frame=None, onset_interval_ms=None):
    if not isinstance(baseline, Baseline) or not isinstance(path, OutputPath): raise ValueError('Frozen calibration required')
    if not isinstance(parameters, StimulationParameters): raise ValueError('Frozen parameter profile required')
    identifier(event_id, 'event_id'); integer(reference_frame, 'reference_frame', baseline.end_frame)
    if condition not in ('stimulus', 'sham'): raise ValueError('Unknown stimulation condition')
    context = observation_key(observation, 'reserves')  # Validate the factual allowlist first.
    player = observation['player']
    for name in ('health', 'thirst'):
        if not 0 <= number(player[name], name) <= 100: raise ValueError('Invalid game vital')
    threats = [number(item['distance'], 'threat distance') for item in observation['threats']]
    delay = 100 if player['thirst'] <= 20 else 200 if player['health'] <= 30 or any(d < 1.6 for d in threats) else 300
    offset_frames = baseline.sample_rate_hz * delay / 1000
    if offset_frames != int(offset_frames): raise ValueError('Unrepresentable timing grid')
    if previous_stimulus_frame is None:
        if onset_interval_ms is not None: raise ValueError('Previous onset required for interval')
    else:
        integer(previous_stimulus_frame, 'previous_stimulus_frame', baseline.end_frame)
        parameters.post_pulse_delay_samples(onset_interval_ms)
        interval_frames, remainder = divmod(baseline.sample_rate_hz * onset_interval_ms, 1000)
        if remainder: raise ValueError('Unrepresentable onset interval')
        scheduled_reference = previous_stimulus_frame + interval_frames - int(offset_frames)
        if scheduled_reference < reference_frame: raise ValueError('Scheduled sequence overlaps previous response')
        reference_frame = scheduled_reference
    cap, lsb, uncertainty = Decimal(baseline.cap_mv_exact), decimal(path.dac_lsb_mv, 'LSB'), decimal(path.uncertainty_mv, 'uncertainty')
    bits = 0 if condition == 'sham' else min(511, int(((cap - uncertainty) / lsb).to_integral_value(rounding=ROUND_FLOOR)))
    if condition == 'stimulus' and bits <= 0: raise ValueError('No nonzero DAC level under frozen 1.2xP90 cap')
    nominal = bits * lsb
    upper = Decimal(0) if condition == 'sham' else nominal + uncertainty
    if upper > cap: raise ValueError('Output upper bound exceeds cap')
    return StimulusPlan(baseline.session_id, baseline.chip_id, baseline.well_id, baseline.channel_map_id,
                        baseline.baseline_id, baseline.electrode_id, event_id, context, path.path_id, path.calibration_id,
                        condition, reference_frame, reference_frame + int(offset_frames), delay, parameters.phase_us, bits,
                        (512 - bits, 512 + bits, 512), float(nominal), float(upper), float(cap), parameters)


def delay_commands(samples):
    """Split a nominal hardware delay at the documented 16-bit bound."""
    integer(samples, 'delay samples')
    if samples > 144_000_000: raise ValueError('Delay exceeds two-hour session')
    full, remainder = divmod(samples, 65535)
    return ([dict(kind='delay_samples', samples=65535) for _ in range(full)]
            + ([dict(kind='delay_samples', samples=remainder)] if remainder else []))


def compile_sequence(plan):
    """Return inert descriptions, never SDK objects (constructors may connect).

    reference_frame denotes virtual sequence start, not host send time. The
    default Sequence delay is INCLUDED in the requested 100/200/300 ms code.
    A real backend must attest stimulus_frame from device data independently.
    """
    if not isinstance(plan, StimulusPlan) or plan.phase_us != plan.parameters.phase_us:
        raise ValueError('Only the registered 200 us per-phase plan is supported')
    integer(plan.dac_offset, 'DAC offset')
    if (plan.dac_offset > 511 or plan.dac_codes != (512-plan.dac_offset, 512+plan.dac_offset, 512)
            or plan.condition not in ('stimulus', 'sham')
            or (plan.dac_offset == 0) != (plan.condition == 'sham')):
        raise ValueError('DAC codes do not match a bounded positive-first biphasic plan')
    parameters = plan.parameters
    initial = parameters.initial_delay_samples
    total_delay = plan.code_delay_ms * 1000 // parameters.time_step_us
    if plan.code_delay_ms not in (100, 200, 300) or total_delay < initial:
        raise ValueError('Unregistered time code')
    commands = delay_commands(total_delay - initial)
    commands.append(dict(kind='event', event_id=plan.event_id))
    phase_steps = parameters.phase_us // parameters.time_step_us
    commands.extend([dict(kind='dac', channel=parameters.dac_channel, code=plan.dac_codes[0]),
                     dict(kind='delay_samples', samples=phase_steps),
                     dict(kind='dac', channel=parameters.dac_channel, code=plan.dac_codes[1]),
                     dict(kind='delay_samples', samples=phase_steps),
                     dict(kind='dac', channel=parameters.dac_channel, code=512)])
    return dict(schema='mea.inert_sequence.v1', simulation_only=True, voltage_mode=True,
                time_step_us=parameters.time_step_us, initial_delay_samples=initial, commands=commands,
                total_duration_us=plan.code_delay_ms*1000 + 2*parameters.phase_us)
