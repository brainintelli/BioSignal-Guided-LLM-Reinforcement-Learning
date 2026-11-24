"""Simulation-only response/action transaction. No hardware backend is loaded."""
from copy import deepcopy
import random
import time

from agent.mea_contract import encode, observation_key
from agent.mea_protocol import ResearchStream, canonical, digest, fields, integer, number
from agent.mea_snapshot import Normalization, build_snapshot
from agent.mea_stimulation import (Baseline, OutputPath, StimulationParameters, DEFAULT_PARAMETERS,
                                   compile_sequence, decimal, make_plan)

ACTOR_FIELDS = ('messages', 'branches', 'target_texts', 'verb_mask', 'wetware_features')
_CLAIMED_SESSIONS = set()


class LoopStopped(ValueError):
    pass


def actor_input(prepared):
    """Only these fields may cross the actor boundary, never audit metadata."""
    return deepcopy({key: prepared[key] for key in ACTOR_FIELDS})


def response_snapshot(data, plan, baseline):
    fields(data, 'sequence receipt packets')
    if data.get('sequence') != compile_sequence(plan): raise ValueError('Simulated waveform sequence mismatch')
    receipt = data['receipt']
    fields(receipt, 'session_id chip_id well_id channel_map_id baseline_id event_id context_id plan_id '
                    'source status actual_frame peak_upper_mv artifact_recovery_ms recovered saturated')
    for key in ('session_id', 'chip_id', 'well_id', 'channel_map_id', 'baseline_id', 'event_id', 'context_id', 'plan_id'):
        if type(receipt[key]) is not type(getattr(plan, key)) or receipt[key] != getattr(plan, key):
            raise ValueError('Receipt identity mismatch: ' + key)
    if receipt.get('source') != 'simulation' or receipt.get('status') != 'simulated_' + plan.condition:
        raise ValueError('Output state unknown or non-simulation')
    if integer(receipt.get('actual_frame'), 'actual_frame') != plan.stimulus_frame:
        raise ValueError('Actual stimulus frame differs from plan')
    peak = decimal(receipt.get('peak_upper_mv'), 'peak upper bound')
    if peak != decimal(plan.peak_upper_mv, 'planned peak upper bound') or peak > decimal(plan.cap_mv, 'cap'):
        raise ValueError('Output peak differs or exceeds cap')
    recovery = number(receipt.get('artifact_recovery_ms'), 'artifact recovery')
    if receipt['saturated'] is not False or receipt.get('recovered') is not True or not 0 <= recovery <= 15:
        raise ValueError('Artifact/recovery gate failed')
    packets = data['packets']
    if not isinstance(packets, list) or not 3 <= len(packets) <= 64:
        raise ValueError('Bounded raw packet evidence required')
    stream = ResearchStream(max_spikes=100000)
    for packet in packets: stream.feed((canonical(packet) + '\n').encode('utf-8'))
    stream.finish()
    session = stream.session
    if (session is None or session.source != 'simulation' or session.session_id != plan.session_id
            or session.well_id != baseline.well_id or session.sample_rate_hz != baseline.sample_rate_hz
            or session.start_frame != plan.reference_frame or session.channel_map_id != baseline.channel_map_id
            or session.channels != baseline.channels):
        raise ValueError('Raw session/mapping mismatch')
    if len(stream.stimuli) != 1: raise ValueError('One unique event per response transaction required')
    event = stream.stimulus(plan.event_id)
    if (event.frame, event.context_id, event.pattern_id) != (plan.stimulus_frame, plan.context_id, plan.plan_id):
        raise ValueError('Raw event does not attest to the requested stimulus')
    # Keep the original stream in the audit. Build a separate immutable
    # analysis view, whose identity explicitly binds the original map and mask.
    channels = [(c, e) for c, e in baseline.channels if e != plan.electrode_id]
    included = {c for c, _ in channels}
    analysis_map = digest(dict(raw_channel_map_id=baseline.channel_map_id, analysis_channels=channels))
    analysis = ResearchStream(max_spikes=100000)
    rows = [dict(type='session', protocol='mea.spikes.v1', source='simulation',
                 sample_rate_hz=session.sample_rate_hz, well_id=session.well_id, start_frame=session.start_frame,
                 channel_map_id=analysis_map, channels=[list(p) for p in channels]),
            dict(type='stimulus', event_id=event.event_id, frame=event.frame,
                 context_id=event.context_id, pattern_id=event.pattern_id),
            dict(type='spikes', events=[list(s) for s in stream.spikes if s[1] in included]),
            dict(type='watermark', end_frame=stream.watermark, complete=True, dropped=0)]
    for seq, row in enumerate(rows):
        row.update(session_id=plan.session_id, seq=seq)
        analysis.feed((canonical(row) + '\n').encode('utf-8'))
    analysis.finish()
    rate = baseline.sample_rate_hz
    if rate % 1000: raise ValueError('Response timing requires an integral millisecond grid')
    return build_snapshot(analysis, event.event_id, blank_frames=rate * 20 // 1000,
                          window_frames=rate * 180 // 1000, normalization=Normalization.identity())


class ClosedLoop:
    def __init__(self, baseline, path, *, clock=time.monotonic, max_seconds=7200, reserve_seconds=900,
                 max_pulses=25, max_shams=12, parameters=DEFAULT_PARAMETERS, _read_only_replay=False):
        if not isinstance(baseline, Baseline) or not isinstance(path, OutputPath): raise ValueError('Frozen inputs required')
        if not isinstance(parameters, StimulationParameters): raise ValueError('Frozen parameter profile required')
        integer(max_seconds, 'max_seconds', 1); integer(reserve_seconds, 'reserve_seconds', 900)
        integer(max_pulses, 'max_pulses'); integer(max_shams, 'max_shams')
        if max_seconds > 7200 or reserve_seconds >= max_seconds or max_pulses > 25 or max_shams > 12:
            raise ValueError('Session budget exceeds registered limits')
        self.baseline, self.path, self.clock = baseline, path, clock
        self.started = number(clock(), 'monotonic start'); self.last_clock = self.started
        if type(_read_only_replay) is not bool: raise ValueError('Explicit replay mode required')
        self._read_only_replay = _read_only_replay
        if not _read_only_replay:
            if baseline.session_id in _CLAIMED_SESSIONS:
                raise ValueError('This session already has a lifecycle; never reset its budget')
            _CLAIMED_SESSIONS.add(baseline.session_id)
        self.cutoff_seconds = max_seconds - reserve_seconds
        self.max_pulses, self.max_shams = max_pulses, max_shams
        self.reference_frame = baseline.end_frame
        self.parameters = parameters
        rng = random.Random(parameters.interval_seed)
        self.onset_intervals_ms = tuple(rng.randint(parameters.onset_interval_min_ms, parameters.onset_interval_max_ms)
                                       for _ in range(max(0, max_pulses + max_shams - 1)))
        self.previous_stimulus_frame = None
        self.pulse_attempts = self.sham_attempts = self.index = 0
        self.state, self.audit = 'READY', []

    @classmethod
    def for_replay(cls, baseline, path, **kwargs):
        """Explicit read-only reconstruction, limited to built-in simulation ports."""
        return cls(baseline, path, _read_only_replay=True, **kwargs)

    def _deadline(self, frame):
        now = number(self.clock(), 'monotonic clock')
        if (now < self.last_clock or now - self.started >= self.cutoff_seconds
                or (frame - self.baseline.start_frame) / self.baseline.sample_rate_hz >= self.cutoff_seconds):
            raise LoopStopped('Session deadline or monotonic clock regression')
        self.last_clock = now

    def transact(self, observation, backend, actor, condition='stimulus'):
        if self.state != 'READY': raise ValueError('Session is not reusable after stop/failure')
        evidence = dict(index=self.index, condition=condition)
        self.audit.append(evidence)
        try:
            if self._read_only_replay:
                from agent.mea_simulator import Simulator, State2Probe
                if type(backend) is not Simulator or type(actor) is not State2Probe:
                    raise ValueError('Read-only replay admits only built-in simulator and probe')
            self._deadline(self.reference_frame)
            if condition not in ('stimulus', 'sham'): raise ValueError('Unknown condition')
            if ((condition == 'stimulus' and self.pulse_attempts >= self.max_pulses)
                    or (condition == 'sham' and self.sham_attempts >= self.max_shams)):
                raise LoopStopped('Attempt budget exhausted')
            interval = self.onset_intervals_ms[self.index - 1] if self.index else None
            plan = make_plan(observation, self.baseline, self.path, f'event-{self.index:04d}', self.reference_frame,
                             condition, parameters=self.parameters, previous_stimulus_frame=self.previous_stimulus_frame,
                             onset_interval_ms=interval)
            evidence['plan'] = plan.to_dict()
            evidence['timing'] = dict(onset_interval_ms=interval,
                                     post_pulse_delay_samples=(self.parameters.post_pulse_delay_samples(interval)
                                                              if interval is not None else None))
            end = plan.stimulus_frame + self.baseline.sample_rate_hz // 5
            self._deadline(end)
            self.state = 'PENDING'
            # Count before submitting; unknown outcomes are never retried.
            if condition == 'stimulus': self.pulse_attempts += 1
            else: self.sham_attempts += 1
            data = backend.capture(plan)
            evidence.update(deepcopy(data))
            self._deadline(end)
            if observation_key(observation, 'reserves') != plan.context_id:
                raise ValueError('Current observation changed while pending')
            snapshot = response_snapshot(data, plan, self.baseline)
            self.state = 'RESPONSE_VALID'
            prepared = encode(observation, snapshot, 'reserves')
            projection = actor_input(prepared)
            evidence.update(snapshot=snapshot.to_dict(), prepared=prepared, actor_input=deepcopy(projection))
            action = actor.choose(deepcopy(projection))
            self._deadline(end)
            if observation_key(observation, 'reserves') != plan.context_id:
                raise ValueError('Current observation changed during inference')
            if action not in [a for branch in prepared['branches'] for a in branch]:
                raise ValueError('Actor returned an illegal action')
            self.state = 'ACTION_READY'
            evidence['action'] = deepcopy(action)
            self.previous_stimulus_frame = plan.stimulus_frame
            self.reference_frame = end
            self.index += 1
            self.state = 'READY'
            return deepcopy(evidence)
        except Exception as exc:
            self.state = 'STOPPED' if isinstance(exc, LoopStopped) else 'INVALID'
            evidence.update(error_type=type(exc).__name__, error=str(exc), state=self.state)
            raise
