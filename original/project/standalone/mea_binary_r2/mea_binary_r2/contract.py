"""Strict simulation-only reward and transaction contracts, not hardware proof."""
from copy import deepcopy
import math
from .encoding import Intent, finite
from .profile import resolve_profile

IDENTITY_FIELDS = frozenset(('session_id','chip_id','organoid_id','route_id',
                             'episode_id','transition_id','context_id'))
BINDING_FIELDS = frozenset(('session_id','chip_id','organoid_id','route_id'))
EVIDENCE_FIELDS = frozenset(('source','identity','event_id','window_id','status','sample_rate_hz',
    'anchor_frame','pulse_frames','max_abs_output_mv','output_verified','route_verified',
    'window_start_frame','window_end_frame','coverage_start_frame','coverage_end_frame',
    'complete','dropped_frames','qc_passed'))


def identifier(value):
    if type(value) is not str or not value or len(value)>256:
        raise ValueError('Nonempty bounded identifier required')
    return value


def identity(value, fields=IDENTITY_FIELDS):
    if type(value) is not dict or set(value)!=fields:
        raise ValueError('Exact identity fields required')
    return {key:identifier(item) for key,item in value.items()}


def integer(value, name, minimum=0):
    if type(value) is not int or value<minimum:
        raise ValueError(name+': nonnegative integer required')
    return value


def _receipt(value, expected, profile=None):
    pulse_profile=resolve_profile(profile)
    required=EVIDENCE_FIELDS | ({'stimulus_profile_sha256'} if profile is not None else set())
    if type(value) is not dict or set(value)!=required:
        raise ValueError('Exact receipt schema required')
    if profile is not None and value['stimulus_profile_sha256']!=pulse_profile.digest:
        raise ValueError('Stimulus profile differs from output receipt')
    if value['source']!='simulation':
        raise ValueError('Offline contract accepts simulation only; live verifier not implemented')
    if identity(value['identity'])!=expected:
        raise ValueError('Receipt identity differs from current action context')
    for field in ('event_id','window_id'):identifier(value[field])
    if value['status'] not in ('confirmed','unknown'):
        raise ValueError('Unknown receipt status')
    for field in ('output_verified','route_verified','complete','qc_passed'):
        if type(value[field]) is not bool:raise ValueError('Boolean required: '+field)
    for field in ('anchor_frame','window_start_frame','window_end_frame',
                  'coverage_start_frame','coverage_end_frame','dropped_frames','sample_rate_hz'):
        integer(value[field],field)
    if value['sample_rate_hz']!=20000:raise ValueError('This profile requires 20 kHz')
    frames=value['pulse_frames']
    if type(frames) is not list or len(frames)>3:
        raise ValueError('At most three observed pulse frames required')
    for frame in frames:integer(frame,'pulse frame')
    if sorted(set(frames))!=frames:raise ValueError('Pulse frames must increase strictly')
    if finite(value['max_abs_output_mv'],'peak voltage')<0:
        raise ValueError('Absolute output voltage cannot be negative')
    return deepcopy(value)


def reward_record(*, r1, duration, gamma, beta, intent, evidence, expected_identity, terminal=False, profile=None):
    finite(r1,'R1');integer(duration,'duration',1);finite(gamma,'gamma');finite(beta,'beta')
    if duration>10000 or not 0<gamma<1 or not 0<=beta<=1:
        raise ValueError('Invalid reward duration/discount/weight')
    if type(terminal) is not bool or type(intent) is not Intent:
        raise ValueError('Boolean terminal and immutable Intent required')
    pulse_profile=resolve_profile(profile)
    if profile is not None and intent.pulses not in (0,pulse_profile.pulses_per_stimulus):
        raise ValueError('Intent pulse count differs from explicit profile')
    expected=identity(expected_identity)
    result=dict(schema='mea.binary_r2.offline.v1',source='simulation',identity=deepcopy(expected),
        r1=r1,duration=duration,gamma_tick=gamma,beta=beta,r2_enabled=True,
        r2_raw=None,r2_train=None,r2_contribution=None,total_reward=None,
        eligible=not terminal,valid=False,reasons=[])
    if profile is not None:result['stimulus_profile_sha256']=pulse_profile.digest
    if terminal:
        if intent.eligible or intent.pulses or evidence is not None:
            raise ValueError('Terminal transition must have no eligible output or receipt')
        result.update(valid=True,r2_train=0.,r2_contribution=0.,total_reward=r1,
                      reasons=['terminal_not_eligible'])
        return result
    if not intent.eligible:raise ValueError('Nonterminal transition needs eligible intent')
    receipt=_receipt(evidence,expected,profile)
    result.update(event_id=receipt['event_id'],window_id=receipt['window_id'])
    reasons=result['reasons'];frames=receipt['pulse_frames'];anchor=receipt['anchor_frame']
    known=(receipt['status']=='confirmed' and receipt['output_verified'] and receipt['route_verified'])
    peak=receipt['max_abs_output_mv']
    if not known:
        reasons.append('output_unknown_or_unverified')
    elif frames and peak>0:
        result['r2_raw']=-1
    elif not frames and peak==0 and intent.pulses==0:
        result['r2_raw']=1
    else:
        reasons.append('output_does_not_attest_to_intent')
    if frames!=[anchor+i*(pulse_profile.pulse_interval_us//50) for i in range(intent.pulses)]:
        reasons.append('pulse_plan_mismatch')
    if peak>pulse_profile.target_mv or (not frames and peak!=0) or (frames and peak==0):
        reasons.append('invalid_output_amplitude')
    start,end=receipt['window_start_frame'],receipt['window_end_frame']
    if (start,end)!=(anchor+240000,anchor+260000):
        reasons.append('response_window_timing_mismatch')
    if (not receipt['complete'] or receipt['dropped_frames'] or
            receipt['coverage_start_frame']>anchor or receipt['coverage_end_frame']<end or
            receipt['coverage_start_frame']>=receipt['coverage_end_frame']):
        reasons.append('incomplete_recording_coverage')
    if not receipt['qc_passed']:reasons.append('recovery_qc_failed')
    if not reasons and result['r2_raw'] is not None:
        duration_factor=math.fsum(gamma**tick for tick in range(duration))
        train=result['r2_raw']*duration_factor
        contribution=beta*train
        total=r1+contribution
        finite(total,'total reward')
        result.update(valid=True,r2_train=train,r2_contribution=contribution,total_reward=total)
    return result


class TransitionGate:
    """Reject duplicates and overlapping responses on one simulated timeline."""
    def __init__(self,binding):
        self.binding=identity(binding,BINDING_FIELDS)
        self.stopped=False
        self.transitions=set();self.events=set();self.windows=set();self.previous_end=None
        self.closed_episodes=set()

    def accept(self,**arguments):
        if self.stopped:raise ValueError('Session stopped; no retry or reuse')
        try:
            current=identity(arguments['expected_identity'])
            if any(current[key]!=value for key,value in self.binding.items()):
                raise ValueError('Foreign biological binding')
            if current['episode_id'] in self.closed_episodes:
                raise ValueError('Episode already terminated; a new episode is required')
            key=(current['episode_id'],current['transition_id'])
            if key in self.transitions:raise ValueError('Transition already consumed')
            self.transitions.add(key)
            evidence=arguments['evidence']
            if evidence is not None:
                receipt=_receipt(evidence,current,arguments.get('profile'))
                if receipt['event_id'] in self.events or receipt['window_id'] in self.windows:
                    raise ValueError('Event or response window already consumed')
                if self.previous_end is not None and receipt['anchor_frame']<self.previous_end:
                    raise ValueError('New event overlaps previously consumed response')
                self.events.add(receipt['event_id']);self.windows.add(receipt['window_id'])
            result=reward_record(**arguments)
            if not result['valid']:
                self.stopped=True
            elif arguments.get('terminal') is True:
                self.closed_episodes.add(current['episode_id'])
            elif evidence is not None:
                self.previous_end=evidence['window_end_frame']
            return result
        except (ValueError,TypeError,KeyError,OverflowError):
            self.stopped=True
            raise


class AttemptBudget:
    """In-memory engineering model only; never replaces a durable live ledger."""
    def __init__(self,started_wall,clock):
        self.clock=clock
        now=finite(clock.time(),'wall time');mono=finite(clock.monotonic(),'monotonic time')
        finite(started_wall,'original start')
        if mono<0 or not 0<=now-started_wall<7200:
            raise ValueError('Original biological start is future/expired')
        self.cutoff_wall=started_wall+5700
        self.cutoff_mono=mono+self.cutoff_wall-now
        self.hard_deadline_wall=started_wall+7200
        self.last_wall=now;self.last_mono=mono
        self.next_slot_mono=mono
        self.events=0;self.pulses=0;self.pending=False;self.stopped=False

    def _remaining(self):
        now=finite(self.clock.time(),'wall time');mono=finite(self.clock.monotonic(),'monotonic time')
        if now<self.last_wall or mono<self.last_mono:
            raise ValueError('Clock regression')
        self.last_wall=now;self.last_mono=mono
        return min(self.cutoff_wall-now,self.cutoff_mono-mono)

    def admit(self,intent):
        if self.stopped:raise ValueError('Budget stopped; never restart it')
        try:
            if self.pending:raise ValueError('Only one outstanding event is allowed')
            if type(intent) is not Intent or not intent.eligible:
                raise ValueError('Only eligible intents may consume an output slot')
            if self._remaining()<20:raise ValueError('Insufficient time before recovery reserve')
            if self.last_mono<self.next_slot_mono:
                raise ValueError('Minimum 20 second observation cycle not complete')
            if self.events>=30 or self.pulses>=90 or self.pulses+intent.pulses>90:
                raise ValueError('Dose/attempt budget exhausted; no quiet bonus')
            if intent.pulses:
                self.events+=1;self.pulses+=intent.pulses
            self.pending=True
            self.next_slot_mono=self.last_mono+20
            return dict(source='simulation',events=self.events,pulses=self.pulses,
                        hard_deadline_wall=self.hard_deadline_wall)
        except (ValueError,TypeError,OverflowError):
            self.stopped=True
            raise

    def complete(self,confirmed):
        if self.stopped or not self.pending or type(confirmed) is not bool:
            self.stopped=True
            raise ValueError('No pending event or invalid completion')
        try:
            remaining=self._remaining()
        except (ValueError,TypeError,OverflowError):
            self.stopped=True
            raise
        self.pending=False
        if not confirmed or remaining<0:self.stopped=True
