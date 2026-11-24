"""Pure-memory engineering fixtures, with no biological validity claim."""
from agent.mea_protocol import integer
from agent.mea_stimulation import compile_sequence


class Simulator:
    def __init__(self, baseline):
        self.baseline = baseline

    def capture(self, plan):
        baseline = self.baseline
        if (plan.baseline_id != baseline.baseline_id or plan.session_id != baseline.session_id
                or plan.chip_id != baseline.chip_id):
            raise ValueError('Simulator baseline identity mismatch')
        rate = baseline.sample_rate_hz
        stimulation_channel = next(c for c, e in baseline.channels if e == plan.electrode_id)
        analysis = [c for c, e in baseline.channels if e != plan.electrode_id]
        # The time-code response below is a deliberately artificial transfer
        # function, not a fitted organoid model or an efficacy prediction.
        count = 0 if plan.condition == 'sham' else 3 + plan.code_delay_ms // 100
        spikes = [[plan.stimulus_frame, analysis[0], -999.],
                  [plan.stimulus_frame + rate * 30 // 1000, stimulation_channel, -888.]]
        spikes += [[plan.stimulus_frame + rate * (40 + i * 10) // 1000,
                    analysis[i % len(analysis)], -float(20 + i)] for i in range(count)]
        packets = [
            dict(type='session', protocol='mea.spikes.v1', source='simulation',
                 sample_rate_hz=rate, well_id=baseline.well_id, start_frame=plan.reference_frame,
                 channel_map_id=baseline.channel_map_id, channels=[list(p) for p in baseline.channels]),
            dict(type='stimulus', event_id=plan.event_id, frame=plan.stimulus_frame,
                 context_id=plan.context_id, pattern_id=plan.plan_id),
            dict(type='spikes', events=spikes),
            dict(type='watermark', end_frame=plan.stimulus_frame + rate // 5, complete=True, dropped=0),
        ]
        for seq, packet in enumerate(packets): packet.update(seq=seq, session_id=baseline.session_id)
        receipt = {name: getattr(plan, name) for name in ('session_id', 'chip_id', 'well_id', 'channel_map_id',
                   'baseline_id', 'event_id', 'context_id', 'plan_id')}
        receipt.update(source='simulation', status='simulated_' + plan.condition,
                       actual_frame=plan.stimulus_frame, peak_upper_mv=plan.peak_upper_mv,
                       artifact_recovery_ms=15, recovered=True, saturated=False)
        return dict(receipt=receipt, packets=packets, sequence=compile_sequence(plan))


class State2Probe:
    """Deterministic action-port probe. This is explicitly not an LLM."""
    action_source = 'state2_probe_not_llm'

    def choose(self, prepared):
        actions = [action for branch in prepared['branches'] for action in branch]
        if not actions: raise ValueError('No legal actions')
        count = prepared['wetware_features'][0]
        if count < 0 or count != int(count): raise ValueError('Probe requires identity-normalized counts')
        return dict(actions[integer(int(count), 'spike count') % len(actions)])
