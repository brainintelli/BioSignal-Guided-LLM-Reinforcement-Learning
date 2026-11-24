"""Frozen digital experiment; no hardware commands are issued or attested."""
from dataclasses import asdict
import math
import random
from mea_binary_r2.encoding import Encoder
from mea_binary_r2.contract import TransitionGate
from mea_binary_r2.offline import simulated_evidence
from mea_binary_r2.profile import PulseProfile

PIN = 'fe86f1e84f4b3239f2f02a7e5fb140503cfac02e8b4bc0587f76e0fbf753ff61'
ARMS = ('A', 'B', 'C', 'D')
GAMMA = .99985
BETA = .001
PROFILE = PulseProfile(schema='mea.pulse_profile.v1', profile_id='h5-22021-000014-8ba000037b05',
    source_sha256='8ba000037b053f47e70c9576808b9d6f2e3609bf6d85dd8e2ffa55496f465771',
    target_mv=200., phase_us=200, pulse_interval_us=10000, pulses_per_stimulus=3, polarity='positive_first')
PROTOCOL = dict(schema='mea.state2_r2.digital.v1', arms=list(ARMS), blocks=8, train_decisions=384,
    rounds=3, decisions_per_round=128, eval_decisions=150, source_pin=PIN,
    normalization_phase='before', normalization_range=[0, 385], training_phase='after',
    train_range=[0, 385], gap=[385, 445], eval_range=[445, 596], beta=BETA, gamma_tick=GAMMA,
    primary_context='after', primary_metric='r1_sum', primary_comparison='C_minus_B',
    other_comparisons=['C_minus_A', 'C_minus_A0', 'C_minus_D', 'B_minus_A'],
    state2_source='historical_28080', r2_source='simulation_from_game_state',
    real_output_verified=False, hardware_commands_sent=0, realtime_training=False,
    biological_replications=1, pulse_profile=asdict(PROFILE), inference='digital_seeds_conditional_on_one_recording')


def cell(block, arm):
    if type(block) is not int or not 0 <= block < 8 or arm not in (*ARMS, 'A0'):
        raise ValueError('Unregistered cell')
    return dict(block=block, arm=arm, initial_seed=6100000, world_seed=11000100 + 1000 * block,
                action_seed=12000100 + 10000 * block, ppo_seed=13000100 + 100 * block,
                eval_world_seed=14000100 + block, eval_action_seed=15000100 + 10000 * block)


def assignment(block, arm, split):
    cell(block, arm)
    if split not in ('train', 'eval'):
        raise ValueError('Unregistered split')
    lo, hi = (0, 384) if split == 'train' else (445, 595)
    order = list(range(lo, hi))
    if arm == 'D':
        rng = random.Random(17000100 + block + (10000 if split == 'eval' else 0))
        while True:
            rng.shuffle(order)
            if all(i != v for i, v in zip(range(lo, hi), order)):
                break
    return order + [hi]


class Rewards:
    def __init__(self, enabled):
        if type(enabled) is not bool:
            raise ValueError('Explicit reward enable flag required')
        self.enabled = enabled
        self.index = 0
        self.episode = 0
        self.encoder = Encoder(pulses_per_stimulus=3)
        self.binding = dict(session_id='SIMULATION-ONLY', chip_id='SIMULATION-ONLY',
                            organoid_id='SIMULATION-ONLY', route_id='SIMULATION-ONLY')
        self.gate = TransitionGate(self.binding)

    def score(self, r1, duration, terminal, reserves, context_id):
        if not math.isfinite(r1) or type(duration) is not int or duration < 1 or not context_id:
            raise ValueError('Finite reward/positive game duration/context required')
        intent = self.encoder.decide(reserves, terminal)
        identity = self.binding | dict(episode_id=str(self.episode), transition_id=str(self.index), context_id=context_id)
        receipt = None if terminal else simulated_evidence(identity, intent, 100000 + 400000 * self.index,
                        'event-' + str(self.index), 'window-' + str(self.index), profile=PROFILE)
        result = self.gate.accept(r1=r1, duration=duration, gamma=GAMMA, beta=BETA if self.enabled else 0.,
                    intent=intent, evidence=receipt, expected_identity=identity, terminal=terminal, profile=PROFILE)
        if not result['valid']:
            raise ValueError('Simulation evidence rejected: ' + str(result['reasons']))
        result.update(r2_enabled=self.enabled, r2_source='simulation_from_game_state' if self.enabled else 'disabled',
                      real_output_verified=False, hardware_commands_sent=0, intent=asdict(intent), simulation_receipt=receipt)
        if not self.enabled:
            result.update(r2_raw=None, r2_train=0., r2_contribution=0., total_reward=r1)
        self.index += 1
        if terminal:
            self.episode += 1
            self.encoder = Encoder(pulses_per_stimulus=3)
        return result
