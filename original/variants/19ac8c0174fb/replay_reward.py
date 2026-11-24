"""Scale an offline feedback label in native game-reward units before mixing it."""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ReplayRewardMix:
    reference_per_tick: float
    scale_fraction: float
    alpha: float
    beta: float
    gamma: float

    def __post_init__(self):
        values = (self.reference_per_tick, self.scale_fraction, self.alpha, self.beta, self.gamma)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Non-finite reward calibration')
        if self.reference_per_tick <= 0 or not 0 < self.scale_fraction < 1:
            raise ValueError('Invalid native reward scale')
        if self.alpha <= 0 or self.beta < 0 or self.beta * self.scale_fraction >= self.alpha:
            raise ValueError('The auxiliary component must stay below the native component')
        if not 0 < self.gamma <= 1:
            raise ValueError('Invalid tick discount')

    @classmethod
    def from_study(cls, study, gamma, enabled):
        config = study['reward_mix']
        if config['schema'] != 'bounded_native_r1.v1' or config['gamma_tick'] != gamma:
            raise ValueError('Reward schema or calibration discount mismatch')
        return cls(config['reference_per_tick'], config['scale_fraction'], study['alpha'],
                   study['beta'] if enabled else 0., gamma)

    def score(self, r1, sign, ticks):
        if not math.isfinite(r1) or type(sign) is not int or sign not in (-1, 0, 1):
            raise ValueError('Expected finite native R1 and an explicit feedback label')
        if type(ticks) is not int or ticks < 1:
            raise ValueError('Expected a positive game duration')
        duration = ticks if self.gamma == 1 else math.expm1(ticks*math.log(self.gamma))/math.expm1(math.log(self.gamma))
        scaled = sign * self.scale_fraction * min(abs(r1), self.reference_per_tick * duration)
        contribution = self.beta * scaled
        return scaled, contribution, self.alpha * r1 + contribution
