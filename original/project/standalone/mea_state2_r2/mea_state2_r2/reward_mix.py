"""Magnitude calibration and a separate mixing coefficient for recorded-event rewards."""
import math
import json
from pathlib import Path


def recording_mix(replay_sha256,registered=None):
    if registered is None:
        frozen=json.loads(Path(__file__).with_name('22554_reward_mix.json').read_bytes())
        if frozen['replay_sha256']!=replay_sha256:
            raise ValueError('This recording requires its own R1 calibration')
        registered=frozen['mix']
    return validate_mix(registered)


def validate_mix(mix):
    if not isinstance(mix,dict) or mix.get('version')!='r1-relative-v1':
        raise ValueError('A frozen R1-relative reward calibration is required')
    if (mix.get('r1_weight')!=1 or not 0<mix['amplitude_fraction']<1 or
            not 0<mix['r2_weight']<1 or not math.isfinite(mix['reference_per_tick'])
            or mix['reference_per_tick']<=0):
        raise ValueError('Invalid calibration or mixing coefficients')
    return mix


def discounted_ticks(ticks,gamma):
    if type(ticks) is not int or ticks<1 or not 0<gamma<1:
        raise ValueError('Invalid duration or tick discount')
    return math.expm1(ticks*math.log(gamma))/math.expm1(math.log(gamma))


def calibrated_reward(r1,sign,ticks,gamma,mix):
    if not math.isfinite(r1) or sign not in (-1,1,None):
        raise ValueError('Invalid native reward or delivery label')
    if sign is None:return 0.,0.
    reference=mix['reference_per_tick']*discounted_ticks(ticks,gamma)
    scaled=sign*mix['amplitude_fraction']*min(abs(r1),reference)
    return scaled,mix['r2_weight']*scaled
