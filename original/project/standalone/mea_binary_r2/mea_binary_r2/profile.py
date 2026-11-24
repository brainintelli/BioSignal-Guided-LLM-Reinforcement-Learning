"""Immutable, explicit pulse descriptions; never device authorization."""
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import re


@dataclass(frozen=True)
class PulseProfile:
    schema: str
    profile_id: str
    source_sha256: str
    target_mv: float
    phase_us: int
    pulse_interval_us: int
    pulses_per_stimulus: int
    polarity: str

    def __post_init__(self):
        if self.schema != 'mea.pulse_profile.v1':
            raise ValueError('Unsupported pulse profile schema')
        if type(self.profile_id) is not str or not 1 <= len(self.profile_id) <= 128:
            raise ValueError('Bounded profile ID required')
        if type(self.source_sha256) is not str or not re.fullmatch('[0-9a-f]{64}',self.source_sha256):
            raise ValueError('Source SHA256 required')
        if (type(self.target_mv) not in (int,float) or not math.isfinite(self.target_mv)
                or not 0 < self.target_mv <= 300):
            raise ValueError('Supported target range is greater than0 and at most300mV')
        if type(self.phase_us) is not int or self.phase_us != 200:
            raise ValueError('This version supports200us per phase')
        if (type(self.pulse_interval_us) is not int or self.pulse_interval_us % 50
                or not 2*self.phase_us < self.pulse_interval_us <= 300000):
            raise ValueError('Pulse onset interval must fit50us grid and recovery envelope')
        if type(self.pulses_per_stimulus) is not int or self.pulses_per_stimulus != 3:
            raise ValueError('An explicit profile uses one3pulse feedback group')
        if self.polarity != 'positive_first':
            raise ValueError('Only existing positive-first candidate is implemented')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls,value):
        if type(value) is not dict or set(value) != {f.name for f in fields(cls)}:
            raise ValueError('Exact pulse profile fields required')
        return cls(**value)

    @property
    def digest(self):
        raw=json.dumps(self.to_dict(),sort_keys=True,separators=(',',':'),allow_nan=False)
        return hashlib.sha256(raw.encode()).hexdigest()


LEGACY_PROFILE=PulseProfile('mea.pulse_profile.v1','legacy-300mv-300ms','0'*64,
                           300.,200,300000,3,'positive_first')


def resolve_profile(value):
    if value is None:return LEGACY_PROFILE
    if type(value) is not PulseProfile:raise ValueError('Validated immutable PulseProfile required')
    return value


def load_profile(path):
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    if type(value) is not dict or value.get('schema') != 'mea.stimulus_profile_import.v1':
        raise ValueError('Explicit imported profile document required')
    return PulseProfile.from_dict(value['execution_profile'])
