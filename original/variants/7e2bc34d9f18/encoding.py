"""Pure game-state encoding and DAC command descriptions, never device I/O."""
from dataclasses import dataclass
from decimal import Decimal
import math

SITES = ('H', 'W', 'F')


def finite(value, name):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(name + ': finite real number required')
    return value


@dataclass(frozen=True)
class Intent:
    site: str | None
    pulses: int
    eligible: bool
    active: tuple

    def __post_init__(self):
        if type(self.eligible) is not bool or type(self.pulses) is not int:
            raise ValueError('Boolean eligibility and integer pulse count required')
        if type(self.active) is not tuple or any(x not in SITES for x in self.active):
            raise ValueError('Known immutable active sites required')
        if tuple(x for x in SITES if x in self.active) != self.active:
            raise ValueError('Active sites must be unique and canonically ordered')
        if self.pulses == 0:
            if self.site is not None or self.active:
                raise ValueError('Quiet or terminal intent cannot contain active sites')
        elif self.pulses not in (1, 3) or not self.eligible or self.site not in self.active:
            raise ValueError('Stimulus intent is inconsistent')


class Encoder:
    """One game episode's hysteresis; create a new instance for each episode."""
    def __init__(self):
        self.active = ()

    def decide(self, reserves, terminal=False):
        if type(terminal) is not bool or type(reserves) is not dict or set(reserves) != set(SITES):
            raise ValueError('Exactly H/W/F reserves and boolean terminal required')
        for site in SITES:
            value = finite(reserves[site], site)
            if not 0 <= value <= 100:
                raise ValueError('Reserves must be normalized to 0..100')
        if terminal:
            self.active = ()
            return Intent(None, 0, False, ())
        active = tuple(site for site in SITES
                       if reserves[site] < (90 if site in self.active else 85))
        if active:
            site = min(active, key=lambda x: (reserves[x], SITES.index(x)))
            intent = Intent(site, 3 if reserves[site] <= 50 else 1, True, active)
        else:
            intent = Intent(None, 0, True, ())
        self.active = active
        return intent


def compile_sequence(intent, lsb_mv):
    """Return immutable ('dac'|'delay', value) pairs, with delays in 50 us steps.

    These are command descriptions, not an armed MaxLab Sequence. Identity,
    routing, markers, deadline and real output verification are separate.
    """
    if type(intent) is not Intent:
        raise ValueError('Validated immutable Intent required')
    finite(lsb_mv, 'DAC LSB')
    if lsb_mv <= 0:
        raise ValueError('DAC LSB must be positive')
    bits = int(Decimal(300) / Decimal(str(lsb_mv)))
    if not 1 <= bits <= 511:
        raise ValueError('Target amplitude cannot be represented')
    pulse = (('dac', 512-bits), ('delay', 4), ('dac', 512+bits),
             ('delay', 4), ('dac', 512))
    result = []
    for index in range(intent.pulses):
        if index:
            result.append(('delay', 5992))
        result.extend(pulse)
    return tuple(result)
