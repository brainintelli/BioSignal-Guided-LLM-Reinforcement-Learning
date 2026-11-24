"""Unified wetware & control signal encoders for Stage 0, Stage 3, and Stage 4 training.

Strict causal isolation contract:
- Encoders accept ONLY observable game sensory fields (`x`, `y`, `health`, `thirst`, `hunger`,
  `berries`, `nearest_water_dist`, `nearest_food_dist`, `threat_nearby`, `nearest_threat_dist`).
  They NEVER accept `visited_count`, `r1`, `reward`, `terminated`, or `next_action`.
- All signal sources (`none`, `ideal`, `noisy`, `wetware_real`, `wetware_shuffled`,
  `wetware_time_shuffled`, `wetware_inverted`) share the exact same interface and feed
  `tools.r2_guard.shape_calibrated` / `tools.r2_guard.shape_for_decoupled_gae`.
- Three modes are supported:
  1) `explore`: 4x4 tile patch (`rx = x // 4, ry = y // 4`) -> site `('H','W','F')[(rx+ry)%3]`
     -> recent-8 habituation bin (`low`, `mid`, `high`) -> positive novelty signal.
  2) `homeo`: observable physiological deficit (`health < 60` or active threat -> `H`;
     `thirst < 35` -> `W`; `hunger < 35` -> `F`; else `Z` sham) with immediate observable
     deficit-relief switching when drinking, eating/gathering, or approaching visible water/food.
  3) `dual`: stress regime gates off exploration bonus and applies `homeo` aversion/relief;
     safe regime applies `explore` novelty.
"""
from dataclasses import dataclass
import math
import random

import numpy as np

from . import design

PATCH_TILES = 4
HISTORY = design.HISTORY  # 8 recent decisions
HEALTH_ALARM = 60.0
THIRST_ALARM = 35.0
HUNGER_ALARM = 35.0
CRITICAL_RESERVE = 20.0

IDEAL_HABITUATION = {'low': 1.0, 'mid': 0.0, 'high': -1.0}
IDEAL_EVOKED = {'real': 1.0, 'sham': -1.0}
MODES = ('explore', 'homeo', 'dual')
SOURCES = (
    'none',
    'ideal',
    'noisy',
    'wetware_real',
    'wetware_shuffled',
    'wetware_time_shuffled',
    'wetware_inverted',
)


def _nearest_visible_kind_dist(obs, kinds):
    player = obs.get('player', {})
    px, py = float(player.get('x', 0)), float(player.get('y', 0))
    best = None
    for item in obs.get('nearby') or ():
        if not isinstance(item, dict) or not item.get('available', True):
            continue
        if item.get('kind') in kinds and 'x' in item and 'y' in item:
            d = abs(float(item['x']) - px) + abs(float(item['y']) - py)
            if best is None or d < best:
                best = d
    return best


def _extract_threat_info(obs):
    player = obs.get('player', {})
    px, py = float(player.get('x', 0)), float(player.get('y', 0))
    threat_nearby = False
    best_dist = None
    for item in obs.get('threats') or obs.get('enemies') or ():
        if isinstance(item, dict):
            if item.get('alive', True) is False:
                continue
            if 'distance' in item:
                d = float(item['distance'])
            elif 'x' in item and 'y' in item:
                d = abs(float(item['x']) - px) + abs(float(item['y']) - py)
            else:
                threat_nearby = True
                continue
            if best_dist is None or d < best_dist:
                best_dist = d
            if d <= 6.0:
                threat_nearby = True
    return threat_nearby, best_dist


@dataclass(frozen=True)
class SensoryState:
    """Strict allowlist of publicly observable sensory fields allowed into stimulus encoding."""
    x: int
    y: int
    health: float
    thirst: float
    hunger: float
    threat_nearby: bool = False
    berries: int = 0
    nearest_water_dist: float | None = None
    nearest_food_dist: float | None = None
    nearest_threat_dist: float | None = None

    @classmethod
    def from_observation(cls, obs):
        """Extract strictly observable fields; ignores visited_count, reward, terminated, etc."""
        player = obs['player']
        inv = obs.get('inventory') or {}
        threat_nearby, threat_dist = _extract_threat_info(obs)
        return cls(
            x=int(player['x']),
            y=int(player['y']),
            health=float(player['health']),
            thirst=float(player['thirst']),
            hunger=float(player['hunger']),
            threat_nearby=threat_nearby,
            berries=int(inv.get('berries', 0)),
            nearest_water_dist=_nearest_visible_kind_dist(obs, ('water',)),
            nearest_food_dist=_nearest_visible_kind_dist(obs, ('berry', 'mushroom')),
            nearest_threat_dist=threat_dist,
        )


def patch_of(x, y, patch_tiles=PATCH_TILES):
    """Map tile coordinates (x, y) to micro-region (rx, ry) of width `patch_tiles`."""
    if patch_tiles < 1:
        raise ValueError('patch_tiles must be >= 1')
    return math.floor(x / patch_tiles), math.floor(y / patch_tiles)


def spatial_site(x, y, patch_tiles=PATCH_TILES):
    """Neighbouring 4x4 micro-regions always map to different stimulation sites."""
    rx, ry = patch_of(x, y, patch_tiles=patch_tiles)
    return design.SITES[(rx + ry) % len(design.SITES)]


def homeostatic_site(state: SensoryState):
    """Return ('H'|'W'|'F') when under physiological/threat stress, else 'Z' (sham)."""
    if state.health < HEALTH_ALARM or state.threat_nearby:
        return 'H'
    if state.thirst < THIRST_ALARM:
        return 'W'
    if state.hunger < HUNGER_ALARM:
        return 'F'
    return design.SHAM


def _observable_deficit_relieved(prev_state: SensoryState, curr_state: SensoryState):
    """Check whether a step transition actively relieved or approached relief for an active alarm.

    Uses ONLY public sensory fields on `(prev_state, curr_state)`. Returns `(was_in_alarm, alarm_site, relieved)`.
    """
    # Priority 1: Health / active threat / acute critical reserve (<20)
    acute_danger = (
        prev_state.health < HEALTH_ALARM
        or prev_state.threat_nearby
        or prev_state.thirst < CRITICAL_RESERVE
        or prev_state.hunger < CRITICAL_RESERVE
    )
    if acute_danger:
        escaped_threat = prev_state.threat_nearby and (
            not curr_state.threat_nearby
            or (
                prev_state.nearest_threat_dist is not None
                and curr_state.nearest_threat_dist is not None
                and curr_state.nearest_threat_dist > prev_state.nearest_threat_dist
            )
        )
        healed = curr_state.health > prev_state.health
        # If critical reserve (<20) triggered danger, drinking/eating or approaching resource relieves it
        thirst_rel = prev_state.thirst < CRITICAL_RESERVE and (
            curr_state.thirst > prev_state.thirst
            or (
                prev_state.nearest_water_dist is not None
                and curr_state.nearest_water_dist is not None
                and curr_state.nearest_water_dist < prev_state.nearest_water_dist
            )
        )
        hunger_rel = prev_state.hunger < CRITICAL_RESERVE and (
            curr_state.hunger > prev_state.hunger
            or curr_state.berries > prev_state.berries
            or (
                prev_state.nearest_food_dist is not None
                and curr_state.nearest_food_dist is not None
                and curr_state.nearest_food_dist < prev_state.nearest_food_dist
            )
        )
        site = (
            'H' if (prev_state.health < HEALTH_ALARM or prev_state.threat_nearby)
            else ('W' if prev_state.thirst <= prev_state.hunger else 'F')
        )
        return True, site, bool(escaped_threat or healed or thirst_rel or hunger_rel)

    # Priority 2: Thirst alarm (<35)
    if prev_state.thirst < THIRST_ALARM:
        closer_water = (
            prev_state.nearest_water_dist is not None
            and curr_state.nearest_water_dist is not None
            and curr_state.nearest_water_dist < prev_state.nearest_water_dist
        )
        drank = curr_state.thirst > prev_state.thirst
        return True, 'W', bool(drank or closer_water)

    # Priority 3: Hunger alarm (<35)
    if prev_state.hunger < HUNGER_ALARM:
        closer_food = (
            prev_state.nearest_food_dist is not None
            and curr_state.nearest_food_dist is not None
            and curr_state.nearest_food_dist < prev_state.nearest_food_dist
        )
        ate_or_gathered = curr_state.hunger > prev_state.hunger or curr_state.berries > prev_state.berries
        return True, 'F', bool(ate_or_gathered or closer_food)

    return False, design.SHAM, False


class EpisodeContext:
    """Tracks recent-site history for one episode without accessing any hidden outcome fields."""

    def __init__(self, patch_tiles=PATCH_TILES, history=HISTORY):
        self.patch_tiles = int(patch_tiles)
        self.history = int(history)
        self.recent_sites = []

    def reset(self):
        self.recent_sites.clear()

    def encode_step(self, state: SensoryState, mode: str, *, prev_state: SensoryState | None = None):
        """Return a pure stimulus-context descriptor for the current sensory state (and optional transition)."""
        if mode not in MODES:
            raise ValueError('Unknown mode: %r' % (mode,))
        s_site = spatial_site(state.x, state.y, patch_tiles=self.patch_tiles)
        count = self.recent_sites.count(s_site)
        self.recent_sites = (self.recent_sites + [s_site])[-self.history:]
        hab_bin = design.bin_of(count)
        h_site = homeostatic_site(state)
        in_stress = h_site != design.SHAM
        relieved = False

        if mode == 'explore':
            channel = 'habituation'
            site = s_site
            bin_name = hab_bin
        elif prev_state is not None:
            was_alarm, alarm_site, relieved = _observable_deficit_relieved(prev_state, state)
            if was_alarm:
                in_stress = True
                channel = 'evoked'
                if relieved:
                    site = design.SHAM
                    bin_name = 'sham'
                else:
                    site = alarm_site
                    bin_name = 'real'
            elif in_stress:
                channel = 'evoked'
                site = h_site
                bin_name = 'real'
            elif mode == 'homeo':
                channel = 'evoked'
                site = design.SHAM
                bin_name = 'sham'
            else:  # dual in fully safe regime
                channel = 'habituation'
                site = s_site
                bin_name = hab_bin
        else:
            if mode == 'homeo':
                channel = 'evoked'
                site = h_site
                bin_name = 'real' if in_stress else 'sham'
            else:  # dual (single-state snapshot mode)
                if in_stress:
                    channel = 'evoked'
                    site = h_site
                    bin_name = 'real'
                else:
                    channel = 'habituation'
                    site = s_site
                    bin_name = hab_bin

        return dict(
            mode=mode,
            channel=channel,
            site=site,
            bin=bin_name,
            spatial_site=s_site,
            spatial_count=count,
            habituation_bin=hab_bin,
            homeostatic_site=h_site,
            in_stress=in_stress,
            deficit_relieved=relieved,
        )


def _validate_library_v2(lib, kind):
    if lib is None:
        raise ValueError(f'{kind} library is required for wetware signal sources')
    sites = lib.get('sites')
    if not isinstance(sites, dict):
        raise ValueError(f'{kind} library missing sites dict')
    return lib


class WetwareSignalSource:
    """Generates raw unshaped per-step signals for a rollout batch across all 7 experimental arms.

    - `none` (Arm A): returns all zeros.
    - `ideal` (Arm D): noise-free ideal value (+1/0/-1 for habituation; -1/+1 for stress/relief).
    - `noisy` (Arm F): ideal value mixed with independent zero-mean unit-variance Gaussian noise
      so corr(raw, ideal) == `rho` (requires `0 < rho <= 1`).
    - `wetware_real` (Arm B): draws recorded organoid response from the matching site & bin.
      Habituation draws enter positively (+y); evoked stress draws enter negatively (-y), while
      sham/relief draws enter as `-sham_y` (positive relief since sham < 0).
    - `wetware_shuffled` (Arm C): draws from the exact same library after permuting the context
      indices within the batch, preserving marginal distribution while destroying context alignment.
    - `wetware_time_shuffled` (Arm C_time): draws context-aligned responses from the pre-build
      timeline-permuted organoid library (`ctime_*_library`), testing whether real biological
      temporal memory is necessary.
    - `wetware_inverted` (Arm E): strictly negates `wetware_real`.
    """

    def __init__(
        self,
        mode,
        source,
        seed,
        *,
        rho=1.0,
        habituation_library=None,
        evoked_library=None,
        ctime_habituation_library=None,
        ctime_evoked_library=None,
    ):
        if mode not in MODES:
            raise ValueError('Unknown mode: %r' % (mode,))
        if source not in SOURCES:
            raise ValueError('Unknown source: %r' % (source,))
        if source == 'noisy' and not (0.0 < float(rho) <= 1.0):
            raise ValueError('rho must lie in (0, 1] for noisy source')
        self.mode = mode
        self.source = source
        self.rho = float(rho)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        if source == 'wetware_time_shuffled':
            self.hab_lib = ctime_habituation_library if ctime_habituation_library is not None else habituation_library
            self.evk_lib = ctime_evoked_library if ctime_evoked_library is not None else evoked_library
        else:
            self.hab_lib = habituation_library
            self.evk_lib = evoked_library

        if source in ('wetware_real', 'wetware_shuffled', 'wetware_time_shuffled', 'wetware_inverted'):
            if mode in ('explore', 'dual'):
                _validate_library_v2(self.hab_lib, 'habituation')
            if mode in ('homeo', 'dual'):
                _validate_library_v2(self.evk_lib, 'evoked')

    def _ideal_value(self, ctx):
        if ctx['channel'] == 'habituation':
            return IDEAL_HABITUATION[ctx['bin']]
        # Evoked channel: real stress stimulus -> negative aversion (-1); sham safe/relief -> (+1)
        return -IDEAL_EVOKED[ctx['bin']]

    def _draw_wetware(self, ctx):
        if ctx['channel'] == 'habituation':
            usable = set(self.hab_lib.get('usable_sites', design.SITES))
            site = ctx['site']
            if site not in usable or site not in self.hab_lib['sites']:
                return 0.0
            pool = self.hab_lib['sites'][site][ctx['bin']]
            return float(self.rng.choice(pool))
        # Evoked channel: high excitable discharge during stress is aversive (-y); refractory sham is relief (-y > 0)
        usable = set(self.evk_lib.get('usable_sites', design.SITES))
        site = ctx['site']
        if site == design.SHAM or ctx['bin'] == 'sham':
            pool = self.evk_lib['sham']
            return -float(self.rng.choice(pool))
        if site not in usable or site not in self.evk_lib['sites']:
            return 0.0
        pool = self.evk_lib['sites'][site]['real']
        return -float(self.rng.choice(pool))

    def score_batch(self, contexts):
        """Return raw signal array of length len(contexts) before R2 shaping."""
        n = len(contexts)
        if n == 0 or self.source == 'none':
            return np.zeros(n, dtype=float)
        if self.source == 'ideal':
            return np.asarray([self._ideal_value(c) for c in contexts], dtype=float)
        if self.source == 'noisy':
            clean = np.asarray([self._ideal_value(c) for c in contexts], dtype=float)
            clean_std = float(clean.std())
            if self.rho >= 1.0 or clean_std < 1e-12:
                return clean
            z_clean = (clean - clean.mean()) / clean_std
            noise = self.np_rng.standard_normal(n)
            return self.rho * z_clean + math.sqrt(1.0 - self.rho ** 2) * noise
        if self.source in ('wetware_real', 'wetware_time_shuffled'):
            return np.asarray([self._draw_wetware(c) for c in contexts], dtype=float)
        if self.source == 'wetware_inverted':
            return -np.asarray([self._draw_wetware(c) for c in contexts], dtype=float)
        if self.source == 'wetware_shuffled':
            perm = list(range(n))
            self.rng.shuffle(perm)
            return np.asarray([self._draw_wetware(contexts[i]) for i in perm], dtype=float)
        raise AssertionError('unreachable')
