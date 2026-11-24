"""Fixed-tick exploration, fear-of-thirst/hunger/death, survival guardrails, and paired bootstrap.

Pure Python + NumPy; operates on recorded episode trajectories without touching GPU or hardware.
All endpoints are anchored to a fixed game tick (default Tick 96 for standard maps; Tick 60/180
for high-stress archives) to eliminate action-duration timing confounds.
"""
import math

import numpy as np

THIRST_LOW = 30.0
HUNGER_LOW = 30.0
DANGER_RESERVE = 20.0
DANGER_HEALTH = 50.0
SAFE_RESERVE = 35.0
SAFE_HEALTH = 60.0

GUARDRAIL_SURVIVAL_DROP_MAX = -0.05  # -5 percentage points lower bound redline
WEAK_POSITIVE_SEED_FRAC = 0.70
CLUE_SEED_FRAC = 0.65
CLUE_CONSECUTIVE_CKPTS = 3

# Canonical 4-regime training curriculum (total weight = 10 -> 40% / 20% / 20% / 20%)
CURRICULUM_SCENARIOS = (
    # 40% Standard long-horizon exploration
    dict(scenario_name='C1_standard_a', weight=2),
    dict(scenario_name='C1_standard_b', weight=2),
    # 20% Fear-of-thirst acute + proactive warning
    dict(scenario_name='C2_thirst_acute', initial_health=45.0, initial_thirst=4.0, initial_hunger=85.0, initial_tick=0, weight=1),
    dict(scenario_name='C2_thirst_warn', initial_health=60.0, initial_thirst=22.0, initial_hunger=85.0, initial_tick=0, weight=1),
    # 20% Fear-of-hunger acute + proactive warning
    dict(scenario_name='C3_hunger_acute', initial_health=45.0, initial_thirst=85.0, initial_hunger=6.0, initial_berries=1, initial_tick=0, weight=1),
    dict(scenario_name='C3_hunger_warn', initial_health=60.0, initial_thirst=85.0, initial_hunger=24.0, initial_berries=0, initial_tick=0, weight=1),
    # 20% Fear-of-death night threat (fixes 100% daytime survival ceiling!)
    dict(scenario_name='C4_night_threat', initial_health=18.0, initial_thirst=65.0, initial_hunger=65.0, initial_tick=140, near_threat=True, weight=1),
    dict(scenario_name='C4_dusk_low_hp', initial_health=20.0, initial_thirst=65.0, initial_hunger=65.0, initial_tick=130, weight=1),
)

# Canonical frozen evaluation suites (zero ceiling saturation, fixed elapsed tick cutoffs)
EVAL_SUITES = {
    'E1_explore_96': dict(scenario=dict(scenario_name='E1_explore_96'), horizon_ticks=96),
    'E1_explore_180': dict(scenario=dict(scenario_name='E1_explore_180'), horizon_ticks=180),
    'E2_thirst_acute': dict(
        scenario=dict(scenario_name='E2_thirst_acute', initial_health=25.0, initial_thirst=1.6, initial_hunger=95.0, initial_tick=0),
        horizon_ticks=60,
    ),
    'E2_thirst_warn': dict(
        scenario=dict(scenario_name='E2_thirst_warn', initial_health=60.0, initial_thirst=24.0, initial_hunger=85.0, initial_tick=0),
        horizon_ticks=120,
    ),
    'E3_hunger_acute': dict(
        scenario=dict(scenario_name='E3_hunger_acute', initial_health=25.0, initial_thirst=95.0, initial_hunger=4.0, initial_berries=1, initial_tick=0),
        horizon_ticks=60,
    ),
    'E3_hunger_warn': dict(
        scenario=dict(scenario_name='E3_hunger_warn', initial_health=60.0, initial_thirst=85.0, initial_hunger=26.0, initial_berries=0, initial_tick=0),
        horizon_ticks=120,
    ),
    'E4_death_night_threat': dict(
        scenario=dict(scenario_name='E4_death_night_threat', initial_health=18.0, initial_thirst=65.0, initial_hunger=65.0, initial_tick=140, near_threat=True),
        horizon_ticks=70,
    ),
}


def sample_curriculum_scenario(step_index: int) -> dict:
    """Deterministic weighted schedule over the 4 survival/exploration curriculum regimes."""
    expanded = []
    for item in CURRICULUM_SCENARIOS:
        spec = {k: v for k, v in item.items() if k != 'weight'}
        expanded.extend([spec] * int(item['weight']))
    return dict(expanded[int(step_index) % len(expanded)])


def _nearest_threat_dist(obs):
    player = obs.get('player', {})
    px, py = float(player.get('x', 0)), float(player.get('y', 0))
    best = None
    for item in obs.get('threats') or obs.get('enemies') or ():
        if isinstance(item, dict):
            if item.get('alive', True) is False:
                continue
            if 'distance' in item:
                d = float(item['distance'])
            elif 'x' in item and 'y' in item:
                d = math.hypot(float(item['x']) - px, float(item['y']) - py)
            else:
                d = 1.0
            if best is None or d < best:
                best = d
    return best


def _camp_near(obs) -> bool:
    camp = obs.get('camp') or {}
    if not camp.get('exists'):
        return False
    player = obs.get('player', {})
    return math.hypot(float(player.get('x', 0)) - float(camp.get('x', 0)),
                      float(player.get('y', 0)) - float(camp.get('y', 0))) <= 2.5


def _nearest_resource_dist(obs, kinds):
    player = obs.get('player', {})
    px, py = float(player.get('x', 0)), float(player.get('y', 0))
    best = None
    for item in obs.get('nearby', ()):
        if not item.get('available', True):
            continue
        if item.get('kind') in kinds:
            d = abs(float(item['x']) - px) + abs(float(item['y']) - py)
            if best is None or d < best:
                best = d
    return best


def _has_threat(obs):
    player = obs.get('player', {})
    px, py = float(player.get('x', 0)), float(player.get('y', 0))
    for item in obs.get('threats') or obs.get('enemies') or ():
        if isinstance(item, dict):
            if item.get('alive', True) is False:
                continue
            if 'distance' in item:
                if float(item['distance']) <= 6.0:
                    return True
            elif 'x' in item and 'y' in item:
                if abs(float(item['x']) - px) + abs(float(item['y']) - py) <= 6.0:
                    return True
            else:
                return True
    return False


def _snapshot_at_tick(rows, horizon_tick):
    """Return state summary at exact `horizon_tick` (or death tick if dead earlier)."""
    if not rows:
        return dict(tick=0, dead=False, visited_count=0, initial_visited=0,
                    health=0.0, thirst=0.0, hunger=0.0)
    first_before = rows[0]['transition']['before']
    init_visited = int(first_before.get('visited_count', 0))
    last_state = dict(first_before)
    for row in rows:
        tr = row['transition']
        trace = tr.get('trace')
        if trace:
            for step in trace:
                t = int(step.get('tick', 0))
                if t <= horizon_tick:
                    last_state = step
                    if step.get('dead', False):
                        return dict(
                            tick=t,
                            dead=True,
                            visited_count=int(step.get('visited_count', tr['after'].get('visited_count', init_visited))),
                            initial_visited=init_visited,
                            health=float(step.get('health', 0.0)),
                            thirst=float(step.get('thirst', 0.0)),
                            hunger=float(step.get('hunger', 0.0)),
                        )
                else:
                    return dict(
                        tick=int(last_state.get('tick', horizon_tick)),
                        dead=bool(last_state.get('dead', False)),
                        visited_count=int(last_state.get('visited_count', init_visited)),
                        initial_visited=init_visited,
                        health=float(last_state.get('health', 0.0)),
                        thirst=float(last_state.get('thirst', 0.0)),
                        hunger=float(last_state.get('hunger', 0.0)),
                    )
        else:
            after = tr['after']
            t = int(after.get('tick', 0))
            if t <= horizon_tick:
                last_state = after
                if after.get('dead', False):
                    break
            else:
                break
    return dict(
        tick=min(int(last_state.get('tick', 0)), int(horizon_tick)),
        dead=bool(last_state.get('dead', False)),
        visited_count=int(last_state.get('visited_count', init_visited)),
        initial_visited=init_visited,
        health=float(last_state.get('health', 0.0)),
        thirst=float(last_state.get('thirst', 0.0)),
        hunger=float(last_state.get('hunger', 0.0)),
    )


def evaluate_episode(rows, horizon_tick=96, patch_tiles=4, relative_horizon=False):
    """Compute locked exploration, fear-of-thirst/hunger/death, and survival metrics for one episode.

    When `relative_horizon=True` (or when `start_tick > 0` in night-start suites such as E4),
    `cutoff_tick = start_tick + horizon_tick`; otherwise `cutoff_tick = horizon_tick`.
    """
    start_tick = int(rows[0]['transition']['before'].get('tick', 0)) if rows else 0
    cutoff_tick = (start_tick + int(horizon_tick)) if (relative_horizon or start_tick > 0) else int(horizon_tick)
    snap = _snapshot_at_tick(rows, cutoff_tick)
    new_cells = max(0, snap['visited_count'] - snap['initial_visited'])

    patches = set()
    durations = []
    thirst_low_total = thirst_low_responsive = 0
    hunger_low_total = hunger_low_responsive = 0
    danger_total = danger_explore = 0
    safe_total = safe_explore = 0
    threat_decisions = threat_evasions = 0
    threat_damage_taken = 0.0

    first_thirst_low_tick = None
    drink_after_low_tick = None
    first_hunger_low_tick = None
    eat_after_low_tick = None

    for row in rows:
        tr = row['transition']
        before = tr['before']
        after = tr['after']
        t0 = int(before.get('tick', 0))
        if t0 >= cutoff_tick:
            break
        t1 = int(after.get('tick', t0 + int(tr.get('elapsed_ticks', 1))))
        durations.append(max(1, int(tr.get('elapsed_ticks', max(1, t1 - t0)))))

        obs = row.get('observation', {})
        next_obs = row.get('next_observation', {})
        action = row.get('action', {})
        verb = action.get('verb', '')

        p_after = next_obs.get('player') or after
        if 'x' in p_after and 'y' in p_after:
            patches.add((math.floor(float(p_after['x']) / patch_tiles),
                         math.floor(float(p_after['y']) / patch_tiles)))

        thirst0 = float(before.get('thirst', obs.get('player', {}).get('thirst', 100.0)))
        hunger0 = float(before.get('hunger', obs.get('player', {}).get('hunger', 100.0)))
        health0 = float(before.get('health', obs.get('player', {}).get('health', 100.0)))
        thirst1 = float(after.get('thirst', thirst0))
        hunger1 = float(after.get('hunger', hunger0))
        health1 = float(after.get('health', health0))

        # Fear of thirst
        if thirst0 < THIRST_LOW:
            thirst_low_total += 1
            if first_thirst_low_tick is None:
                first_thirst_low_tick = t0
            d0 = _nearest_resource_dist(obs, ('water',))
            d1 = _nearest_resource_dist(next_obs, ('water',))
            closer = d0 is not None and d1 is not None and d1 < d0
            if verb == 'drink' or thirst1 > thirst0 or closer:
                thirst_low_responsive += 1
        if first_thirst_low_tick is not None and drink_after_low_tick is None:
            if verb == 'drink' and thirst1 > thirst0:
                drink_after_low_tick = min(t1, cutoff_tick)

        # Fear of hunger
        if hunger0 < HUNGER_LOW:
            hunger_low_total += 1
            if first_hunger_low_tick is None:
                first_hunger_low_tick = t0
            d0 = _nearest_resource_dist(obs, ('berry', 'mushroom'))
            d1 = _nearest_resource_dist(next_obs, ('berry', 'mushroom'))
            closer = d0 is not None and d1 is not None and d1 < d0
            if verb in ('eat', 'gather') or hunger1 > hunger0 or closer:
                hunger_low_responsive += 1
        if first_hunger_low_tick is not None and eat_after_low_tick is None:
            if verb == 'eat' and hunger1 > hunger0:
                eat_after_low_tick = min(t1, cutoff_tick)

        # Active threat evasion (fear of death under immediate night/slime danger)
        td0 = _nearest_threat_dist(obs)
        td1 = _nearest_threat_dist(next_obs)
        if td0 is not None and td0 <= 6.0:
            threat_decisions += 1
            evaded = (
                (td1 is None)
                or (td1 > td0 + 1e-6)
                or _camp_near(next_obs)
                or verb == 'build_camp'
            )
            if evaded:
                threat_evasions += 1
            if thirst0 > 0.0 and hunger0 > 0.0 and health1 < health0:
                threat_damage_taken += max(0.0, health0 - health1)

        # Fear of death: stepping onto new tile under danger vs safe
        stepped_new = int(after.get('visited_count', 0)) > int(before.get('visited_count', 0))
        threat = _has_threat(obs)
        in_danger = health0 < DANGER_HEALTH or thirst0 < DANGER_RESERVE or hunger0 < DANGER_RESERVE or threat
        in_safe = health0 >= SAFE_HEALTH and thirst0 >= SAFE_RESERVE and hunger0 >= SAFE_RESERVE and not threat
        if in_danger:
            danger_total += 1
            danger_explore += int(stepped_new)
        elif in_safe:
            safe_total += 1
            safe_explore += int(stepped_new)

    thirst_rate = (thirst_low_responsive / thirst_low_total) if thirst_low_total else None
    hunger_rate = (hunger_low_responsive / hunger_low_total) if hunger_low_total else None
    threat_evasion_rate = (threat_evasions / threat_decisions) if threat_decisions else None
    thirst_latency = None if first_thirst_low_tick is None else float(
        (cutoff_tick if drink_after_low_tick is None else drink_after_low_tick) - first_thirst_low_tick
    )
    hunger_latency = None if first_hunger_low_tick is None else float(
        (cutoff_tick if eat_after_low_tick is None else eat_after_low_tick) - first_hunger_low_tick
    )
    if danger_total and safe_total:
        danger_delta = (danger_explore / danger_total) - (safe_explore / safe_total)
        caution_score = -danger_delta  # positive = reduces risky exploration when in danger
    else:
        danger_delta = None
        caution_score = None

    return dict(
        start_tick=int(start_tick),
        horizon_tick=int(cutoff_tick),
        elapsed_horizon_ticks=int(cutoff_tick - start_tick),
        new_cells=int(new_cells),
        unique_patches=len(patches),
        alive=not snap['dead'],
        survival_ticks=int(snap['tick'] - start_tick),
        final_health=snap['health'],
        final_thirst=snap['thirst'],
        final_hunger=snap['hunger'],
        decisions=len(durations),
        mean_ticks_per_action=float(np.mean(durations)) if durations else 0.0,
        fear_of_thirst_rate=thirst_rate,
        thirst_low_decisions=thirst_low_total,
        thirst_drink_latency=thirst_latency,
        drank_after_low=bool(drink_after_low_tick is not None),
        fear_of_hunger_rate=hunger_rate,
        hunger_low_decisions=hunger_low_total,
        hunger_eat_latency=hunger_latency,
        ate_after_low=bool(eat_after_low_tick is not None),
        threat_decisions=threat_decisions,
        threat_evasion_rate=threat_evasion_rate,
        threat_damage_taken=float(threat_damage_taken),
        fear_of_death_delta=danger_delta,
        fear_of_death_caution=caution_score,
        danger_decisions=danger_total,
        safe_decisions=safe_total,
    )


def paired_resource_decomposition(lat_a, succ_a, lat_b, succ_b):
    """Decompose restricted wait latency diff (A - B) into concordant & discordant pairs + CIF diff.

    Addresses R8 reviewer item R0-007 (competing risk of death before first drink/eat).
    """
    la = np.asarray(lat_a, dtype=float).ravel()
    lb = np.asarray(lat_b, dtype=float).ravel()
    sa = np.asarray(succ_a, dtype=bool).ravel()
    sb = np.asarray(succ_b, dtype=bool).ravel()
    valid = np.isfinite(la) & np.isfinite(lb)
    la, lb, sa, sb = la[valid], lb[valid], sa[valid], sb[valid]
    n = len(la)
    if n == 0:
        return dict(n_pairs=0, cif_diff=None, overall_wait_diff=None)
    both = sa & sb
    only_a = sa & ~sb
    only_b = ~sa & sb
    neither = ~sa & ~sb
    diff = la - lb
    return dict(
        n_pairs=int(n),
        cif_a=float(sa.mean()),
        cif_b=float(sb.mean()),
        cif_diff=float(sa.mean() - sb.mean()),
        overall_wait_diff=float(diff.mean()),
        concordant_both_success=int(both.sum()),
        concordant_wait_diff=float(diff[both].mean()) if both.any() else None,
        discordant_only_a=int(only_a.sum()),
        discordant_only_b=int(only_b.sum()),
        neither_success=int(neither.sum()),
        discordant_net_contribution=float((diff[only_a].sum() + diff[only_b].sum()) / n),
    )


def paired_bootstrap_2way(arm_a, arm_b, seed, n_boot=10000, alpha=0.05):
    """Two-way (seeds x worlds) paired bootstrap of (arm_a - arm_b).

    `arm_a` and `arm_b` are 2D arrays of shape (n_seeds, n_worlds); NaNs are ignored per resample.
    """
    a = np.asarray(arm_a, dtype=float)
    b = np.asarray(arm_b, dtype=float)
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError('arm_a and arm_b must have identical 2D shape (n_seeds, n_worlds)')
    diff = a - b
    n_seeds, n_worlds = diff.shape
    seed_means = np.nanmean(diff, axis=1)
    valid_seeds = ~np.isnan(seed_means)
    if not valid_seeds.any():
        return dict(mean=None, ci_low=None, ci_high=None, seed_positive_frac=None,
                    seed_std=None, n_seeds=int(n_seeds), n_worlds=int(n_worlds))
    observed = float(np.nanmean(seed_means))
    pos_frac = float(np.mean(seed_means[valid_seeds] > 0.0))
    seed_std = float(np.std(seed_means[valid_seeds], ddof=1)) if valid_seeds.sum() > 1 else 0.0

    rng = np.random.default_rng(seed)
    s_idx = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
    w_idx = rng.integers(0, n_worlds, size=(n_boot, n_worlds))
    # Vectorized chunked two-way resampling to bound peak RAM
    samples = np.empty(n_boot, dtype=float)
    chunk = 1000
    for start in range(0, n_boot, chunk):
        end = min(start + chunk, n_boot)
        si = s_idx[start:end, :, None]
        wi = w_idx[start:end, None, :]
        grid = diff[si, wi]
        samples[start:end] = np.nanmean(grid, axis=(1, 2))
    lo = float(np.nanquantile(samples, alpha / 2.0))
    hi = float(np.nanquantile(samples, 1.0 - alpha / 2.0))
    return dict(
        mean=observed,
        ci_low=lo,
        ci_high=hi,
        seed_positive_frac=pos_frac,
        seed_std=seed_std,
        n_seeds=int(n_seeds),
        n_worlds=int(n_worlds),
        n_boot=int(n_boot),
    )


def required_seeds_power90(effect_size, seed_diff_std, min_seeds=32, max_seeds=64):
    """One-sided alpha=0.05, power=0.90 paired-t normal approximation: N = ((z_a + z_b) * s / d)^2."""
    if effect_size is None or effect_size <= 0 or seed_diff_std is None or seed_diff_std <= 0:
        return int(max_seeds)
    # z_0.95 = 1.64485, z_0.90 = 1.28155 => sum = 2.9264
    raw = math.ceil((2.9264 * float(seed_diff_std) / float(effect_size)) ** 2)
    return int(max(min_seeds, min(max_seeds, raw)))


def classify_stage3_outcome(
    b_minus_c,
    b_minus_a,
    d_minus_a,
    e_minus_a,
    survival_b_minus_a,
    *,
    b_minus_ctime=None,
    drive_b_minus_c=None,
    survival_b_minus_c=None,
    checkpoint_b_gt_c=None,
    second_organoid_replicated=False,
):
    """Apply frozen grading criteria to Stage 3 / Stage 4 results (including optional C_time gate)."""
    if d_minus_a['ci_low'] is None or d_minus_a['ci_low'] <= 0.0:
        return dict(verdict='inconclusive', survival_warning=False,
                    reason='Positive control D - A lower 95% CI is not above 0; pipeline cannot adjudicate')

    surv_warn = bool(
        survival_b_minus_a is not None
        and survival_b_minus_a.get('ci_low') is not None
        and survival_b_minus_a['ci_low'] < GUARDRAIL_SURVIVAL_DROP_MAX
    )
    e_not_positive = e_minus_a is None or e_minus_a.get('ci_low') is None or e_minus_a['ci_low'] <= 0.0
    ctime_ok = (
        b_minus_ctime is None
        or (
            b_minus_ctime.get('ci_low') is not None
            and b_minus_ctime['ci_low'] > 0.0
            and (b_minus_ctime.get('seed_positive_frac') or 0.0) >= WEAK_POSITIVE_SEED_FRAC
        )
    )

    weak_pos = bool(
        b_minus_c['ci_low'] is not None and b_minus_c['ci_low'] > 0.0
        and b_minus_a['ci_low'] is not None and b_minus_a['ci_low'] > 0.0
        and (b_minus_c['seed_positive_frac'] or 0.0) >= WEAK_POSITIVE_SEED_FRAC
        and ctime_ok
        and not surv_warn
        and e_not_positive
    )
    if weak_pos:
        if second_organoid_replicated:
            return dict(verdict='stable_positive', survival_warning=False,
                        reason='Weak positive replicated on an independent second organoid')
        improved_drives = 0
        if drive_b_minus_c:
            for stats in drive_b_minus_c.values():
                if stats and stats.get('ci_low') is not None and stats['ci_low'] > 0.0:
                    improved_drives += 1
        surv_nonneg = (survival_b_minus_c is not None and survival_b_minus_c.get('mean') is not None
                       and survival_b_minus_c['mean'] >= 0.0)
        if improved_drives >= 2 and surv_nonneg:
            return dict(verdict='dual_weak_positive', survival_warning=False,
                        reason='Primary endpoint and >=2 survival-drive metrics improved with non-negative survival')
        return dict(verdict='weak_positive', survival_warning=False,
                    reason='B beats both C and A (95% CI > 0, >=70% seeds) within survival guardrail')

    # Check exploratory process clue
    consec = 0
    max_consec = 0
    for flag in (checkpoint_b_gt_c or ()):
        consec = consec + 1 if flag else 0
        max_consec = max(max_consec, consec)
    is_clue = bool(
        max_consec >= CLUE_CONSECUTIVE_CKPTS
        and b_minus_c['mean'] is not None and b_minus_c['mean'] > 0.0
        and (b_minus_c['seed_positive_frac'] or 0.0) >= CLUE_SEED_FRAC
    )
    if is_clue:
        return dict(verdict='process_clue', survival_warning=surv_warn,
                    reason='Exploratory process clue (>=3 consecutive checkpoints B>C, >=65% seeds positive; CI crosses 0)')

    return dict(verdict='negative', survival_warning=surv_warn,
                reason='Positive control D-A passed, but B-C did not meet weak-positive criteria')
