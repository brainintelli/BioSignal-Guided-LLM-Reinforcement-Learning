"""Strictly causal endogenous short-term synaptic depression (STD) dual-library builder.

Extracts `habituation_library.json` (`mea.oddball.habituation_library.v2`) and
`evoked_library.json` (`mea.oddball.evoked_library.v2`) from existing multi-electrode
organoid recordings WITHOUT any new lab collection and WITHOUT circular label-sorting:

1) Strict causal past-only indexing:
   Every window `t` is assigned its habituation load bin (`low` / `mid` / `high`) and
   homeostatic excitability state (`real` excitable vs `sham` post-burst refractory)
   using ONLY past windows `[t - history_k, t - 1]` within the same contiguous recording
   block. Window `t`'s own spike activity is NEVER inspected to assign its bin or state.
2) Strict train/held-out split:
   Activity thresholds, non-negative channel SNR weights, and library standardization
   parameters `(center, scale)` are fit strictly on `[0, fit_end)` (rounds 1-2), then
   evaluated untouched on `[fit_end, T)` (round 3 held-out) via `analysis_v2.offline_gates_v2`.
3) Pre-build timeline-permuted null control (`C_time`):
   Permuting the window timeline BEFORE running the identical extraction & library builder
   destroys biological multi-second synaptic memory while preserving marginal distributions,
   providing an airtight control against pipeline circularity.
"""
import math
import numpy as np

from . import analysis_v2
from . import design_v2 as design
from . import library_v2

HISTORY_K_DEFAULT = 4
SECTOR_QUANTILE = 70.0
GLOBAL_BURST_QUANTILE = 75.0


def maxone_electrode_xy_um(electrode_id: int):
    """MaxOne 220x120 array with 17.5 um center-to-center pitch."""
    elec = int(electrode_id)
    return float((elec % 220) * 17.5), float((elec // 220) * 17.5)


def partition_spatial_sectors(channel_ids, channel_xy_um=None):
    """Deterministically partition channels into 3 equal-sized spatial sectors ('H', 'W', 'F').

    Uses polar angle around the active-array centroid when physical `(x_um, y_um)` coordinates
    are available for >= 50% of channels; otherwise falls back to sorted channel IDs.
    Returns dict mapping `'H'|'W'|'F'` -> 1D integer array of column indices into `channel_ids`.
    """
    ch_list = [int(c) for c in channel_ids]
    n_ch = len(ch_list)
    if n_ch < 6:
        raise ValueError(f'At least 6 channels required to form 3 spatial sectors (got {n_ch})')
    xy_map = channel_xy_um or {}
    known = [c for c in ch_list if c in xy_map]
    if len(known) >= max(6, n_ch // 2):
        cx = float(np.mean([xy_map[c][0] for c in known]))
        cy = float(np.mean([xy_map[c][1] for c in known]))
        keyed = []
        for idx, c in enumerate(ch_list):
            if c in xy_map:
                x, y = float(xy_map[c][0]), float(xy_map[c][1])
                ang = math.atan2(y - cy, x - cx)
                rad = math.hypot(x - cx, y - cy)
                keyed.append((0, ang, rad, c, idx))
            else:
                keyed.append((1, 0.0, 0.0, c, idx))
        keyed.sort()
        ordered_indices = np.asarray([row[-1] for row in keyed], dtype=np.int64)
    else:
        ordered_indices = np.argsort(np.asarray(ch_list, dtype=np.int64), kind='stable')

    splits = np.array_split(ordered_indices, len(design.SITES))
    return {site: np.sort(sp) for site, sp in zip(design.SITES, splits)}


def _contiguous_past_valid(block_ids, t, history_k):
    if t < history_k:
        return False
    b = block_ids[t]
    for k in range(1, history_k + 1):
        if block_ids[t - k] != b:
            return False
    return True


def extract_causal_std_rows(
    spike_counts,
    *,
    block_ids=None,
    fit_end=None,
    sectors=None,
    channel_ids=None,
    channel_xy_um=None,
    history_k=HISTORY_K_DEFAULT,
    sector_quantile=SECTOR_QUANTILE,
    global_quantile=GLOBAL_BURST_QUANTILE,
):
    """Extract habituation and homeostatic-contrast trial rows using strictly causal past history.

    `spike_counts`: 2D array `(n_windows, n_channels)` of spike counts per window.
    Returns dict with `habituation_rows`, `evoked_rows`, `channel_weights`, `fit_end`, `sectors`.
    """
    counts = np.asarray(spike_counts, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[0] < 30 or counts.shape[1] < 6:
        raise ValueError(f'spike_counts must be 2D with >=30 windows and >=6 channels; got {counts.shape}')
    n_t, n_ch = counts.shape
    if fit_end is None:
        fit_end = (2 * n_t) // 3
    fit_end = int(fit_end)
    if not (history_k + 10 <= fit_end <= n_t - 10):
        raise ValueError(f'Invalid fit_end={fit_end} for n_windows={n_t}')

    if block_ids is None:
        blk = np.zeros(n_t, dtype=np.int64)
    else:
        blk = np.asarray(block_ids, dtype=np.int64)
        if blk.shape != (n_t,):
            raise ValueError('block_ids length must match spike_counts rows')

    if sectors is None:
        c_ids = np.arange(n_ch) if channel_ids is None else channel_ids
        sectors = partition_spatial_sectors(c_ids, channel_xy_um=channel_xy_um)

    # Split [0, fit_end) into Round 1 and Round 2; [fit_end, n_t) is Round 3 (strictly held-out).
    # Each round is divided into 4 chronological sub-blocks so `v1_analysis.split_rounds`
    # recognizes 4 complete blocks per round without leaking any future boundary.
    r1_end = fit_end // 2

    def round_and_block_of(t):
        if t < r1_end:
            rnd = 1
            sub = min(3, (4 * t) // max(1, r1_end))
        elif t < fit_end:
            rnd = 2
            sub = min(3, (4 * (t - r1_end)) // max(1, fit_end - r1_end))
        else:
            rnd = 3
            sub = min(3, (4 * (t - fit_end)) // max(1, n_t - fit_end))
        return rnd, f'r{rnd}_b{sub}'

    valid_hist = np.asarray([_contiguous_past_valid(blk, t, history_k) for t in range(n_t)], dtype=bool)

    # Global synchronous burst indicator strictly thresholded on [0, fit_end)
    global_sum = counts.sum(axis=1)
    global_thresh = float(np.percentile(global_sum[:fit_end], global_quantile))
    global_burst = global_sum > global_thresh

    sector_past_load = {}
    sector_weights = {}
    sector_responses = {}

    for site in design.SITES:
        cols = sectors[site]
        sec_counts = counts[:, cols]
        sec_sum = sec_counts.sum(axis=1)
        sec_thresh = float(np.percentile(sec_sum[:fit_end], sector_quantile))
        sec_active = (sec_sum > sec_thresh).astype(np.int64)

        past_load = np.full(n_t, -1, dtype=np.int64)
        for t in range(n_t):
            if valid_hist[t]:
                past_load[t] = int(sec_active[t - history_k:t].sum())
        sector_past_load[site] = past_load

        # Fit non-negative channel weights strictly on [0, fit_end) using causal past labels only
        fit_mask_rest = (np.arange(n_t) < fit_end) & valid_hist & (past_load == 0)
        fit_mask_dep = (np.arange(n_t) < fit_end) & valid_hist & (past_load >= 2)
        if fit_mask_rest.sum() >= 5 and fit_mask_dep.sum() >= 5:
            w = analysis_v2.fit_channel_weights(sec_counts[fit_mask_rest], sec_counts[fit_mask_dep])
        else:
            w = np.full(len(cols), 1.0 / len(cols), dtype=np.float64)
        sector_weights[site] = w
        sector_responses[site] = analysis_v2.weighted_response(sec_counts, w)

    hab_rows = []
    evk_rows = []
    for t in range(n_t):
        if not valid_hist[t]:
            continue
        rnd, blk_name = round_and_block_of(t)
        just_global_burst = bool(global_burst[t - 1])

        # Habituation rows: every causal-valid window contributes 1 observation per spatial sector
        for site in design.SITES:
            pl = int(sector_past_load[site][t])
            bin_name = 'low' if pl == 0 else ('mid' if pl == 1 else 'high')
            resp = float(sector_responses[site][t])
            hab_rows.append(dict(
                t=t,
                round=rnd,
                block=blk_name,
                site=site,
                real=True,
                bin=bin_name,
                past_load=pl,
                response=resp,
                valid=True,
            ))

        # Evoked / Homeostatic contrast rows (strictly causal past-only):
        # - Post-global-burst window (t-1 was synchronous burst) -> post-discharge refractory ('sham' / Z)
        # - Rested sector window (past_load == 0 and t-1 not global burst) -> pre-discharge excitable ('real')
        if just_global_burst:
            mean_refractory = float(np.mean([sector_responses[s][t] for s in design.SITES]))
            evk_rows.append(dict(
                t=t,
                round=rnd,
                block=blk_name,
                site=design.SHAM,
                real=False,
                bin='sham',
                response=mean_refractory,
                valid=True,
            ))
        else:
            for site in design.SITES:
                if int(sector_past_load[site][t]) == 0:
                    evk_rows.append(dict(
                        t=t,
                        round=rnd,
                        block=blk_name,
                        site=site,
                        real=True,
                        bin='low',
                        response=float(sector_responses[site][t]),
                        valid=True,
                    ))

    return dict(
        habituation_rows=hab_rows,
        evoked_rows=evk_rows,
        channel_weights=sector_weights,
        sectors=sectors,
        fit_end=fit_end,
        n_windows=n_t,
    )


def evaluate_and_build_bundle(
    spike_counts,
    *,
    dataset_id,
    chip_id,
    block_ids=None,
    fit_end=None,
    channel_ids=None,
    channel_xy_um=None,
    history_k=HISTORY_K_DEFAULT,
    rho_min=0.20,
    gate_seed=20261123,
    permute_timeline_seed=None,
    force_export_for_control=False,
):
    """Run causal extraction, evaluate held-out G2/G3 gates, and build library JSON dicts.

    When `permute_timeline_seed` is set, permutes window order within each contiguous block
    BEFORE extraction to construct the `C_time` timeline-shuffled control.
    """
    counts = np.asarray(spike_counts, dtype=np.float64)
    n_t = counts.shape[0]
    blk = np.zeros(n_t, dtype=np.int64) if block_ids is None else np.asarray(block_ids, dtype=np.int64)

    if permute_timeline_seed is not None:
        rng = np.random.default_rng(int(permute_timeline_seed))
        perm_counts = np.empty_like(counts)
        for b_val in np.unique(blk):
            idx = np.flatnonzero(blk == b_val)
            perm_counts[idx] = counts[idx[rng.permutation(len(idx))]]
        counts = perm_counts

    extracted = extract_causal_std_rows(
        counts,
        block_ids=blk,
        fit_end=fit_end,
        channel_ids=channel_ids,
        channel_xy_um=channel_xy_um,
        history_k=history_k,
    )
    hab_rows = extracted['habituation_rows']
    evk_rows = extracted['evoked_rows']

    # Evaluate G2 on evoked_rows (held-out round 3) and G3 on habituation_rows (held-out round 3)
    evk_gates = analysis_v2.offline_gates_v2(evk_rows, seed=gate_seed, rho_min=rho_min)
    # Combine sham from evk_rows with hab_rows so offline_gates_v2 evaluates both G2 & G3 coherently
    sham_only = [r for r in evk_rows if not r['real']]
    combined_for_hab = hab_rows + sham_only
    hab_gates = analysis_v2.offline_gates_v2(combined_for_hab, seed=gate_seed + 1000, rho_min=rho_min)

    # Also evaluate pure G3 habituation gate directly regardless of evoked sham pairing
    lib_hab, held_hab, _ = analysis_v2.v1_analysis.split_rounds(hab_rows)
    g3_held = analysis_v2._strictly_decreasing_habituation(held_hab, gate_seed + 200) if held_hab else None
    g3_lib = analysis_v2._strictly_decreasing_habituation(lib_hab, gate_seed + 300) if lib_hab else None

    usable_evoked = list(evk_gates.get('usable_evoked_sites', []))
    usable_hab = []
    rho_lib = None
    if g3_held and g3_lib:
        usable_hab = [
            s for s in design.SITES
            if s in g3_held['passed_sites'] and (g3_lib['sites'][s]['auc'] or 0.0) > 0.5
        ]
        lib_stats = {}
        for s in usable_hab:
            mine = [r for r in lib_hab if r['real'] and r['site'] == s and r.get('valid', True)]
            by_bin = {b: [float(r['response']) for r in mine if r['bin'] == b] for b in design.BIN_NAMES}
            if all(by_bin.values()):
                lib_stats[s] = dict(
                    center=float(np.mean([np.mean(v) for v in by_bin.values()])),
                    scale=float(np.std([float(r['response']) for r in mine])),
                )
        rho_lib = analysis_v2.compute_rho_lib(held_hab, lib_stats, usable_hab)
        # Also compute 3-site diagnostic rho across all sites for null-control reporting
        all_stats = {}
        for s in design.SITES:
            mine = [r for r in lib_hab if r['real'] and r['site'] == s and r.get('valid', True)]
            by_bin = {b: [float(r['response']) for r in mine if r['bin'] == b] for b in design.BIN_NAMES}
            if all(by_bin.values()):
                all_stats[s] = dict(
                    center=float(np.mean([np.mean(v) for v in by_bin.values()])),
                    scale=float(np.std([float(r['response']) for r in mine])),
                )
        rho_all_sites = analysis_v2.compute_rho_lib(held_hab, all_stats, list(design.SITES))
    else:
        rho_all_sites = None

    g2_passed = len(usable_evoked) >= 2
    g3_passed = len(usable_hab) >= 2 and rho_lib is not None and rho_lib >= float(rho_min)

    evoked_lib = None
    habituation_lib = None
    lib_evk_rows = [r for r in evk_rows if r['round'] in (1, 2)]
    lib_hab_rows = [r for r in hab_rows if r['round'] in (1, 2)]

    if g2_passed or force_export_for_control:
        sites_for_evk = usable_evoked if len(usable_evoked) >= 2 else list(design.SITES)
        evoked_lib = library_v2.build_evoked_library(
            lib_evk_rows, sites_for_evk, channel_weights=extracted['channel_weights']
        )
    if (g2_passed and g3_passed) or force_export_for_control:
        sites_for_hab = usable_hab if len(usable_hab) >= 2 else list(design.SITES)
        habituation_lib = library_v2.build_habituation_library(
            lib_hab_rows, sites_for_hab, channel_weights=extracted['channel_weights']
        )

    gate_summary = dict(
        dataset_id=dataset_id,
        chip_id=str(chip_id),
        timeline_permuted=permute_timeline_seed is not None,
        permute_timeline_seed=permute_timeline_seed,
        n_windows=int(n_t),
        fit_end=int(extracted['fit_end']),
        held_windows=int(n_t - extracted['fit_end']),
        n_channels=int(counts.shape[1]),
        sector_sizes={s: int(len(extracted['sectors'][s])) for s in design.SITES},
        g2_passed=bool(g2_passed),
        g3_passed=bool(g3_passed),
        dual_drive_unlocked=bool(g2_passed and g3_passed),
        usable_evoked_sites=usable_evoked,
        usable_habituation_sites=usable_hab,
        rho_lib=rho_lib,
        rho_all_sites=rho_all_sites,
        evoked_held_sites={
            s: dict(
                auc=evk_gates['evoked']['sites'][s]['auc'],
                p=evk_gates['evoked']['sites'][s]['p'],
                passed=evk_gates['evoked']['sites'][s]['passed'],
            )
            for s in design.SITES
        } if 'evoked' in evk_gates else {},
        habituation_held_sites={
            s: dict(
                auc=g3_held['sites'][s]['auc'],
                p=g3_held['sites'][s]['p'],
                monotonic=g3_held['sites'][s]['monotonic'],
                bin_means=g3_held['sites'][s]['bin_means'],
                passed=g3_held['sites'][s]['passed'],
            )
            for s in design.SITES
        } if g3_held else {},
        hab_gates_combined=dict(
            evoked_allowed=hab_gates.get('evoked_training_allowed'),
            habituation_allowed=hab_gates.get('habituation_training_allowed'),
        ),
    )
    return dict(
        summary=gate_summary,
        evoked_library=evoked_lib,
        habituation_library=habituation_lib,
    )
