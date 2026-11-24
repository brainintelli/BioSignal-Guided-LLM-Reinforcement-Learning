"""Checks an auxiliary reward (R2) has to pass before it may be mixed into the game reward (R1).

Works on recorded per-decision arrays only; nothing here trains a policy or contacts a device.
The order is fixed: leak gate first, then the structure decision, then the proportions.
"""
import numpy as np

LEAK_MARGIN = .05       # held-out AUC of the R2-only probe above its shuffled control
LEAK_P_MAX = .05        # and the excess must be rarer than this among the shuffles
SHUFFLES = 1000
PARTIAL_P_MAX = .01     # R2 still related to the label once the input is held fixed
BETA_DEFAULT = .05
BETA_MAX = .15
KAPPA = .2              # per-decision cap relative to |R1| of the same decision
DELTA_FRAC = .02        # floor of that cap, relative to the tracked R1 scale
SHARE_TARGET = .10
SHARE_REDLINE = .20
REDUNDANT_RHO = .8
NOISE_RHO = .05
EMA_DECAY = .9


def auc(score, label):
    """Probability that a positive outranks a negative; ties count half."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if not n_pos or not n_neg:
        return None
    order = np.argsort(score, kind='stable')
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    for value in np.unique(score):
        tied = score == value
        ranks[tied] = ranks[tied].mean()
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _design(x, center, scale):
    return np.column_stack([np.ones(len(x)), (x - center) / scale])


def probe_auc(x_train, y_train, x_test, y_test, ridge=1e-2, steps=25):
    """Held-out AUC of a logistic regression probe. None when a split has one class."""
    x_train, x_test = np.asarray(x_train, float), np.asarray(x_test, float)
    y_train, y_test = np.asarray(y_train, bool), np.asarray(y_test, bool)
    if y_train.all() or not y_train.any() or y_test.all() or not y_test.any():
        return None
    center, scale = x_train.mean(axis=0), x_train.std(axis=0)
    scale[scale == 0] = 1.
    a = _design(x_train, center, scale)
    w = np.zeros(a.shape[1])
    penalty = ridge * np.eye(a.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(a @ w, -30, 30)))
        gradient = a.T @ (p - y_train) / len(a) + penalty @ w
        hessian = (a * (p * (1 - p))[:, None]).T @ a / len(a) + penalty
        w -= np.linalg.solve(hessian, gradient)
    return auc(_design(x_test, center, scale) @ w, y_test)


def leak_gate(features, label, group, test_groups, seed, shuffles=SHUFFLES):
    """Can R2 alone predict the label on held-out groups better than shuffled R2?"""
    features = np.asarray(features, float).reshape(len(label), -1)
    label, test = np.asarray(label, bool), np.isin(group, list(test_groups))
    real = probe_auc(features[~test], label[~test], features[test], label[test])
    if real is None:
        return dict(auc_real=None, auc_shuffled=None, margin=None, leaked=None, n_train=int((~test).sum()),
                    n_test=int(test.sum()), reason='one class in a split')
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(shuffles):
        mixed = features[rng.permutation(len(features))]
        null.append(probe_auc(mixed[~test], label[~test], mixed[test], label[test]))
    null = np.asarray(null)
    margin = real - float(null.mean())
    p = float((1 + (null >= real).sum()) / (1 + shuffles))
    # With few held-out decisions the margin alone is reached by chance; it also has to beat the shuffles.
    return dict(auc_real=real, auc_shuffled=float(null.mean()), auc_shuffled_p95=float(np.quantile(null, .95)),
                margin=margin, p=p, leaked=bool(margin > LEAK_MARGIN and p <= LEAK_P_MAX),
                n_train=int((~test).sum()), n_test=int(test.sum()), positives_test=int(label[test].sum()))


def partial_correlation(r2, label, inputs, seed, permutations=2000):
    """Correlation of R2 and the label after a linear fit on the inputs is removed from both."""
    a = np.column_stack([np.ones(len(r2)), np.asarray(inputs, float)])
    residual = lambda v: np.asarray(v, float) - a @ np.linalg.lstsq(a, np.asarray(v, float), rcond=None)[0]
    u, v = residual(r2), residual(label)
    if u.std() < 1e-12 or v.std() < 1e-12:
        return dict(r=None, p=None, leaked=None, reason='nothing left after the inputs are removed')
    r = float(np.corrcoef(u, v)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.asarray([np.corrcoef(rng.permutation(u), v)[0, 1] for _ in range(permutations)])
    p = float((1 + (np.abs(null) >= abs(r)).sum()) / (1 + permutations))
    return dict(r=r, p=p, leaked=bool(p <= PARTIAL_P_MAX))


def shares(r1, r2):
    """The two shares that have to be reported together, plus the per-decision extremes."""
    r1, r2 = np.asarray(r1, float), np.asarray(r2, float)
    v1, v2 = r1.var(), r2.var()
    nonzero = r1 != 0
    constant = v1 == 0 or v2 == 0
    return dict(magnitude=float(np.abs(r2).mean() / np.abs(r1).mean()),
                variance=float(v2 / (v1 + v2)) if v1 + v2 else 0.,
                rho=None if constant else float(np.corrcoef(r1, r2)[0, 1]),
                max_step_ratio=float(np.abs(r2[nonzero] / r1[nonzero]).max()) if nonzero.any() else None,
                larger_than_r1=int((np.abs(r2) > np.abs(r1)).sum()),
                sign_flips=int((np.sign(r1 + r2) != np.sign(r1)).sum()), steps=len(r1))


def verdict(share):
    worst = max(share['magnitude'], share['variance'])
    level = 'redline' if worst > SHARE_REDLINE else 'above_target' if worst >= SHARE_TARGET else 'ok'
    flags = []
    if share['rho'] is not None and share['rho'] > REDUNDANT_RHO:
        flags.append('redundant')
    if share['rho'] is not None and abs(share['rho']) < NOISE_RHO and share['variance'] > SHARE_TARGET:
        flags.append('noise')
    return level, flags


def track_scale(previous, r1):
    """EMA of median |R1|. The median keeps a few large decisions from setting the scale."""
    now = float(np.median(np.abs(np.asarray(r1, float))))
    return now if previous is None else EMA_DECAY * previous + (1 - EMA_DECAY) * now


def shape(r1, signal, s1, beta, eligible=None):
    """R2 relative to the tracked R1 scale, centered within the batch and capped per decision."""
    if not 0 <= beta <= BETA_MAX or not s1 > 0:
        raise ValueError('beta must lie in [0, BETA_MAX] and the R1 scale must be positive')
    r1 = np.asarray(r1, float)
    eligible = np.ones(len(r1), bool) if eligible is None else np.asarray(eligible, bool)
    out = np.zeros(len(r1))
    if eligible.any():
        u = np.tanh(np.asarray(signal, float)[eligible])
        cap = KAPPA * np.abs(r1[eligible]) + DELTA_FRAC * s1
        out[eligible] = np.clip(beta * s1 * (u - u.mean()), -cap, cap)
    return out


CALIBRATED_SHARE_MAX = .15  # below SHARE_REDLINE (0.20) with safety headroom
STEP_RATIO_CAP = .85        # never let |R2| exceed |R1| on standard steps
ZERO_R1_CAP_FRAC = .10      # cap when R1 == 0, relative to tracked mean(|R1|)
EXPLORE_FLOOR_FRAC = .25    # floor relative to mean(|R1|) so landmark-departure steps (|R1|~0.001) are not crushed
SHORT_GAMMA_DEFAULT = .95   # biological short-horizon GAE discount (~10-20 decisions)
SHORT_LAM_DEFAULT = .90
ADV_BETA_MAX = .35          # upper cap on decoupled advantage injection gain


def track_r1_stats(previous, r1):
    """EMA tracker for std(R1) and mean(|R1|) so calibrated R2 stays in the 5%-10% share band."""
    arr = np.asarray(r1, float)
    now_std = float(arr.std())
    now_abs = float(np.abs(arr).mean())
    if previous is None:
        return dict(std=now_std, mean_abs=now_abs)
    return dict(
        std=EMA_DECAY * previous['std'] + (1 - EMA_DECAY) * now_std,
        mean_abs=EMA_DECAY * previous['mean_abs'] + (1 - EMA_DECAY) * now_abs,
    )


def shape_calibrated(r1, signal, r1_stats, target_share, eligible=None):
    """Scale centered tanh(signal) so realized magnitude & variance shares target `target_share`.

    Unlike `shape()` (which scales by median(|R1|) and collapses variance share to <0.5% when R1
    contains potential jumps), this scales against tracked (std(R1), mean(|R1|)) while strictly
    bounding every step below STEP_RATIO_CAP * |R1| and below SHARE_REDLINE.
    """
    if not 0. <= target_share <= CALIBRATED_SHARE_MAX:
        raise ValueError('target_share must lie in [0, CALIBRATED_SHARE_MAX]')
    r1 = np.asarray(r1, float)
    sig = np.asarray(signal, float)
    if len(r1) != len(sig):
        raise ValueError('r1 and signal must have the same length')
    eligible = np.ones(len(r1), bool) if eligible is None else np.asarray(eligible, bool)
    out = np.zeros(len(r1), dtype=float)
    if target_share == 0. or not eligible.any():
        return out
    s_std = float(r1_stats['std'])
    s_abs = float(r1_stats['mean_abs'])
    if s_abs <= 0:
        raise ValueError('Tracked R1 mean(|R1|) must be positive')
    u = np.tanh(sig[eligible])
    u_centered = u - u.mean()
    u_std = float(u_centered.std())
    u_abs = float(np.abs(u_centered).mean())
    if u_std < 1e-12 or u_abs < 1e-12:
        return out
    frac_active = float(eligible.mean())
    mag_gain = (target_share * s_abs) / (u_abs * max(frac_active, 1e-6))
    if s_std > 1e-12:
        var_budget_std = np.sqrt(target_share / max(1. - target_share, 1e-6)) * s_std
        var_gain = var_budget_std / (u_std * np.sqrt(max(frac_active, 1e-6)))
        gain = min(mag_gain, var_gain)
    else:
        gain = mag_gain
    raw = gain * u_centered
    r1_e = np.abs(r1[eligible])
    cap = np.where(r1_e > 0, STEP_RATIO_CAP * r1_e, ZERO_R1_CAP_FRAC * s_abs)
    out[eligible] = np.clip(raw, -cap, cap)
    return out


def shape_for_decoupled_gae(r1, signal, r1_stats, target_share, eligible=None, floor_frac=EXPLORE_FLOOR_FRAC):
    """Calibrated R2 shaping for Dual-Timescale Decoupled GAE.

    Prevents the `|r1_t| ~ 0.001` landmark-departure step-crushing bug by setting a floor
    `floor_frac * mean(|R1|)` on per-step clipping, while `|clip(raw, -cap, cap)| <= |raw|`
    mathematically guarantees realized batch magnitude & variance shares remain <= `target_share`
    (well below `SHARE_REDLINE = 0.20`).
    """
    if not 0. <= target_share <= CALIBRATED_SHARE_MAX:
        raise ValueError('target_share must lie in [0, CALIBRATED_SHARE_MAX]')
    if not 0. <= floor_frac <= 1.0:
        raise ValueError('floor_frac must lie in [0, 1]')
    r1 = np.asarray(r1, float)
    sig = np.asarray(signal, float)
    if len(r1) != len(sig):
        raise ValueError('r1 and signal must have the same length')
    eligible = np.ones(len(r1), bool) if eligible is None else np.asarray(eligible, bool)
    out = np.zeros(len(r1), dtype=float)
    if target_share == 0. or not eligible.any():
        return out
    s_std = float(r1_stats['std'])
    s_abs = float(r1_stats['mean_abs'])
    if s_abs <= 0:
        raise ValueError('Tracked R1 mean(|R1|) must be positive')
    u = np.tanh(sig[eligible])
    u_centered = u - u.mean()
    u_std = float(u_centered.std())
    u_abs = float(np.abs(u_centered).mean())
    if u_std < 1e-12 or u_abs < 1e-12:
        return out
    frac_active = float(eligible.mean())
    mag_gain = (target_share * s_abs) / (u_abs * max(frac_active, 1e-6))
    if s_std > 1e-12:
        var_budget_std = np.sqrt(target_share / max(1. - target_share, 1e-6)) * s_std
        var_gain = var_budget_std / (u_std * np.sqrt(max(frac_active, 1e-6)))
        gain = min(mag_gain, var_gain)
    else:
        gain = mag_gain
    raw = gain * u_centered
    r1_e = np.abs(r1[eligible])
    cap = np.maximum(STEP_RATIO_CAP * r1_e, floor_frac * s_abs)
    out[eligible] = np.clip(raw, -cap, cap)
    return out


def short_horizon_gae(r2_shaped, durations, terminals, boundaries,
                      gamma_r2=SHORT_GAMMA_DEFAULT, lam_r2=SHORT_LAM_DEFAULT):
    """Compute zero-baseline short-horizon GAE for wetware signal R2 without polluting the R1 Critic."""
    r2 = np.asarray(r2_shaped, dtype=float)
    dur = np.maximum(1, np.asarray(durations, dtype=int))
    term = np.asarray(terminals, dtype=bool)
    bnd = np.asarray(boundaries, dtype=bool)
    n = len(r2)
    adv = np.zeros(n, dtype=float)
    running = 0.0
    for i in range(n - 1, -1, -1):
        if term[i] or bnd[i]:
            running = float(r2[i])
        else:
            w = (gamma_r2 * lam_r2) ** int(dur[i])
            running = float(r2[i]) + w * running
        adv[i] = running
    return adv


def mix_decoupled_advantages(adv_r1, adv_r2, beta_adv, clip_z=2.5):
    """Combine R1 survival advantages with bounded short-horizon R2 advantages.

    Returns `(mixed_adv, adv_snr)` in R1's native scale so `m8_train.update` normalizes cleanly
    while `policy.critic` continues training strictly on uncorrupted `R1` returns.
    """
    if not 0.0 <= float(beta_adv) <= ADV_BETA_MAX:
        raise ValueError(f'beta_adv must lie in [0, {ADV_BETA_MAX}]')
    a1 = np.asarray(adv_r1, dtype=float)
    a2 = np.asarray(adv_r2, dtype=float)
    if a1.shape != a2.shape:
        raise ValueError('adv_r1 and adv_r2 must have matching shape')
    if beta_adv == 0.0 or a2.std() < 1e-12 or a1.std() < 1e-12:
        return a1.copy(), 0.0
    s1 = float(a1.std())
    z2 = np.clip((a2 - float(a2.mean())) / float(a2.std()), -float(clip_z), float(clip_z))
    delta = float(beta_adv) * s1 * z2
    mixed = a1 + delta
    adv_snr = float(np.median(np.abs(delta)) / max(s1, 1e-12))
    return mixed, adv_snr


def channel_bound_check(r2_real, r2_ideal, r2_shuffled, label, group, test_groups, seed,
                        shuffles=SHUFFLES, r2_ctime=None):
    """Pre-training check for causal wetware signals:
    1) Wetware B cannot predict the target better than the ideal stimulus encoding D (+ LEAK_MARGIN).
    2) Shuffled control C carries no information above chance (|AUC - 0.5| <= LEAK_MARGIN).
    3) Optional timeline-permuted control C_time carries no information above chance.
    """
    real_gate = leak_gate(r2_real, label, group, test_groups, seed=seed, shuffles=shuffles)
    ideal_gate = leak_gate(r2_ideal, label, group, test_groups, seed=seed + 1, shuffles=shuffles)
    shuf_gate = leak_gate(r2_shuffled, label, group, test_groups, seed=seed + 2, shuffles=shuffles)
    ctime_gate = (
        leak_gate(r2_ctime, label, group, test_groups, seed=seed + 3, shuffles=shuffles)
        if r2_ctime is not None else None
    )
    if real_gate['auc_real'] is None or ideal_gate['auc_real'] is None or shuf_gate['auc_real'] is None:
        return dict(passed=None, reason='one class in a split', real=real_gate, ideal=ideal_gate,
                    shuffled=shuf_gate, ctime=ctime_gate)
    exceeds_ideal = bool(real_gate['auc_real'] > ideal_gate['auc_real'] + LEAK_MARGIN)
    shuffled_leaks = bool(abs(shuf_gate['auc_real'] - .5) > LEAK_MARGIN and shuf_gate['p'] <= LEAK_P_MAX)
    ctime_leaks = bool(
        ctime_gate is not None
        and ctime_gate['auc_real'] is not None
        and abs(ctime_gate['auc_real'] - .5) > LEAK_MARGIN
        and ctime_gate['p'] <= LEAK_P_MAX
    )
    out = dict(
        passed=not exceeds_ideal and not shuffled_leaks and not ctime_leaks,
        exceeds_ideal=exceeds_ideal,
        shuffled_leaks=shuffled_leaks,
        ctime_leaks=ctime_leaks,
        real_auc=real_gate['auc_real'],
        ideal_auc=ideal_gate['auc_real'],
        shuffled_auc=shuf_gate['auc_real'],
        real=real_gate,
        ideal=ideal_gate,
        shuffled=shuf_gate,
    )
    if ctime_gate is not None:
        out['ctime_auc'] = ctime_gate['auc_real']
        out['ctime'] = ctime_gate
    return out
