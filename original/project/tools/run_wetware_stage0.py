"""Stage 0 GPU positive-control calibration, mixed-regime curriculum, and decoupled GAE training.

Upgrades for zero-new-collection training on existing MEA data:
1) Mixed-Regime Curriculum (`MixedRegimeEnvPool`): interleaves persistent `m9` exploration
   episodes (up to 240 ticks across updates) with SHA-pinned `thirst`, `hunger`, and `low_health`
   high-stress rollouts so the 1B LLM experiences genuine homeostatic crises during training.
2) Dual-Timescale Decoupled Advantage Injection:
   - The warm-started 108-dim Critic trains STRICTLY on uncorrupted long-horizon `R1` returns
     (`gamma_tick = 0.99985, gae_lambda = 0.999`).
   - Wetware `R2` uses `shape_for_decoupled_gae` (fixing the `|R1|~0.001` landmark-departure
     clipping dead-zone) and short-horizon GAE (`gamma_r2 = 0.95, lam_r2 = 0.90`), injected into
     Actor advantages at bounded gain `beta_adv`.
3) Joint Frozen Evaluation: evaluates standard unseen maps at fixed `Tick 96` (`new_cells`) AND
   high-stress archives at fixed `Tick 60` (`fear_of_thirst`, `fear_of_hunger`, `fear_of_death`,
   and high-stress `alive` rate) with zero MEA input at test time.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT, ROOT / 'standalone' / 'mea_oddball'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from oddball.signals import EpisodeContext, MODES, SensoryState, WetwareSignalSource
from tools import eval_survival_drives as ev
from tools import r2_guard as guard
from tools.stress_curriculum import MixedRegimeEnvPool, STRESS_HORIZON_TICK

# Ordered Stage 0 budget ladder (low -> high cost; freeze on first passing tier)
STAGE0_LADDER = (
    dict(tier=1, decisions=256, updates=4, rollout_steps=64, actor_lr=5e-6,
         target_share=0.06, beta_adv=0.15, reference_kl=0.02),
    dict(tier=2, decisions=512, updates=8, rollout_steps=64, actor_lr=5e-6,
         target_share=0.08, beta_adv=0.18, reference_kl=0.01),
    dict(tier=3, decisions=1024, updates=16, rollout_steps=64, actor_lr=5e-6,
         target_share=0.08, beta_adv=0.20, reference_kl=0.005),
    dict(tier=4, decisions=2048, updates=32, rollout_steps=64, actor_lr=3e-6,
         target_share=0.08, beta_adv=0.22, reference_kl=0.005),
)
NOISE_RHO_GRID = (0.70, 0.50, 0.35, 0.25, 0.15)
PROBE_EVERY_UPDATES = 2
HORIZON_TICK = 96


def digest_obj(obj):
    raw = json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def select_rho_min(rho_results):
    """Lowest rho such that this level and every higher level have 95% CI lower bound > 0."""
    ordered = sorted(rho_results, key=lambda r: r['rho'], reverse=True)
    rho_min = None
    for row in ordered:
        ci_low = row['stats'].get('ci_low')
        if ci_low is not None and ci_low > 0.0:
            rho_min = float(row['rho'])
        else:
            break
    return rho_min


def apply_calibrated_r2(
    rows,
    mode,
    source_kind,
    seed,
    r1_stats,
    target_share,
    *,
    rho=1.0,
    habituation_library=None,
    evoked_library=None,
    ctime_habituation_library=None,
    ctime_evoked_library=None,
    decoupled=True,
):
    """Attach calibrated R2 to rollout `rows` and return audit stats + updated r1_stats."""
    ctx_tracker = EpisodeContext()
    contexts = []
    r1_vals = []
    for row in rows:
        if row.get('episode_start'):
            ctx_tracker.reset()
        prev_sensory = (
            SensoryState.from_observation(row['observation'])
            if 'observation' in row and row['observation'] is not None else None
        )
        sensory = SensoryState.from_observation(row['next_observation'])
        contexts.append(ctx_tracker.encode_step(sensory, mode, prev_state=prev_sensory))
        r1_vals.append(float(row['r1_reward']))

    r1_arr = np.asarray(r1_vals, dtype=float)
    new_stats = guard.track_r1_stats(r1_stats, r1_arr)
    source = WetwareSignalSource(
        mode=mode,
        source=source_kind,
        seed=seed,
        rho=rho,
        habituation_library=habituation_library,
        evoked_library=evoked_library,
        ctime_habituation_library=ctime_habituation_library,
        ctime_evoked_library=ctime_evoked_library,
    )
    raw = source.score_batch(contexts)
    shaper = guard.shape_for_decoupled_gae if decoupled else guard.shape_calibrated
    r2 = shaper(r1_arr, raw, new_stats, target_share)
    share = guard.shares(r1_arr, r2)
    level, flags = guard.verdict(share)
    if level == 'redline':
        raise RuntimeError(f'R2 exceeded redline share: {share}')

    for row, c, raw_v, r2_v in zip(rows, contexts, raw, r2):
        row['r2_context'] = c
        row['r2_raw'] = float(raw_v)
        row['r2_shaped'] = float(r2_v)
        # Keep row['reward'] as pure R1 when decoupled so Critic targets remain 100% pure R1
        row['reward'] = float(row['r1_reward']) if decoupled else (float(row['r1_reward']) + float(r2_v))
    return dict(shares=share, level=level, flags=flags, r1_stats=new_stats, contexts=contexts, raw=raw, r2=r2)


def inject_decoupled_advantage(rows, boot_r1, gamma_tick, gae_lambda, beta_adv):
    """Compute pure R1 Critic returns + bounded short-horizon R2 Actor advantages in-place."""
    from agent.m9_run import finish_trajectory
    for r in rows:
        r['reward'] = float(r['r1_reward'])
    finish_trajectory(rows, boot_r1, gamma_tick, gae_lambda)
    adv_r1 = np.asarray([float(r['advantage']) for r in rows], dtype=float)
    r2_arr = np.asarray([float(r.get('r2_shaped', 0.0)) for r in rows], dtype=float)
    durations = [int(r.get('duration', 1)) for r in rows]
    terminals = [bool(r.get('terminal', False)) for r in rows]
    boundaries = [bool(r.get('boundary', False)) for r in rows]
    adv_r2 = guard.short_horizon_gae(r2_arr, durations, terminals, boundaries)
    mixed_adv, adv_snr = guard.mix_decoupled_advantages(adv_r1, adv_r2, beta_adv=beta_adv)
    for r, ma in zip(rows, mixed_adv):
        # `m8_train.update` trains Actor on `row['advantage']` and Critic on `row['return']` (pure R1!)
        r['advantage'] = float(ma)
    return dict(adv_snr=adv_snr, adv_r1_std=float(adv_r1.std()), adv_r2_std=float(adv_r2.std()))


def _run_episode_to_horizon(env, policy, world_seed, horizon_tick, rng_base):
    import torch
    state = env.reset(int(world_seed))
    obs = state['observation']
    rows = []
    elapsed = 0
    step_i = 0
    while elapsed < horizon_tick:
        prep = policy.prepare([obs])
        torch.manual_seed(int(rng_base + step_i))
        sampled = policy.act(prep, with_reference=False)
        act = sampled['actions'][0]
        nxt = env.step(act)
        rows.append(dict(observation=obs, next_observation=nxt['observation'],
                         action=act, transition=nxt['transition']))
        elapsed = int(nxt['transition']['after'].get('tick', elapsed + 1))
        obs = nxt['observation']
        step_i += 1
        if nxt['terminated']:
            break
    return ev.evaluate_episode(rows, horizon_tick=horizon_tick)


def _gpu_train_and_eval_one_seed(
    *,
    mode,
    source_kind,
    seed,
    tier_cfg,
    model_config_path,
    godot_path,
    project_path,
    eval_worlds,
    probe_worlds,
    rho=1.0,
    habituation_library=None,
    evoked_library=None,
    ctime_habituation_library=None,
    ctime_evoked_library=None,
):
    """Real 108-dim M9 PPO training (Mixed-Regime + Decoupled GAE) & standalone frozen evaluation."""
    import torch
    from agent.m8_train import update as ppo_update
    from agent.m9_contract import Reward
    from agent.m9_policy import Policy
    from tools.mea_offline import CONFIG as BASE_CONFIG

    model_paths = json.loads(Path(model_config_path).read_text(encoding='utf-8'))
    cfg = SimpleNamespace(
        **(BASE_CONFIG | model_paths | dict(
            actor_lr=tier_cfg['actor_lr'],
            reference_kl=tier_cfg['reference_kl'],
            minibatch=min(32, tier_cfg['rollout_steps']),
            critic_minibatch=min(32, tier_cfg['rollout_steps']),
            rng_seed=seed,
        ))
    )
    beta_adv = float(tier_cfg.get('beta_adv', 0.18)) if source_kind != 'none' else 0.0
    torch.manual_seed(seed)
    policy = Policy(cfg, reference=True)
    actor_opt = torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad],
                                  lr=cfg.actor_lr, eps=1e-5, weight_decay=0.0)
    critic_opt = torch.optim.AdamW(policy.critic.parameters(), lr=cfg.critic_lr, eps=1e-5, weight_decay=0.0)

    r1_stats = None
    probe_curve = []
    share_history = []
    regime_counts = {}

    with MixedRegimeEnvPool(godot_path, project_path,
                            map_mode=cfg.map_mode, navigation_steps=cfg.navigation_steps) as pool:

        def eval_frozen(worlds, tag_offset):
            ep_metrics = []
            stress_keys = sorted(pool.stress_envs.keys())
            with torch.no_grad():
                for idx, w in enumerate(worlds):
                    std_m = _run_episode_to_horizon(
                        pool.standard_env, policy, int(w), HORIZON_TICK,
                        rng_base=seed * 10000 + tag_offset + idx * 200,
                    )
                    if stress_keys:
                        s_scene = stress_keys[idx % len(stress_keys)]
                        str_m = _run_episode_to_horizon(
                            pool.stress_envs[s_scene], policy, int(w), STRESS_HORIZON_TICK,
                            rng_base=seed * 10000 + tag_offset + idx * 200 + 100,
                        )
                        # Standard Tick 96 provides `new_cells`; stress scene provides true crisis survival & drives
                        std_m['alive'] = str_m['alive']
                        for dk in ('fear_of_thirst_rate', 'thirst_drink_latency',
                                   'fear_of_hunger_rate', 'hunger_eat_latency',
                                   'fear_of_death_caution', 'fear_of_death_delta'):
                            if str_m.get(dk) is not None:
                                std_m[dk] = str_m[dk]
                    ep_metrics.append(std_m)
            return ep_metrics

        for upd in range(1, tier_cfg['updates'] + 1):
            world_seed = seed * 1000 + upd
            env, obs, tracker, ep_start, regime = pool.begin_rollout(
                upd, world_seed, Reward, cfg.reward_variant, cfg.gamma_tick
            )
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            rows = []
            for step in range(tier_cfg['rollout_steps']):
                prep = policy.prepare([obs])
                torch.manual_seed(seed * 100000 + upd * 1000 + step)
                sampled = policy.act(prep, with_reference=True)
                act = sampled['actions'][0]
                nxt = env.step(act)
                scored = tracker.score(nxt['transition'], nxt['observation'])
                rows.append(dict(
                    episode_start=bool(ep_start),
                    regime=regime,
                    prepared=prep[0],
                    observation=obs,
                    next_observation=nxt['observation'],
                    action=act,
                    transition=nxt['transition'],
                    verb=sampled['verbs'][0],
                    target=sampled['targets'][0],
                    value=sampled['values'][0],
                    logp=sampled['logp'][0],
                    verb_logp=sampled['verb_logp'][0],
                    reference_verb=sampled['reference_verb'][0],
                    reference_target=sampled['reference_target'][0],
                    r1_reward=float(scored['reward']),
                    reward=float(scored['reward']),
                    duration=int(scored['elapsed_ticks']),
                    terminal=bool(nxt['terminated']),
                    boundary=bool(nxt['terminated']),
                    next_obs=nxt['observation'],
                ))
                obs, tracker, ep_start = pool.update_persistent_state(
                    regime, nxt['observation'], tracker, bool(nxt['terminated']),
                    world_seed * 17 + step + 1, Reward, cfg.reward_variant, cfg.gamma_tick,
                )

            audit = apply_calibrated_r2(
                rows, mode, source_kind, seed * 997 + upd, r1_stats, tier_cfg['target_share'],
                rho=rho, habituation_library=habituation_library, evoked_library=evoked_library,
                ctime_habituation_library=ctime_habituation_library,
                ctime_evoked_library=ctime_evoked_library,
                decoupled=True,
            )
            r1_stats = audit['r1_stats']
            share_history.append(audit['shares'])
            with torch.no_grad():
                boot = 0.0 if rows[-1]['terminal'] else float(policy.values(policy.prepare([rows[-1]['next_obs']]))[0])
            adv_diag = inject_decoupled_advantage(rows, boot, cfg.gamma_tick, cfg.gae_lambda, beta_adv=beta_adv)
            ppo_update(policy, actor_opt, critic_opt, rows, cfg)

            if upd % PROBE_EVERY_UPDATES == 0 and probe_worlds:
                pm = eval_frozen(probe_worlds, tag_offset=upd * 500)
                probe_curve.append(dict(
                    update=upd,
                    new_cells=float(np.mean([m['new_cells'] for m in pm])),
                    alive_rate=float(np.mean([m['alive'] for m in pm])),
                    adv_snr=adv_diag['adv_snr'],
                    r2_mag_share=audit['shares']['magnitude'],
                    r2_var_share=audit['shares']['variance'],
                ))

        final_eval = eval_frozen(eval_worlds, tag_offset=999000)

    return dict(eval=final_eval, probe_curve=probe_curve, shares=share_history, regime_counts=regime_counts)


def _dry_run_simulate(mode, source_kind, seed, tier_cfg, eval_worlds, rho=1.0):
    """Deterministic lightweight dry-run model to verify Stage 0 pipeline logic without GPU."""
    rng = np.random.default_rng(seed * 131 + tier_cfg['tier'] * 17 + int(round(rho * 1000)))
    eff = 0.0
    if source_kind == 'ideal':
        eff = 1.0
    elif source_kind == 'noisy':
        eff = float(rho)
    beta_scale = float(tier_cfg.get('beta_adv', 0.15)) / 0.15
    strength = eff * np.sqrt(tier_cfg['updates'] / 4.0) * beta_scale * 2.4
    out = []
    for w in eval_worlds:
        world_rng = np.random.default_rng(int(w) * 31 + seed)
        base_cells = 40.0 + world_rng.normal(0.0, 4.0)
        cells = max(0, int(round(base_cells + strength + rng.normal(0.0, 2.5))))
        out.append(dict(
            new_cells=cells,
            alive=True,
            fear_of_thirst_rate=float(np.clip(0.55 + 0.04 * strength + rng.normal(0, 0.05), 0, 1)),
            fear_of_hunger_rate=float(np.clip(0.50 + 0.04 * strength + rng.normal(0, 0.05), 0, 1)),
            fear_of_death_caution=float(0.05 + 0.02 * strength + rng.normal(0, 0.03)),
            mean_ticks_per_action=1.35,
        ))
    probe = [
        dict(update=u, new_cells=40.0 + strength * (u / tier_cfg['updates']), alive_rate=1.0,
             adv_snr=float(tier_cfg.get('beta_adv', 0.15)) * 0.85 if eff > 0 else 0.0,
             r2_mag_share=tier_cfg['target_share'] if eff > 0 else 0.0,
             r2_var_share=tier_cfg['target_share'] * 0.8 if eff > 0 else 0.0)
        for u in range(PROBE_EVERY_UPDATES, tier_cfg['updates'] + 1, PROBE_EVERY_UPDATES)
    ]
    return dict(eval=out, probe_curve=probe, shares=[])


def run_stage0(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tune_seeds = list(range(args.seed_base, args.seed_base + args.tune_seeds))
    confirm_seeds = list(range(args.seed_base + 1000, args.seed_base + 1000 + args.confirm_seeds))
    eval_worlds = list(range(args.world_base, args.world_base + args.eval_worlds))
    probe_worlds = eval_worlds[:min(16, len(eval_worlds))]

    def run_arm(source_kind, seeds, tier_cfg, rho=1.0):
        mat = []
        curves = []
        for s in seeds:
            if args.dry_run:
                res = _dry_run_simulate(args.mode, source_kind, s, tier_cfg, eval_worlds, rho=rho)
            else:
                res = _gpu_train_and_eval_one_seed(
                    mode=args.mode, source_kind=source_kind, seed=s, tier_cfg=tier_cfg,
                    model_config_path=args.model_config, godot_path=args.godot,
                    project_path=args.project, eval_worlds=eval_worlds, probe_worlds=probe_worlds,
                    rho=rho,
                )
            key = 'new_cells' if args.mode in ('explore', 'dual') else 'fear_of_thirst_rate'
            mat.append([float(m[key]) if m[key] is not None else float('nan') for m in res['eval']])
            curves.append(dict(seed=s, probe=res['probe_curve']))
        return np.asarray(mat, dtype=float), curves

    # Step 4.1: Ordered budget ladder
    ladder_reports = []
    frozen_tier = None
    for tier_cfg in STAGE0_LADDER:
        mat_a, curves_a = run_arm('none', tune_seeds, tier_cfg)
        mat_d, curves_d = run_arm('ideal', tune_seeds, tier_cfg)
        stats = ev.paired_bootstrap_2way(mat_d, mat_a, seed=args.seed_base + tier_cfg['tier'], n_boot=args.n_boot)
        passed = bool(stats['ci_low'] is not None and stats['ci_low'] > 0.0)
        ladder_reports.append(dict(tier=tier_cfg, stats=stats, passed=passed,
                                   probe_a=curves_a[:2], probe_d=curves_d[:2]))
        if passed:
            frozen_tier = tier_cfg
            break

    report = dict(
        schema='wetware.stage0.v2',
        mode=args.mode,
        dry_run=bool(args.dry_run),
        curriculum='mixed_persistent_exploration_and_pinned_stress_50_50',
        advantage_injection='dual_timescale_decoupled_gae',
        tune_seeds=tune_seeds,
        confirm_seeds=confirm_seeds,
        eval_worlds=eval_worlds,
        ladder=ladder_reports,
        frozen_tier=frozen_tier,
        confirmed=False,
        rho_min=None,
        recommended_stage3_seeds=None,
    )

    if frozen_tier is not None:
        mat_a, _ = run_arm('none', confirm_seeds, frozen_tier)
        mat_d, _ = run_arm('ideal', confirm_seeds, frozen_tier)
        confirm_stats = ev.paired_bootstrap_2way(mat_d, mat_a, seed=args.seed_base + 500, n_boot=args.n_boot)
        report['confirm'] = confirm_stats
        report['confirmed'] = bool(confirm_stats['ci_low'] is not None and confirm_stats['ci_low'] > 0.0)

        if report['confirmed']:
            rho_rows = []
            for idx, rho in enumerate(NOISE_RHO_GRID):
                mat_f, _ = run_arm('noisy', confirm_seeds, frozen_tier, rho=rho)
                st = ev.paired_bootstrap_2way(mat_f, mat_a, seed=args.seed_base + 600 + idx, n_boot=args.n_boot)
                rho_rows.append(dict(rho=rho, stats=st))
            rho_min = select_rho_min(rho_rows)
            report['noise_curve'] = rho_rows
            report['rho_min'] = rho_min
            if rho_min is not None:
                min_row = next(r for r in rho_rows if r['rho'] == rho_min)
                report['recommended_stage3_seeds'] = ev.required_seeds_power90(
                    min_row['stats']['mean'], min_row['stats']['seed_std'], min_seeds=32, max_seeds=64
                )

    report['sha256'] = digest_obj(report)
    (out_dir / 'stage0_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=MODES, default='dual')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--model-config', type=Path)
    p.add_argument('--godot', type=Path)
    p.add_argument('--project', type=Path, default=ROOT)
    p.add_argument('--tune-seeds', type=int, default=16)
    p.add_argument('--confirm-seeds', type=int, default=32)
    p.add_argument('--eval-worlds', type=int, default=48)
    p.add_argument('--seed-base', type=int, default=20261123)
    p.add_argument('--world-base', type=int, default=900000)
    p.add_argument('--n-boot', type=int, default=10000)
    return p


if __name__ == '__main__':
    res = run_stage0(build_parser().parse_args())
    print(json.dumps(dict(
        mode=res['mode'], confirmed=res['confirmed'], frozen_tier=res['frozen_tier'],
        rho_min=res['rho_min'], recommended_stage3_seeds=res['recommended_stage3_seeds']
    ), ensure_ascii=False))
