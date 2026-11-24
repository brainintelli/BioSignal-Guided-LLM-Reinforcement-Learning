"""Self-contained 8-GPU parallel runner for Existing-Data Wetware Training (Phase 1, 2, 3).

Supports:
  - Worker mode (`--worker`): runs ONE `(arm, seed)` on its assigned `CUDA_VISIBLE_DEVICES` GPU
    using B4-batched 4-env curriculum rollouts (`godot_tests/m9_curriculum_environment.gd`),
    optional Decoupled GAE (pure R1 Critic + bounded short-horizon Dual-Drive R2 Actor),
    optional M1 (`WetwareDynamicsExplorer.sample_batch`) + M2 (`WetwareEntropyModulator`),
    and B4-batched pure 108-dim frozen evaluation across E1-E4 non-ceiling suites.
  - Orchestrator mode (`--orchestrate smoke | phase1 | phase2 | phase3`): schedules jobs across
    all specified GPUs (e.g. 0..7 on 8x RTX 4090), supports atomic crash-safe resume, and writes
    two-way paired bootstrap + CIF decomposition + graded verdict reports.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT, ROOT / 'standalone' / 'mea_oddball'):
    s = str(extra)
    if s not in sys.path:
        sys.path.insert(0, s)

from agent import mea_dynamics_explorer as mde
from oddball import library_v2
from oddball.signals import EpisodeContext, MODES, SensoryState, WetwareSignalSource
from tools import eval_survival_drives as ev
from tools import r2_guard as guard
from tools import run_wetware_spontaneous_study as sp
from tools import run_wetware_stage0 as s0
from tools import run_wetware_stage3 as s3

CURRICULUM_ENV_SCRIPT = 'godot_tests/m9_curriculum_environment.gd'
DEFAULT_SUITES = ('E1_explore_96', 'E2_thirst_acute', 'E3_hunger_acute', 'E4_death_night_threat')
LIBRARY_SEVEN_ARMS = (
    ('A', 'none'),
    ('B', 'wetware_real'),
    ('C', 'wetware_shuffled'),
    ('C_time', 'wetware_time_shuffled'),
    ('D', 'ideal'),
    ('E', 'wetware_inverted'),
    ('F', 'noisy'),
)
NOISE_RHO_GRID = (0.70, 0.50, 0.35, 0.25, 0.15)
STAGE0_LADDER_TIERS = (1, 2, 3, 4)

TIERS = {
    0: dict(tier=0, decisions=256, updates=2, rollout_steps=128, envs=4, actor_lr=5e-6, target_share=0.06, beta_adv=0.18, reference_kl=0.01),
    1: dict(tier=1, decisions=512, updates=4, rollout_steps=128, envs=4, actor_lr=5e-6, target_share=0.06, beta_adv=0.18, reference_kl=0.01),
    2: dict(tier=2, decisions=1024, updates=8, rollout_steps=128, envs=4, actor_lr=5e-6, target_share=0.08, beta_adv=0.18, reference_kl=0.005),
    3: dict(tier=3, decisions=2048, updates=16, rollout_steps=128, envs=4, actor_lr=5e-6, target_share=0.08, beta_adv=0.18, reference_kl=0.005),
    4: dict(tier=4, decisions=4096, updates=32, rollout_steps=128, envs=4, actor_lr=3e-6, target_share=0.08, beta_adv=0.18, reference_kl=0.003),
}


def encode_rollout_contexts(all_rows: list[dict], r2_mode: str = 'dual') -> tuple[list[dict], np.ndarray]:
    """Encode rollout transitions with strict observable `prev_state` for crisis relief switching."""
    ctx_tracker = EpisodeContext()
    contexts = []
    r1_vals = []
    for r in all_rows:
        if r.get('episode_start'):
            ctx_tracker.reset()
        prev_obs = r.get('observation')
        prev_s = SensoryState.from_observation(prev_obs) if prev_obs is not None else None
        next_s = SensoryState.from_observation(r['next_observation'])
        contexts.append(ctx_tracker.encode_step(next_s, r2_mode, prev_state=prev_s))
        r1_vals.append(float(r['r1_reward']))
    return contexts, np.asarray(r1_vals, dtype=float)


def _load_checked_library(path: Path | None, schema: str) -> dict | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'Required library not found: {p}')
    return library_v2.check_library(json.loads(p.read_text(encoding='utf-8')), schema)


def digest_obj(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def atomic_write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def run_one_gpu_job(
    *,
    arm_name: str,
    r2_source: str,
    explorer_arm: str,
    seed: int,
    tier_cfg: dict,
    model_config_path: Path,
    godot_path: Path,
    project_path: Path,
    eval_worlds: list[int],
    eval_suites: tuple[str, ...] = DEFAULT_SUITES,
    library_manifest: Path | None = None,
    allow_non_gate_s: bool = False,
    eps_base: float = 0.10,
    enable_m2: bool = False,
    entropy_eta: float = 0.35,
    use_curriculum: bool = True,
    microbatch: int = 2,
    r2_mode: str = 'dual',
    habituation_library_path: Path | None = None,
    evoked_library_path: Path | None = None,
    ctime_habituation_library_path: Path | None = None,
    ctime_evoked_library_path: Path | None = None,
    rho_lib: float = 1.0,
) -> dict:
    """Train one 108-dim M9 policy on one GPU and run B4-batched frozen evaluation across E1-E4."""
    import torch
    from agent.m8_train import update as ppo_update
    from agent.m9_contract import Reward
    from agent.m9_env import GodotEnv
    from agent.m9_policy import Policy
    from agent.m9_run import finish_trajectory
    from tools.mea_offline import CONFIG as BASE_CONFIG

    t0 = time.monotonic()
    model_paths = json.loads(Path(model_config_path).read_text(encoding='utf-8'))
    cfg = SimpleNamespace(
        **(BASE_CONFIG | model_paths | dict(
            actor_lr=tier_cfg['actor_lr'],
            reference_kl=tier_cfg['reference_kl'],
            minibatch=min(32, tier_cfg['rollout_steps']),
            critic_minibatch=min(32, tier_cfg['rollout_steps']),
            microbatch=int(microbatch),
            rng_seed=seed,
        ))
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    policy = Policy(cfg, reference=True)
    initial_digest = policy.digest()
    actor_opt = torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad],
                                  lr=cfg.actor_lr, eps=1e-5, weight_decay=0.0)
    critic_opt = torch.optim.AdamW(policy.critic.parameters(), lr=cfg.critic_lr, eps=1e-5, weight_decay=0.0)
    env_script = CURRICULUM_ENV_SCRIPT if use_curriculum else 'godot_tests/m9_environment.gd'

    lib = None
    if explorer_arm in ('real', 'shuffled', 'white', 'iaaft'):
        if library_manifest is None:
            raise ValueError(f'explorer_arm={explorer_arm} requires --library-manifest')
        lib = mde.load_dynamics_library(library_manifest, require_gate_s=not allow_non_gate_s)
    explorer = (
        None if explorer_arm == 'none'
        else mde.WetwareDynamicsExplorer(explorer_arm, lib, seed=seed, eps_base=eps_base, homeostatic_gate=True)
    )
    ent_mod = mde.WetwareEntropyModulator(
        eta=entropy_eta,
        enabled=bool(enable_m2 and explorer_arm not in ('none', 'static')),
    )
    hab_lib = _load_checked_library(habituation_library_path, library_v2.HABITUATION_SCHEMA)
    evk_lib = _load_checked_library(evoked_library_path, library_v2.EVOKED_SCHEMA)
    ctime_hab_lib = _load_checked_library(ctime_habituation_library_path, library_v2.HABITUATION_SCHEMA)
    ctime_evk_lib = _load_checked_library(ctime_evoked_library_path, library_v2.EVOKED_SCHEMA)
    library_sha256 = {
        k: v['sha256']
        for k, v in (
            ('habituation', hab_lib),
            ('evoked', evk_lib),
            ('ctime_habituation', ctime_hab_lib),
            ('ctime_evoked', ctime_evk_lib),
        )
        if v is not None and 'sha256' in v
    }

    def eval_frozen_suite(worlds: list[int], tag_offset: int, suite_name: str, batch_envs: int = 4) -> list[dict]:
        suite = ev.EVAL_SUITES[suite_name]
        # Frozen evaluation MUST ALWAYS use the exact same E1-E4 scenarios across ALL arms (including A0_native_r1)!
        scenario = suite['scenario']
        horizon = int(suite['horizon_ticks'])
        n_slots = min(max(1, int(batch_envs)), len(worlds))
        results_by_idx: list[dict | None] = [None] * len(worlds)
        eenvs = [
            GodotEnv(str(godot_path), str(project_path),
                     map_mode=cfg.map_mode, navigation_steps=cfg.navigation_steps,
                     env_script=CURRICULUM_ENV_SCRIPT)
            for _ in range(n_slots)
        ]
        try:
            next_w = 0
            active = {}

            def start_slot(slot: int):
                nonlocal next_w
                if next_w >= len(worlds):
                    return False
                w_idx = next_w
                next_w += 1
                st = eenvs[slot].reset(int(worlds[w_idx]), scenario=scenario)
                obs = st['observation']
                st_tick = int(obs.get('tick', 0))
                gen = torch.Generator(device=policy.device_name).manual_seed(
                    int(seed) * 100003 + int(tag_offset) + int(worlds[w_idx]) * 17
                )
                active[slot] = dict(idx=w_idx, obs=obs, cutoff=st_tick + horizon, rows=[], gen=gen)
                return True

            for sl in range(n_slots):
                start_slot(sl)

            with torch.no_grad():
                while active:
                    slots = sorted(active.keys())
                    prep_batch = policy.prepare([active[sl]['obs'] for sl in slots])
                    with policy._scope(prep_batch, False):
                        vd = policy.verb_distribution(prep_batch, reference=False)
                        vs = [int(torch.multinomial(vd.probs[j], 1, generator=active[sl]['gen']).item())
                              for j, sl in enumerate(slots)]
                        td = policy.target_distribution(prep_batch, vs, reference=False)
                        ts = [int(torch.multinomial(td.probs[j], 1, generator=active[sl]['gen']).item())
                              for j, sl in enumerate(slots)]
                    done_slots = []
                    for j, sl in enumerate(slots):
                        act = prep_batch[j]['branches'][vs[j]][ts[j]]
                        nxt = eenvs[sl].step(act)
                        rec = active[sl]
                        rec['rows'].append(dict(
                            observation=rec['obs'], next_observation=nxt['observation'],
                            action=act, transition=nxt['transition'],
                        ))
                        elapsed = int(nxt['transition']['after'].get('tick', rec['cutoff']))
                        rec['obs'] = nxt['observation']
                        if nxt['terminated'] or elapsed >= rec['cutoff']:
                            results_by_idx[rec['idx']] = ev.evaluate_episode(
                                rec['rows'], horizon_tick=horizon, relative_horizon=True
                            )
                            done_slots.append(sl)
                    for sl in done_slots:
                        active.pop(sl)
                        start_slot(sl)
        finally:
            for eenv in eenvs:
                eenv.close()
        return results_by_idx

    # Record initial verb probabilities on 4 fixed probe states to measure real policy plasticity (TV shift)
    def measure_anchor_verb_probs() -> np.ndarray:
        with torch.no_grad(), GodotEnv(
            str(godot_path), str(project_path),
            map_mode=cfg.map_mode, navigation_steps=cfg.navigation_steps,
            env_script=CURRICULUM_ENV_SCRIPT,
        ) as penv:
            obs4 = []
            for si, sname in enumerate(('E1_explore_96', 'E2_thirst_acute', 'E3_hunger_acute', 'E4_death_night_threat')):
                sc = ev.EVAL_SUITES[sname]['scenario']
                obs4.append(penv.reset(int(eval_worlds[0]) + si, scenario=sc)['observation'])
            prep4 = policy.prepare(obs4)
            with policy._scope(prep4, False):
                vd = policy.verb_distribution(prep4, reference=False)
                return vd.probs.detach().cpu().numpy()

    anchor_probs_before = measure_anchor_verb_probs()

    # Pre-warm frozen R1 scale stats across all 4 curriculum regimes before update 1
    def warmup_r1_stats():
        r1_samples = []
        wenvs = [
            GodotEnv(str(godot_path), str(project_path),
                     map_mode=cfg.map_mode, navigation_steps=cfg.navigation_steps,
                     env_script=env_script)
            for _ in range(4)
        ]
        try:
            obs_l, trks, alive = [], [], [True] * 4
            for sc_i in range(4):
                sc = ev.sample_curriculum_scenario(sc_i * 2) if use_curriculum else None
                st = wenvs[sc_i].reset(seed * 73 + sc_i, scenario=sc)
                obs_l.append(st['observation'])
                trks.append(Reward(st['observation'], cfg.reward_variant, cfg.gamma_tick))
            with torch.no_grad():
                for stp in range(16):
                    live = [i for i in range(4) if alive[i]]
                    if not live:
                        break
                    prep = policy.prepare([obs_l[i] for i in live])
                    torch.manual_seed(seed * 50000 + stp)
                    smp = policy.act(prep, with_reference=False)
                    for j, i in enumerate(live):
                        nxt = wenvs[i].step(smp['actions'][j])
                        r1_samples.append(float(trks[i].score(nxt['transition'], nxt['observation'])['reward']))
                        obs_l[i] = nxt['observation']
                        if nxt['terminated']:
                            alive[i] = False
        finally:
            for wenv in wenvs:
                wenv.close()
        return guard.track_r1_stats(None, r1_samples) if r1_samples else None

    r1_stats = warmup_r1_stats()
    beta_adv = float(tier_cfg.get('beta_adv', 0.18)) if r2_source != 'none' else 0.0
    n_envs = max(1, int(tier_cfg.get('envs', 4)))
    steps_per_env = max(1, int(tier_cfg['rollout_steps']) // n_envs)

    envs = [
        GodotEnv(str(godot_path), str(project_path),
                 map_mode=cfg.map_mode, navigation_steps=cfg.navigation_steps,
                 env_script=env_script)
        for _ in range(n_envs)
    ]
    scenario_cursor = 0
    update_logs = []
    t_train_start = time.monotonic()
    try:
        for upd in range(1, int(tier_cfg['updates']) + 1):
            env_trajs = [[] for _ in range(n_envs)]
            obs_l, trks, wseeds = [], [], []
            ep_starts = [True] * n_envs
            for ei, env in enumerate(envs):
                ws = seed * 1000 + upd * 17 + ei
                wseeds.append(ws)
                sc = ev.sample_curriculum_scenario(scenario_cursor) if use_curriculum else None
                scenario_cursor += 1
                init = env.reset(ws, scenario=sc)
                if explorer is not None:
                    explorer.reset_episode(ei)
                obs_l.append(init['observation'])
                trks.append(Reward(init['observation'], cfg.reward_variant, cfg.gamma_tick))

            for step in range(steps_per_env):
                prep_b = policy.prepare(obs_l)
                gens = [
                    torch.Generator(device=policy.device_name).manual_seed(
                        seed * 100000 + upd * 1000 + ei * 200 + step
                    )
                    for ei in range(n_envs)
                ]
                if explorer is not None:
                    smp = explorer.sample_batch(policy, prep_b, obs_l, generators=gens)
                else:
                    torch.manual_seed(seed * 100000 + upd * 1000 + step)
                    smp = policy.act(prep_b, with_reference=True)

                for ei, env in enumerate(envs):
                    act = smp['actions'][ei]
                    nxt = env.step(act)
                    scored = trks[ei].score(nxt['transition'], nxt['observation'])
                    row_item = dict(
                        episode_start=ep_starts[ei],
                        prepared=prep_b[ei],
                        observation=obs_l[ei],
                        next_observation=nxt['observation'],
                        action=act,
                        transition=nxt['transition'],
                        verb=smp['verbs'][ei],
                        target=smp['targets'][ei],
                        value=smp['values'][ei],
                        logp=smp['logp'][ei],
                        verb_logp=smp['verb_logp'][ei],
                        reference_verb=smp['reference_verb'][ei],
                        reference_target=smp['reference_target'][ei],
                        r1_reward=float(scored['reward']),
                        reward=float(scored['reward']),
                        duration=int(scored['elapsed_ticks']),
                        terminal=bool(nxt['terminated']),
                        boundary=bool(nxt['terminated']),
                        next_obs=nxt['observation'],
                        mea_burst_z=float(smp['mea_burst_z'][ei]) if 'mea_burst_z' in smp else 0.0,
                    )
                    if 'beh_logp' in smp:
                        row_item['beh_logp'] = float(smp['beh_logp'][ei])
                    env_trajs[ei].append(row_item)
                    if nxt['terminated']:
                        sc = ev.sample_curriculum_scenario(scenario_cursor) if use_curriculum else None
                        scenario_cursor += 1
                        init = env.reset(wseeds[ei] * 17 + step + 1, scenario=sc)
                        if explorer is not None:
                            explorer.reset_episode(ei)
                        obs_l[ei] = init['observation']
                        trks[ei] = Reward(init['observation'], cfg.reward_variant, cfg.gamma_tick)
                        ep_starts[ei] = True
                    else:
                        obs_l[ei] = nxt['observation']
                        ep_starts[ei] = False

            for tr in env_trajs:
                if tr:
                    tr[-1]['boundary'] = True

            all_rows = [r for tr in env_trajs for r in tr]
            # Calibrate R2 signal with strict observable `prev_state` for crisis relief switching
            contexts, r1_arr = encode_rollout_contexts(all_rows, r2_mode=r2_mode)
            r1_stats = guard.track_r1_stats(r1_stats, r1_arr)
            src = WetwareSignalSource(
                mode=r2_mode,
                source=r2_source,
                seed=seed * 997 + upd,
                rho=float(rho_lib),
                habituation_library=hab_lib,
                evoked_library=evk_lib,
                ctime_habituation_library=ctime_hab_lib,
                ctime_evoked_library=ctime_evk_lib,
            )
            raw_sig = src.score_batch(contexts)
            r2_shaped = guard.shape_for_decoupled_gae(r1_arr, raw_sig, r1_stats, float(tier_cfg['target_share']))
            sh = guard.shares(r1_arr, r2_shaped)
            lvl, _ = guard.verdict(sh)
            if lvl == 'redline':
                raise RuntimeError(f'R2 exceeded redline share: {sh}')
            for r, r2v in zip(all_rows, r2_shaped):
                r['r2_shaped'] = float(r2v)

            # Pure R1 GAE per trajectory -> Critic learns strictly uncorrupted R1 returns!
            for tr in env_trajs:
                with torch.no_grad():
                    boot = 0.0 if tr[-1]['terminal'] else float(policy.values(policy.prepare([tr[-1]['next_obs']]))[0])
                finish_trajectory(tr, boot, cfg.gamma_tick, cfg.gae_lambda)

            # Decoupled short-horizon R2 advantage mixing for Actor (only when beta_adv > 0)
            adv_snr = 0.0
            if beta_adv > 0.0:
                adv_r1 = np.asarray([r['advantage'] for r in all_rows], dtype=float)
                adv_r2 = guard.short_horizon_gae(
                    [r['r2_shaped'] for r in all_rows],
                    [r['duration'] for r in all_rows],
                    [r['terminal'] for r in all_rows],
                    [r['boundary'] for r in all_rows],
                )
                mixed_adv, adv_snr = guard.mix_decoupled_advantages(adv_r1, adv_r2, beta_adv=beta_adv)
                for r, ma in zip(all_rows, mixed_adv):
                    r['advantage'] = float(ma)

            upd_cfg = SimpleNamespace(**vars(cfg))
            if ent_mod.enabled:
                upd_cfg.entropy_weights = ent_mod.batch_weights(all_rows)
            um = ppo_update(policy, actor_opt, critic_opt, all_rows, upd_cfg)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            update_logs.append(dict(
                update=upd,
                approx_kl=float(um['approx_kl']),
                clip_fraction=float(um['clip_fraction']),
                kl_early_stop=bool(um['kl_early_stop']),
                optimizer_steps=int(um['optimizer_steps']),
                zero_update_logp_max=float(um['zero_update_logp_max']),
                explained_variance_after=um['explained_variance_after'],
                adv_snr=float(adv_snr),
                r2_mag_share=float(sh['magnitude']),
                r2_var_share=float(sh['variance']),
                in_stress_frac=float(np.mean([c['in_stress'] for c in contexts])),
                relieved_frac=float(np.mean([1.0 if c.get('deficit_relieved') else 0.0 for c in contexts])),
            ))
    finally:
        for env in envs:
            env.close()

    train_seconds = time.monotonic() - t_train_start
    final_digest = policy.digest()
    anchor_probs_after = measure_anchor_verb_probs()
    anchor_tv_shift = float(0.5 * np.abs(anchor_probs_after - anchor_probs_before).sum(axis=1).mean())

    # Pure 108-dim frozen evaluation across E1-E4
    t_eval_start = time.monotonic()
    suites_out = {}
    for si, sname in enumerate(eval_suites):
        suites_out[sname] = eval_frozen_suite(eval_worlds, tag_offset=999000 + si * 10000, suite_name=sname, batch_envs=4)
    eval_seconds = time.monotonic() - t_eval_start
    peak_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0

    # Compact per-suite summary means
    suite_means = {}
    for sname, eps in suites_out.items():
        def _nanmean(key):
            vals = [float(e[key]) for e in eps if e.get(key) is not None]
            return float(np.mean(vals)) if vals else None
        suite_means[sname] = dict(
            new_cells=_nanmean('new_cells'),
            alive_rate=float(np.mean([1.0 if e['alive'] else 0.0 for e in eps])),
            fear_of_thirst_rate=_nanmean('fear_of_thirst_rate'),
            fear_of_hunger_rate=_nanmean('fear_of_hunger_rate'),
            threat_evasion_rate=_nanmean('threat_evasion_rate'),
            fear_of_death_caution=_nanmean('fear_of_death_caution'),
            threat_damage_taken=_nanmean('threat_damage_taken'),
            mean_ticks_per_action=_nanmean('mean_ticks_per_action'),
        )

    return dict(
        schema='wetware.gpu.job.v1',
        arm_name=arm_name,
        r2_mode=r2_mode,
        r2_source=r2_source,
        rho_lib=float(rho_lib),
        library_sha256=library_sha256,
        explorer_arm=explorer_arm,
        seed=int(seed),
        dry_run=False,
        tier=tier_cfg,
        use_curriculum=bool(use_curriculum),
        enable_m2=bool(enable_m2),
        initial_actor_sha256=initial_digest,
        final_actor_sha256=final_digest,
        weights_changed=(initial_digest != final_digest),
        anchor_tv_shift=anchor_tv_shift,
        cuda_peak_mb=round(peak_mb, 1),
        train_seconds=round(train_seconds, 2),
        eval_seconds=round(eval_seconds, 2),
        total_seconds=round(time.monotonic() - t0, 2),
        update_logs=update_logs,
        suite_means=suite_means,
        suites=suites_out,
    )


def _extract_matrix(job_list: list[dict], suite_name: str, metric_key: str) -> np.ndarray:
    mat = []
    for job in job_list:
        eps = job['suites'][suite_name]
        row = []
        for e in eps:
            v = e.get(metric_key)
            if isinstance(v, bool):
                row.append(1.0 if v else 0.0)
            elif v is None:
                row.append(float('nan'))
            else:
                row.append(float(v))
        mat.append(row)
    return np.asarray(mat, dtype=float)


def simulate_library_job(job: dict, eval_worlds: list[int], *, dry_run_strength: float = 1.0) -> dict:
    """Deterministic 0-GPU dry-run job matching the exact `wetware.gpu.job.v1` schema.

    When real library JSON paths are provided, measures actual empirical alignment against
    the ideal Dual-Drive signal via `s3._measure_library_effective_alignment` (so `B` reflects
    real library SNR while `C` and `C_time` naturally collapse near 0).
    """
    arm_name = job['arm_name']
    r2_mode = job.get('r2_mode', 'dual')
    r2_source = job['r2_source']
    seed = int(job['seed'])
    tier_cfg = TIERS[int(job['tier'])]
    use_curr = bool(job.get('use_curriculum', True))
    rho_lib = float(job.get('rho_lib', 1.0))

    hab_lib = _load_checked_library(job.get('habituation_library'), library_v2.HABITUATION_SCHEMA)
    evk_lib = _load_checked_library(job.get('evoked_library'), library_v2.EVOKED_SCHEMA)
    ctime_hab = _load_checked_library(job.get('ctime_habituation_library'), library_v2.HABITUATION_SCHEMA)
    ctime_evk = _load_checked_library(job.get('ctime_evoked_library'), library_v2.EVOKED_SCHEMA)
    lib_sha = {
        k: v['sha256']
        for k, v in (('habituation', hab_lib), ('evoked', evk_lib),
                     ('ctime_habituation', ctime_hab), ('ctime_evoked', ctime_evk))
        if v is not None and 'sha256' in v
    }

    has_libs = (
        (r2_mode in ('explore', 'dual') and hab_lib is not None)
        or (r2_mode in ('homeo', 'dual') and evk_lib is not None)
    )
    if has_libs and r2_source in ('wetware_real', 'wetware_shuffled', 'wetware_time_shuffled', 'wetware_inverted'):
        emp_corr = s3._measure_library_effective_alignment(
            r2_mode, r2_source, seed, rho_lib, hab_lib, evk_lib, ctime_hab, ctime_evk
        )
        eff = emp_corr * float(dry_run_strength)
    elif r2_source in ('none', 'wetware_shuffled', 'wetware_time_shuffled'):
        eff = 0.0
    elif r2_source == 'ideal':
        eff = 1.0
    elif r2_source == 'noisy':
        eff = float(rho_lib)
    elif r2_source == 'wetware_real':
        eff = float(rho_lib) * float(dry_run_strength)
    elif r2_source == 'wetware_inverted':
        eff = -float(rho_lib) * float(dry_run_strength)
    else:
        raise ValueError(f'Unsupported r2_source: {r2_source}')

    rng = np.random.default_rng(seed * 257 + sum(ord(ch) for ch in arm_name) * 31 + int(tier_cfg['tier']) * 13)
    scale = np.sqrt(tier_cfg['updates'] / 4.0) * (float(tier_cfg['beta_adv']) / 0.18)
    gain = eff * scale * 3.0
    curr_surv_boost = 0.04 if use_curr else 0.0

    suites_out = {s: [] for s in DEFAULT_SUITES}
    for w in eval_worlds:
        wrng = np.random.default_rng(int(w) * 19 + seed)
        e1_cells = max(0, int(round(43.0 + wrng.normal(0.0, 3.6) + gain + rng.normal(0.0, 2.0))))
        suites_out['E1_explore_96'].append(dict(
            new_cells=e1_cells, alive=True, mean_ticks_per_action=float(1.36 + rng.normal(0.0, 0.01)),
            fear_of_thirst_rate=None, fear_of_hunger_rate=None, threat_evasion_rate=None,
            fear_of_death_caution=None, threat_damage_taken=0.0,
        ))
        drank = bool(wrng.random() < np.clip(0.48 + curr_surv_boost + 0.025 * gain, 0.05, 0.98))
        suites_out['E2_thirst_acute'].append(dict(
            new_cells=max(0, int(round(12.0 + rng.normal(0.0, 2.0)))),
            alive=drank,
            fear_of_thirst_rate=float(np.clip(0.52 + curr_surv_boost + 0.035 * gain + rng.normal(0.0, 0.04), 0.0, 1.0)),
            thirst_drink_latency=float(np.clip(14.0 - 0.6 * gain + rng.normal(0.0, 2.0), 1.0, 60.0)) if drank else 60.0,
            drank_after_low=drank,
            fear_of_hunger_rate=None, threat_evasion_rate=None, fear_of_death_caution=None,
            threat_damage_taken=0.0, mean_ticks_per_action=1.28,
        ))
        ate = bool(wrng.random() < np.clip(0.56 + curr_surv_boost + 0.022 * gain, 0.05, 0.98))
        suites_out['E3_hunger_acute'].append(dict(
            new_cells=max(0, int(round(11.0 + rng.normal(0.0, 2.0)))),
            alive=ate,
            fear_of_hunger_rate=float(np.clip(0.50 + curr_surv_boost + 0.035 * gain + rng.normal(0.0, 0.04), 0.0, 1.0)),
            hunger_eat_latency=float(np.clip(16.0 - 0.6 * gain + rng.normal(0.0, 2.2), 1.0, 60.0)) if ate else 60.0,
            ate_after_low=ate,
            fear_of_thirst_rate=None, threat_evasion_rate=None, fear_of_death_caution=None,
            threat_damage_taken=0.0, mean_ticks_per_action=1.30,
        ))
        evaded = bool(wrng.random() < np.clip(0.54 + curr_surv_boost + 0.028 * gain, 0.05, 0.98))
        suites_out['E4_death_night_threat'].append(dict(
            new_cells=max(0, int(round(10.0 + rng.normal(0.0, 2.0)))),
            alive=evaded,
            threat_evasion_rate=float(np.clip(0.54 + curr_surv_boost + 0.035 * gain + rng.normal(0.0, 0.04), 0.0, 1.0)),
            fear_of_death_caution=float(0.05 + 0.020 * gain + rng.normal(0.0, 0.02)),
            threat_damage_taken=float(np.clip(4.0 - 0.4 * gain + rng.normal(0.0, 1.0), 0.0, 18.0)) if evaded else 18.0,
            fear_of_thirst_rate=None, fear_of_hunger_rate=None, mean_ticks_per_action=1.22,
        ))

    suite_means = {}
    for sname, eps in suites_out.items():
        def _nm(key):
            vals = [float(e[key]) for e in eps if e.get(key) is not None]
            return float(np.mean(vals)) if vals else None
        suite_means[sname] = dict(
            new_cells=_nm('new_cells'),
            alive_rate=float(np.mean([1.0 if e['alive'] else 0.0 for e in eps])),
            fear_of_thirst_rate=_nm('fear_of_thirst_rate'),
            fear_of_hunger_rate=_nm('fear_of_hunger_rate'),
            threat_evasion_rate=_nm('threat_evasion_rate'),
            fear_of_death_caution=_nm('fear_of_death_caution'),
            threat_damage_taken=_nm('threat_damage_taken'),
            mean_ticks_per_action=_nm('mean_ticks_per_action'),
        )

    r2_active = r2_source != 'none'
    update_logs = [
        dict(
            update=u,
            approx_kl=0.0025,
            clip_fraction=0.04,
            kl_early_stop=False,
            optimizer_steps=4,
            zero_update_logp_max=0.0,
            explained_variance_after=0.55,
            adv_snr=float(tier_cfg['beta_adv'] * 0.85) if r2_active else 0.0,
            r2_mag_share=float(tier_cfg['target_share']) if r2_active else 0.0,
            r2_var_share=float(tier_cfg['target_share'] * 0.78) if r2_active else 0.0,
            in_stress_frac=0.45 if use_curr else 0.0,
            relieved_frac=0.18 if use_curr else 0.0,
        )
        for u in range(1, int(tier_cfg['updates']) + 1)
    ]
    init_sha = digest_obj({'dry_init': seed})
    final_sha = digest_obj({'dry_final': arm_name, 'seed': seed, 'tier': tier_cfg['tier']})
    return dict(
        schema='wetware.gpu.job.v1',
        arm_name=arm_name,
        r2_mode=r2_mode,
        r2_source=r2_source,
        rho_lib=rho_lib,
        library_sha256=lib_sha,
        explorer_arm=job.get('explorer_arm', 'none'),
        seed=seed,
        dry_run=True,
        tier=tier_cfg,
        use_curriculum=use_curr,
        enable_m2=bool(job.get('enable_m2', False)),
        initial_actor_sha256=init_sha,
        final_actor_sha256=final_sha,
        weights_changed=True,
        anchor_tv_shift=float(0.015 + 0.006 * abs(gain)),
        cuda_peak_mb=0.0,
        train_seconds=0.01,
        eval_seconds=0.01,
        total_seconds=0.02,
        update_logs=update_logs,
        suite_means=suite_means,
        suites=suites_out,
    )


def run_parallel_jobs(
    jobs: list[dict],
    gpus: list[int],
    out_dir: Path,
    common_cli: list[str],
    *,
    dry_run: bool = False,
    eval_worlds: list[int] | None = None,
    dry_run_strength: float = 1.0,
) -> dict[tuple[str, int], dict]:
    """Execute independent `(arm, seed)` worker subprocesses across all specified GPUs with atomic resume."""
    jobs_dir = out_dir / 'jobs'
    jobs_dir.mkdir(parents=True, exist_ok=True)
    gpu_q: queue.Queue[int] = queue.Queue()
    for g in gpus:
        gpu_q.put(g)

    results: dict[tuple[str, int], dict] = {}
    pending = []
    for job in jobs:
        jfile = jobs_dir / f"{job['arm_name']}_seed{job['seed']}.json"
        if jfile.is_file():
            results[(job['arm_name'], job['seed'])] = json.loads(jfile.read_text(encoding='utf-8'))
        else:
            pending.append((job, jfile))

    print(f"[ORCH] Total jobs={len(jobs)}, cached={len(results)}, running={len(pending)} across GPUs={gpus} (dry_run={dry_run})", flush=True)
    if not pending:
        return results

    if dry_run:
        worlds_list = list(eval_worlds or range(960000, 960048))
        for job, jfile in pending:
            data = simulate_library_job(job, worlds_list, dry_run_strength=dry_run_strength)
            atomic_write_json(jfile, data)
            results[(job['arm_name'], job['seed'])] = data
        return results

    def _exec_one(job_and_path):
        job, jfile = job_and_path
        gpu_id = gpu_q.get()
        t_s = time.monotonic()
        try:
            env = dict(os.environ)
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            cmd = [
                sys.executable, str(Path(__file__).resolve()), '--worker',
                '--arm-name', job['arm_name'],
                '--r2-mode', str(job.get('r2_mode', 'dual')),
                '--r2-source', job['r2_source'],
                '--explorer-arm', job.get('explorer_arm', 'none'),
                '--seed', str(job['seed']),
                '--tier', str(job['tier']),
                '--out-file', str(jfile),
            ] + common_cli
            if job.get('library_manifest'):
                cmd += ['--library-manifest', str(job['library_manifest'])]
            if job.get('habituation_library'):
                cmd += ['--habituation-library', str(job['habituation_library'])]
            if job.get('evoked_library'):
                cmd += ['--evoked-library', str(job['evoked_library'])]
            if job.get('ctime_habituation_library'):
                cmd += ['--ctime-habituation-library', str(job['ctime_habituation_library'])]
            if job.get('ctime_evoked_library'):
                cmd += ['--ctime-evoked-library', str(job['ctime_evoked_library'])]
            if job.get('rho_lib') is not None:
                cmd += ['--rho-lib', str(job['rho_lib'])]
            if job.get('allow_non_gate_s'):
                cmd += ['--allow-non-gate-s']
            if job.get('enable_m2'):
                cmd += ['--enable-m2']
            if not job.get('use_curriculum', True):
                cmd += ['--no-curriculum']
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if proc.returncode != 0:
                msg = (
                    f"[FAIL gpu={gpu_id}] {job['arm_name']} seed={job['seed']} rc={proc.returncode}\n"
                    f"STDOUT:\n{proc.stdout[-1500:]}\nSTDERR:\n{proc.stderr[-1500:]}"
                )
                print(msg, flush=True)
                raise RuntimeError(msg)
            data = json.loads(jfile.read_text(encoding='utf-8'))
            dt = time.monotonic() - t_s
            e1c = data['suite_means']['E1_explore_96']['new_cells']
            e2s = data['suite_means']['E2_thirst_acute']['alive_rate']
            e3s = data['suite_means']['E3_hunger_acute']['alive_rate']
            e4s = data['suite_means']['E4_death_night_threat']['alive_rate']
            print(
                f"[DONE gpu={gpu_id}] {job['arm_name']} seed={job['seed']} ({dt:.1f}s, VRAM={data['cuda_peak_mb']:.0f}MB, "
                f"TV={data['anchor_tv_shift']:.4f}) | E1={e1c:.2f} E2={e2s:.2%} E3={e3s:.2%} E4={e4s:.2%}",
                flush=True,
            )
            return (job['arm_name'], job['seed']), data
        finally:
            gpu_q.put(gpu_id)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futs = [pool.submit(_exec_one, item) for item in pending]
        for fut in as_completed(futs):
            key, data = fut.result()
            results[key] = data
    return results


def summarize_phase1(results: dict, seeds: list[int], out_dir: Path, n_boot: int, tier: int) -> dict:
    arms = ('A0_native_r1', 'A1_curr_r1', 'D_curr_dual')
    by_arm = {a: [results[(a, s)] for s in seeds] for a in arms if (a, seeds[0]) in results}
    comps = {}
    for pair_name, hi, lo in (('D_minus_A0', 'D_curr_dual', 'A0_native_r1'),
                              ('D_minus_A1', 'D_curr_dual', 'A1_curr_r1'),
                              ('A1_minus_A0', 'A1_curr_r1', 'A0_native_r1')):
        if hi not in by_arm or lo not in by_arm:
            continue
        m_hi, m_lo = by_arm[hi], by_arm[lo]
        stress_hi = (_extract_matrix(m_hi, 'E2_thirst_acute', 'alive')
                     + _extract_matrix(m_hi, 'E3_hunger_acute', 'alive')
                     + _extract_matrix(m_hi, 'E4_death_night_threat', 'alive')) / 3.0
        stress_lo = (_extract_matrix(m_lo, 'E2_thirst_acute', 'alive')
                     + _extract_matrix(m_lo, 'E3_hunger_acute', 'alive')
                     + _extract_matrix(m_lo, 'E4_death_night_threat', 'alive')) / 3.0
        comps[pair_name] = dict(
            E1_new_cells=ev.paired_bootstrap_2way(_extract_matrix(m_hi, 'E1_explore_96', 'new_cells'),
                                                  _extract_matrix(m_lo, 'E1_explore_96', 'new_cells'), seed=101, n_boot=n_boot),
            stress_survival=ev.paired_bootstrap_2way(stress_hi, stress_lo, seed=102, n_boot=n_boot),
            E2_thirst_surv=ev.paired_bootstrap_2way(_extract_matrix(m_hi, 'E2_thirst_acute', 'alive'),
                                                    _extract_matrix(m_lo, 'E2_thirst_acute', 'alive'), seed=103, n_boot=n_boot),
            E3_hunger_surv=ev.paired_bootstrap_2way(_extract_matrix(m_hi, 'E3_hunger_acute', 'alive'),
                                                    _extract_matrix(m_lo, 'E3_hunger_acute', 'alive'), seed=104, n_boot=n_boot),
            E4_death_surv=ev.paired_bootstrap_2way(_extract_matrix(m_hi, 'E4_death_night_threat', 'alive'),
                                                   _extract_matrix(m_lo, 'E4_death_night_threat', 'alive'), seed=105, n_boot=n_boot),
            E2_fear_thirst=ev.paired_bootstrap_2way(_extract_matrix(m_hi, 'E2_thirst_acute', 'fear_of_thirst_rate'),
                                                    _extract_matrix(m_lo, 'E2_thirst_acute', 'fear_of_thirst_rate'), seed=106, n_boot=n_boot),
            E3_fear_hunger=ev.paired_bootstrap_2way(_extract_matrix(m_hi, 'E3_hunger_acute', 'fear_of_hunger_rate'),
                                                    _extract_matrix(m_lo, 'E3_hunger_acute', 'fear_of_hunger_rate'), seed=107, n_boot=n_boot),
            E4_threat_evasion=ev.paired_bootstrap_2way(_extract_matrix(m_hi, 'E4_death_night_threat', 'threat_evasion_rate'),
                                                       _extract_matrix(m_lo, 'E4_death_night_threat', 'threat_evasion_rate'), seed=108, n_boot=n_boot),
        )
    arm_means = {}
    for a, jl in by_arm.items():
        arm_means[a] = dict(
            E1_new_cells=float(np.nanmean(_extract_matrix(jl, 'E1_explore_96', 'new_cells'))),
            E2_thirst_surv=float(np.nanmean(_extract_matrix(jl, 'E2_thirst_acute', 'alive'))),
            E3_hunger_surv=float(np.nanmean(_extract_matrix(jl, 'E3_hunger_acute', 'alive'))),
            E4_death_surv=float(np.nanmean(_extract_matrix(jl, 'E4_death_night_threat', 'alive'))),
            E2_fear_thirst=float(np.nanmean(_extract_matrix(jl, 'E2_thirst_acute', 'fear_of_thirst_rate'))),
            E3_fear_hunger=float(np.nanmean(_extract_matrix(jl, 'E3_hunger_acute', 'fear_of_hunger_rate'))),
            E4_threat_evasion=float(np.nanmean(_extract_matrix(jl, 'E4_death_night_threat', 'threat_evasion_rate'))),
            mean_tv_shift=float(np.mean([j['anchor_tv_shift'] for j in jl])),
        )
    d_a1 = comps.get('D_minus_A1', {})
    d_a0 = comps.get('D_minus_A0', {})
    p1_explore_pass = bool(
        d_a1.get('E1_new_cells', {}).get('ci_low') is not None
        and d_a1['E1_new_cells']['ci_low'] > 0.0
    )
    p1_surv_nonneg = bool(
        d_a0.get('stress_survival', {}).get('ci_low') is not None
        and d_a0['stress_survival']['ci_low'] >= -0.05
    )
    report = dict(
        schema='wetware.gpu.phase1.v1',
        tier=TIERS[tier],
        n_seeds=len(seeds),
        arm_means=arm_means,
        comparisons=comps,
        phase1_explore_gate_passed=p1_explore_pass,
        phase1_survival_guardrail_passed=p1_surv_nonneg,
    )
    report['sha256'] = digest_obj(report)
    atomic_write_json(out_dir / 'phase1_summary.json', report)
    return report


def summarize_spontaneous(results: dict, seeds: list[int], out_dir: Path, n_boot: int,
                          tier: int, chip: str, substudy: str, replicated: bool) -> dict:
    arms = ('A', 'B', 'C', 'G1', 'G2', 'D')
    by_arm = {a: [results[(a, s)] for s in seeds] for a in arms if (a, seeds[0]) in results}

    def _stress(jl):
        return (_extract_matrix(jl, 'E2_thirst_acute', 'alive')
                + _extract_matrix(jl, 'E3_hunger_acute', 'alive')
                + _extract_matrix(jl, 'E4_death_night_threat', 'alive')) / 3.0

    b, c, a, g1, g2, d = (by_arm[k] for k in ('B', 'C', 'A', 'G1', 'G2', 'D'))
    b_c = ev.paired_bootstrap_2way(_extract_matrix(b, 'E1_explore_96', 'new_cells'),
                                   _extract_matrix(c, 'E1_explore_96', 'new_cells'), seed=201, n_boot=n_boot)
    b_a = ev.paired_bootstrap_2way(_extract_matrix(b, 'E1_explore_96', 'new_cells'),
                                   _extract_matrix(a, 'E1_explore_96', 'new_cells'), seed=202, n_boot=n_boot)
    b_g1 = ev.paired_bootstrap_2way(_extract_matrix(b, 'E1_explore_96', 'new_cells'),
                                    _extract_matrix(g1, 'E1_explore_96', 'new_cells'), seed=203, n_boot=n_boot)
    b_g2 = ev.paired_bootstrap_2way(_extract_matrix(b, 'E1_explore_96', 'new_cells'),
                                    _extract_matrix(g2, 'E1_explore_96', 'new_cells'), seed=204, n_boot=n_boot)
    d_a = ev.paired_bootstrap_2way(_extract_matrix(d, 'E1_explore_96', 'new_cells'),
                                   _extract_matrix(a, 'E1_explore_96', 'new_cells'), seed=205, n_boot=n_boot)
    surv_b_a = ev.paired_bootstrap_2way(_stress(b), _stress(a), seed=206, n_boot=n_boot)
    surv_b_c = ev.paired_bootstrap_2way(_stress(b), _stress(c), seed=207, n_boot=n_boot)
    drive_b_c = dict(
        fear_of_thirst=ev.paired_bootstrap_2way(_extract_matrix(b, 'E2_thirst_acute', 'fear_of_thirst_rate'),
                                                _extract_matrix(c, 'E2_thirst_acute', 'fear_of_thirst_rate'), seed=211, n_boot=n_boot),
        fear_of_hunger=ev.paired_bootstrap_2way(_extract_matrix(b, 'E3_hunger_acute', 'fear_of_hunger_rate'),
                                                _extract_matrix(c, 'E3_hunger_acute', 'fear_of_hunger_rate'), seed=212, n_boot=n_boot),
        fear_of_death_evasion=ev.paired_bootstrap_2way(_extract_matrix(b, 'E4_death_night_threat', 'threat_evasion_rate'),
                                                       _extract_matrix(c, 'E4_death_night_threat', 'threat_evasion_rate'), seed=213, n_boot=n_boot),
    )
    thirst_decomp = ev.paired_resource_decomposition(
        _extract_matrix(b, 'E2_thirst_acute', 'thirst_drink_latency'),
        _extract_matrix(b, 'E2_thirst_acute', 'drank_after_low') > 0.5,
        _extract_matrix(c, 'E2_thirst_acute', 'thirst_drink_latency'),
        _extract_matrix(c, 'E2_thirst_acute', 'drank_after_low') > 0.5,
    )
    hunger_decomp = ev.paired_resource_decomposition(
        _extract_matrix(b, 'E3_hunger_acute', 'hunger_eat_latency'),
        _extract_matrix(b, 'E3_hunger_acute', 'ate_after_low') > 0.5,
        _extract_matrix(c, 'E3_hunger_acute', 'hunger_eat_latency'),
        _extract_matrix(c, 'E3_hunger_acute', 'ate_after_low') > 0.5,
    )
    outcome = sp.classify_spontaneous_outcome(
        b_c, b_a, b_g2, d_a, surv_b_a, surv_b_c, drive_b_c,
        replicated_on_second_chip=replicated,
    )
    arm_means = {
        k: dict(
            E1_new_cells=float(np.nanmean(_extract_matrix(jl, 'E1_explore_96', 'new_cells'))),
            stress_survival=float(np.nanmean(_stress(jl))),
            E2_thirst_surv=float(np.nanmean(_extract_matrix(jl, 'E2_thirst_acute', 'alive'))),
            E3_hunger_surv=float(np.nanmean(_extract_matrix(jl, 'E3_hunger_acute', 'alive'))),
            E4_death_surv=float(np.nanmean(_extract_matrix(jl, 'E4_death_night_threat', 'alive'))),
            mean_tv_shift=float(np.mean([j['anchor_tv_shift'] for j in jl])),
        )
        for k, jl in by_arm.items()
    }
    report = dict(
        schema='wetware.gpu.spontaneous.v1',
        chip=chip,
        substudy=substudy,
        tier=TIERS[tier],
        n_seeds=len(seeds),
        arm_means=arm_means,
        comparisons=dict(
            B_minus_C=b_c,
            B_minus_A=b_a,
            B_minus_G1=b_g1,
            B_minus_G2_iaaft=b_g2,
            D_minus_A=d_a,
            stress_survival_B_minus_A=surv_b_a,
            stress_survival_B_minus_C=surv_b_c,
            drives_B_minus_C=drive_b_c,
            thirst_cif_B_vs_C=thirst_decomp,
            hunger_cif_B_vs_C=hunger_decomp,
        ),
        outcome=outcome,
    )
    report['sha256'] = digest_obj(report)
    atomic_write_json(out_dir / 'spontaneous_summary.json', report)
    return report


def resolve_library_bundle(
    *,
    libraries_dir: Path | None,
    dataset_id: str | None = None,
    role: str = 'primary',
    habituation_library: Path | None = None,
    evoked_library: Path | None = None,
    ctime_habituation_library: Path | None = None,
    ctime_evoked_library: Path | None = None,
    rho_lib: float | None = None,
) -> dict:
    """Resolve and gate-check a Dual-Drive wetware library bundle from manifest or explicit paths."""
    if habituation_library is not None or evoked_library is not None:
        hab_p = Path(habituation_library) if habituation_library else None
        evk_p = Path(evoked_library) if evoked_library else None
        ch_p = Path(ctime_habituation_library) if ctime_habituation_library else (
            (hab_p.parent / 'ctime_habituation_library.json') if hab_p and (hab_p.parent / 'ctime_habituation_library.json').is_file() else None
        )
        ce_p = Path(ctime_evoked_library) if ctime_evoked_library else (
            (evk_p.parent / 'ctime_evoked_library.json') if evk_p and (evk_p.parent / 'ctime_evoked_library.json').is_file() else None
        )
        return dict(
            dataset_id=dataset_id or (hab_p.parent.name if hab_p else 'custom_library'),
            chip_id='custom',
            rho_lib=float(rho_lib if rho_lib is not None else 0.45),
            habituation_library=hab_p,
            evoked_library=evk_p,
            ctime_habituation_library=ch_p,
            ctime_evoked_library=ce_p,
            manifest_entry=None,
        )

    if libraries_dir is None:
        raise ValueError('Must provide --libraries-dir or explicit --habituation-library / --evoked-library')
    lib_dir = Path(libraries_dir)
    manifest_path = lib_dir / 'gate_manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Library gate_manifest.json not found under {lib_dir}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    ds_key = dataset_id or (
        manifest['primary_dataset'] if role == 'primary' else manifest['replication_dataset']
    )
    entry = manifest['datasets'][ds_key]
    real_gate = entry['real_gate']
    if not real_gate.get('dual_drive_unlocked', False):
        raise RuntimeError(f'Dataset {ds_key} did not pass offline dual gates; training is blocked')
    ds_base = lib_dir / ds_key
    return dict(
        dataset_id=ds_key,
        chip_id=str(real_gate.get('chip_id', ds_key)),
        rho_lib=float(rho_lib if rho_lib is not None else real_gate['rho_lib']),
        habituation_library=ds_base / 'habituation_library.json',
        evoked_library=ds_base / 'evoked_library.json',
        ctime_habituation_library=ds_base / 'ctime_habituation_library.json',
        ctime_evoked_library=ds_base / 'ctime_evoked_library.json',
        manifest_entry=entry,
    )


def _aggregate_training_audit(job_list: list[dict]) -> dict:
    all_logs = [u for j in job_list for u in j.get('update_logs', [])]
    return dict(
        n_jobs=len(job_list),
        weights_changed_all=all(bool(j.get('weights_changed', False)) for j in job_list),
        max_zero_update_logp=max((float(u.get('zero_update_logp_max', 0.0)) for u in all_logs), default=0.0),
        mean_anchor_tv_shift=float(np.mean([float(j.get('anchor_tv_shift', 0.0)) for j in job_list])) if job_list else 0.0,
        mean_approx_kl=float(np.mean([float(u.get('approx_kl', 0.0)) for u in all_logs])) if all_logs else 0.0,
        kl_early_stop_frac=float(np.mean([1.0 if u.get('kl_early_stop') else 0.0 for u in all_logs])) if all_logs else 0.0,
        mean_r2_mag_share=float(np.mean([float(u.get('r2_mag_share', 0.0)) for u in all_logs])) if all_logs else 0.0,
        mean_r2_var_share=float(np.mean([float(u.get('r2_var_share', 0.0)) for u in all_logs])) if all_logs else 0.0,
        mean_adv_snr=float(np.mean([float(u.get('adv_snr', 0.0)) for u in all_logs])) if all_logs else 0.0,
        mean_in_stress_frac=float(np.mean([float(u.get('in_stress_frac', 0.0)) for u in all_logs])) if all_logs else 0.0,
        mean_relieved_frac=float(np.mean([float(u.get('relieved_frac', 0.0)) for u in all_logs])) if all_logs else 0.0,
        mean_train_seconds=float(np.mean([float(j.get('train_seconds', 0.0)) for j in job_list])) if job_list else 0.0,
        mean_eval_seconds=float(np.mean([float(j.get('eval_seconds', 0.0)) for j in job_list])) if job_list else 0.0,
    )


def _stress_survival_matrix(job_list: list[dict]) -> np.ndarray:
    return (
        _extract_matrix(job_list, 'E2_thirst_acute', 'alive')
        + _extract_matrix(job_list, 'E3_hunger_acute', 'alive')
        + _extract_matrix(job_list, 'E4_death_night_threat', 'alive')
    ) / 3.0


def _compare_pair_full(hi_jobs: list[dict], lo_jobs: list[dict], seed_base: int, n_boot: int) -> dict:
    return dict(
        E1_new_cells=ev.paired_bootstrap_2way(
            _extract_matrix(hi_jobs, 'E1_explore_96', 'new_cells'),
            _extract_matrix(lo_jobs, 'E1_explore_96', 'new_cells'),
            seed=seed_base + 1, n_boot=n_boot,
        ),
        stress_survival=ev.paired_bootstrap_2way(
            _stress_survival_matrix(hi_jobs), _stress_survival_matrix(lo_jobs),
            seed=seed_base + 2, n_boot=n_boot,
        ),
        E2_thirst_surv=ev.paired_bootstrap_2way(
            _extract_matrix(hi_jobs, 'E2_thirst_acute', 'alive'),
            _extract_matrix(lo_jobs, 'E2_thirst_acute', 'alive'),
            seed=seed_base + 3, n_boot=n_boot,
        ),
        E3_hunger_surv=ev.paired_bootstrap_2way(
            _extract_matrix(hi_jobs, 'E3_hunger_acute', 'alive'),
            _extract_matrix(lo_jobs, 'E3_hunger_acute', 'alive'),
            seed=seed_base + 4, n_boot=n_boot,
        ),
        E4_death_surv=ev.paired_bootstrap_2way(
            _extract_matrix(hi_jobs, 'E4_death_night_threat', 'alive'),
            _extract_matrix(lo_jobs, 'E4_death_night_threat', 'alive'),
            seed=seed_base + 5, n_boot=n_boot,
        ),
        E2_fear_thirst=ev.paired_bootstrap_2way(
            _extract_matrix(hi_jobs, 'E2_thirst_acute', 'fear_of_thirst_rate'),
            _extract_matrix(lo_jobs, 'E2_thirst_acute', 'fear_of_thirst_rate'),
            seed=seed_base + 6, n_boot=n_boot,
        ),
        E3_fear_hunger=ev.paired_bootstrap_2way(
            _extract_matrix(hi_jobs, 'E3_hunger_acute', 'fear_of_hunger_rate'),
            _extract_matrix(lo_jobs, 'E3_hunger_acute', 'fear_of_hunger_rate'),
            seed=seed_base + 7, n_boot=n_boot,
        ),
        E4_threat_evasion=ev.paired_bootstrap_2way(
            _extract_matrix(hi_jobs, 'E4_death_night_threat', 'threat_evasion_rate'),
            _extract_matrix(lo_jobs, 'E4_death_night_threat', 'threat_evasion_rate'),
            seed=seed_base + 8, n_boot=n_boot,
        ),
        E4_fear_death_caution=ev.paired_bootstrap_2way(
            _extract_matrix(hi_jobs, 'E4_death_night_threat', 'fear_of_death_caution'),
            _extract_matrix(lo_jobs, 'E4_death_night_threat', 'fear_of_death_caution'),
            seed=seed_base + 9, n_boot=n_boot,
        ),
        E4_threat_damage=ev.paired_bootstrap_2way(
            _extract_matrix(hi_jobs, 'E4_death_night_threat', 'threat_damage_taken'),
            _extract_matrix(lo_jobs, 'E4_death_night_threat', 'threat_damage_taken'),
            seed=seed_base + 10, n_boot=n_boot,
        ),
        E1_ticks_per_action=ev.paired_bootstrap_2way(
            _extract_matrix(hi_jobs, 'E1_explore_96', 'mean_ticks_per_action'),
            _extract_matrix(lo_jobs, 'E1_explore_96', 'mean_ticks_per_action'),
            seed=seed_base + 11, n_boot=n_boot,
        ),
    )


def _safe_nanmean(arr: np.ndarray) -> float | None:
    a = np.asarray(arr, dtype=float)
    valid = a[np.isfinite(a)]
    return float(valid.mean()) if valid.size > 0 else None


def _arm_summary_row(job_list: list[dict]) -> dict:
    return dict(
        E1_new_cells=_safe_nanmean(_extract_matrix(job_list, 'E1_explore_96', 'new_cells')),
        E1_ticks_per_action=_safe_nanmean(_extract_matrix(job_list, 'E1_explore_96', 'mean_ticks_per_action')),
        stress_survival=_safe_nanmean(_stress_survival_matrix(job_list)),
        E2_thirst_surv=_safe_nanmean(_extract_matrix(job_list, 'E2_thirst_acute', 'alive')),
        E2_fear_thirst=_safe_nanmean(_extract_matrix(job_list, 'E2_thirst_acute', 'fear_of_thirst_rate')),
        E2_drink_latency=_safe_nanmean(_extract_matrix(job_list, 'E2_thirst_acute', 'thirst_drink_latency')),
        E3_hunger_surv=_safe_nanmean(_extract_matrix(job_list, 'E3_hunger_acute', 'alive')),
        E3_fear_hunger=_safe_nanmean(_extract_matrix(job_list, 'E3_hunger_acute', 'fear_of_hunger_rate')),
        E3_eat_latency=_safe_nanmean(_extract_matrix(job_list, 'E3_hunger_acute', 'hunger_eat_latency')),
        E4_death_surv=_safe_nanmean(_extract_matrix(job_list, 'E4_death_night_threat', 'alive')),
        E4_threat_evasion=_safe_nanmean(_extract_matrix(job_list, 'E4_death_night_threat', 'threat_evasion_rate')),
        E4_fear_death_caution=_safe_nanmean(_extract_matrix(job_list, 'E4_death_night_threat', 'fear_of_death_caution')),
        E4_threat_damage=_safe_nanmean(_extract_matrix(job_list, 'E4_death_night_threat', 'threat_damage_taken')),
        mean_tv_shift=float(np.mean([float(j.get('anchor_tv_shift', 0.0)) for j in job_list])),
    )


def orchestrate_library_stage0(
    *,
    out_dir: Path,
    gpus: list[int],
    common_cli: list[str],
    r2_mode: str,
    tune_seeds: list[int],
    confirm_seeds: list[int],
    eval_worlds: list[int],
    ladder_tiers: tuple[int, ...],
    n_boot: int,
    dry_run: bool,
    dry_run_strength: float = 1.0,
) -> dict:
    """Multi-GPU Stage 0 positive-control calibration & noise sensitivity curve (`rho_min`)."""
    ladder_reports = []
    frozen_tier_id = None
    for t_id in ladder_tiers:
        t_dir = out_dir / f'tier{t_id}'
        jobs = []
        for s in tune_seeds:
            jobs.append(dict(arm_name='A0_native_r1', r2_mode=r2_mode, r2_source='none', explorer_arm='none', seed=s, tier=t_id, use_curriculum=False))
            jobs.append(dict(arm_name='A', r2_mode=r2_mode, r2_source='none', explorer_arm='none', seed=s, tier=t_id, use_curriculum=True))
            jobs.append(dict(arm_name='D', r2_mode=r2_mode, r2_source='ideal', explorer_arm='none', seed=s, tier=t_id, use_curriculum=True))
        res = run_parallel_jobs(jobs, gpus, t_dir, common_cli, dry_run=dry_run, eval_worlds=eval_worlds, dry_run_strength=dry_run_strength)
        a0_j = [res[('A0_native_r1', s)] for s in tune_seeds]
        a_j = [res[('A', s)] for s in tune_seeds]
        d_j = [res[('D', s)] for s in tune_seeds]
        d_vs_a = _compare_pair_full(d_j, a_j, seed_base=1000 + t_id * 50, n_boot=n_boot)
        d_vs_a0 = _compare_pair_full(d_j, a0_j, seed_base=2000 + t_id * 50, n_boot=n_boot)
        explore_ok = bool(d_vs_a['E1_new_cells']['ci_low'] is not None and d_vs_a['E1_new_cells']['ci_low'] > 0.0)
        surv_ok = bool(d_vs_a0['stress_survival']['ci_low'] is not None and d_vs_a0['stress_survival']['ci_low'] >= ev.GUARDRAIL_SURVIVAL_DROP_MAX)
        passed = explore_ok and surv_ok
        ladder_reports.append(dict(
            tier=TIERS[t_id],
            arm_means=dict(A0_native_r1=_arm_summary_row(a0_j), A=_arm_summary_row(a_j), D=_arm_summary_row(d_j)),
            D_minus_A=d_vs_a,
            D_minus_A0=d_vs_a0,
            explore_passed=explore_ok,
            survival_guardrail_passed=surv_ok,
            passed=passed,
        ))
        if passed:
            frozen_tier_id = t_id
            break

    report = dict(
        schema='wetware.gpu.library_stage0.v1',
        r2_mode=r2_mode,
        dry_run=bool(dry_run),
        tune_seeds=tune_seeds,
        confirm_seeds=confirm_seeds,
        n_worlds=len(eval_worlds),
        ladder=ladder_reports,
        frozen_tier=TIERS[frozen_tier_id] if frozen_tier_id is not None else None,
        confirmed=False,
        rho_min=None,
        recommended_stage3_seeds=None,
    )

    if frozen_tier_id is not None:
        c_dir = out_dir / 'confirm'
        c_jobs = []
        for s in confirm_seeds:
            c_jobs.append(dict(arm_name='A0_native_r1', r2_mode=r2_mode, r2_source='none', explorer_arm='none', seed=s, tier=frozen_tier_id, use_curriculum=False))
            c_jobs.append(dict(arm_name='A', r2_mode=r2_mode, r2_source='none', explorer_arm='none', seed=s, tier=frozen_tier_id, use_curriculum=True))
            c_jobs.append(dict(arm_name='D', r2_mode=r2_mode, r2_source='ideal', explorer_arm='none', seed=s, tier=frozen_tier_id, use_curriculum=True))
        c_res = run_parallel_jobs(c_jobs, gpus, c_dir, common_cli, dry_run=dry_run, eval_worlds=eval_worlds, dry_run_strength=dry_run_strength)
        a0_c = [c_res[('A0_native_r1', s)] for s in confirm_seeds]
        a_c = [c_res[('A', s)] for s in confirm_seeds]
        d_c = [c_res[('D', s)] for s in confirm_seeds]
        conf_da = _compare_pair_full(d_c, a_c, seed_base=3100, n_boot=n_boot)
        conf_da0 = _compare_pair_full(d_c, a0_c, seed_base=3200, n_boot=n_boot)
        confirmed = bool(
            conf_da['E1_new_cells']['ci_low'] is not None and conf_da['E1_new_cells']['ci_low'] > 0.0
            and conf_da0['stress_survival']['ci_low'] is not None and conf_da0['stress_survival']['ci_low'] >= ev.GUARDRAIL_SURVIVAL_DROP_MAX
        )
        report['confirm'] = dict(
            arm_means=dict(A0_native_r1=_arm_summary_row(a0_c), A=_arm_summary_row(a_c), D=_arm_summary_row(d_c)),
            D_minus_A=conf_da,
            D_minus_A0=conf_da0,
        )
        report['confirmed'] = confirmed

        if confirmed:
            noise_rows = []
            for idx, rho in enumerate(NOISE_RHO_GRID):
                n_dir = out_dir / f'noise_rho{int(round(rho * 100)):02d}'
                n_jobs = [
                    dict(arm_name='F', r2_mode=r2_mode, r2_source='noisy', rho_lib=float(rho),
                         explorer_arm='none', seed=s, tier=frozen_tier_id, use_curriculum=True)
                    for s in confirm_seeds
                ]
                n_res = run_parallel_jobs(n_jobs, gpus, n_dir, common_cli, dry_run=dry_run, eval_worlds=eval_worlds, dry_run_strength=dry_run_strength)
                f_c = [n_res[('F', s)] for s in confirm_seeds]
                st = ev.paired_bootstrap_2way(
                    _extract_matrix(f_c, 'E1_explore_96', 'new_cells'),
                    _extract_matrix(a_c, 'E1_explore_96', 'new_cells'),
                    seed=4000 + idx * 17, n_boot=n_boot,
                )
                noise_rows.append(dict(rho=float(rho), stats=st))
            rho_min = s0.select_rho_min(noise_rows)
            report['noise_curve'] = noise_rows
            report['rho_min'] = rho_min
            if rho_min is not None:
                min_row = next(r for r in noise_rows if r['rho'] == rho_min)
                report['recommended_stage3_seeds'] = ev.required_seeds_power90(
                    min_row['stats']['mean'], min_row['stats']['seed_std'], min_seeds=16, max_seeds=64
                )

    report['sha256'] = digest_obj(report)
    atomic_write_json(out_dir / 'stage0_summary.json', report)
    return report


def summarize_library_stage3(
    *,
    results: dict[tuple[str, int], dict],
    seeds: list[int],
    out_dir: Path,
    n_boot: int,
    tier_cfg: dict,
    r2_mode: str,
    bundle: dict,
    channel_bound_audit: dict | None,
    stage0_sha256: str | None,
    dry_run: bool,
    second_organoid_replicated: bool = False,
    summary_filename: str = 'stage3_summary.json',
) -> dict:
    """Compute complete 7-arm paired comparison, CIF decomposition, audit, and frozen graded verdict."""
    by_arm = {arm: [results[(arm, s)] for s in seeds] for arm, _ in LIBRARY_SEVEN_ARMS}
    arm_means = {arm: _arm_summary_row(jl) for arm, jl in by_arm.items()}
    training_audit = {arm: _aggregate_training_audit(jl) for arm, jl in by_arm.items()}

    pair_specs = (
        ('B_minus_C', 'B', 'C', 5100),
        ('B_minus_Ctime', 'B', 'C_time', 5200),
        ('B_minus_A', 'B', 'A', 5300),
        ('D_minus_A', 'D', 'A', 5400),
        ('E_minus_A', 'E', 'A', 5500),
        ('B_minus_F', 'B', 'F', 5600),
    )
    comparisons_full = {
        name: _compare_pair_full(by_arm[hi], by_arm[lo], seed_base=sb, n_boot=n_boot)
        for name, hi, lo, sb in pair_specs
    }

    cif = {}
    for tag, ref_arm in (('B_vs_C', 'C'), ('B_vs_Ctime', 'C_time'), ('B_vs_A', 'A')):
        cif[tag] = dict(
            thirst=ev.paired_resource_decomposition(
                _extract_matrix(by_arm['B'], 'E2_thirst_acute', 'thirst_drink_latency'),
                _extract_matrix(by_arm['B'], 'E2_thirst_acute', 'drank_after_low') > 0.5,
                _extract_matrix(by_arm[ref_arm], 'E2_thirst_acute', 'thirst_drink_latency'),
                _extract_matrix(by_arm[ref_arm], 'E2_thirst_acute', 'drank_after_low') > 0.5,
            ),
            hunger=ev.paired_resource_decomposition(
                _extract_matrix(by_arm['B'], 'E3_hunger_acute', 'hunger_eat_latency'),
                _extract_matrix(by_arm['B'], 'E3_hunger_acute', 'ate_after_low') > 0.5,
                _extract_matrix(by_arm[ref_arm], 'E3_hunger_acute', 'hunger_eat_latency'),
                _extract_matrix(by_arm[ref_arm], 'E3_hunger_acute', 'ate_after_low') > 0.5,
            ),
        )

    primary_key = 'E1_new_cells' if r2_mode in ('explore', 'dual') else 'E2_fear_thirst'
    drives_b_c = dict(
        fear_of_thirst=comparisons_full['B_minus_C']['E2_fear_thirst'],
        fear_of_hunger=comparisons_full['B_minus_C']['E3_fear_hunger'],
        fear_of_death=comparisons_full['B_minus_C']['E4_threat_evasion'],
    )
    outcome = ev.classify_stage3_outcome(
        comparisons_full['B_minus_C'][primary_key],
        comparisons_full['B_minus_A'][primary_key],
        comparisons_full['D_minus_A'][primary_key],
        comparisons_full['E_minus_A'][primary_key],
        comparisons_full['B_minus_A']['stress_survival'],
        b_minus_ctime=comparisons_full['B_minus_Ctime'][primary_key],
        drive_b_minus_c=drives_b_c,
        survival_b_minus_c=comparisons_full['B_minus_C']['stress_survival'],
        second_organoid_replicated=second_organoid_replicated,
    )

    report = dict(
        schema='wetware.gpu.library_stage3.v1',
        dataset_id=bundle['dataset_id'],
        chip_id=bundle['chip_id'],
        rho_lib=float(bundle['rho_lib']),
        r2_mode=r2_mode,
        primary_metric=primary_key,
        dry_run=bool(dry_run),
        tier=tier_cfg,
        n_seeds=len(seeds),
        n_worlds=len(by_arm['A'][0]['suites']['E1_explore_96']),
        seeds=seeds,
        arms=[a for a, _ in LIBRARY_SEVEN_ARMS],
        stage0_sha256=stage0_sha256,
        channel_bound_audit=channel_bound_audit,
        arm_means=arm_means,
        training_audit=training_audit,
        comparisons=comparisons_full,
        cif_decomposition=cif,
        outcome=outcome,
    )
    report['sha256'] = digest_obj(report)
    atomic_write_json(out_dir / summary_filename, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', action='store_true')
    p.add_argument(
        '--orchestrate',
        choices=('smoke', 'phase1', 'phase2', 'phase3', 'library_smoke', 'library_stage0', 'library_stage3', 'library_stage4'),
    )
    p.add_argument('--model-config', type=Path)
    p.add_argument('--godot', type=Path)
    p.add_argument('--project', type=Path, default=ROOT)
    p.add_argument('--out', type=Path)
    p.add_argument('--out-file', type=Path)
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    p.add_argument('--tier', type=int, default=1)
    p.add_argument('--ladder-tiers', default='1,2,3,4')
    p.add_argument('--seeds', type=int, default=16)
    p.add_argument('--confirm-seeds', type=int)
    p.add_argument('--seed-base', type=int, default=20261123)
    p.add_argument('--eval-worlds', type=int, default=48)
    p.add_argument('--world-base', type=int, default=960000)
    p.add_argument('--n-boot', type=int, default=10000)
    # Worker / dual-library options
    p.add_argument('--arm-name', default='D')
    p.add_argument('--r2-mode', choices=MODES, default='dual')
    p.add_argument('--r2-source', default='none')
    p.add_argument('--explorer-arm', default='none')
    p.add_argument('--seed', type=int, default=20261123)
    p.add_argument('--library-manifest', type=Path)
    p.add_argument('--libraries-dir', type=Path, default=ROOT / 'output/existing_wetware_libraries')
    p.add_argument('--dataset-id', default=None)
    p.add_argument('--habituation-library', type=Path)
    p.add_argument('--evoked-library', type=Path)
    p.add_argument('--ctime-habituation-library', type=Path)
    p.add_argument('--ctime-evoked-library', type=Path)
    p.add_argument('--rho-lib', type=float, default=None)
    p.add_argument('--stage0-report', type=Path)
    p.add_argument('--allow-unconfirmed-stage0', action='store_true')
    p.add_argument('--allow-non-gate-s', action='store_true')
    p.add_argument('--eps-base', type=float, default=0.10)
    p.add_argument('--enable-m2', action='store_true')
    p.add_argument('--entropy-eta', type=float, default=0.35)
    p.add_argument('--microbatch', type=int, default=1)
    p.add_argument('--no-curriculum', action='store_true')
    p.add_argument('--substudy', choices=('2A_pure_r1', '2B_dual_drive'), default='2B_dual_drive')
    p.add_argument('--replicated-on-second-chip', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--dry-run-strength', type=float, default=1.0)
    args = p.parse_args()

    worlds = list(range(args.world_base, args.world_base + args.eval_worlds))

    if args.worker:
        if not args.model_config or not args.godot:
            raise ValueError('--model-config and --godot are required in --worker mode')
        res = run_one_gpu_job(
            arm_name=args.arm_name,
            r2_mode=args.r2_mode,
            r2_source=args.r2_source,
            explorer_arm=args.explorer_arm,
            seed=args.seed,
            tier_cfg=TIERS[args.tier],
            model_config_path=args.model_config,
            godot_path=args.godot,
            project_path=args.project,
            eval_worlds=worlds,
            library_manifest=args.library_manifest,
            allow_non_gate_s=args.allow_non_gate_s,
            eps_base=args.eps_base,
            enable_m2=args.enable_m2,
            entropy_eta=args.entropy_eta,
            use_curriculum=not args.no_curriculum,
            microbatch=args.microbatch,
            habituation_library_path=args.habituation_library,
            evoked_library_path=args.evoked_library,
            ctime_habituation_library_path=args.ctime_habituation_library,
            ctime_evoked_library_path=args.ctime_evoked_library,
            rho_lib=float(args.rho_lib if args.rho_lib is not None else 1.0),
        )
        atomic_write_json(args.out_file, res)
        return

    if not args.orchestrate or not args.out:
        raise ValueError('Must specify --worker or (--orchestrate and --out)')
    if not args.dry_run and (not args.model_config or not args.godot):
        raise ValueError('--model-config and --godot are required unless --dry-run is set')
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    gpus = [int(x.strip()) for x in args.gpus.split(',') if x.strip()]
    common_cli = [] if args.dry_run else [
        '--model-config', str(args.model_config),
        '--godot', str(args.godot),
        '--project', str(args.project),
        '--eval-worlds', str(args.eval_worlds),
        '--world-base', str(args.world_base),
        '--eps-base', str(args.eps_base),
        '--entropy-eta', str(args.entropy_eta),
        '--microbatch', str(args.microbatch),
    ]

    if args.orchestrate == 'library_smoke':
        bundle = resolve_library_bundle(
            libraries_dir=args.libraries_dir,
            dataset_id=args.dataset_id,
            role='primary',
            habituation_library=args.habituation_library,
            evoked_library=args.evoked_library,
            ctime_habituation_library=args.ctime_habituation_library,
            ctime_evoked_library=args.ctime_evoked_library,
            rho_lib=args.rho_lib,
        )
        lib_kw = dict(
            r2_mode=args.r2_mode,
            habituation_library=bundle['habituation_library'],
            evoked_library=bundle['evoked_library'],
            ctime_habituation_library=bundle['ctime_habituation_library'],
            ctime_evoked_library=bundle['ctime_evoked_library'],
            rho_lib=bundle['rho_lib'],
        )
        jobs = [
            dict(arm_name='A', r2_source='none', explorer_arm='none', seed=args.seed_base, tier=0, use_curriculum=True, r2_mode=args.r2_mode),
            dict(arm_name='B', r2_source='wetware_real', explorer_arm='none', seed=args.seed_base, tier=0, use_curriculum=True, **lib_kw),
            dict(arm_name='C_time', r2_source='wetware_time_shuffled', explorer_arm='none', seed=args.seed_base, tier=0, use_curriculum=True, **lib_kw),
            dict(arm_name='D', r2_source='ideal', explorer_arm='none', seed=args.seed_base, tier=0, use_curriculum=True, r2_mode=args.r2_mode),
        ]
        res = run_parallel_jobs(jobs, gpus[:4], out_dir, common_cli, dry_run=args.dry_run, eval_worlds=worlds, dry_run_strength=args.dry_run_strength)
        summary = dict(
            schema='wetware.gpu.library_smoke.v1',
            dataset_id=bundle['dataset_id'],
            rho_lib=bundle['rho_lib'],
            dry_run=bool(args.dry_run),
            arms={
                k[0]: dict(
                    weights_changed=bool(v['weights_changed']),
                    anchor_tv_shift=float(v['anchor_tv_shift']),
                    vram_mb=float(v['cuda_peak_mb']),
                    train_s=float(v['train_seconds']),
                    eval_s=float(v['eval_seconds']),
                    audit=_aggregate_training_audit([v]),
                    suite_means=v['suite_means'],
                )
                for k, v in res.items()
            },
        )
        summary['sha256'] = digest_obj(summary)
        atomic_write_json(out_dir / 'smoke_summary.json', summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.orchestrate == 'library_stage0':
        tune_seeds = list(range(args.seed_base, args.seed_base + args.seeds))
        n_conf = int(args.confirm_seeds or args.seeds)
        confirm_seeds = list(range(args.seed_base + 10000, args.seed_base + 10000 + n_conf))
        ladder_tiers = tuple(int(x.strip()) for x in args.ladder_tiers.split(',') if x.strip())
        rep = orchestrate_library_stage0(
            out_dir=out_dir,
            gpus=gpus,
            common_cli=common_cli,
            r2_mode=args.r2_mode,
            tune_seeds=tune_seeds,
            confirm_seeds=confirm_seeds,
            eval_worlds=worlds,
            ladder_tiers=ladder_tiers,
            n_boot=args.n_boot,
            dry_run=args.dry_run,
            dry_run_strength=args.dry_run_strength,
        )
        print(json.dumps(dict(
            confirmed=rep['confirmed'],
            frozen_tier=rep['frozen_tier'],
            rho_min=rep['rho_min'],
            recommended_stage3_seeds=rep['recommended_stage3_seeds'],
        ), indent=2, ensure_ascii=False))
        return

    if args.orchestrate in ('library_stage3', 'library_stage4'):
        role = 'primary' if args.orchestrate == 'library_stage3' else 'replication'
        bundle = resolve_library_bundle(
            libraries_dir=args.libraries_dir,
            dataset_id=args.dataset_id,
            role=role,
            habituation_library=args.habituation_library,
            evoked_library=args.evoked_library,
            ctime_habituation_library=args.ctime_habituation_library,
            ctime_evoked_library=args.ctime_evoked_library,
            rho_lib=args.rho_lib,
        )
        tier_id = int(args.tier)
        stage0_sha = None
        if args.stage0_report is not None:
            st0 = json.loads(Path(args.stage0_report).read_text(encoding='utf-8'))
            stage0_sha = st0.get('sha256')
            if not args.allow_unconfirmed_stage0:
                if not st0.get('confirmed') or not st0.get('frozen_tier'):
                    raise RuntimeError('Stage 0 did not pass confirmation; Stage 3/4 training is blocked')
                if st0.get('rho_min') is not None and float(bundle['rho_lib']) < float(st0['rho_min']):
                    raise RuntimeError(
                        f"Library rho_lib={bundle['rho_lib']:.4f} < Stage 0 rho_min={st0['rho_min']:.4f}; blocked"
                    )
            if st0.get('frozen_tier'):
                tier_id = int(st0['frozen_tier']['tier'])

        hab_lib = _load_checked_library(bundle['habituation_library'], library_v2.HABITUATION_SCHEMA)
        evk_lib = _load_checked_library(bundle['evoked_library'], library_v2.EVOKED_SCHEMA)
        ctime_hab = _load_checked_library(bundle['ctime_habituation_library'], library_v2.HABITUATION_SCHEMA)
        ctime_evk = _load_checked_library(bundle['ctime_evoked_library'], library_v2.EVOKED_SCHEMA)
        seed_offset = 0 if args.orchestrate == 'library_stage3' else 50000
        seeds = list(range(args.seed_base + seed_offset, args.seed_base + seed_offset + args.seeds))
        eval_w = [w + seed_offset for w in worlds]
        if not args.dry_run and seed_offset:
            # Ensure worker CLI uses offset world base for Stage 4 fresh maps
            common_cli = [
                str(args.world_base + seed_offset) if (i > 0 and common_cli[i - 1] == '--world-base') else tok
                for i, tok in enumerate(common_cli)
            ]

        bound_audit = s3.pretrain_channel_audit(
            args.r2_mode, seeds[0], bundle['rho_lib'], hab_lib, evk_lib, ctime_hab, ctime_evk
        )
        if bound_audit is not None and bound_audit.get('passed') is False:
            raise RuntimeError(f'Pre-training channel_bound_check failed: {bound_audit}')

        jobs = []
        for s in seeds:
            for aname, r2_src in LIBRARY_SEVEN_ARMS:
                job = dict(
                    arm_name=aname,
                    r2_mode=args.r2_mode,
                    r2_source=r2_src,
                    explorer_arm='none',
                    seed=s,
                    tier=tier_id,
                    use_curriculum=True,
                    rho_lib=float(bundle['rho_lib']),
                )
                if r2_src in ('wetware_real', 'wetware_shuffled', 'wetware_time_shuffled', 'wetware_inverted'):
                    job.update(
                        habituation_library=bundle['habituation_library'],
                        evoked_library=bundle['evoked_library'],
                        ctime_habituation_library=bundle['ctime_habituation_library'],
                        ctime_evoked_library=bundle['ctime_evoked_library'],
                    )
                jobs.append(job)

        res = run_parallel_jobs(jobs, gpus, out_dir, common_cli, dry_run=args.dry_run, eval_worlds=eval_w, dry_run_strength=args.dry_run_strength)
        rep = summarize_library_stage3(
            results=res,
            seeds=seeds,
            out_dir=out_dir,
            n_boot=args.n_boot,
            tier_cfg=TIERS[tier_id],
            r2_mode=args.r2_mode,
            bundle=bundle,
            channel_bound_audit=bound_audit,
            stage0_sha256=stage0_sha,
            dry_run=args.dry_run,
            second_organoid_replicated=(args.orchestrate == 'library_stage4' or args.replicated_on_second_chip),
            summary_filename='stage3_summary.json' if args.orchestrate == 'library_stage3' else 'stage4_summary.json',
        )
        print(json.dumps(dict(
            dataset_id=rep['dataset_id'],
            tier=tier_id,
            outcome=rep['outcome'],
            arm_means={k: round(v['E1_new_cells'], 3) for k, v in rep['arm_means'].items()},
            B_minus_C_E1=rep['comparisons']['B_minus_C']['E1_new_cells'],
            B_minus_Ctime_E1=rep['comparisons']['B_minus_Ctime']['E1_new_cells'],
            B_minus_A_stress_surv=rep['comparisons']['B_minus_A']['stress_survival'],
        ), indent=2, ensure_ascii=False))
        return

    if args.orchestrate == 'smoke':
        jobs = [
            dict(arm_name='A0_native_r1', r2_source='none', explorer_arm='none', seed=args.seed_base, tier=0, use_curriculum=False),
            dict(arm_name='D_curr_dual', r2_source='ideal', explorer_arm='none', seed=args.seed_base, tier=0, use_curriculum=True),
            dict(arm_name='B_22554_m1m2', r2_source='ideal', explorer_arm='real', seed=args.seed_base, tier=0,
                 use_curriculum=True, library_manifest=args.library_manifest, enable_m2=True),
            dict(arm_name='C_22554_shuf', r2_source='ideal', explorer_arm='shuffled', seed=args.seed_base, tier=0,
                 use_curriculum=True, library_manifest=args.library_manifest, enable_m2=True),
        ]
        res = run_parallel_jobs(jobs, gpus[:4], out_dir, common_cli)
        summary = {k[0]: dict(suite_means=v['suite_means'], tv=v['anchor_tv_shift'],
                              vram_mb=v['cuda_peak_mb'], train_s=v['train_seconds'],
                              eval_s=v['eval_seconds'], updates=v['update_logs'])
                   for k, v in res.items()}
        atomic_write_json(out_dir / 'smoke_summary.json', summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    seeds = list(range(args.seed_base, args.seed_base + args.seeds))

    if args.orchestrate == 'phase1':
        jobs = []
        for s in seeds:
            jobs.append(dict(arm_name='A0_native_r1', r2_source='none', explorer_arm='none', seed=s, tier=args.tier, use_curriculum=False))
            jobs.append(dict(arm_name='A1_curr_r1', r2_source='none', explorer_arm='none', seed=s, tier=args.tier, use_curriculum=True))
            jobs.append(dict(arm_name='D_curr_dual', r2_source='ideal', explorer_arm='none', seed=s, tier=args.tier, use_curriculum=True))
        res = run_parallel_jobs(jobs, gpus, out_dir, common_cli)
        rep = summarize_phase1(res, seeds, out_dir, args.n_boot, args.tier)
        print(json.dumps(dict(
            tier=args.tier,
            explore_pass=rep['phase1_explore_gate_passed'],
            surv_guardrail=rep['phase1_survival_guardrail_passed'],
            arm_means=rep['arm_means'],
            D_minus_A1_E1=rep['comparisons']['D_minus_A1']['E1_new_cells'],
            D_minus_A0_stress_surv=rep['comparisons']['D_minus_A0']['stress_survival'],
        ), indent=2, ensure_ascii=False))
        return

    if args.orchestrate in ('phase2', 'phase3'):
        if not args.library_manifest:
            raise ValueError('--library-manifest required for phase2/phase3')
        lib = mde.load_dynamics_library(args.library_manifest, require_gate_s=not args.allow_non_gate_s)
        chip = lib['manifest']['chip']
        r2_src = 'none' if args.substudy == '2A_pure_r1' else 'ideal'
        arm_specs = (
            ('A', 'static'),
            ('B', 'real'),
            ('C', 'shuffled'),
            ('G1', 'white'),
            ('G2', 'iaaft'),
            ('D', 'ideal'),
        )
        jobs = []
        for s in seeds:
            for aname, akind in arm_specs:
                jobs.append(dict(
                    arm_name=aname, r2_source=r2_src, explorer_arm=akind,
                    seed=s, tier=args.tier, use_curriculum=True,
                    library_manifest=args.library_manifest if akind in ('real', 'shuffled', 'white', 'iaaft') else None,
                    allow_non_gate_s=args.allow_non_gate_s,
                    enable_m2=args.enable_m2,
                ))
        res = run_parallel_jobs(jobs, gpus, out_dir, common_cli)
        rep = summarize_spontaneous(
            res, seeds, out_dir, args.n_boot, args.tier, chip, args.substudy,
            replicated=(args.orchestrate == 'phase3' or args.replicated_on_second_chip),
        )
        print(json.dumps(dict(
            chip=chip, substudy=args.substudy, outcome=rep['outcome'],
            arm_means=rep['arm_means'],
            B_minus_C=rep['comparisons']['B_minus_C'],
            B_minus_A=rep['comparisons']['B_minus_A'],
            B_minus_G2=rep['comparisons']['B_minus_G2_iaaft'],
            D_minus_A=rep['comparisons']['D_minus_A'],
        ), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
