"""Training-only wetware spontaneous dynamics exploration (M1) & entropy modulation (M2).

Strict scientific & causal isolation guarantees:
1. Frozen evaluation NEVER imports or calls this module; evaluated models are pure 108-dim
   M9 policies with zero MEA prompt/critic text or timing confounds.
2. Homeostatic safety gating reads ONLY publicly observable SensoryState fields
   (`x, y, health, thirst, hunger, threat_nearby`) — never `visited_count`, `r1`, `reward`,
   or `terminated`.
3. Mean exploration rate across every arm (`B` real / `C` shuffled / `G1` white / `G2` IAAFT)
   is strictly calibrated to `eps_base` (default 0.10) to rule out trivial "higher average
   exploration rate" confounds.
4. M2 entropy modulation strictly normalizes batch mean weight to 1.000000, preserving total
   entropy budget identically across arms.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ODDBALL_ROOT = ROOT / 'standalone' / 'mea_oddball'
if str(ODDBALL_ROOT) not in sys.path:
    sys.path.insert(0, str(ODDBALL_ROOT))

from oddball.signals import SensoryState, homeostatic_site, design as odd_design

ARMS = ('static', 'real', 'shuffled', 'white', 'iaaft', 'ideal')
DIR_VECTORS = {
    (0, -1): 0,   # N
    (1, 0): 1,    # E
    (0, 1): 2,    # S
    (-1, 0): 3,   # W
}
CARDINAL_XY = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dynamics_library(manifest_path: Path, *, require_gate_s: bool = True) -> dict:
    """Load and SHA-256 verify a frozen Phase 0 spontaneous dynamics library."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema') != 'wetware.spontaneous.dynamics.v1':
        raise ValueError(f'Unexpected dynamics library schema: {manifest.get("schema")}')
    if require_gate_s and not manifest.get('gate_s_passed', False):
        raise ValueError(f'Chip {manifest.get("chip")} failed Gate-S and is blocked from primary confirmation')
    raw_npz = str(manifest['library_npz'])
    npz_path = Path(raw_npz)
    if not npz_path.is_file():
        npz_name = raw_npz.replace('\\', '/').rsplit('/', 1)[-1]
        npz_path = manifest_path.parent / npz_name
    actual_sha = _file_sha256(npz_path)
    if actual_sha != manifest['library_npz_sha256']:
        raise ValueError(f'Library NPZ SHA-256 mismatch for {npz_path}')
    data = np.load(npz_path)
    arrays = {k: np.asarray(data[k], dtype=np.float64) for k in data.files if k != 'active_channels'}
    return dict(manifest=manifest, arrays=arrays, npz_path=str(npz_path))


def calibrate_epsilon_stream(burst_z: np.ndarray, eps_base: float = 0.10,
                             kappa: float = 0.55, eps_min: float = 0.02,
                             eps_max: float = 0.25) -> np.ndarray:
    """Map burst z-score stream to [eps_min, eps_max] with EXACT mean == eps_base."""
    z = np.clip(np.asarray(burst_z, dtype=np.float64), -3.0, 3.0)
    if len(z) == 0:
        return np.array([], dtype=np.float64)
    if float(z.std()) < 1e-12:
        return np.full(len(z), float(eps_base), dtype=np.float64)
    raw = np.exp(kappa * z)
    # Bisection on scale multiplier c so mean(clip(c * raw, eps_min, eps_max)) == eps_base
    lo, hi = 1e-6, 1e3
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        val = float(np.clip(mid * raw, eps_min, eps_max).mean())
        if val < eps_base:
            lo = mid
        else:
            hi = mid
    return np.clip(0.5 * (lo + hi) * raw, eps_min, eps_max)


def _action_unit_direction(action: dict, obs: dict) -> np.ndarray | None:
    """Compute observable 2D unit direction vector of a legal move/navigate action."""
    verb = action.get('verb', '')
    player = obs.get('player', {})
    px, py = float(player.get('x', 0.0)), float(player.get('y', 0.0))
    tid = str(action.get('target_id', ''))
    if verb == 'move' and tid.startswith('tile_'):
        parts = tid.split('_')
        if len(parts) == 3:
            dx, dy = float(int(parts[1])) - px, float(int(parts[2])) - py
            norm = math.hypot(dx, dy)
            if norm > 1e-6:
                return np.array([dx / norm, dy / norm], dtype=np.float64)
    elif verb == 'navigate':
        if tid == 'camp' and isinstance(obs.get('camp'), dict) and obs['camp'].get('exists'):
            dx = float(obs['camp']['x']) - px
            dy = float(obs['camp']['y']) - py
            norm = math.hypot(dx, dy)
            if norm > 1e-6:
                return np.array([dx / norm, dy / norm], dtype=np.float64)
        for item in obs.get('nearby') or ():
            if str(item.get('id', '')) == tid and 'x' in item and 'y' in item:
                dx, dy = float(item['x']) - px, float(item['y']) - py
                norm = math.hypot(dx, dy)
                if norm > 1e-6:
                    return np.array([dx / norm, dy / norm], dtype=np.float64)
        if tid.startswith('tile_'):
            parts = tid.split('_')
            if len(parts) == 3:
                dx, dy = float(int(parts[1])) - px, float(int(parts[2])) - py
                norm = math.hypot(dx, dy)
                if norm > 1e-6:
                    return np.array([dx / norm, dy / norm], dtype=np.float64)
    return None


class WetwareDynamicsExplorer:
    """M1: Homeostatic-gated correlated exploration driven by verified MEA spontaneous streams."""

    def __init__(
        self,
        arm: str,
        library: dict | None = None,
        *,
        seed: int = 0,
        eps_base: float = 0.10,
        kappa: float = 0.55,
        dir_alpha: float = 1.25,
        homeostatic_gate: bool = True,
    ):
        if arm not in ARMS:
            raise ValueError(f'Unknown arm: {arm!r}')
        self.arm = arm
        self.seed = int(seed)
        self.eps_base = float(eps_base)
        self.kappa = float(kappa)
        self.dir_alpha = float(dir_alpha)
        self.homeostatic_gate = bool(homeostatic_gate)
        self.cursor = 0
        self.env_cursors: dict[int, int] = {}
        self.recent_xy: list[tuple[int, int]] = []
        self.recent_xy_by_env: dict[int, list[tuple[int, int]]] = {}

        if arm in ('real', 'shuffled', 'white', 'iaaft'):
            if library is None:
                raise ValueError(f'Arm {arm} requires a loaded Phase 0 dynamics library')
            arrs = library['arrays']
            self.burst_z = np.asarray(arrs[f'{arm}_burst_z'], dtype=np.float64)
            self.entropy = np.asarray(arrs[f'{arm}_entropy'], dtype=np.float64)
            self.u4 = np.asarray(arrs[f'{arm}_u4'], dtype=np.float64)
            self.eps_stream = calibrate_epsilon_stream(self.burst_z, eps_base=self.eps_base, kappa=self.kappa)
            # Random start offset per training seed so every seed reads a distinct phase slice
            rng = np.random.default_rng(self.seed * 7919 + 13)
            self.offset = int(rng.integers(0, max(1, len(self.burst_z))))
        else:
            self.burst_z = None
            self.entropy = None
            self.u4 = None
            self.eps_stream = None
            self.offset = 0

    def reset_episode(self, env_idx: int | None = None):
        if env_idx is None:
            self.recent_xy.clear()
            self.recent_xy_by_env.clear()
        else:
            self.recent_xy_by_env.pop(int(env_idx), None)
            if int(env_idx) == 0:
                self.recent_xy.clear()

    def step_signal(self, obs: dict, step_index: int | None = None, *, env_idx: int = 0) -> dict:
        """Return strictly leak-free exploration parameters `(eps, dir_xy, burst_z, gated_safe)`.

        Each parallel `env_idx` advances along its own contiguous 1-bin-per-decision slice
        (quarter-library phase separated) so multi-env B4 batching NEVER strides or breaks
        biological burst temporal persistence (Lag-1 ACF / Hurst H / burst run lengths).
        """
        sensory = SensoryState.from_observation(obs)
        in_stress = homeostatic_site(sensory) != odd_design.SHAM
        gated_safe = (not in_stress) or (not self.homeostatic_gate)
        ei = int(env_idx)

        if step_index is None:
            idx = self.env_cursors.get(ei, 0)
            self.env_cursors[ei] = idx + 1
            if ei == 0:
                self.cursor = idx + 1
        else:
            idx = int(step_index)

        if self.arm == 'static':
            eps = self.eps_base if gated_safe else 0.0
            return dict(eps=eps, dir_xy=np.zeros(2, dtype=np.float64), burst_z=0.0, in_stress=in_stress, gated_safe=gated_safe)

        if self.arm == 'ideal':
            # Deterministic software upper bound: push away from centroid of recent 8 positions when safe
            if not gated_safe:
                return dict(eps=0.0, dir_xy=np.zeros(2, dtype=np.float64), burst_z=-1.0, in_stress=in_stress, gated_safe=False)
            hist = (self.recent_xy_by_env.get(ei, []) + [(sensory.x, sensory.y)])[-8:]
            self.recent_xy_by_env[ei] = hist
            if ei == 0:
                self.recent_xy = list(hist)
            if len(hist) >= 2:
                pts = np.asarray(hist[:-1], dtype=np.float64)
                vec = np.array([sensory.x, sensory.y], dtype=np.float64) - pts.mean(axis=0)
                nrm = float(np.linalg.norm(vec))
                dir_xy = (vec / nrm) if nrm > 1e-6 else np.array([1.0, 0.0], dtype=np.float64)
            else:
                dir_xy = np.array([1.0, 0.0], dtype=np.float64)
            return dict(eps=min(0.22, self.eps_base * 1.5), dir_xy=dir_xy, burst_z=1.0, in_stress=False, gated_safe=True)

        # Biological or matched-surrogate stream (real / shuffled / white / iaaft)
        n_bins = len(self.burst_z)
        phase_shift = (ei * max(1, n_bins // 4)) % n_bins
        pos = (self.offset + phase_shift + idx) % n_bins
        bz = float(self.burst_z[pos])
        eps = float(self.eps_stream[pos]) if gated_safe else 0.0
        # Project 4-quadrant unit vector u4 = (N, E, S, W) onto 2D direction vector
        u = self.u4[pos]
        vec2 = u @ CARDINAL_XY
        nrm = float(np.linalg.norm(vec2))
        dir_xy = (vec2 / nrm) if (gated_safe and nrm > 1e-8) else np.zeros(2, dtype=np.float64)
        return dict(eps=eps, dir_xy=dir_xy, burst_z=bz, in_stress=in_stress, gated_safe=gated_safe)

    def blend_verb_probs(self, base_probs: np.ndarray, verb_mask: np.ndarray, sig: dict) -> np.ndarray:
        """Pure-NumPy behavioral verb distribution (testable without torch)."""
        p = np.asarray(base_probs, dtype=np.float64)
        m = np.asarray(verb_mask, dtype=bool)
        eps = float(sig['eps'])
        if eps <= 0.0 or not m.any():
            out = np.where(m, p, 0.0)
            return out / max(float(out.sum()), 1e-12)
        # Boost move (0) and navigate (1) within the exploration mix when burst_z > 0 and safe
        q = np.zeros_like(p)
        legal_idx = np.where(m)[0]
        q[legal_idx] = 1.0
        if sig['gated_safe'] and sig['burst_z'] > 0.25:
            for exp_v in (0, 1):
                if exp_v < len(m) and m[exp_v]:
                    q[exp_v] *= 1.0 + 0.5 * min(2.0, float(sig['burst_z']))
        q /= max(float(q.sum()), 1e-12)
        blended = (1.0 - eps) * np.where(m, p, 0.0) + eps * q
        return blended / max(float(blended.sum()), 1e-12)

    def blend_target_probs(self, base_probs: np.ndarray, branch_actions: list[dict], obs: dict, sig: dict) -> np.ndarray:
        """Pure-NumPy behavioral target distribution biased along organoid spatial manifold dir_xy."""
        p = np.asarray(base_probs, dtype=np.float64)
        if len(p) <= 1 or not sig['gated_safe'] or float(np.linalg.norm(sig['dir_xy'])) < 1e-8 or sig['eps'] <= 0.0:
            return p / max(float(p.sum()), 1e-12)
        dir_xy = sig['dir_xy']
        logits = np.zeros(len(p), dtype=np.float64)
        has_dir = False
        for i, act in enumerate(branch_actions):
            u = _action_unit_direction(act, obs)
            if u is not None:
                logits[i] = self.dir_alpha * float(np.dot(u, dir_xy))
                has_dir = True
        if not has_dir:
            return p / max(float(p.sum()), 1e-12)
        logits -= logits.max()
        q = np.exp(logits)
        q /= max(float(q.sum()), 1e-12)
        mix = min(0.5, 2.0 * float(sig['eps']))
        blended = (1.0 - mix) * p + mix * q
        return blended / max(float(blended.sum()), 1e-12)

    def sample_batch(self, policy, prepared_list: list[dict], obs_list: list[dict], *,
                     generators=None) -> dict:
        """Batched B4 sampling under behavioral policy mu_t using exact policy._scope().

        Records exact on-policy `logp` / `verb_logp` (preserving zero_update_logp_tolerance)
        and behavioral `beh_logp` (for V-trace truncated off-policy IS in m8_train.update).
        """
        import torch

        n = len(prepared_list)
        sigs = [self.step_signal(obs, env_idx=i) for i, obs in enumerate(obs_list)]
        with torch.no_grad():
            vals = policy.values(prepared_list).detach().cpu().tolist()
            with policy._scope(prepared_list, False):
                vd = policy.verb_distribution(prepared_list, reference=False)
                base_vp_all = vd.probs.detach().cpu().numpy()
                v_indices = []
                beh_v_probs = []
                for i in range(n):
                    vmask = np.asarray(prepared_list[i]['verb_mask'], dtype=bool)
                    bvp = self.blend_verb_probs(base_vp_all[i], vmask, sigs[i])
                    beh_v_probs.append(bvp)
                    bt = torch.tensor(bvp, device=vd.probs.device, dtype=torch.float32)
                    gen = generators[i] if generators is not None else None
                    v_idx = int(torch.multinomial(bt, 1, generator=gen).item())
                    v_indices.append(v_idx)

                td = policy.target_distribution(prepared_list, v_indices, reference=False)
                base_tp_all = td.probs.detach().cpu().numpy()
                t_indices = []
                actions = []
                beh_logps = []
                for i in range(n):
                    branch = prepared_list[i]['branches'][v_indices[i]]
                    btp = self.blend_target_probs(base_tp_all[i][:len(branch)], branch, obs_list[i], sigs[i])
                    bt = torch.tensor(btp, device=td.probs.device, dtype=torch.float32)
                    gen = generators[i] if generators is not None else None
                    t_idx = int(torch.multinomial(bt, 1, generator=gen).item())
                    t_indices.append(t_idx)
                    actions.append(branch[t_idx])
                    beh_logps.append(float(
                        math.log(max(float(beh_v_probs[i][v_indices[i]]), 1e-12))
                        + math.log(max(float(btp[t_idx]), 1e-12))
                    ))

                v_t = torch.tensor(v_indices, device=vd.probs.device)
                t_t = torch.tensor(t_indices, device=td.probs.device)
                v_logp = vd.log_prob(v_t).detach().cpu().tolist()
                t_logp = td.log_prob(t_t).detach().cpu().tolist()

            with policy._scope(prepared_list, True):
                rv = policy.verb_distribution(prepared_list, reference=True)
                rt = policy.target_distribution(prepared_list, v_indices, reference=True)
                ref_v = rv.probs.detach().cpu().tolist()
                ref_t = rt.probs.detach().cpu().tolist()

        return dict(
            actions=actions,
            verbs=v_indices,
            targets=t_indices,
            values=[float(v) for v in vals],
            logp=[float(vl + tl) for vl, tl in zip(v_logp, t_logp)],
            verb_logp=[float(vl) for vl in v_logp],
            beh_logp=beh_logps,
            reference_verb=ref_v,
            reference_target=ref_t,
            mea_burst_z=[float(s['burst_z']) for s in sigs],
        )

    def sample_action(self, policy, prepared_row: dict, obs: dict, *, upd: int = 0, step: int = 0, env_i: int = 0) -> dict:
        """Single-row convenience wrapper around `sample_batch`."""
        res = self.sample_batch(policy, [prepared_row], [obs])
        res['mea_burst_z'] = res['mea_burst_z'][0]
        return res


class WetwareEntropyModulator:
    """M2: Burst-consolidate entropy rhythm modulator with exact batch-mean budget conservation."""

    def __init__(self, eta: float = 0.35, enabled: bool = True):
        self.eta = float(eta)
        self.enabled = bool(enabled)

    def batch_weights(self, rows: list[dict]) -> list[float]:
        n = len(rows)
        if n == 0:
            return []
        if not self.enabled or self.eta == 0.0:
            return [1.0] * n
        z = np.clip(np.asarray([float(r.get('mea_burst_z', 0.0)) for r in rows], dtype=np.float64), -3.0, 3.0)
        raw = np.exp(self.eta * z)
        normed = raw / max(float(raw.mean()), 1e-12)
        return [float(x) for x in normed]
