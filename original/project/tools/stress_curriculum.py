"""Non-invasive Mixed-Regime Curriculum & High-Stress Scene Staging for Wetware PPO.

Solves the historical training blind-spot (`min_thirst = 92.84`, zero crisis states seen during
short 48/64-step resets) WITHOUT modifying any hash-sealed files in `agent/`, `scripts/`, or `configs/`:

1) Stages the SHA-256-verified stress projects (`low_health`, `thirst`, `hunger`) from
   `recovered-scenes.tar.gz` into an owned temporary directory under `output/runtime_logs/`.
2) Interleaves:
   - Even updates: persistent `m9` exploration episodes (continuing across updates up to
     `max_persistent_ticks = 240` before resetting unless dead), covering both fresh-map
     exploration (`0..96` ticks) and natural mid-game resource depletion (`96..240` ticks).
   - Odd updates: deterministic rotation over `('thirst', 'hunger', 'low_health')` high-stress
     openings, paired identically across all experimental arms for each seed.
"""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VERIFICATION_DIR = (
    ROOT / 'output/mea_predictability_22021_20261119T042416Z/offline-confirmatory-results/verification'
)
STRESS_SCENES = ('thirst', 'hunger', 'low_health')
MAX_PERSISTENT_TICKS = 240
STRESS_HORIZON_TICK = 60


def _sha256_file(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _canonical_digest(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def stage_pinned_stress_projects(target_dir: Path, verification_dir: Path = DEFAULT_VERIFICATION_DIR):
    """Extract and SHA-verify `thirst`, `hunger`, and `low_health` plain Godot projects."""
    tar_path = verification_dir / 'recovered-scenes.tar.gz'
    rec_path = verification_dir / 'source-recovery.json'
    if not tar_path.is_file() or not rec_path.is_file():
        return {}

    pins = json.loads(rec_path.read_text(encoding='utf-8'))['restored_scene_pins']
    staged = {}
    with tarfile.open(tar_path, 'r:gz') as tar:
        for entry in tar:
            parts = entry.name.split('/', 3)
            if entry.isdir() or len(parts) != 4 or parts[0] != 'pinned-scenes':
                continue
            scene, role, filename = parts[1:]
            if scene not in STRESS_SCENES or role != 'plain' or not entry.isfile():
                continue
            dest = (target_dir / scene / role / filename).resolve()
            root_allowed = (target_dir / scene / role).resolve()
            if not dest.is_relative_to(root_allowed):
                raise ValueError('Unsafe tar path in recovered-scenes.tar.gz')
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('wb') as f:
                f.write(tar.extractfile(entry).read())

    for scene in STRESS_SCENES:
        if scene not in pins or 'plain' not in pins[scene]:
            continue
        scene_root = target_dir / scene / 'plain'
        file_pins = {p.relative_to(scene_root).as_posix(): _sha256_file(p) for p in scene_root.rglob('*') if p.is_file()}
        if _canonical_digest(file_pins) != pins[scene]['plain']:
            raise ValueError(f'SHA-256 mismatch when staging pinned stress scene: {scene}')
        staged[scene] = scene_root
    return staged


class MixedRegimeEnvPool:
    """Manages persistent standard `m9` exploration + pinned `thirst/hunger/low_health` stress envs."""

    def __init__(
        self,
        godot_path,
        project_path,
        *,
        map_mode='infinite_freeze',
        navigation_steps=4,
        max_persistent_ticks=MAX_PERSISTENT_TICKS,
        verification_dir=DEFAULT_VERIFICATION_DIR,
        enable_stress=True,
    ):
        self.godot_path = Path(godot_path)
        self.project_path = Path(project_path).resolve()
        self.map_mode = map_mode
        self.navigation_steps = navigation_steps
        self.max_persistent_ticks = int(max_persistent_ticks)
        self.verification_dir = Path(verification_dir)
        self.enable_stress = bool(enable_stress)
        self._stack = ExitStack()
        self.standard_env = None
        self.stress_envs = {}
        self._p_obs = None
        self._p_tracker = None
        self._p_fresh_start = True

    def __enter__(self):
        from agent.m9_env import GodotEnv
        log_root = self.project_path / 'output/runtime_logs'
        log_root.mkdir(parents=True, exist_ok=True)
        self.standard_env = self._stack.enter_context(
            GodotEnv(str(self.godot_path), str(self.project_path),
                     map_mode=self.map_mode, navigation_steps=self.navigation_steps)
        )
        if self.enable_stress:
            tmp_dir = Path(self._stack.enter_context(
                tempfile.TemporaryDirectory(prefix='wetware-stress-scenes-', dir=log_root)
            ))
            staged = stage_pinned_stress_projects(tmp_dir, self.verification_dir)
            for scene, scene_root in staged.items():
                self.stress_envs[scene] = self._stack.enter_context(
                    GodotEnv(str(self.godot_path), str(scene_root),
                             map_mode=self.map_mode, navigation_steps=self.navigation_steps)
                )
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._stack.__exit__(exc_type, exc, tb)

    def regime_for_update(self, upd: int) -> str:
        """Odd updates (1, 3, 5, ...) run persistent exploration; even updates rotate stress scenes."""
        if upd % 2 == 1 or not self.stress_envs:
            return 'standard'
        ordered = [s for s in STRESS_SCENES if s in self.stress_envs]
        return ordered[((upd // 2) - 1) % len(ordered)]

    def begin_rollout(self, upd: int, world_seed: int, reward_cls, reward_variant: str, gamma_tick: float):
        """Return `(env, obs, tracker, episode_start, regime)` for update `upd`."""
        regime = self.regime_for_update(upd)
        if regime == 'standard':
            cur_tick = int(self._p_obs.get('world_tick', self._p_obs.get('tick', 0))) if self._p_obs else 0
            if self._p_obs is None or cur_tick >= self.max_persistent_ticks:
                init = self.standard_env.reset(int(world_seed))
                self._p_obs = init['observation']
                self._p_tracker = reward_cls(self._p_obs, reward_variant, gamma_tick)
                self._p_fresh_start = True
            ep_start = self._p_fresh_start
            self._p_fresh_start = False
            return self.standard_env, self._p_obs, self._p_tracker, ep_start, 'standard'

        s_env = self.stress_envs[regime]
        init = s_env.reset(int(world_seed))
        obs = init['observation']
        tracker = reward_cls(obs, reward_variant, gamma_tick)
        return s_env, obs, tracker, True, regime

    def update_persistent_state(self, regime: str, obs, tracker, reset_needed: bool, next_seed: int,
                                reward_cls, reward_variant: str, gamma_tick: float):
        """Keep persistent `standard` state across rollout boundaries; reset immediately on death/cap."""
        if regime == 'standard':
            cur_tick = int(obs.get('world_tick', obs.get('tick', 0)))
            if reset_needed or cur_tick >= self.max_persistent_ticks:
                init = self.standard_env.reset(int(next_seed))
                self._p_obs = init['observation']
                self._p_tracker = reward_cls(self._p_obs, reward_variant, gamma_tick)
                self._p_fresh_start = True
                return self._p_obs, self._p_tracker, True
            self._p_obs = obs
            self._p_tracker = tracker
            self._p_fresh_start = False
            return obs, tracker, False

        s_env = self.stress_envs[regime]
        if reset_needed:
            init = s_env.reset(int(next_seed))
            new_obs = init['observation']
            new_tracker = reward_cls(new_obs, reward_variant, gamma_tick)
            return new_obs, new_tracker, True
        return obs, tracker, False
