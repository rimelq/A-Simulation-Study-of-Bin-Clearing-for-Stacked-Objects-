"""BinClearingGymEnv, Gymnasium env for bin clearing with candidate selection.

Episode = clear one source bin (move every item to the destination bin).
Step    = one pick attempt:

    1. Robot at the fixed overhead sensing pose.
    2. Overhead depth -> candidate generator (GG-CNN / CC / PPO) -> top-K poses.
    3. Build flat observation.
    4. Policy picks a in [0, K*27 - 1]: candidate slot k + 3x3x3 (dx,dy,dyaw)
       refinement. Empty slot -> invalid action.
    5. Associate the chosen candidate with the closest source-bin item.
    6. ``reward_mode``:
         "hybrid"          (default): geometric grasp predicate + deterministic
                                       geometric neighbour disturbance.
         "hybrid_physics" : geometric predicate + real-physics disturbance.
         "physics"        : full attempt_grasp_physical (legacy).
         "geometric"      : predicate only, no disturbance.
    7. ``compute_reward`` (rl/reward.py).
    8. Re-perceive. terminated iff bin empty. truncated iff step_count >= max_steps.
"""
import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sim.robosuite_env import BinClearingEnv
from sim.sensing_pose import BIN_HALF_SIZE
from control.sensing_pose_controller import SensingPoseController
from control.pick_place_primitive import PickPlacePrimitive
from control.grasp_success_predicate import evaluate_grasp, _FINGER_TO_WRIST
from control.geometric_disturbance import compute_and_apply_disturbance
from perception.ggcnn_infer import GGCNNInference
from perception.perceive import perceive_grasp_candidates
from perception.perceive_cc import perceive_cc_candidates
from perception.perceive_ppo import perceive_ppo_candidates
from rl.observation_builder import build_observation, obs_dim_for_K
from rl.reward import compute_reward, compute_reward_geometric

# square_22 assets: cubes ~5.9x5.0x5.9 cm, realistic gravity-fall stack
_ASSETS_DIR = os.path.join(_ROOT, "assets")
_DEFAULT_STL_DIR = os.path.join(_ASSETS_DIR, "square_22_pkg", "stl_vis")
_DEFAULT_POSES_CSV = os.path.join(_ASSETS_DIR, "square_22_pkg", "poses_relative_to_container.csv")
_DEFAULT_GENERATED_XML_DIR = os.path.join(_ASSETS_DIR, "square_22_xml")

# descent below depth-read item-top before wrist target
_GRASP_DESCENT_OFFSET = 0.012

# half-height of a square_22 cube
_ITEM_HALF_HEIGHT = 0.029

# XY tolerance (metres) for associating a candidate with a sim item
_CANDIDATE_ITEM_XY_TOL = 0.06

# 2x K to absorb the ~65% empty-grab filter loss while keeping K=10 post-filter
_CANDIDATE_RAW_K = 20

# v5 action-space refinement: Discrete(K * 27) over (slot, 3x3x3 (dx,dy,dyaw)).
# m = 13 is the zero-offset cell so selection-only baselines nest in v5.
REFINEMENT_LEVELS = 3
N_REFINEMENT      = 27
REFINEMENT_STEP_XY  = 0.015
REFINEMENT_STEP_YAW = 0.1745  # ~10 deg


def _decode_action_v5(action_id: int):
    """Decode flat v5 action id into (k, dx, dy, dyaw).

    action_id = k * N_REFINEMENT + m
    m = dx_idx * 9 + dy_idx * 3 + dyaw_idx, each idx in {0,1,2}
    offset(idx) = (idx - 1) * step
    """
    action_id = int(action_id)
    k, m = divmod(action_id, N_REFINEMENT)
    dx_idx = m // 9
    dy_idx = (m % 9) // 3
    dyaw_idx = m % 3
    dx = (dx_idx - 1) * REFINEMENT_STEP_XY
    dy = (dy_idx - 1) * REFINEMENT_STEP_XY
    dyaw = (dyaw_idx - 1) * REFINEMENT_STEP_YAW
    return k, float(dx), float(dy), float(dyaw)


def _yaw_quat_wxyz(yaw_rad: float) -> np.ndarray:
    """World-frame quat (wxyz) for a rotation about world z by yaw_rad."""
    c = float(np.cos(0.5 * float(yaw_rad)))
    s = float(np.sin(0.5 * float(yaw_rad)))
    return np.array([c, 0.0, 0.0, s], dtype=float)


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions: q1 * q2.

    Applying yaw in the WORLD frame left-multiplies the grasp quat
    (q_out = q_yaw * q_grasp).
    """
    w1, x1, y1, z1 = float(q1[0]), float(q1[1]), float(q1[2]), float(q1[3])
    w2, x2, y2, z2 = float(q2[0]), float(q2[1]), float(q2[2]), float(q2[3])
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=float)


# Dest-bin tiling: 3x3 grid per layer, 6.5 cm > 5.9 cm cube so deliveries
# don't merge, layers stacked 6.2 cm in z.
_DEST_SLOT_SPACING = 0.065
_DEST_ITEM_LIFT = 0.030
_DEST_LAYER_H = 0.062
_DEST_PER_LAYER = 9


class BinClearingGymEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    # v2 (2026-05-17) retune: prev defaults left returns deeply negative because
    # disturbance dominated. invalid_penalty unused under MaskablePPO (mask
    # eliminates invalid actions at sampling).
    _DEFAULT_REWARD_WEIGHTS = dict(
        step_penalty=-0.02,
        deliver_reward=3.0,
        grasp_quality_coef=0.30,
        empty_grab_penalty=-0.50,
        invalid_penalty=-1.0,
        # v5: bumped 1.0 -> 2.0. Refinement genuinely interacts with neighbours
        disturb_coef=2.0,
        eject_penalty=1.0,
        # v5: predictive shaping for post-refinement descent-column clearance
        approach_clear_pred_coef=0.50,
        # v5 L4 sequential-reasoning shaping
        lookahead_bonus_coef=0.20,
        keystone_penalty_coef=0.30,
    )

    def __init__(self,
                 n_objects: int = 20,
                 K: int = 10,
                 object_subset_seed: int = 42,
                 reward_mode: str = "hybrid",
                 slow_physics: bool = False,
                 render_mode: str = None,
                 max_steps: int = None,
                 max_steps_per_object: float = 3.0,
                 layout_jitter: float = 0.02,
                 reward_weights: dict = None,
                 stl_dir: str = _DEFAULT_STL_DIR,
                 poses_csv: str = _DEFAULT_POSES_CSV,
                 generated_xml_dir: str = _DEFAULT_GENERATED_XML_DIR,
                 ggcnn_device: str = None,
                 orientation_source: str = "snap",
                 filter_candidates: bool = True,
                 candidate_source: str = "ggcnn",
                 ppo_visibility_mode: str = "raycast",
                 ppo_quality_mode: str = "analytical",
                 failure_mask_size: int = 3,
                 use_snap_xy: bool = False,
                 use_snap_z: bool = True):
        super().__init__()
        self.n_objects = int(n_objects)
        self.K = int(K)
        self.object_subset_seed = int(object_subset_seed)
        # orientation_source: "snap" uses GT short-axis yaw, "raw" uses GG-CNN's
        # own angle (needed for paper-faithful greedy baselines).
        if orientation_source not in ("snap", "raw"):
            raise ValueError(f"orientation_source must be 'snap' or 'raw', got {orientation_source!r}")
        self.orientation_source = orientation_source
        # filter_candidates: True drops raw GG-CNN candidates with no item
        # within 6 cm. False is paper-faithful for the greedy baseline.
        self.filter_candidates = bool(filter_candidates)
        # candidate_source: "ggcnn" | "cc" (NOT recommended, merges cubes) | "ppo"
        if candidate_source not in ("ggcnn", "cc", "ppo"):
            raise ValueError(f"candidate_source must be 'ggcnn', 'cc' or "
                             f"'ppo', got {candidate_source!r}")
        self.candidate_source = candidate_source
        if ppo_visibility_mode not in ("raycast", "omniscient"):
            raise ValueError(f"ppo_visibility_mode must be 'raycast' or "
                             f"'omniscient', got {ppo_visibility_mode!r}")
        if ppo_quality_mode not in ("analytical", "uniform"):
            raise ValueError(f"ppo_quality_mode must be 'analytical' or "
                             f"'uniform', got {ppo_quality_mode!r}")
        self.ppo_visibility_mode = ppo_visibility_mode
        self.ppo_quality_mode = ppo_quality_mode
        # failure_mask_size: ring buffer of recently-failed source_body_ids
        # 0 disables. Breaks L2 fixation across all selection policies.
        if int(failure_mask_size) < 0:
            raise ValueError(f"failure_mask_size must be >= 0, "
                             f"got {failure_mask_size}")
        self.failure_mask_size = int(failure_mask_size)
        # reward_mode:
        # "hybrid" (default, Option B): geometric predicate + geometric disturbance
        # "hybrid_physics": geometric predicate + real-physics disturbance
        # "physics": legacy full attempt_grasp_physical
        # "geometric": predicate only, no disturbance (smoke tests)
        if reward_mode not in ("hybrid", "hybrid_physics", "physics", "geometric"):
            raise ValueError(f"reward_mode must be 'hybrid', 'hybrid_physics', "
                             f"'physics' or 'geometric', got {reward_mode!r}")
        self.reward_mode = reward_mode
        self.slow_physics = bool(slow_physics)
        self.render_mode = render_mode
        # Episode step cap: explicit max_steps wins. else
        # max_steps_per_object * n_objects (3x is the room needed for retries).
        if max_steps is not None:
            self.max_steps = int(max_steps)
        else:
            self.max_steps = max(1, int(round(float(max_steps_per_object) * self.n_objects)))
        # Per-episode XY jitter so the layout (and candidates) varies. uses
        # self.np_random for reproducibility per seed. 0 disables.
        self.layout_jitter = float(layout_jitter)
        self.reward_weights = dict(self._DEFAULT_REWARD_WEIGHTS)
        if reward_weights:
            unknown = set(reward_weights) - set(self._DEFAULT_REWARD_WEIGHTS)
            if unknown:
                raise ValueError(f"Unknown reward_weights keys: {sorted(unknown)}. "
                                 f"Allowed: {sorted(self._DEFAULT_REWARD_WEIGHTS)}")
            self.reward_weights.update({k: float(v) for k, v in reward_weights.items()})

        # use_camera_obs=False: skip robosuite's per-step camera render. perception
        # uses its own renderer. ~halves per-step cost.
        self.env = BinClearingEnv(
            n_objects=self.n_objects,
            has_renderer=False,
            max_episode_steps=5000,
            object_subset_seed=self.object_subset_seed,
            stl_dir=stl_dir,
            poses_csv=poses_csv,
            generated_xml_dir=generated_xml_dir,
            use_camera_obs=False,
        )
        self.sensing_ctrl = SensingPoseController(self.env)
        self.pick_place = PickPlacePrimitive(
            self.env, self.env.get_dst_bin_world_pos(), self.sensing_ctrl)
        # device=None -> auto-pick (cuda when available). GPU is ~10x faster than
        # CPU on the cluster.
        self.ggcnn = GGCNNInference(device=ggcnn_device)

        # v5 snap gating. Pre-v5 the XY/yaw snap unconditionally overwrote any
        # policy-chosen pose with GT, making refinement a no-op. With
        # use_snap_xy=False (v5 default) horizontal pose is policy-controlled
        # end-to-end. Z snap is PRIMITIVE CALIBRATION and stays on by default.
        self.use_snap_xy = bool(use_snap_xy)
        self.use_snap_z  = bool(use_snap_z)

        self._obs_dim = obs_dim_for_K(self.K)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)
        # v5: Discrete(K * 27) joint (candidate slot, refinement cell).
        self.action_space = spaces.Discrete(self.K * N_REFINEMENT)

        self._src_bin_world = self.env.get_src_bin_world_pos()
        self._dst_bin_world = self.env.get_dst_bin_world_pos()
        self._candidates = []
        self._meta = {}
        self._step_count = 0
        self._prev_success = False
        from collections import deque as _deque
        self._recent_failed_bids = _deque(maxlen=max(1, self.failure_mask_size))
        # Per-episode aggregates surfaced in info on every step so SB3
        # (Vec)Monitor logs the end-of-episode values to monitor.csv.
        self._init_episode_counters()
        self._n_delivered = 0
        self._delivered_names = []

        # v5 L4 sequential state: 5-step delivered ring + prior-step visible count
        from collections import deque as _deque5
        self._delivered_last5 = _deque5(maxlen=5)
        self._n_visible_prev_step = 0
        self._clear_l4_cache()

    # helpers

    def _source_bin_items(self):
        """[(name, world_pos)] for items currently in the source bin."""
        src = self._src_bin_world
        out = []
        for name, pos in self.env.get_object_positions().items():
            pos = np.asarray(pos, dtype=float)
            if (abs(pos[0] - src[0]) < BIN_HALF_SIZE[0]
                    and abs(pos[1] - src[1]) < BIN_HALF_SIZE[1]
                    and pos[2] > src[2] - 0.05):
                out.append((name, pos))
        return out

    def _n_items_remaining(self):
        return len(self._source_bin_items())

    def _approach_clear_pred(self, candidate_pos, source_items=None,
                              gripper_radius: float = 0.02,
                              exclude_assoc: bool = True) -> float:
        """Vertical-cylinder descent-clearance predicate (1.0 = clear)."""
        cp = np.asarray(candidate_pos, dtype=float)
        if source_items is None:
            source_items = self._source_bin_items()
        if not source_items:
            return 1.0
        item_xy = np.asarray([p[:2] for _, p in source_items], dtype=float)
        item_z  = np.asarray([float(p[2]) for _, p in source_items], dtype=float)
        d_xy = np.linalg.norm(item_xy - cp[:2][None, :], axis=1)
        # exclude associated item (nearest within 6 cm)
        keep = np.ones(len(source_items), dtype=bool)
        if exclude_assoc and len(d_xy) > 0 and d_xy.min() <= 0.06:
            keep[int(d_xy.argmin())] = False
        # 1 mm tolerance avoids spurious self-block if exclusion missed
        blockers = keep & (d_xy <= float(gripper_radius)) & (item_z <= float(cp[2]) + 1e-3)
        return float(0.0 if blockers.any() else 1.0)

    # v5 L4 sequential features
    def _clear_l4_cache(self):
        # per-step cache shared across the K candidates within a step
        self._l4_cache = {
            "exposure": {},
            "keystone": {},
            "tilt":     {},
        }

    def _candidate_tilt(self, body_name: str) -> float:
        """|R[2,2]|, 1.0 = upright, 0.0 = on a side. Falls back to 1.0 on error."""
        if not hasattr(self, "_l4_cache"):
            self._clear_l4_cache()
        cache = self._l4_cache["tilt"]
        if body_name in cache:
            return cache[body_name]
        val = 1.0
        try:
            bid = self.env.sim.model.body_name2id(body_name)
            R = np.array(self.env.sim.data.body_xmat[bid]).reshape(3, 3)
            val = float(abs(R[2, 2]))
        except Exception:
            val = 1.0
        cache[body_name] = val
        return val

    def _exposure_value(self, body_name: str, source_items=None) -> int:
        """Currently-hidden cubes that would become visible from above if
        ``body_name`` were removed. Clamped to [0, 5].

        Mirrors perceive_ppo._is_visible_from_above (XY-footprint occlusion,
        +0.005 tolerance, 5 mm z margin).
        """
        if not hasattr(self, "_l4_cache"):
            self._clear_l4_cache()
        cache = self._l4_cache["exposure"]
        if body_name in cache:
            return cache[body_name]
        val = 0
        try:
            from perception.perceive_ppo import (
                _is_visible_from_above as _vis,
                _ITEM_HALF_XY as _PPO_HALF_XY,
                _OCCLUSION_XY_TOL as _PPO_XY_TOL,
                _OCCLUSION_Z_TOL as _PPO_Z_TOL,
            )
            if source_items is None:
                source_items = self._source_bin_items()
            if source_items:
                items_with_top = [(n, np.asarray(p, dtype=float),
                                   float(p[2]) + _ITEM_HALF_HEIGHT)
                                  for n, p in source_items]
                base_hidden = []
                for i, (n_i, p_i, top_i) in enumerate(items_with_top):
                    if n_i == body_name:
                        continue
                    others = [(n_j, p_j)
                              for j, (n_j, p_j, _) in enumerate(items_with_top)
                              if j != i]
                    if not _vis(p_i, top_i, others):
                        base_hidden.append(i)
                newly_visible = 0
                for i in base_hidden:
                    n_i, p_i, top_i = items_with_top[i]
                    others = [(n_j, p_j)
                              for j, (n_j, p_j, _) in enumerate(items_with_top)
                              if j != i and n_j != body_name]
                    if _vis(p_i, top_i, others):
                        newly_visible += 1
                val = int(np.clip(newly_visible, 0, 5))
        except Exception:
            val = 0
        cache[body_name] = val
        return val

    def _is_keystone_geometric(self, body_name: str, source_items=None) -> int:
        """# other cubes resting on body_name with no intermediate cube in
        the same XY column. Clamped to [0, 3]."""
        if not hasattr(self, "_l4_cache"):
            self._clear_l4_cache()
        cache = self._l4_cache["keystone"]
        if body_name in cache:
            return cache[body_name]
        val = 0
        try:
            if source_items is None:
                source_items = self._source_bin_items()
            tgt = None
            for n, p in source_items:
                if n == body_name:
                    tgt = (n, np.asarray(p, dtype=float))
                    break
            if tgt is not None:
                _, tp = tgt
                tx, ty, tz = float(tp[0]), float(tp[1]), float(tp[2])
                count = 0
                for n_j, p_j in source_items:
                    if n_j == body_name:
                        continue
                    jx, jy, jz = float(p_j[0]), float(p_j[1]), float(p_j[2])
                    if abs(jx - tx) >= 0.05 or abs(jy - ty) >= 0.05:
                        continue
                    if jz <= tz + _ITEM_HALF_HEIGHT:
                        continue
                    has_intermediate = False
                    for n_m, p_m in source_items:
                        if n_m == body_name or n_m == n_j:
                            continue
                        mx, my, mz = float(p_m[0]), float(p_m[1]), float(p_m[2])
                        if abs(mx - tx) >= 0.05 or abs(my - ty) >= 0.05:
                            continue
                        if tz < mz < jz:
                            has_intermediate = True
                            break
                    if not has_intermediate:
                        count += 1
                val = int(np.clip(count, 0, 3))
        except Exception:
            val = 0
        cache[body_name] = val
        return val

    def _wrist_target_from_candidate(self, cand):
        """Wrist (OSC EEF) target: candidate world_pos with z = item_top -
        descent + finger_to_wrist. XY clamped 6 cm inside the bin."""
        wp = np.asarray(cand["world_pos"], dtype=float).copy()
        item_top_z = float(wp[2])
        wrist_z = item_top_z - _GRASP_DESCENT_OFFSET + _FINGER_TO_WRIST
        src = self._src_bin_world
        delta = wp[:2] - src[:2]
        hx, hy = BIN_HALF_SIZE[0], BIN_HALF_SIZE[1]
        margin = 0.06
        delta = np.clip(delta, [-hx + margin, -hy + margin], [hx - margin, hy - margin])
        out = np.array([src[0] + delta[0], src[1] + delta[1], wrist_z], dtype=float)
        return out

    def _associate_item(self, grasp_pos):
        """Closest source-bin item to grasp XY, or None if none within
        ``_CANDIDATE_ITEM_XY_TOL``."""
        items = self._source_bin_items()
        if not items:
            return None
        name, pos = min(items, key=lambda np_: float(np.linalg.norm(
            np.asarray(np_[1][:2]) - np.asarray(grasp_pos[:2]))))
        d = float(np.linalg.norm(np.asarray(pos[:2]) - np.asarray(grasp_pos[:2])))
        return name if d <= _CANDIDATE_ITEM_XY_TOL else None

    def _grasp_quat_for_item(self, item_name):
        """World gripper quat (wxyz): top-down, jaws aligned to the item's
        true short horizontal axis (Ori-snap, project doc Sec D primitive support).
        Falls back to fixed-yaw down-grasp on error."""
        try:
            from scipy.spatial.transform import Rotation as _Rot
            bid = self.env.sim.model.body_name2id(item_name)
            R_item = np.array(self.env.sim.data.body_xmat[bid]).reshape(3, 3)
            gid = None
            for g in range(self.env.sim.model.ngeom):
                if self.env.sim.model.geom_bodyid[g] == bid:
                    gid = g
                    break
            if gid is None:
                raise RuntimeError("no geom for item")
            half = np.array(self.env.sim.model.geom_aabb[gid], dtype=float)[3:6]
            # shortest *horizontal* local axis: vertical projects to ~0
            xy_ext = np.array([half[i] * np.sqrt(max(1.0 - float(R_item[2, i]) ** 2, 0.0))
                               for i in range(3)])
            short_idx = int(np.argsort(xy_ext)[1])
            short_local = np.zeros(3); short_local[short_idx] = 1.0
            short_world = R_item @ short_local
            short_xy = np.array([short_world[0], short_world[1]])
            n = float(np.linalg.norm(short_xy))
            if n < 1e-6:
                raise RuntimeError("degenerate short axis")
            short_xy /= n
            gz = np.array([0.0, 0.0, -1.0])
            gy = np.array([short_xy[0], short_xy[1], 0.0])
            gx = np.cross(gy, gz); gx /= max(np.linalg.norm(gx), 1e-9)
            R_w = np.column_stack([gx, gy, gz])
            xyzw = _Rot.from_matrix(R_w).as_quat()
            return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=float)  # wxyz
        except Exception:
            # 180 deg about world X = top-down at sensing yaw
            return np.array([0.0, 1.0, 0.0, 0.0], dtype=float)

    def _jitter_layout(self):
        """Per-episode XY jitter on source-bin items (3 cm wall clearance)."""
        if self.layout_jitter <= 0.0:
            return
        src = self._src_bin_world
        hx, hy = BIN_HALF_SIZE[0] - 0.03, BIN_HALF_SIZE[1] - 0.03
        for name, pos in self._source_bin_items():
            pos = np.asarray(pos, dtype=float)
            dx = float(self.np_random.uniform(-self.layout_jitter, self.layout_jitter))
            dy = float(self.np_random.uniform(-self.layout_jitter, self.layout_jitter))
            nx = float(np.clip(pos[0] + dx, src[0] - hx, src[0] + hx))
            ny = float(np.clip(pos[1] + dy, src[1] - hy, src[1] + hy))
            try:
                bid = self.env.sim.model.body_name2id(name)
                cur_quat = np.array(self.env.sim.data.body_xquat[bid], dtype=float)
            except Exception:
                cur_quat = np.array([1.0, 0.0, 0.0, 0.0])
            self.env._set_object_pose(name, np.array([nx, ny, pos[2]]), cur_quat)
        self.env._forward()

    def _dest_slot_pose(self, index: int):
        """World (pos, quat) for the index-th delivered item: 3x3 per layer,
        non-overlapping 6.5 cm spacing, layers stacked in z."""
        dst = self._dst_bin_world
        layer = index // _DEST_PER_LAYER
        slot  = index %  _DEST_PER_LAYER
        row, col = slot // 3, slot % 3
        pos = np.array([
            dst[0] + (col - 1) * _DEST_SLOT_SPACING,
            dst[1] + (row - 1) * _DEST_SLOT_SPACING,
            dst[2] + _DEST_ITEM_LIFT + layer * _DEST_LAYER_H,
        ], dtype=float)
        return pos, np.array([1.0, 0.0, 0.0, 0.0])

    def _deliver_item(self, item_name):
        """Teleport item_name to its dest-bin slot.

        Index is len(_delivered_names) because step() appends AFTER calling this.
        """
        pos, quat = self._dest_slot_pose(len(self._delivered_names))
        self.env._set_object_pose(item_name, pos, quat)
        self.env._forward()

    def _hybrid_geometric_outcome(self, grasp_pos, grasp_quat,
                                  target_item_name, predicate) -> dict:
        """reward_mode='hybrid' (Option B): geometric grasp-success predicate +
        deterministic geometric neighbour-disturbance model.

        compute_and_apply_disturbance MOVES the disturbed items in the sim and
        ejects any shoved past the rim.
        """
        disturb = compute_and_apply_disturbance(
            self.env, grasp_pos, grasp_quat, target_item_name)
        # graded grasp quality = weighted fraction of the four predicate
        # sub-criteria, gives near-misses smooth partial credit
        grasp_quality = float(
            0.40 * bool(predicate.get("item_between_jaws", False)) +
            0.20 * bool(predicate.get("jaws_aligned",      False)) +
            0.20 * bool(predicate.get("mid_height",        False)) +
            0.20 * bool(predicate.get("approach_clear",    False)))
        return {
            "grasp_ok":                bool(predicate.get("success", False)),
            "grasp_quality":           grasp_quality,
            "neighbour_disturbance_m": float(disturb["neighbour_disturbance_m"]),
            "items_ejected":           int(disturb["items_ejected"]),
            "n_disturbed":             int(disturb["n_disturbed"]),
            "n_in_footprint":          int(disturb["n_in_footprint"]),
            "picked_item":             predicate.get("picked_item"),
            "predicate_reason":        predicate.get("reason", ""),
        }

    def _re_pin_delivered(self):
        """Re-pin delivered items in case sim reset / step nudged them."""
        for i, name in enumerate(self._delivered_names):
            pos, quat = self._dest_slot_pose(i)
            self.env._set_object_pose(name, pos, quat)
        if self._delivered_names:
            self.env._forward()

    def _filter_candidates_to_items(self, candidates: list) -> list:
        """Drop candidates with no source-bin item centre within 6 cm.

        ~65% of raw GG-CNN candidates land at empty positions (centroid snap on
        a gap, or back-projection samples the bin floor). Perception-layer
        cleanup (project doc Sec D), not a learning hint.
        """
        items = self._source_bin_items()
        if not items:
            return []
        item_xy = np.asarray([p[:2] for _, p in items], dtype=np.float64)
        out = []
        for c in candidates:
            gp_xy = np.asarray(c["world_pos"][:2], dtype=np.float64)
            if np.min(np.linalg.norm(item_xy - gp_xy, axis=1)) <= _CANDIDATE_ITEM_XY_TOL:
                out.append(c)
        return out

    def _perceive(self):
        # all sources share a common candidate-dict schema
        if self.candidate_source == "ppo":
            # Perfect-Perception Oracle: one candidate per visible cube
            # source_body_id emitted directly.
            raw, self._meta = perceive_ppo_candidates(
                self, self.sensing_ctrl, ggcnn=None, K=self.K,
                visibility_mode=self.ppo_visibility_mode,
                quality_mode=self.ppo_quality_mode)
            self._candidates = raw[:self.K]
        elif self.candidate_source == "cc":
            # depth-only connected components. merges touching cubes, NOT recommended
            raw, self._meta = perceive_cc_candidates(
                self.env, self.sensing_ctrl, ggcnn=None, K=self.K)
            self._candidates = raw[:self.K]
        else:
            raw, self._meta = perceive_grasp_candidates(
                self.env, self.sensing_ctrl, self.ggcnn, K=_CANDIDATE_RAW_K)
            if self.filter_candidates:
                filtered = self._filter_candidates_to_items(raw)
            else:
                # paper-faithful: top-K by quality, no GT filter
                filtered = sorted(raw, key=lambda c: -c["quality"])
            self._candidates = filtered[:self.K]
        # failure mask keys on body id. PPO emits it, GG-CNN / CC don't
        self._annotate_candidates_with_body_ids()
        try:
            self._meta["n_candidates_raw"] = len(raw)
            if self.candidate_source in ("cc", "ppo"):
                self._meta["n_candidates_after_filter"] = len(self._candidates)
            else:
                self._meta["n_candidates_after_filter"] = len(filtered)
        except Exception:
            pass

    def _annotate_candidates_with_body_ids(self):
        """Backfill candidate['source_body_id'] for GG-CNN / CC. -1 = no
        association (failure mask treats -1 as never-matches)."""
        for c in self._candidates:
            if c.get("source_body_id", None) is not None and\
               c.get("source_body_id", -1) >= 0:
                continue   # PPO already set it
            assoc_name = self._associate_item(np.asarray(c["world_pos"]))
            if assoc_name is None:
                c["source_body_id"] = -1
                continue
            try:
                bid = int(self.env.sim.model.body_name2id(assoc_name))
            except Exception:
                bid = -1
            c["source_body_id"] = bid

    def _build_obs(self):
        # v5 L4 per-candidate sequential features
        source_items = self._source_bin_items()
        l4_tilt, l4_expo, l4_keystone = [], [], []
        for c in self._candidates:
            try:
                bid = int(c.get("source_body_id", -1))
            except Exception:
                bid = -1
            body_name = None
            if bid >= 0:
                try:
                    body_name = self.env.sim.model.body_id2name(bid)
                except Exception:
                    body_name = None
            if body_name is None:
                l4_tilt.append(1.0)
                l4_expo.append(0)
                l4_keystone.append(0)
            else:
                l4_tilt.append(float(self._candidate_tilt(body_name)))
                l4_expo.append(int(self._exposure_value(
                    body_name, source_items=source_items)))
                l4_keystone.append(int(self._is_keystone_geometric(
                    body_name, source_items=source_items)))

        # v5 L4 globals
        n_vis_prev_norm = float(np.clip(int(self._n_visible_prev_step),
                                        0, 10)) / 10.0
        if len(self._delivered_last5) > 0:
            recent_rate = float(sum(self._delivered_last5)) / 5.0
        else:
            recent_rate = 0.0
        try:
            bin_floor_z = float(self._src_bin_world[2]) - float(BIN_HALF_SIZE[2])
            bin_height  = 2.0 * float(BIN_HALF_SIZE[2])
            if source_items:
                max_z = max(float(p[2]) for _, p in source_items)
                pile_h_norm = float(np.clip(
                    (max_z - bin_floor_z) / max(bin_height, 1e-6), 0.0, 1.0))
            else:
                pile_h_norm = 0.0
        except Exception:
            pile_h_norm = 0.0

        return build_observation(
            self._candidates, self.env, self._src_bin_world,
            prev_success=self._prev_success,
            step_count=self._step_count,
            K=self.K, max_steps=self.max_steps,
            n_items_remaining=self._n_items_remaining(),
            n_objects=self.n_objects,
            l4_tilt=l4_tilt,
            l4_exposure=l4_expo,
            l4_keystone=l4_keystone,
            n_visible_prev_step_norm=n_vis_prev_norm,
            recent_clearing_rate=recent_rate,
            pile_height_normalised=pile_h_norm,
        )

    def _init_episode_counters(self):
        self._ep_n_invalid = 0
        self._ep_n_empty_grab = 0
        self._ep_n_physics_attempts = 0
        self._ep_sum_grasp_quality = 0.0
        self._ep_sum_disturb_m = 0.0
        self._ep_sum_disturb_raw_m = 0.0
        self._ep_max_disturb_m = 0.0
        self._ep_n_ejected = 0
        self._ep_n_predicate_succ = 0
        self._ep_sum_reward = 0.0
        self._ep_n_failure_masked = 0

    def _info_episode_aggregates(self) -> dict:
        """Snapshot of per-episode counters. SB3 (Vec)Monitor logs LAST step's
        values to monitor.csv as extra columns (listed in train.py info_keywords)."""
        return {
            "n_invalid":          int(self._ep_n_invalid),
            "n_empty_grab":       int(self._ep_n_empty_grab),
            "n_physics_attempts": int(self._ep_n_physics_attempts),
            "sum_grasp_quality":  float(self._ep_sum_grasp_quality),
            "sum_disturb_m":      float(self._ep_sum_disturb_m),
            "sum_disturb_raw_m":  float(self._ep_sum_disturb_raw_m),
            "max_disturb_m":      float(self._ep_max_disturb_m),
            "n_ejected":          int(self._ep_n_ejected),
            "n_predicate_succ":   int(self._ep_n_predicate_succ),
            "n_failure_masked":   int(getattr(self, "_ep_n_failure_masked", 0)),
            "n_objects_initial":  int(self.n_objects),
        }

    # gym API

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()
        self.sensing_ctrl.set_sensing_pose_direct()
        self.env._forward()
        self._src_bin_world = self.env.get_src_bin_world_pos()
        self._dst_bin_world = self.env.get_dst_bin_world_pos()
        self._jitter_layout()
        self._step_count = 0
        self._prev_success = False
        self._n_delivered = 0
        self._delivered_names = []
        # failure mask is INTRA-episode. clear at episode start
        self._recent_failed_bids.clear()
        self._init_episode_counters()
        self._delivered_last5.clear()
        self._n_visible_prev_step = 0
        self._clear_l4_cache()
        self._perceive()
        # seed prev_step from first perception so initial obs isn't biased to 0
        self._n_visible_prev_step = len(self._candidates)
        obs = self._build_obs()
        info = {
            "n_items_remaining": self._n_items_remaining(),
            "n_candidates": len(self._candidates),
            "n_delivered": self._n_delivered,
            **self._info_episode_aggregates(),
        }
        return obs.astype(np.float32), info

    def step(self, action):
        action = int(action)
        self._step_count += 1
        # fresh per-step L4 cache so K candidates share work cleanly
        self._clear_l4_cache()

        # pre-step visible-candidate count for the L4 lookahead / cascade flags
        n_visible_prev = int(len(self._candidates))

        # v5 joint action decode. m = 13 is the zero-offset cell (selection-only)
        cand_idx, dx, dy, dyaw = _decode_action_v5(action)
        # invalidity decided over the slot, all 27 refinements share validity
        invalid_action = not (0 <= cand_idx < len(self._candidates))
        success = False
        delivered_one = False
        item_present = False
        picked_item = None
        predicate = {}
        outcome = None
        cand_used = None
        grasp_pos = None
        grasp_quat = None

        if not invalid_action:
            cand_used = self._candidates[cand_idx]
            grasp_pos = self._wrist_target_from_candidate(cand_used)
            # apply XY refinement BEFORE any snap so policy can move when
            # use_snap_xy=False (v5 default)
            grasp_pos = grasp_pos + np.array([float(dx), float(dy), 0.0],
                                             dtype=float)
            # carry yaw refinement into the raw orientation branch
            _raw_quat_refined = _quat_mul_wxyz(
                _yaw_quat_wxyz(float(dyaw)),
                np.asarray(cand_used["world_quat"], dtype=float))
            # pre-v5 unconditionally snapped to GT (project doc Sec D primitive
            # support). v5 gates by use_snap_xy / use_snap_z, XY off, Z on by
            # default so the policy owns horizontal pose end-to-end.
            assoc_name = self._associate_item(grasp_pos)
            if assoc_name is not None:
                item_present = True
                item_pos = np.asarray(
                    self.env.get_object_positions()[assoc_name], dtype=float)
                item_top = float(item_pos[2]) + _ITEM_HALF_HEIGHT
                if self.use_snap_xy:
                    xy = np.array([float(item_pos[0]), float(item_pos[1])],
                                  dtype=float)
                else:
                    xy = np.array(grasp_pos[:2], dtype=float)
                if self.use_snap_z:
                    z = item_top - _GRASP_DESCENT_OFFSET + _FINGER_TO_WRIST
                else:
                    z = float(grasp_pos[2])
                grasp_pos = np.array([xy[0], xy[1], z], dtype=float)
                # yaw snap tied to use_snap_xy so the horizontal-cheat policy is
                # coherent end-to-end
                if self.use_snap_xy and self.orientation_source == "snap":
                    grasp_quat = self._grasp_quat_for_item(assoc_name)
                else:
                    grasp_quat = _raw_quat_refined
            else:
                grasp_quat = _raw_quat_refined
            # always-computed diagnostic predicate. drives reward only in "geometric"
            predicate = evaluate_grasp(self.env, grasp_pos, grasp_quat, assoc_name)

            if self.reward_mode in ("hybrid", "hybrid_physics", "physics"):
                if item_present:
                    try:
                        if self.reward_mode == "hybrid":
                            outcome = self._hybrid_geometric_outcome(
                                grasp_pos, grasp_quat, assoc_name, predicate)
                        elif self.reward_mode == "hybrid_physics":
                            outcome = self.pick_place.attempt_grasp_hybrid(
                                grasp_pos, grasp_quat, assoc_name)
                        else:  # "physics"
                            outcome = self.pick_place.attempt_grasp_physical(
                                grasp_pos, grasp_quat, assoc_name)
                    except Exception as e:
                        print(f"[BinClearingGymEnv] grasp attempt "
                              f"({self.reward_mode}) failed: {e}")
                        outcome = None
                    # snap back to sensing pose so the next perception render is clean
                    self.sensing_ctrl.set_sensing_pose_direct()
                    self.env._forward()
                    if outcome is not None and outcome.get("grasp_ok", False):
                        picked_item = assoc_name
                        self._deliver_item(picked_item)
                        self._delivered_names.append(picked_item)
                        self._n_delivered += 1
                        delivered_one = True
                # else: empty grab, no attempt
            else:  # reward_mode == "geometric"
                geo_ok = bool(predicate.get("success", False))
                picked_item = predicate.get("picked_item")
                if geo_ok and picked_item is not None:
                    if self.slow_physics:
                        try:
                            self.pick_place.execute(grasp_pos, grasp_quat,
                                                    target_item_name=picked_item,
                                                    max_steps_per_phase=400)
                        except Exception as e:
                            print(f"[BinClearingGymEnv] slow_physics execute failed: {e}")
                        self.sensing_ctrl.set_sensing_pose_direct()
                        self.env._forward()
                    self._deliver_item(picked_item)
                    self._delivered_names.append(picked_item)
                    self._n_delivered += 1
                    delivered_one = True

        success = bool(delivered_one)

        # failure mask: record source_body_id of just-failed picks. Breaks L2
        # fixation across all policies. Records on real attempts that didn't
        # deliver OR valid-slot empty-grabs with an associated body.
        if (not invalid_action) and (not delivered_one)\
                and self.failure_mask_size > 0:
            try:
                failed_bid = int(cand_used["source_body_id"])\
                    if cand_used is not None else -1
            except Exception:
                failed_bid = -1
            if failed_bid >= 0:
                self._recent_failed_bids.append(failed_bid)

        # disjoint categorisation: invalid / empty_grab / physics_attempt
        if invalid_action:
            self._ep_n_invalid += 1
        elif not item_present:
            self._ep_n_empty_grab += 1
        else:
            self._ep_n_physics_attempts += 1
            if outcome is not None:
                self._ep_sum_grasp_quality += float(outcome.get("grasp_quality", 0.0))
                self._ep_sum_disturb_m    += float(outcome.get("neighbour_disturbance_m", 0.0))
                self._ep_sum_disturb_raw_m += float(outcome.get("neighbour_disturbance_raw_m", 0.0))
                self._ep_max_disturb_m = max(self._ep_max_disturb_m,
                                              float(outcome.get("neighbour_disturbance_max_m", 0.0)))
                self._ep_n_ejected        += int(outcome.get("items_ejected", 0))
        if predicate.get("success", False):
            self._ep_n_predicate_succ += 1

        # v5 L4 recent-clearing ring
        self._delivered_last5.append(1 if delivered_one else 0)

        self._re_pin_delivered()
        self._perceive()
        # n_visible_prev_step for the NEXT obs. obs builder normalises /10
        self._n_visible_prev_step = len(self._candidates)

        n_visible_next = int(len(self._candidates))
        # Lookahead: max(0, n_next - (n_prev - 1)) / 5, clipped to [0, 1].
        # The (-1) accounts for the pick. Rewards picks that uncover MORE
        # than one cube. 5 newly-exposed saturates. Reward gates on delivery.
        _delta_lookahead = n_visible_next - (n_visible_prev - 1)
        if _delta_lookahead < 0:
            _delta_lookahead = 0
        lookahead_bonus_norm = float(_delta_lookahead) / 5.0
        if lookahead_bonus_norm > 1.0:
            lookahead_bonus_norm = 1.0
        # Keystone collapse: pile fell apart (>3 new visible) AND >=1 ejected.
        # Conjunction filters benign settling-into-view.
        _items_ejected = int((outcome or {}).get("items_ejected", 0))
        cascade_collapse_flag = bool(
            ((n_visible_next - n_visible_prev) > 3)
            and (_items_ejected > 0)
            and (not invalid_action))

        # post-grasp descent-cylinder clearance evaluated on the executed pose.
        # Zero for invalid / no grasp.
        if invalid_action or grasp_pos is None:
            approach_clear_pred_val = 0.0
        else:
            approach_clear_pred_val = self._approach_clear_pred(
                grasp_pos, source_items=None, gripper_radius=0.02,
                exclude_assoc=True)

        if self.reward_mode in ("hybrid", "hybrid_physics", "physics"):
            reward = compute_reward(
                invalid_action=invalid_action,
                item_present=item_present,
                delivered=delivered_one,
                outcome=outcome,
                approach_clear_pred=approach_clear_pred_val,
                lookahead_bonus_norm=lookahead_bonus_norm,
                cascade_collapse_flag=cascade_collapse_flag,
                **self.reward_weights)
        else:
            reward = compute_reward_geometric(
                success=bool(predicate.get("success", False)) and delivered_one,
                invalid_action=invalid_action,
                delivered_one=delivered_one)
        self._ep_sum_reward += float(reward)

        # signed per-component breakdown for diagnostics. entries sum to `reward`
        # in physics modes (except invalid-only path).
        _rw = self.reward_weights
        _o = outcome or {}
        if invalid_action:
            r_components = {
                "step_penalty": float(_rw.get("step_penalty", 0.0)),
                "invalid_penalty": float(_rw.get("invalid_penalty", 0.0)),
                "grasp_quality": 0.0,
                "deliver": 0.0,
                "empty_grab": 0.0,
                "disturb": 0.0,
                "eject": 0.0,
                "approach_clear_pred": 0.0,
                "lookahead_bonus": 0.0,
                "keystone_penalty": 0.0,
            }
        else:
            _disturb = float(_o.get("neighbour_disturbance_m", 0.0))
            _ejected = int(_o.get("items_ejected", 0))
            _gq = float(_o.get("grasp_quality", 0.0))
            r_components = {
                "step_penalty": float(_rw.get("step_penalty", 0.0)),
                "invalid_penalty": 0.0,
                "grasp_quality": (float(_rw.get("grasp_quality_coef", 0.0))
                                  * _gq if item_present else 0.0),
                "deliver": (float(_rw.get("deliver_reward", 0.0))
                            if delivered_one else 0.0),
                "empty_grab": (float(_rw.get("empty_grab_penalty", 0.0))
                               if not item_present else 0.0),
                "disturb": -float(_rw.get("disturb_coef", 0.0)) * _disturb,
                "eject": -float(_rw.get("eject_penalty", 0.0)) * _ejected,
                "approach_clear_pred": (
                    float(_rw.get("approach_clear_pred_coef", 0.0))
                    * float(approach_clear_pred_val)),
                "lookahead_bonus": (
                    float(_rw.get("lookahead_bonus_coef", 0.0))
                    * float(lookahead_bonus_norm)
                    if delivered_one else 0.0),
                "keystone_penalty": (
                    -float(_rw.get("keystone_penalty_coef", 0.0))
                    if cascade_collapse_flag else 0.0),
            }

        n_remaining = self._n_items_remaining()
        terminated = bool(n_remaining == 0)
        truncated = bool(self._step_count >= self.max_steps)

        self._prev_success = success
        obs = self._build_obs()

        info = {
            "action": action,
            "action_cand_idx": int(cand_idx),
            "action_dx": float(dx),
            "action_dy": float(dy),
            "action_dyaw": float(dyaw),
            "approach_clear_pred": float(approach_clear_pred_val),
            "lookahead_bonus_norm": float(lookahead_bonus_norm),
            "cascade_collapse_flag": bool(cascade_collapse_flag),
            "n_visible_prev": int(n_visible_prev),
            "n_visible_next": int(n_visible_next),
            "r_components": r_components,
            "invalid_action": invalid_action,
            "item_present": item_present,
            "success": success,
            "delivered_one": delivered_one,
            "picked_item": picked_item,
            "predicate": predicate,
            "physics_outcome": outcome,
            "neighbour_disturbance_raw_m": float(outcome.get("neighbour_disturbance_raw_m", 0.0)) if outcome else 0.0,
            "neighbour_disturbance_max_m": float(outcome.get("neighbour_disturbance_max_m", 0.0)) if outcome else 0.0,
            "n_items_remaining": n_remaining,
            "n_delivered": self._n_delivered,
            "n_candidates": len(self._candidates),
            "step_count": self._step_count,
            "grasp_pos": None if grasp_pos is None else grasp_pos.tolist(),
            "terminated": terminated,
            "reward": float(reward),
            **self._info_episode_aggregates(),
        }
        if self.render_mode == "rgb_array":
            try:
                info["frame"] = self.env.get_wrist_pov_rgb()
            except Exception:
                pass
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            try:
                return self.env.get_wrist_pov_rgb()
            except Exception:
                return None
        return None

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

    # Action masking (sb3_contrib.MaskablePPO)

    def action_masks(self) -> np.ndarray:
        """Boolean mask over the v5 action space, shape (K * N_REFINEMENT,).

        Slot k is valid iff k < len(self._candidates) AND its source_body_id
        is not in the recent-failures buffer. Validity broadcasts over all 27
        refinement cells. If the failure mask would drop every valid slot we
        fall back to validity alone (forced retry over an invalid no-op).
        """
        K_slots = self.K
        N = K_slots * N_REFINEMENT
        n_valid = len(self._candidates)

        if n_valid == 0:
            # unmask the zero-refinement cell of slot 0 (action_id = 13) so the
            # env can still advance one step (registers invalid, ends episode)
            mask = np.zeros(N, dtype=bool)
            mask[N_REFINEMENT // 2] = True
            return mask

        slot_mask = np.array([i < n_valid for i in range(K_slots)], dtype=bool)
        if self.failure_mask_size > 0 and len(self._recent_failed_bids) > 0:
            recent = set(int(b) for b in self._recent_failed_bids)
            failure_slot_mask = np.ones(K_slots, dtype=bool)
            n_masked = 0
            for i in range(n_valid):
                try:
                    bid = int(self._candidates[i].get("source_body_id", -1))
                except Exception:
                    bid = -1
                if bid >= 0 and bid in recent:
                    failure_slot_mask[i] = False
                    n_masked += 1
            combined = slot_mask & failure_slot_mask
            if not combined.any():
                # failure mask would drop everything -> fall back to validity
                pass
            else:
                slot_mask = combined
                if n_masked > 0:
                    try:
                        self._ep_n_failure_masked += 1
                    except AttributeError:
                        pass

        # broadcast over 27 refinement cells. layout matches _decode_action_v5
        mask_2d = np.broadcast_to(slot_mask[:, None],
                                  (K_slots, N_REFINEMENT)).copy()
        return mask_2d.reshape(N)
