"""
4-policy comparison driver: random / topdown / greedy_ggcnn / rl on paired seeds.

Builds BinClearingGymEnv twice (candidate_source='ggcnn' for greedy_ggcnn,
candidate_source='ppo' for random/topdown/rl), runs paired-seed episodes, and
writes per_attempt.csv + per_episode.csv. Outer loop: n_objects, then seed,
INNER: policy (so a given seed is replayed by every policy back-to-back).

Action selection (all policies consult env.action_masks()):
    random        : uniform over unmasked indices, per-episode RNG
    topdown       : argmax world_pos[2] among unmasked
    greedy_ggcnn  : argmax quality among unmasked, tie-break by source_body_id asc
    rl            : MaskablePPO.predict(obs, action_masks=mask, deterministic=True)

Usage: see README.md for the exact command and flags.
"""
from __future__ import annotations

# CSV schema version 2: adds neighbour_disturbance_raw_m and neighbour_disturbance_max_m
import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl.bin_clearing_env import BinClearingGymEnv  # noqa: E402

# Default RL model path resolved from the submission root.
_DEFAULT_RL_MODEL_PATH = os.path.join(_ROOT, "rl", "models", "best_model.zip")

ALL_POLICIES = ["greedy_ggcnn", "random", "topdown", "heuristic_augmented", "rl"]

# Joint action_id = k * N_REFINEMENTS + m. m encodes a 3x3x3 (dx, dy, dyaw) grid:
# dx, dy in {-0.015, 0.0, +0.015} m
# dyaw in {-10 deg, 0, +10 deg}
# m = i*9 + j*3 + l, zero-offset cell -> m = 13.
N_REFINEMENTS = 27
ZERO_REFINEMENT = 13
DX_BINS = (-0.015, 0.0, 0.015)
DY_BINS = (-0.015, 0.0, 0.015)
DYAW_BINS = (-np.deg2rad(10.0), 0.0, np.deg2rad(10.0))


def _encode_action(k: int, m: int = ZERO_REFINEMENT) -> int:
    return int(k) * N_REFINEMENTS + int(m)


def _decode_action(action_id: int) -> Tuple[int, int]:
    return int(action_id) // N_REFINEMENTS, int(action_id) % N_REFINEMENTS


def _encode_dyaw(dyaw_target: float) -> int:
    diffs = [abs(float(dyaw_target) - b) for b in DYAW_BINS]
    return int(np.argmin(diffs))


# CSV column orders are the spec. do not reorder.

PER_ATTEMPT_COLUMNS = [
    "policy", "n_objects", "seed", "episode_idx", "step_idx",
    "action",
    # chosen_candidate_idx is the slot k = action // 27 (distinct from raw action_id).
    "chosen_candidate_idx", "chosen_dx", "chosen_dy", "chosen_dyaw",
    "n_candidates_valid", "chosen_source_body_id", "chosen_quality",
    "chosen_world_x", "chosen_world_y", "chosen_world_z",
    "invalid_action", "item_present", "predicate_success",
    "jaws_aligned", "item_between_jaws", "mid_height", "approach_clear",
    "delivered_one", "picked_item",
    "attempt_masked", "grasp_attempted",
    "neighbour_disturbance_m",
    "neighbour_disturbance_raw_m",
    "neighbour_disturbance_max_m",
    "items_ejected",
    "grasp_quality",
    "reward", "n_items_remaining", "n_delivered",
]

PER_EPISODE_COLUMNS = [
    "policy", "n_objects", "seed", "episode_idx",
    "cleared", "n_delivered",
    "n_steps",
    # steps_to_clear: n_steps if cleared else -1 (unambiguous).
    "steps_to_clear",
    "total_reward",
    "sum_disturbance_m",
    "sum_disturb_raw_m",
    "max_disturb_m",
    "mean_disturbance_per_attempt",
    "n_invalid", "n_empty_grab", "n_physics_attempts", "n_predicate_succ",
    "n_failure_masked", "n_ejected",
    "wall_time_s",
]


def _env_kwargs_for_policy(policy: str) -> Dict:
    """greedy_ggcnn consumes the GG-CNN proposal pipeline. the other three
    share the PPO oracle pipeline."""
    if policy == "greedy_ggcnn":
        return dict(
            candidate_source="ggcnn",
            filter_candidates=True,
            orientation_source="snap",
        )
    if policy in ("random", "topdown", "heuristic_augmented", "rl"):
        return dict(
            candidate_source="ppo",
            ppo_visibility_mode="raycast",
            ppo_quality_mode="analytical",
            orientation_source="snap",
            filter_candidates=True,
        )
    raise ValueError("unknown policy: {!r}".format(policy))


def _env_config_key(policy: str) -> Tuple:
    """Two policies share an env iff their kwargs match."""
    kw = _env_kwargs_for_policy(policy)
    return tuple(sorted(kw.items()))


def build_env(policy: str, n_objects: int, K: int, reward_mode: str,
              layout_jitter: float, failure_mask: int,
              ggcnn_device: Optional[str]) -> BinClearingGymEnv:
    kw = _env_kwargs_for_policy(policy)
    return BinClearingGymEnv(
        n_objects=int(n_objects),
        K=int(K),
        reward_mode=str(reward_mode),
        layout_jitter=float(layout_jitter),
        failure_mask_size=int(failure_mask),
        ggcnn_device=ggcnn_device,
        **kw,
    )


def _unmasked_joint_indices(env: BinClearingGymEnv) -> np.ndarray:
    """Indices into the full Discrete(K * 27) joint space that survive the
    validity + failure mask."""
    n_total = int(env.action_space.n)
    try:
        mask = np.asarray(env.action_masks(), dtype=bool)
        if mask.size != n_total:
            # Defensive: env may still return a K-element mask. broadcast.
            if mask.size * N_REFINEMENTS == n_total:
                mask = np.repeat(mask, N_REFINEMENTS)
            else:
                raise ValueError("mask size {} != action_space.n {}".format(
                    mask.size, n_total))
    except Exception:
        n_valid = len(env._candidates)
        mask = np.zeros(n_total, dtype=bool)
        if n_valid > 0:
            for k in range(min(n_valid, n_total // N_REFINEMENTS)):
                mask[k * N_REFINEMENTS:(k + 1) * N_REFINEMENTS] = True
    n_valid = len(env._candidates)
    if n_valid == 0:
        # No candidates: emit slot 0 (zero refinement) so env registers an
        # invalid-action no-op and the episode advances.
        return np.asarray([0], dtype=int)
    return np.nonzero(mask)[0].astype(int)


def _unmasked_candidate_indices(env: BinClearingGymEnv) -> np.ndarray:
    """Candidate slots (in [0, K)) that have >= 1 valid joint action."""
    joint = _unmasked_joint_indices(env)
    if joint.size == 0:
        return np.asarray([0], dtype=int)
    cand_ids = np.unique(joint // N_REFINEMENTS)
    n_valid = len(env._candidates)
    if n_valid == 0:
        return np.asarray([0], dtype=int)
    cand_ids = cand_ids[cand_ids < n_valid]
    if cand_ids.size == 0:
        return np.asarray([0], dtype=int)
    return cand_ids.astype(int)


def pick_action_random(env: BinClearingGymEnv,
                       rng: np.random.Generator) -> int:
    """Uniform over the masked joint Discrete(K * 27) action space."""
    valid = _unmasked_joint_indices(env)
    if len(valid) == 0:
        return 0
    return int(rng.choice(valid))


def pick_action_topdown(env: BinClearingGymEnv) -> int:
    """Pick the highest-z candidate among unmasked indices. tie-break by lower source_body_id."""
    valid_k = _unmasked_candidate_indices(env)
    cands = env._candidates
    if len(valid_k) == 0 or len(cands) == 0:
        return _encode_action(0, ZERO_REFINEMENT)
    best_k = int(valid_k[0])
    best_z = -np.inf
    best_bid = np.inf
    for i in valid_k:
        i = int(i)
        if not (0 <= i < len(cands)):
            continue
        cz = float(np.asarray(cands[i]["world_pos"])[2])
        bid = int(cands[i].get("source_body_id", -1))
        if (cz > best_z) or (cz == best_z and bid < best_bid):
            best_z = cz
            best_bid = bid
            best_k = i
    return _encode_action(best_k, ZERO_REFINEMENT)


def pick_action_greedy_ggcnn(env: BinClearingGymEnv) -> int:
    """argmax(quality) among unmasked, tie-break source_body_id asc."""
    valid_k = _unmasked_candidate_indices(env)
    cands = env._candidates
    if len(valid_k) == 0 or len(cands) == 0:
        return _encode_action(0, ZERO_REFINEMENT)
    best_k = int(valid_k[0])
    best_q = -np.inf
    best_bid = np.inf
    for i in valid_k:
        i = int(i)
        if not (0 <= i < len(cands)):
            continue
        q = float(cands[i].get("quality", 0.0))
        bid = int(cands[i].get("source_body_id", -1))
        if (q > best_q) or (q == best_q and bid < best_bid):
            best_q = q
            best_bid = bid
            best_k = i
    return _encode_action(best_k, ZERO_REFINEMENT)


# L4-aware heuristic_augmented per-candidate obs feature offsets.
# Per-candidate slot layout (observation_builder):
# idx 1 : quality
# idx 13 : approach_clear_pred (vertical descent clearance)
# idx 15-18 : approach_clear_d{x,y}_{plus,minus} probes
# idx 20 : exposure_value (L4: # cubes this pick would expose, normalized)
# idx 21 : is_keystone (L4: # cubes resting on this one, normalized)
# Slot offsets are read defensively: if obs lacks L4 slots, fall back to pre-L4.
_HA_FEAT_QUALITY        = 1
_HA_FEAT_APPROACH_CLEAR = 13
_HA_FEAT_DX_PLUS        = 15
_HA_FEAT_DX_MINUS       = 16
_HA_FEAT_DY_PLUS        = 17
_HA_FEAT_DY_MINUS       = 18
_HA_FEAT_EXPOSURE       = 20
_HA_FEAT_KEYSTONE       = 21

_HA_KEYSTONE_THRESHOLD  = 0.66   # > 0.66 -> 2+ cubes resting on it
_HA_EXPOSURE_THRESHOLD  = 0.2    # > 0.2 -> would expose >= 1 cube

# Refinement m = i*9 + j*3 + l with (i, j, l) in {0,1,2}^3.
# dx=+0.015 -> i=2, dx=-0.015 -> i=0. dy=+0.015 -> j=2, dy=-0.015 -> j=0.
def _ha_m_for_probe(direction: str, l: int) -> int:
    if direction == "+x":
        i, j = 2, 1
    elif direction == "-x":
        i, j = 0, 1
    elif direction == "+y":
        i, j = 1, 2
    elif direction == "-y":
        i, j = 1, 0
    else:
        i, j = 1, 1
    return int(i) * 9 + int(j) * 3 + int(l)


# Trailing global features reserved by the heuristic slot-bounds guard below.
# The live observation_builder.py uses 9 globals.
_GLOBAL_FEATS_HA = 6


def _ha_per_cand_slot(obs: Optional[np.ndarray], k: int, feat_idx: int,
                       feat_per_cand: int = 19, n_feats_total: int = 22) -> Optional[float]:
    """Read per-candidate feature `feat_idx` for slot `k` from obs.

    Probes the extended (L4, 22-feat) layout first. falls back to the legacy
    19-feat layout. Returns None if the requested feature is unavailable in
    either layout (signals "feature not in obs, fall back to pre-L4 ranking")."""
    if obs is None:
        return None
    arr = np.asarray(obs).ravel()
    base_ext = int(k) * int(n_feats_total) + int(feat_idx)
    if 0 <= feat_idx < n_feats_total and base_ext < arr.size - _GLOBAL_FEATS_HA:
        if arr.size >= (int(k) + 1) * int(n_feats_total) + _GLOBAL_FEATS_HA:
            return float(arr[base_ext])
    if feat_idx < feat_per_cand:
        base_leg = int(k) * int(feat_per_cand) + int(feat_idx)
        if arr.size >= (int(k) + 1) * int(feat_per_cand) + _GLOBAL_FEATS_HA:
            return float(arr[base_leg])
    return None


def pick_action_heuristic_augmented(env: BinClearingGymEnv,
                                     obs: Optional[np.ndarray] = None) -> int:
    """L4-aware heuristic: drop keystone -> prefer exposure -> prefer
    descent-clear -> argmax quality. Refines XY via directional probes when
    vertical descent is blocked, and dyaw via predicted_angle - cand.angle.

    Gracefully reduces to the pre-L4 heuristic when obs lacks exposure/keystone."""
    valid_k = _unmasked_candidate_indices(env)
    cands = env._candidates
    if len(valid_k) == 0 or len(cands) == 0:
        return _encode_action(0, ZERO_REFINEMENT)

    valid_k_list: List[int] = [int(i) for i in valid_k if 0 <= int(i) < len(cands)]
    if not valid_k_list:
        return _encode_action(0, ZERO_REFINEMENT)

    # L4 filter 1: drop keystones (keep all if they're all keystones).
    non_keystone: List[int] = []
    for i in valid_k_list:
        ks = _ha_per_cand_slot(obs, i, _HA_FEAT_KEYSTONE)
        if ks is None or ks <= _HA_KEYSTONE_THRESHOLD:
            non_keystone.append(i)
    pool = non_keystone if non_keystone else valid_k_list

    # L4 filter 2: prefer high-exposure picks.
    high_exposure = [i for i in pool
                     if (_ha_per_cand_slot(obs, i, _HA_FEAT_EXPOSURE) or 0.0)
                        > _HA_EXPOSURE_THRESHOLD]

    best_k: int
    if high_exposure:
        best_k = int(high_exposure[0])
        best_exp = -np.inf
        best_bid = np.inf
        for i in high_exposure:
            exp = float(_ha_per_cand_slot(obs, i, _HA_FEAT_EXPOSURE) or 0.0)
            bid = int(cands[i].get("source_body_id", -1))
            if (exp > best_exp) or (exp == best_exp and bid < best_bid):
                best_exp = exp
                best_bid = bid
                best_k = i
    else:
        # L4 filter 3: prefer descent-clear, then argmax quality.
        clear_pool = [i for i in pool
                      if (_ha_per_cand_slot(obs, i, _HA_FEAT_APPROACH_CLEAR) or 0.0)
                         >= 0.5]
        ranking_pool = clear_pool if clear_pool else pool

        best_k = int(ranking_pool[0])
        best_q = -np.inf
        best_bid = np.inf
        for i in ranking_pool:
            q = float(cands[i].get("quality", 0.0))
            bid = int(cands[i].get("source_body_id", -1))
            if (q > best_q) or (q == best_q and bid < best_bid):
                best_q = q
                best_bid = bid
                best_k = i

    cand = cands[best_k]

    cand_angle = float(cand.get("angle_rad", 0.0))
    pred_angle = float(cand.get("predicted_angle", cand_angle))
    dyaw_target = pred_angle - cand_angle
    dyaw_target = float(np.arctan2(np.sin(dyaw_target), np.cos(dyaw_target)))
    l = _encode_dyaw(dyaw_target)

    # XY refinement only fires when vertical descent is blocked.
    approach_clear = _ha_per_cand_slot(obs, best_k, _HA_FEAT_APPROACH_CLEAR)
    if approach_clear is not None and approach_clear < 0.5:
        probe_order = (
            ("+x", _HA_FEAT_DX_PLUS),
            ("-x", _HA_FEAT_DX_MINUS),
            ("+y", _HA_FEAT_DY_PLUS),
            ("-y", _HA_FEAT_DY_MINUS),
        )
        chosen_dir = None
        for direction, fidx in probe_order:
            v = _ha_per_cand_slot(obs, best_k, fidx)
            if v is not None and v >= 0.5:
                chosen_dir = direction
                break
        if chosen_dir is not None:
            m = _ha_m_for_probe(chosen_dir, l)
        else:
            m = _ha_m_for_probe("zero", l)
    else:
        m = _ha_m_for_probe("zero", l)

    return _encode_action(best_k, m)


def pick_action_rl(env: BinClearingGymEnv, obs: np.ndarray, model) -> int:
    """MaskablePPO predict with the env's current Discrete(K * 27) mask."""
    try:
        mask = np.asarray(env.action_masks(), dtype=bool)
    except Exception:
        mask = None
    if mask is not None:
        a, _ = model.predict(obs, action_masks=mask, deterministic=True)
    else:
        a, _ = model.predict(obs, deterministic=True)
    return int(a)


def select_action(policy: str, env: BinClearingGymEnv, obs: np.ndarray,
                  rl_model, rng: np.random.Generator) -> int:
    if policy == "random":
        return pick_action_random(env, rng)
    if policy == "topdown":
        return pick_action_topdown(env)
    if policy == "greedy_ggcnn":
        return pick_action_greedy_ggcnn(env)
    if policy == "heuristic_augmented":
        return pick_action_heuristic_augmented(env, obs)
    if policy == "rl":
        if rl_model is None:
            raise RuntimeError("policy='rl' requires --rl_model_path")
        return pick_action_rl(env, obs, rl_model)
    raise ValueError("unknown policy: {!r}".format(policy))


def _decode_refinement_bins(m: int) -> Tuple[float, float, float]:
    m = int(m)
    if not (0 <= m < N_REFINEMENTS):
        return (float("nan"), float("nan"), float("nan"))
    i = m // 9
    j = (m // 3) % 3
    l = m % 3
    return (float(DX_BINS[i]), float(DY_BINS[j]), float(DYAW_BINS[l]))


def _chosen_candidate_fields(env: BinClearingGymEnv, action: int) -> Dict:
    """Pull the chosen candidate's fields + decoded refinement bins from a
    joint Discrete(K * 27) action_id. Returns NaN/-1 for invalid actions."""
    cands = env._candidates
    k, m = _decode_action(action)
    dx, dy, dyaw = _decode_refinement_bins(m)
    if not (0 <= k < len(cands)):
        return dict(
            chosen_candidate_idx=int(k),
            chosen_dx=dx,
            chosen_dy=dy,
            chosen_dyaw=dyaw,
            chosen_source_body_id=-1,
            chosen_quality=float("nan"),
            chosen_world_x=float("nan"),
            chosen_world_y=float("nan"),
            chosen_world_z=float("nan"),
        )
    c = cands[k]
    wp = np.asarray(c.get("world_pos", [float("nan")] * 3), dtype=float)
    return dict(
        chosen_candidate_idx=int(k),
        chosen_dx=dx,
        chosen_dy=dy,
        chosen_dyaw=dyaw,
        chosen_source_body_id=int(c.get("source_body_id", -1)),
        chosen_quality=float(c.get("quality", float("nan"))),
        chosen_world_x=float(wp[0]) if wp.size >= 1 else float("nan"),
        chosen_world_y=float(wp[1]) if wp.size >= 2 else float("nan"),
        chosen_world_z=float(wp[2]) if wp.size >= 3 else float("nan"),
    )


def _build_attempt_row(policy: str, n_objects: int, seed: int,
                       episode_idx: int, step_idx: int,
                       action: int, n_valid_at_pick: int,
                       cand_fields: Dict, info: Dict, reward: float,
                       attempt_masked: bool = False) -> Dict:
    predicate = info.get("predicate") or {}
    physics_outcome = info.get("physics_outcome") or {}

    def _b(v) -> int:
        return int(bool(v))

    item_present = bool(info.get("item_present", False))
    invalid_action = bool(info.get("invalid_action", False))
    grasp_attempted = bool(item_present and not invalid_action)

    return {
        "policy":                  policy,
        "n_objects":               int(n_objects),
        "seed":                    int(seed),
        "episode_idx":             int(episode_idx),
        "step_idx":                int(step_idx),
        "action":                  int(action),
        "chosen_candidate_idx":    int(cand_fields["chosen_candidate_idx"]),
        "chosen_dx":               float(cand_fields["chosen_dx"]),
        "chosen_dy":               float(cand_fields["chosen_dy"]),
        "chosen_dyaw":             float(cand_fields["chosen_dyaw"]),
        "n_candidates_valid":      int(n_valid_at_pick),
        "chosen_source_body_id":   int(cand_fields["chosen_source_body_id"]),
        "chosen_quality":          float(cand_fields["chosen_quality"]),
        "chosen_world_x":          float(cand_fields["chosen_world_x"]),
        "chosen_world_y":          float(cand_fields["chosen_world_y"]),
        "chosen_world_z":          float(cand_fields["chosen_world_z"]),
        "invalid_action":          _b(invalid_action),
        "item_present":            _b(item_present),
        "predicate_success":       _b(predicate.get("success", False)),
        "jaws_aligned":            _b(predicate.get("jaws_aligned", False)),
        "item_between_jaws":       _b(predicate.get("item_between_jaws", False)),
        "mid_height":              _b(predicate.get("mid_height", False)),
        "approach_clear":          _b(predicate.get("approach_clear", False)),
        "delivered_one":           _b(info.get("delivered_one", False)),
        "picked_item":             "" if info.get("picked_item") is None else str(info.get("picked_item")),
        "attempt_masked":          _b(attempt_masked),
        "grasp_attempted":         _b(grasp_attempted),
        "neighbour_disturbance_m": float(physics_outcome.get("neighbour_disturbance_m", float("nan"))),
        "neighbour_disturbance_raw_m": float(info.get("neighbour_disturbance_raw_m", 0.0)),
        "neighbour_disturbance_max_m": float(info.get("neighbour_disturbance_max_m", 0.0)),
        "items_ejected":           int(physics_outcome.get("items_ejected", 0)),
        "grasp_quality":           float(physics_outcome.get("grasp_quality", float("nan"))),
        "reward":                  float(reward),
        "n_items_remaining":       int(info.get("n_items_remaining", -1)),
        "n_delivered":             int(info.get("n_delivered", 0)),
    }


def run_episode(env: BinClearingGymEnv, policy: str, rl_model,
                seed: int, n_objects: int, episode_idx: int,
                attempt_writer: csv.DictWriter, attempt_fh,
                verbose: bool) -> Dict:
    """One episode of `policy` on `env`. Writes per-attempt rows live."""
    # Per-episode RNG deterministically seeded so 'random' is reproducible.
    ep_rng = np.random.default_rng(int(seed) * 31 + int(episode_idx))

    t_ep0 = time.time()
    obs, info = env.reset(seed=int(seed))
    n_initial = int(info.get("n_items_remaining", n_objects))

    done = False
    step_idx = 0
    total_reward = 0.0
    n_delivered = 0
    last_info = info

    while not done:
        step_idx += 1
        # Count distinct candidates, NOT joint actions, so the column keeps
        # its v4 meaning (joint mask has up to 27 entries per candidate).
        n_valid_at_pick = int(len(_unmasked_candidate_indices(env)))
        action = int(select_action(policy, env, obs, rl_model, ep_rng))
        cand_fields = _chosen_candidate_fields(env, action)

        # attempt_masked: chosen body_id is in the recent-failure ring buffer.
        attempt_masked = False
        try:
            recent = getattr(env, "_recent_failed_bids", None)
            if recent is not None and len(recent) > 0:
                recent_set = set(int(b) for b in recent)
                chosen_bid = int(cand_fields.get("chosen_source_body_id", -1))
                if chosen_bid >= 0 and chosen_bid in recent_set:
                    attempt_masked = True
        except Exception:
            attempt_masked = False

        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        total_reward += float(reward)
        n_delivered = int(info.get("n_delivered", n_delivered))
        last_info = info

        row = _build_attempt_row(
            policy=policy, n_objects=n_objects, seed=seed,
            episode_idx=episode_idx, step_idx=step_idx,
            action=action, n_valid_at_pick=n_valid_at_pick,
            cand_fields=cand_fields, info=info, reward=float(reward),
            attempt_masked=attempt_masked,
        )
        attempt_writer.writerow(row)

    attempt_fh.flush()
    try:
        os.fsync(attempt_fh.fileno())
    except (OSError, AttributeError):
        pass

    wall_time = time.time() - t_ep0
    n_invalid = int(last_info.get("n_invalid", 0))
    n_empty_grab = int(last_info.get("n_empty_grab", 0))
    n_physics_attempts = int(last_info.get("n_physics_attempts", 0))
    n_predicate_succ = int(last_info.get("n_predicate_succ", 0))
    n_failure_masked = int(last_info.get("n_failure_masked", 0))
    n_ejected = int(last_info.get("n_ejected", 0))
    sum_disturb = float(last_info.get("sum_disturb_m", 0.0))
    cleared = bool(int(last_info.get("n_items_remaining", -1)) == 0)
    mean_disturb_per_attempt = (
        sum_disturb / n_physics_attempts if n_physics_attempts > 0 else float("nan")
    )

    summary = {
        "policy":                       policy,
        "n_objects":                    int(n_objects),
        "seed":                         int(seed),
        "episode_idx":                  int(episode_idx),
        "cleared":                      int(cleared),
        "n_delivered":                  int(n_delivered),
        "n_steps":                      int(step_idx),
        "steps_to_clear":               int(step_idx) if cleared else -1,
        "total_reward":                 float(total_reward),
        "sum_disturbance_m":            float(sum_disturb),
        "sum_disturb_raw_m":            float(last_info.get("sum_disturb_raw_m", 0.0)),
        "max_disturb_m":                float(last_info.get("max_disturb_m", 0.0)),
        "mean_disturbance_per_attempt": float(mean_disturb_per_attempt),
        "n_invalid":                    int(n_invalid),
        "n_empty_grab":                 int(n_empty_grab),
        "n_physics_attempts":           int(n_physics_attempts),
        "n_predicate_succ":             int(n_predicate_succ),
        "n_failure_masked":             int(n_failure_masked),
        "n_ejected":                    int(n_ejected),
        "wall_time_s":                  float(wall_time),
    }

    if verbose:
        print(
            "    [{policy} n={n} seed={s} ep={e}] "
            "delivered={d}/{ni} cleared={c} steps={st} "
            "reward={r:+.2f} disturb={dist:.3f} masked={fm} wall={w:.0f}s".format(
                policy=policy, n=n_objects, s=seed, e=episode_idx,
                d=n_delivered, ni=n_initial, c=int(cleared), st=step_idx,
                r=total_reward, dist=sum_disturb, fm=n_failure_masked,
                w=wall_time,
            )
        )
    return summary


def _load_completed_cells(per_episode_path: str) -> Set[Tuple[str, int, int, int]]:
    """Read existing per_episode.csv (if any) so a resumed run can skip completed cells."""
    done: Set[Tuple[str, int, int, int]] = set()
    if not os.path.exists(per_episode_path):
        return done
    try:
        with open(per_episode_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    key = (
                        str(row["policy"]),
                        int(row["n_objects"]),
                        int(row["seed"]),
                        int(row["episode_idx"]),
                    )
                    done.add(key)
                except (KeyError, ValueError):
                    continue
    except Exception as exc:
        print("[resume] WARNING: could not parse {}: {}".format(per_episode_path, exc))
    return done


def _verify_paired_layout(envs_by_key: Dict[Tuple, BinClearingGymEnv],
                          n_objects: int, seed: int) -> None:
    """Reset every env at this seed and verify they all spawn the same number
    of items. Each policy will reset again so this does not consume state."""
    counts = {}
    for key, env in envs_by_key.items():
        try:
            env.reset(seed=int(seed))
            counts[key] = int(env._n_items_remaining())
        except Exception as exc:
            counts[key] = -1
            print("[paired-check] env reset failed: {}".format(exc))
    vals = set(counts.values())
    if len(vals) > 1:
        print("[paired-check] WARNING n_objects={} seed={}: "
              "env item counts differ at frame 0: {}".format(
                  n_objects, seed, counts))


def parse_csv_str(s: str, type_fn) -> List:
    return [type_fn(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--policies", type=str, default=",".join(ALL_POLICIES))
    ap.add_argument("--n_objects", type=str, default="10,15,20")
    ap.add_argument("--n_episodes", type=int, default=60)
    ap.add_argument("--seed_offset", type=int, default=1000)
    ap.add_argument("--rl_model_path", type=str, default=_DEFAULT_RL_MODEL_PATH,
                    help="trained MaskablePPO .zip (default: rl/models/best_model.zip)")
    ap.add_argument("--reward_mode",
                    choices=["hybrid_physics", "hybrid", "physics", "geometric"],
                    default="hybrid_physics")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--failure_mask", type=int, default=3)
    ap.add_argument("--layout_jitter", type=float, default=0.02,
                    help="default 0.02 matches the RL training distribution")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--ggcnn_device", type=str, default=None)
    args = ap.parse_args()

    policies = parse_csv_str(args.policies, str)
    bad = [p for p in policies if p not in ALL_POLICIES]
    if bad:
        ap.error("unknown policies: {}; valid: {}".format(bad, ALL_POLICIES))
    n_objects_list = parse_csv_str(args.n_objects, int)
    if not n_objects_list:
        ap.error("--n_objects must contain at least one value")
    if "rl" in policies and not args.rl_model_path:
        ap.error("--policies includes 'rl' but --rl_model_path not provided")

    os.makedirs(args.output_dir, exist_ok=True)
    per_attempt_path = os.path.join(args.output_dir, "per_attempt.csv")
    per_episode_path = os.path.join(args.output_dir, "per_episode.csv")
    config_path = os.path.join(args.output_dir, "config.json")

    config = {
        "policies":       policies,
        "n_objects":      n_objects_list,
        "n_episodes":     int(args.n_episodes),
        "seed_offset":    int(args.seed_offset),
        "rl_model_path":  args.rl_model_path,
        "reward_mode":    args.reward_mode,
        "K":              int(args.K),
        "failure_mask":   int(args.failure_mask),
        "layout_jitter":  float(args.layout_jitter),
        "output_dir":     args.output_dir,
        "resume":         bool(args.resume),
        "verbose":        bool(args.verbose),
        "ggcnn_device":   args.ggcnn_device,
        "started":        time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("=" * 72)
    print("4-POLICY COMPARISON")
    print("=" * 72)
    print("  policies      : {}".format(policies))
    print("  n_objects     : {}".format(n_objects_list))
    print("  n_episodes    : {}".format(args.n_episodes))
    print("  seed_offset   : {}".format(args.seed_offset))
    print("  reward_mode   : {}".format(args.reward_mode))
    print("  K             : {}".format(args.K))
    print("  failure_mask  : {}".format(args.failure_mask))
    print("  layout_jitter : {}".format(args.layout_jitter))
    print("  rl_model_path : {}".format(args.rl_model_path))
    print("  output_dir    : {}".format(args.output_dir))
    print("  resume        : {}".format(args.resume))
    print("=" * 72)

    completed: Set[Tuple[str, int, int, int]] = set()
    if args.resume and args.no_resume:
        ap.error("--resume and --no_resume are mutually exclusive")
    if args.resume and not args.no_resume:
        completed = _load_completed_cells(per_episode_path)
        print("[resume] loaded {} completed cells from {}".format(
            len(completed), per_episode_path))
    elif args.no_resume:
        print("[no_resume] explicit fresh start; ignoring any existing CSVs")

    rl_model = None
    if "rl" in policies:
        from sb3_contrib import MaskablePPO  # noqa: E402
        print("[rl] loading MaskablePPO from {} ...".format(args.rl_model_path))
        rl_model = MaskablePPO.load(args.rl_model_path)
        print("[rl] model loaded")

    write_attempt_header = not (os.path.exists(per_attempt_path)
                                and os.path.getsize(per_attempt_path) > 0)
    write_episode_header = not (os.path.exists(per_episode_path)
                                and os.path.getsize(per_episode_path) > 0)
    attempt_fh = open(per_attempt_path, "a", newline="")
    episode_fh = open(per_episode_path, "a", newline="")
    attempt_writer = csv.DictWriter(attempt_fh, fieldnames=PER_ATTEMPT_COLUMNS)
    episode_writer = csv.DictWriter(episode_fh, fieldnames=PER_EPISODE_COLUMNS)
    if write_attempt_header:
        attempt_writer.writeheader()
        attempt_fh.flush()
    if write_episode_header:
        episode_writer.writeheader()
        episode_fh.flush()

    coverage: Dict[Tuple[str, int], int] = {
        (p, n): 0 for p in policies for n in n_objects_list
    }

    # Envs are built lazily per n_objects (robosuite build is ~12-22 s).
    # Policies that share kwargs share the env. failure-mask buffer resets at
    # every env.reset() so cross-policy contamination is impossible.
    try:
        for n_obj in n_objects_list:
            print("\n===== n_objects = {} =====".format(n_obj))

            envs_by_key: Dict[Tuple, BinClearingGymEnv] = {}
            try:
                for p in policies:
                    key = _env_config_key(p)
                    if key in envs_by_key:
                        continue
                    print("  [build] env for key={} ...".format(dict(key)))
                    t_b0 = time.time()
                    envs_by_key[key] = build_env(
                        policy=p,
                        n_objects=n_obj,
                        K=args.K,
                        reward_mode=args.reward_mode,
                        layout_jitter=args.layout_jitter,
                        failure_mask=args.failure_mask,
                        ggcnn_device=args.ggcnn_device,
                    )
                    print("  [build] done in {:.1f}s".format(time.time() - t_b0))

                for ep in range(int(args.n_episodes)):
                    seed = int(args.seed_offset) + int(ep)

                    if ep % 10 == 0 and len(envs_by_key) > 1:
                        _verify_paired_layout(envs_by_key, n_obj, seed)

                    for policy in policies:
                        cell_key = (policy, int(n_obj), int(seed), int(ep))
                        if cell_key in completed:
                            if args.verbose:
                                print("    [skip] {} n={} seed={} ep={} "
                                      "(already in CSV)".format(
                                          policy, n_obj, seed, ep))
                            continue

                        env = envs_by_key[_env_config_key(policy)]
                        if args.verbose:
                            print("    [start] {} n={} seed={} ep={}".format(
                                policy, n_obj, seed, ep))
                        try:
                            summary = run_episode(
                                env=env, policy=policy, rl_model=rl_model,
                                seed=seed, n_objects=n_obj, episode_idx=ep,
                                attempt_writer=attempt_writer,
                                attempt_fh=attempt_fh,
                                verbose=args.verbose,
                            )
                        except Exception as exc:
                            print("    [error] {} n={} seed={} ep={}: {}".format(
                                policy, n_obj, seed, ep, exc))
                            import traceback
                            traceback.print_exc()
                            continue

                        episode_writer.writerow(summary)
                        episode_fh.flush()
                        try:
                            os.fsync(episode_fh.fileno())
                        except (OSError, AttributeError):
                            pass
                        coverage[(policy, n_obj)] += 1

                        print(
                            "  done {policy:13s} n={n:<3d} seed={s:<6d} ep={e:<4d} "
                            "deliv={d:<3d} cleared={c} steps={st:<3d} "
                            "rew={r:+7.2f} disturb={dist:.3f} masked={fm} "
                            "{w:5.1f}s".format(
                                policy=policy, n=n_obj, s=seed, e=ep,
                                d=summary["n_delivered"],
                                c=summary["cleared"],
                                st=summary["n_steps"],
                                r=summary["total_reward"],
                                dist=summary["sum_disturbance_m"],
                                fm=summary["n_failure_masked"],
                                w=summary["wall_time_s"],
                            )
                        )
            finally:
                for env in envs_by_key.values():
                    try:
                        env.close()
                    except Exception:
                        pass
    finally:
        try:
            attempt_fh.close()
        except Exception:
            pass
        try:
            episode_fh.close()
        except Exception:
            pass

    print("\n" + "=" * 72)
    print("COVERAGE (episodes completed this run)")
    print("=" * 72)
    header = "  {:<14s}".format("policy") + "".join(
        " n={:<3d}  ".format(n) for n in n_objects_list)
    print(header)
    for p in policies:
        row = "  {:<14s}".format(p)
        for n in n_objects_list:
            row += " {:<6d} ".format(coverage[(p, n)])
        print(row)
    print("=" * 72)
    print("\n[eval_4method] wrote {}".format(per_attempt_path))
    print("[eval_4method] wrote {}".format(per_episode_path))
    print("[eval_4method] config -> {}".format(config_path))


if __name__ == "__main__":
    main()
