"""Observation builder for the bin-clearing RL environment.

Flat float32 vector: ``[ K candidate slots x 22 features ] ++ [ 9 globals ]``.

Per-candidate features (slot i, zeros if no i-th candidate)::

     0: valid_flag             1.0 if real candidate
     1: q                      GG-CNN quality in [0, 1]
     2: cos(angle_rad)
     3: sin(angle_rad)
     4: width_m                gripper opening width, metres
     5: gx_rel                 world grasp x - source-bin centre x
     6: gy_rel                 world grasp y - source-bin centre y
     7: gz_rel                 world grasp z - source-bin centre z
     8: dist_to_bin_center
     9: min_neighbour_dist_xy  metres to nearest other source-bin item centre
    10: n_neighbours_within_3cm /5
    11: q_relative_rank        1 - rank/(K_valid-1)
    12: is_top_of_pile
    13: approach_clear_pred    descent-cylinder clearance (2 cm radius)
    14: n_blocking_above /5    items above target within 3 cm XY
    15: approach_clear_dx_plus   approach_clear at (gx + 0.015, gy)
    16: approach_clear_dx_minus  approach_clear at (gx - 0.015, gy)
    17: approach_clear_dy_plus   approach_clear at (gx, gy + 0.015)
    18: approach_clear_dy_minus  approach_clear at (gx, gy - 0.015)
    19: tilt                   |R[2,2]|, 1 = upright. L4
    20: exposure_value /5      newly-visible cubes if this one were removed. L4
    21: is_keystone /3         cubes resting on this one without intermediaries. L4

Features 15-18 expose the local descent-clearance gradient at the v5
refinement-grid step (1.5 cm).

Global features::

    0: n_items_remaining_norm
    1: n_valid_candidates_norm
    2: prev_success_flag
    3: step_count_norm
    4: q_spread                max(q) - min(q) over valid candidates
    5: mean_q
    6: n_visible_prev_step_norm  L4
    7: recent_clearing_rate      deliveries in last 5 steps / 5. L4
    8: pile_height_normalised    max source-cube z above bin floor / bin height. L4

Total = K * 22 + 9 = 229 for K = 10.

The L1 features use env source-bin item GT positions (the candidate filter
already uses GT, so no new leak beyond what the env does). They are a
PERCEIVED proxy for candidate isolation. computing from depth is a v2 task.
"""
import numpy as np

_FEAT_PER_CANDIDATE = 22
_GLOBAL_FEATS = 9

OBS_DIM = 10 * _FEAT_PER_CANDIDATE + _GLOBAL_FEATS  # K=10 contract

# matches REFINEMENT_STEP_XY in rl/bin_clearing_env.py
_APPROACH_PROBE_DXY = 0.015


def obs_dim_for_K(K: int) -> int:
    return K * _FEAT_PER_CANDIDATE + _GLOBAL_FEATS


def _candidate_features(c: dict, source_bin_center: np.ndarray,
                         q_rank_norm: float,
                         min_neighbour_dist_xy: float,
                         n_neighbours_3cm: int,
                         is_top_of_pile: float,
                         approach_clear_pred: float,
                         n_blocking_above: int,
                         approach_clear_dx_plus: float,
                         approach_clear_dx_minus: float,
                         approach_clear_dy_plus: float,
                         approach_clear_dy_minus: float,
                         tilt: float = 1.0,
                         exposure_value: int = 0,
                         is_keystone: int = 0) -> np.ndarray:
    q = float(c.get("quality", 0.0))
    ang = float(c.get("angle_rad", 0.0))
    width_m = float(c.get("width_m", 0.0))
    world_pos = np.asarray(c.get("world_pos", np.zeros(3)), dtype=np.float64)
    rel = world_pos - source_bin_center
    dist = float(np.linalg.norm(rel))
    # 0.20 m is past anything meaningful inside the bin (half-size 0.13 m)
    mnd_clamped = float(np.clip(min_neighbour_dist_xy, 0.0, 0.20))
    n_norm = float(np.clip(n_neighbours_3cm, 0, 5)) / 5.0
    n_blk_norm = float(np.clip(n_blocking_above, 0, 5)) / 5.0
    tilt_v = float(np.clip(tilt, 0.0, 1.0))
    expo_norm = float(np.clip(exposure_value, 0, 5)) / 5.0
    keystone_norm = float(np.clip(is_keystone, 0, 3)) / 3.0
    return np.array([
        1.0,
        q,
        float(np.cos(ang)),
        float(np.sin(ang)),
        width_m,
        float(rel[0]), float(rel[1]), float(rel[2]),
        dist,
        mnd_clamped,
        n_norm,
        float(q_rank_norm),
        float(is_top_of_pile),
        float(approach_clear_pred),
        n_blk_norm,
        float(approach_clear_dx_plus),
        float(approach_clear_dx_minus),
        float(approach_clear_dy_plus),
        float(approach_clear_dy_minus),
        tilt_v,
        expo_norm,
        keystone_norm,
    ], dtype=np.float32)


def _per_candidate_neighbour_info(candidates: list, env):
    """Per-candidate auxiliary features using env source-bin GT positions.

    Returns 9 parallel lists (one entry per candidate):
        min_neighbour_dist_xy, n_neighbours_3cm, is_top_of_pile,
        approach_clear_pred, n_blocking_above,
        and four directional clearance probes at +/- 1.5 cm.

    The "neighbour" set excludes the candidate's associated item (nearest
    within 6 cm of the grasp XY).
    """
    n = len(candidates)
    out_dist  = [float("inf")] * n
    out_count = [0] * n
    out_top   = [0.0] * n
    out_clear = [1.0] * n
    out_blk   = [0]   * n
    out_dxp   = [1.0] * n
    out_dxm   = [1.0] * n
    out_dyp   = [1.0] * n
    out_dym   = [1.0] * n
    if n == 0 or env is None:
        return (out_dist, out_count, out_top, out_clear, out_blk,
                out_dxp, out_dxm, out_dyp, out_dym)
    try:
        items = env._source_bin_items()
    except Exception:
        items = []
    if not items:
        return (out_dist, out_count, out_top, out_clear, out_blk,
                out_dxp, out_dxm, out_dyp, out_dym)

    item_xy = np.asarray([p[:2] for _, p in items], dtype=np.float64)
    item_z  = np.asarray([float(p[2]) for _, p in items], dtype=np.float64)

    # Prefer env's predicate helper so obs and reward shaping use identical geometry
    _env_clear = getattr(env, "_approach_clear_pred", None)

    def _clear_at(xy3: np.ndarray) -> float:
        if _env_clear is not None:
            try:
                return float(_env_clear(xy3, source_items=items,
                                        gripper_radius=0.02,
                                        exclude_assoc=True))
            except Exception:
                pass
        d = np.linalg.norm(item_xy - xy3[:2][None, :], axis=1)
        keep = np.ones(len(items), dtype=bool)
        if len(d) > 0 and d.min() <= 0.06:
            keep[int(d.argmin())] = False
        blockers = keep & (d <= 0.02) & (item_z <= float(xy3[2]) + 1e-3)
        return float(0.0 if blockers.any() else 1.0)

    for i, c in enumerate(candidates):
        try:
            wp = np.asarray(c["world_pos"], dtype=np.float64)
            gxy = wp[:2]
        except Exception:
            continue
        d = np.linalg.norm(item_xy - gxy[None, :], axis=1)
        # exclude the associated item (nearest within 6 cm)
        if d.min() <= 0.06:
            mask = np.ones(len(d), dtype=bool)
            assoc_idx = int(d.argmin())
            mask[assoc_idx] = False
            d_others = d[mask]
            z_others = item_z[mask]
            # is associated item highest among nearby items (within 8 cm)?
            assoc_z = float(item_z[assoc_idx])
            nearby_mask = d_others <= 0.08
            if nearby_mask.any():
                top_z = max(z_others[nearby_mask].max(), assoc_z)
                out_top[i] = float(assoc_z >= top_z - 1e-6)
            else:
                out_top[i] = 1.0
            # items stacked/overhanging above target at this XY
            out_blk[i] = int(((d_others <= 0.03) &
                              (z_others > assoc_z + 1e-3)).sum())
        else:
            d_others = d
            out_blk[i] = int((d <= 0.03).sum())
        out_dist[i]  = float(d_others.min()) if len(d_others) else float("inf")
        out_count[i] = int((d_others <= 0.03).sum()) if len(d_others) else 0

        # v5 cylinder-clearance probes at candidate XY and +/-1.5 cm shifts
        out_clear[i] = _clear_at(wp)
        wp_dxp = wp.copy(); wp_dxp[0] += _APPROACH_PROBE_DXY
        wp_dxm = wp.copy(); wp_dxm[0] -= _APPROACH_PROBE_DXY
        wp_dyp = wp.copy(); wp_dyp[1] += _APPROACH_PROBE_DXY
        wp_dym = wp.copy(); wp_dym[1] -= _APPROACH_PROBE_DXY
        out_dxp[i] = _clear_at(wp_dxp)
        out_dxm[i] = _clear_at(wp_dxm)
        out_dyp[i] = _clear_at(wp_dyp)
        out_dym[i] = _clear_at(wp_dym)
    return (out_dist, out_count, out_top, out_clear, out_blk,
            out_dxp, out_dxm, out_dyp, out_dym)


def build_observation(candidates: list,
                      env,
                      source_bin_center: np.ndarray,
                      prev_success: bool,
                      step_count: int,
                      K: int,
                      max_steps: int,
                      n_items_remaining: int = None,
                      n_objects: int = None,
                      l4_tilt: list = None,
                      l4_exposure: list = None,
                      l4_keystone: list = None,
                      n_visible_prev_step_norm: float = 0.0,
                      recent_clearing_rate: float = 0.0,
                      pile_height_normalised: float = 0.0) -> np.ndarray:
    """Build the flat (K*22 + 9,) float32 observation.

    Args:
        candidates: list of candidate dicts (quality, angle_rad, width_m, world_pos).
        env: BinClearingGymEnv (provides source-bin items). None for unit tests.
        source_bin_center: (3,) world-frame source-bin centre.
        prev_success: previous step's grasp succeeded.
        step_count: steps taken so far this episode.
        K: number of candidate slots.
        max_steps: episode step limit.
        n_items_remaining: optional override.
        n_objects: optional override.
        l4_tilt/l4_exposure/l4_keystone: per-candidate L4 features (defaults
            make missing lists behave as "upright, exposes nothing, not keystone").
        n_visible_prev_step_norm/recent_clearing_rate/pile_height_normalised:
            globals 6/7/8, pre-normalised by caller.

    Returns: np.ndarray, shape (K*22 + 9,), dtype float32.
    """
    source_bin_center = np.asarray(source_bin_center, dtype=np.float64)

    qs = np.asarray([float(c.get("quality", 0.0)) for c in candidates], dtype=float)
    if len(qs) > 1:
        order = np.argsort(-qs, kind="stable")
        rank_of = np.empty(len(qs), dtype=int)
        rank_of[order] = np.arange(len(qs))
        q_rank_norm_per_cand = 1.0 - rank_of / max(len(qs) - 1, 1)
    else:
        q_rank_norm_per_cand = np.ones(len(qs), dtype=float)

    (mnd_list, n3_list, top_list, clear_list, blk_list,
     dxp_list, dxm_list, dyp_list, dym_list) =\
        _per_candidate_neighbour_info(candidates, env)

    def _l4_get(lst, i, default):
        if lst is None or i >= len(lst):
            return default
        return lst[i]

    parts = []
    for i in range(K):
        if i < len(candidates):
            parts.append(_candidate_features(
                candidates[i], source_bin_center,
                q_rank_norm=float(q_rank_norm_per_cand[i]),
                min_neighbour_dist_xy=float(mnd_list[i]),
                n_neighbours_3cm=int(n3_list[i]),
                is_top_of_pile=float(top_list[i]),
                approach_clear_pred=float(clear_list[i]),
                n_blocking_above=int(blk_list[i]),
                approach_clear_dx_plus=float(dxp_list[i]),
                approach_clear_dx_minus=float(dxm_list[i]),
                approach_clear_dy_plus=float(dyp_list[i]),
                approach_clear_dy_minus=float(dym_list[i]),
                tilt=float(_l4_get(l4_tilt, i, 1.0)),
                exposure_value=int(_l4_get(l4_exposure, i, 0)),
                is_keystone=int(_l4_get(l4_keystone, i, 0)),
            ))
        else:
            parts.append(np.zeros(_FEAT_PER_CANDIDATE, dtype=np.float32))
    cand_block = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    if n_objects is None:
        n_objects = len(env.get_obj_names()) if env is not None else max(1, K)
    if n_items_remaining is None:
        try:
            from sim.sensing_pose import BIN_HALF_SIZE
            src = env.get_src_bin_world_pos()
            n_items_remaining = sum(
                1 for p in env.get_object_positions().values()
                if abs(p[0] - src[0]) < BIN_HALF_SIZE[0]
                and abs(p[1] - src[1]) < BIN_HALF_SIZE[1]
                and p[2] > src[2] - 0.05
            )
        except Exception:
            n_items_remaining = len(candidates)

    n_rem_norm = float(n_items_remaining) / float(max(1, n_objects))
    n_valid_norm = float(min(len(candidates), K)) / float(max(1, K))
    prev_s = 1.0 if prev_success else 0.0
    step_norm = float(step_count) / float(max(1, max_steps))

    if len(qs) > 0:
        q_spread = float(qs.max() - qs.min())
        mean_q = float(qs.mean())
    else:
        q_spread = 0.0
        mean_q = 0.0

    n_vis_prev_n = float(np.clip(n_visible_prev_step_norm, 0.0, 1.0))
    recent_rate  = float(np.clip(recent_clearing_rate,     0.0, 1.0))
    pile_h_n     = float(np.clip(pile_height_normalised,   0.0, 1.0))

    global_block = np.array([n_rem_norm, n_valid_norm, prev_s, step_norm,
                              q_spread, mean_q,
                              n_vis_prev_n, recent_rate, pile_h_n],
                             dtype=np.float32)

    return np.concatenate([cand_block, global_block]).astype(np.float32)
