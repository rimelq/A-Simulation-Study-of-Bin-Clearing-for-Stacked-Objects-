"""Geometric disturbance model for reward_mode="hybrid".

Replaces a physics shove with a deterministic, position-based model:
the gripper's two finger pads sweep a known footprint through the bin. any
non-target item whose centre lies inside that footprint is pushed out along
its shortest exit, with item-item overlaps resolved iteratively. Items are
clamped inside the bin walls (no ejection in this engine).

Public entry point::

    compute_and_apply_disturbance(env, grasp_pos, grasp_quat, target_item_name)
        -> dict(neighbour_disturbance_m, items_ejected, n_disturbed, n_in_footprint)

SIDE EFFECT: kinematically MOVES disturbed items via ``env._set_object_pose``
and calls ``env._forward()``, the next render reflects the new layout.
"""
import numpy as np

from sim.sensing_pose import BIN_HALF_SIZE
from control.grasp_success_predicate import _quat_to_R


# Footprint in gripper frame (metres). Y = jaw-closing (~8 cm open), X = pad-long (~5 cm).
_FOOTPRINT_HALF_Y = 0.040
_FOOTPRINT_HALF_X = 0.025
# square_22 cubes ~5.9 cm -> half ~3 cm. used to inflate footprint and as
# item-item separation in the cascade resolver.
_ITEM_HALF = 0.030
_ITEM_DIAM = 2.0 * _ITEM_HALF
_CASCADE_ITERS = 6
# Solid walls: every item centre is clamped this far inside the nominal bin
# half-extent so cube faces rest flush on the wall.
_WALL_CLAMP_MARGIN = 0.030
# Items at or above target level are in the descending fingers' path. items
# clearly below are shielded by the target and the layers between.
_Z_MARGIN = 0.030
_DISTURB_FLOOR_M = 0.005
# Per-item cap caps a pathological cascade. a real ejection (0.15-0.30 m) is
# still fully counted.
_DISTURB_CAP_M = 0.30


def _item_quat(env, name):
    """Current world quaternion (wxyz) of an item body, or identity."""
    try:
        bid = env.sim.model.body_name2id(name)
        return np.array(env.sim.data.body_xquat[bid], dtype=np.float64)
    except Exception:
        return np.array([1.0, 0.0, 0.0, 0.0])


def _source_bin_items(env):
    """{name: (3,) world pos} for items currently inside the source bin."""
    src = env.get_src_bin_world_pos()
    out = {}
    for name, p in env.get_object_positions().items():
        p = np.asarray(p, dtype=np.float64)
        if (abs(p[0] - src[0]) < BIN_HALF_SIZE[0]
                and abs(p[1] - src[1]) < BIN_HALF_SIZE[1]
                and p[2] > src[2] - 0.05):
            out[name] = p.copy()
    return out


def compute_and_apply_disturbance(env, grasp_pos, grasp_quat,
                                  target_item_name,
                                  disturb_floor_m: float = _DISTURB_FLOOR_M) -> dict:
    """Deterministic geometric disturbance for a grasp at (grasp_pos, grasp_quat).

    Returns dict with neighbour_disturbance_m, items_ejected, n_disturbed,
    n_in_footprint. SIDE EFFECT: moves disturbed items in the sim.
    """
    out = {"neighbour_disturbance_m": 0.0, "items_ejected": 0,
           "n_disturbed": 0, "n_in_footprint": 0}

    grasp_pos = np.asarray(grasp_pos, dtype=np.float64)
    R = _quat_to_R(np.asarray(grasp_quat, dtype=np.float64))
    gx_xy = R[:2, 0]; nx = np.linalg.norm(gx_xy)
    gy_xy = R[:2, 1]; ny = np.linalg.norm(gy_xy)
    if nx < 1e-9 or ny < 1e-9:
        return out                      # degenerate orientation
    gx_xy = gx_xy / nx
    gy_xy = gy_xy / ny
    g_center = grasp_pos[:2]

    items = _source_bin_items(env)
    if not items:
        return out
    start_xy = {n: p[:2].copy() for n, p in items.items()}
    cur_xy   = {n: p[:2].copy() for n, p in items.items()}
    cur_z    = {n: float(p[2])  for n, p in items.items()}

    win_x = _FOOTPRINT_HALF_X + _ITEM_HALF
    win_y = _FOOTPRINT_HALF_Y + _ITEM_HALF

    # Items at/above target level are in the descending fingers' path. items
    # clearly below are shielded, makes "clear top first" the winning strategy.
    target_z = cur_z.get(target_item_name, None)

    # 1. z-aware footprint sweep
    in_footprint = []
    for name, xy in cur_xy.items():
        if name == target_item_name:
            continue
        rel = xy - g_center
        x_loc = float(rel @ gx_xy)
        y_loc = float(rel @ gy_xy)
        if abs(x_loc) >= win_x or abs(y_loc) >= win_y:
            continue
        if target_z is not None and cur_z[name] < target_z - _Z_MARGIN:
            continue                      # below the target, shielded
        in_footprint.append(name)
    out["n_in_footprint"] = len(in_footprint)

    # 2. push each footprint item out along its shortest exit
    for name in in_footprint:
        rel = cur_xy[name] - g_center
        x_loc = float(rel @ gx_xy)
        y_loc = float(rel @ gy_xy)
        exit_x = win_x - abs(x_loc)
        exit_y = win_y - abs(y_loc)
        if exit_y <= exit_x:
            sgn = 1.0 if y_loc >= 0 else -1.0
            cur_xy[name] = cur_xy[name] + gy_xy * sgn * (exit_y + 0.003)
        else:
            sgn = 1.0 if x_loc >= 0 else -1.0
            cur_xy[name] = cur_xy[name] + gx_xy * sgn * (exit_x + 0.003)

    # 3. cascade with solid-wall clamps. Per-item rotation-aware footprint half:
    # a cube of half-size h at yaw theta has footprint half h*(|costheta|+|sintheta|). Clamping
    # the centre to wall - fp_half makes the rotated corner rest exactly on the
    # wall (the fixed 3 cm margin missed this and rotated cubes poked through).
    src = env.get_src_bin_world_pos()
    wall = float(BIN_HALF_SIZE[0])

    fp_half = {}
    for name in cur_xy:
        try:
            R = _quat_to_R(_item_quat(env, name))
            yaw = float(np.arctan2(R[1, 0], R[0, 0]))
            fp_half[name] = _ITEM_HALF * (abs(np.cos(yaw)) + abs(np.sin(yaw)))
        except Exception:
            fp_half[name] = _ITEM_HALF * 1.42

    def _clamp_inside_walls(name, xy):
        lim = wall - fp_half.get(name, _ITEM_HALF * 1.42)
        return np.array([
            float(np.clip(xy[0], src[0] - lim, src[0] + lim)),
            float(np.clip(xy[1], src[1] - lim, src[1] + lim)),
        ])

    names = list(cur_xy.keys())
    for name in names:
        if name != target_item_name:
            cur_xy[name] = _clamp_inside_walls(name, cur_xy[name])
    for _ in range(_CASCADE_ITERS):
        moved_any = False
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                d = cur_xy[a] - cur_xy[b]
                dn = float(np.linalg.norm(d))
                if dn < 1e-6:
                    d = np.array([1.0, 0.0]); dn = 1.0
                if dn < _ITEM_DIAM:
                    overlap = _ITEM_DIAM - dn
                    push = (d / dn) * (overlap / 2.0 + 0.001)
                    if a != target_item_name:
                        cur_xy[a] = cur_xy[a] + push
                    if b != target_item_name:
                        cur_xy[b] = cur_xy[b] - push
                    moved_any = True
        for name in names:
            if name != target_item_name:
                cur_xy[name] = _clamp_inside_walls(name, cur_xy[name])
        if not moved_any:
            break

    # 4. apply moves. No ejection in this engine: every item is clamped inside
    # the solid walls. Physics reward modes handle real over-the-wall ejection.
    total_disturb = 0.0
    n_disturbed = 0
    for name in cur_xy:
        if name == target_item_name:
            continue
        final_xy = _clamp_inside_walls(name, cur_xy[name])
        moved = float(np.linalg.norm(final_xy - start_xy[name]))
        if moved < disturb_floor_m:
            continue
        n_disturbed += 1
        total_disturb += min(_DISTURB_CAP_M, moved)
        quat = _item_quat(env, name)
        env._set_object_pose(name, np.array([final_xy[0], final_xy[1],
                                             cur_z[name]]), quat)

    try:
        env._forward()
    except Exception:
        pass

    out["neighbour_disturbance_m"] = float(total_disturb)
    out["items_ejected"] = 0
    out["n_disturbed"] = int(n_disturbed)
    return out
