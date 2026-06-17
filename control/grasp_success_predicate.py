"""Deterministic geometric grasp-success predicate.

Public entry point: ``evaluate_grasp(env, grasp_pos, grasp_quat, candidate_item_name)``.

Conventions:
  grasp_pos  : (3,) world-frame wrist target (OSC end-effector point).
  grasp_quat : (4,) world-frame gripper orientation, wxyz. Gripper frame:
               Z = approach axis (down for top grasp),
               Y = jaw-closing direction,
               X = pad-long axis.
"""
import numpy as np

# Pad centre is 9.7 cm below the robot0_right_hand body centre (Panda gripper).
_FINGER_TO_WRIST = 0.097

_JAW_HALF_SPAN       = 0.040   # jaws open ~8 cm
_PAD_HALF_LENGTH     = 0.025   # pads ~5 cm long
# env snaps grasp orientation to item short axis, so correct grasp arrives ~0 deg here
_JAW_ALIGN_MAX_RAD   = np.deg2rad(30.0)
_HAND_FOOTPRINT_HALF = 0.05
_CANDIDATE_MATCH_XY  = 0.06


def _quat_to_R(quat_wxyz):
    """wxyz quaternion -> 3x3 rotation matrix (world <- gripper)."""
    w, x, y, z = quat_wxyz
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def _item_geom_id(env, body_id):
    """First geom belonging to body_id (items have a single mesh geom)."""
    model = env.sim.model
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == body_id:
            return g
    return None


def _item_aabb_half(env, body_id):
    """Local mesh half-extents [hx, hy, hz] for an item body, or None."""
    gid = _item_geom_id(env, body_id)
    if gid is None:
        return None
    aabb = np.array(env.sim.model.geom_aabb[gid], dtype=float)  # [cx,cy,cz,hx,hy,hz]
    return aabb[3:6]


def _item_short_horizontal_axis_world(R_item, half):
    """Short horizontal axis of the item, in world frame, with diagnostics.

    For each local axis i, its XY-projected half-extent is
    ``half_i * sqrt(1 - R_item[2, i]**2)``. The (near-)vertical axis projects
    to ~0, so sorting yields [vertical, short-horizontal, long-horizontal].
    Index 1 is the short horizontal axis (the jaw-closing direction).
    """
    xy_ext = np.array([
        half[i] * np.sqrt(max(1.0 - float(R_item[2, i]) ** 2, 0.0))
        for i in range(3)
    ])
    short_idx = int(np.argsort(xy_ext)[1])
    short_local = np.zeros(3)
    short_local[short_idx] = 1.0
    short_world = R_item @ short_local
    return short_world, xy_ext, short_idx


def _item_half_height(R_item, half):
    """World-Z extent (half) of the item."""
    return float(np.sum([half[i] * abs(float(R_item[2, i])) for i in range(3)]))


def _list_source_bin_items(env):
    """Return [(name, world_pos)] for items currently inside the source bin."""
    from sim.sensing_pose import BIN_HALF_SIZE
    src = env.get_src_bin_world_pos()
    out = []
    for name, pos in env.get_object_positions().items():
        pos = np.asarray(pos, dtype=float)
        if (abs(pos[0] - src[0]) < BIN_HALF_SIZE[0]
                and abs(pos[1] - src[1]) < BIN_HALF_SIZE[1]
                and pos[2] > src[2] - 0.05):
            out.append((name, pos))
    return out


def evaluate_grasp(env, grasp_pos, grasp_quat, candidate_item_name=None):
    """Geometric success predicate. Returns dict with success + sub-criteria.

    ``candidate_item_name`` is the env's association hint (nearest centre to
    grasp XY). May be None -> automatic failure. We still recompute the best
    picked_item from gripper-frame geometry.
    """
    grasp_pos = np.asarray(grasp_pos, dtype=float)
    grasp_quat = np.asarray(grasp_quat, dtype=float)

    result = {
        "success": False,
        "item_between_jaws": False,
        "jaws_aligned": False,
        "mid_height": False,
        "approach_clear": False,
        "reason": "",
        "picked_item": None,
    }

    items = _list_source_bin_items(env)
    if not items:
        result["reason"] = "source bin empty"
        return result

    R_grip = _quat_to_R(grasp_quat)            # world <- gripper
    R_grip_T = R_grip.T
    pad_center = grasp_pos + R_grip @ np.array([0.0, 0.0, -_FINGER_TO_WRIST])

    # 1) item_between_jaws: y half-window shrinks by item half-width along the
    # jaw-closing direction so the body must fully enter the jaw span.
    best_item = None
    best_item_pos = None
    best_score = None
    for name, pos in items:
        rel_world = pos - pad_center
        rel_grip = R_grip_T @ rel_world
        try:
            bid = env.sim.model.body_name2id(name)
        except Exception:
            continue
        R_item = np.array(env.sim.data.body_xmat[bid]).reshape(3, 3)
        half = _item_aabb_half(env, bid)
        if half is None:
            continue
        gy_world = R_grip[:, 0 + 1]
        item_half_along_y = float(np.sum([
            half[i] * abs(float(np.dot(R_item[:, i], gy_world))) for i in range(3)
        ]))
        gx_world = R_grip[:, 0]
        item_half_along_x = float(np.sum([
            half[i] * abs(float(np.dot(R_item[:, i], gx_world))) for i in range(3)
        ]))
        y_window = max(_JAW_HALF_SPAN - item_half_along_y, 0.005)
        x_window = _PAD_HALF_LENGTH + item_half_along_x
        if abs(rel_grip[1]) < y_window and abs(rel_grip[0]) < x_window:
            score = abs(rel_grip[1]) + 0.5 * abs(rel_grip[0])
            if best_score is None or score < best_score:
                best_score = score
                best_item = name
                best_item_pos = pos

    if best_item is None:
        hint = candidate_item_name
        if hint is not None and any(n == hint for n, _ in items):
            result["reason"] = "no item fits between jaws (hint item too far/misaligned)"
        else:
            result["reason"] = "no item between jaws"
        return result

    result["item_between_jaws"] = True
    result["picked_item"] = best_item

    bid = env.sim.model.body_name2id(best_item)
    R_item = np.array(env.sim.data.body_xmat[bid]).reshape(3, 3)
    half = _item_aabb_half(env, bid)

    # 2) jaws_aligned
    gy_world = R_grip[:, 1]
    gy_xy = np.array([gy_world[0], gy_world[1]])
    short_world, xy_ext, short_idx = _item_short_horizontal_axis_world(R_item, half)
    short_xy = np.array([short_world[0], short_world[1]])
    n1, n2 = np.linalg.norm(gy_xy), np.linalg.norm(short_xy)
    if n1 < 1e-9 or n2 < 1e-9:
        align_rad = np.pi / 2
    else:
        cos_a = abs(float(np.dot(gy_xy / n1, short_xy / n2)))
        align_rad = float(np.arccos(np.clip(cos_a, -1.0, 1.0)))
    result["jaws_aligned"] = bool(align_rad < _JAW_ALIGN_MAX_RAD)

    # 3) mid_height
    item_cz = float(best_item_pos[2])
    item_half_h = _item_half_height(R_item, half)
    pad_z = float(grasp_pos[2] - _FINGER_TO_WRIST)
    result["mid_height"] = bool(abs(pad_z - item_cz) < item_half_h)

    # 4) approach_clear: no other source-bin item in the hand footprint AND
    # above the target top (would collide during descent).
    target_top_z = item_cz + item_half_h
    approach_clear = True
    blocker = None
    for name, pos in items:
        if name == best_item:
            continue
        rel_grip = R_grip_T @ (np.asarray(pos, float) - pad_center)
        in_footprint = (abs(rel_grip[0]) < _HAND_FOOTPRINT_HALF
                        and abs(rel_grip[1]) < _HAND_FOOTPRINT_HALF)
        above_target = float(pos[2]) > target_top_z - 0.002
        if in_footprint and above_target:
            approach_clear = False
            blocker = name
            break
    result["approach_clear"] = bool(approach_clear)

    result["success"] = bool(result["item_between_jaws"] and result["jaws_aligned"]
                             and result["mid_height"] and result["approach_clear"])

    if result["success"]:
        result["reason"] = f"OK: would pick '{best_item}'"
    else:
        missing = []
        if not result["jaws_aligned"]:
            missing.append(f"jaws_misaligned({np.degrees(align_rad):.1f}deg)")
        if not result["mid_height"]:
            missing.append(f"bad_height(pad_z={pad_z:.3f} vs item_z={item_cz:.3f}+-{item_half_h:.3f})")
        if not result["approach_clear"]:
            missing.append(f"blocked_by({blocker})")
        result["reason"] = "fail: " + ", ".join(missing) if missing else "fail"

    return result
