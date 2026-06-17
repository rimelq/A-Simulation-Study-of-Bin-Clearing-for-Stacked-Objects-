"""Perfect-Perception Oracle (PPO) candidate generator.

Reads simulator GT to emit one candidate per visible cube in the source bin,
in the same dict schema as perceive.py / perceive_cc.py. Sits ABOVE the
perception layer: used as the upper-bound oracle in the 4-method comparison.
Feeding every policy the same clean candidate stream isolates the selection
policy from the perception stage, so the two effects can be measured apart.

visibility_mode:
    'raycast', skip cubes occluded from above (XY-footprint check)
    'omniscient', emit one candidate per cube regardless of occlusion

quality_mode:
    'analytical', 0.4*clearance + 0.3*height + 0.3*tilt
    'uniform', 1.0 for every candidate (ablation knob)
"""
import numpy as np

from scipy.spatial.transform import Rotation as _Rot

from sim.sensing_pose import BIN_HALF_SIZE


_ITEM_HALF_HEIGHT = 0.029
_ITEM_HALF_XY = 0.0295

# z tolerance for "another cube is above", robust to numerical noise without rejecting same-layer touches
_OCCLUSION_Z_TOL = 0.005
# XY centre-to-centre tolerance for the occlusion check (one cube half-extent + margin)
_OCCLUSION_XY_TOL = _ITEM_HALF_XY + 0.005

_DEFAULT_WIDTH_M = 0.064
_DEFAULT_WIDTH_PX = 28.0

_W_CLEARANCE = 0.4
_W_HEIGHT    = 0.3
_W_TILT      = 0.3

_CAMERA_FOV = 60.0
_CAMERA_W   = 640
_CAMERA_H   = 480


def _is_visible_from_above(target_pos, target_top_z, other_items):
    """XY-footprint occlusion check: True iff no other cube sits above the target."""
    tx, ty = float(target_pos[0]), float(target_pos[1])
    for _name, opos in other_items:
        ox, oy, oz = float(opos[0]), float(opos[1]), float(opos[2])
        if oz < target_top_z + _OCCLUSION_Z_TOL:
            continue
        dxy = (ox - tx) ** 2 + (oy - ty) ** 2
        if dxy < _OCCLUSION_XY_TOL ** 2:
            return False
    return True


def _analytical_quality(target_pos, R_target, other_xy, bin_aabb_z):
    """Quality = 0.4*clearance + 0.3*height + 0.3*tilt, all in [0, 1]."""
    if len(other_xy) == 0:
        clearance_score = 1.0
    else:
        gxy = np.array([target_pos[0], target_pos[1]], dtype=float)
        dists = np.linalg.norm(other_xy - gxy[None, :], axis=1)
        nearest = float(np.min(dists))
        # clearance cap at 6 cm
        clearance_score = float(np.clip(nearest / 0.06, 0.0, 1.0))

    z_floor, z_ceiling = bin_aabb_z
    z_range = max(z_ceiling - z_floor, 1e-6)
    height_score = float(np.clip((float(target_pos[2]) - z_floor) / z_range,
                                 0.0, 1.0))

    # R[2,2] is the world-Z component of the cube's local Z, |.| == 1 means upright
    tilt_score = float(abs(float(R_target[2, 2])))

    return float(np.clip(
        _W_CLEARANCE * clearance_score
        + _W_HEIGHT    * height_score
        + _W_TILT      * tilt_score,
        0.0, 1.0))


def _gripper_yaw_for_short_axis(quat_world):
    """Recover image-frame grasp angle (rad) from gripper world_quat (wxyz).

    Gripper Y axis is the jaw-closing direction. the angle is the world-XY yaw of Y.
    """
    try:
        w, x, y, z = float(quat_world[0]), float(quat_world[1]),\
                     float(quat_world[2]), float(quat_world[3])
        R = _Rot.from_quat([x, y, z, w]).as_matrix()
        gy = R[:, 1]
        return float(np.arctan2(float(gy[1]), float(gy[0])))
    except Exception:
        return 0.0


def perceive_ppo_candidates(env, sensing_ctrl, ggcnn=None, K: int = 10,
                             visibility_mode: str = "raycast",
                             quality_mode: str = "analytical"):
    """Perfect-Perception Oracle ghost method.

    Drop-in for perception.perceive.perceive_grasp_candidates. ``ggcnn`` is
    accepted for API compatibility and IGNORED. Still moves the arm to the
    sensing pose so the sim state matches what GG-CNN/CC paths see.
    """
    if visibility_mode not in ("raycast", "omniscient"):
        raise ValueError(f"visibility_mode must be 'raycast' or 'omniscient', "
                         f"got {visibility_mode!r}")
    if quality_mode not in ("analytical", "uniform"):
        raise ValueError(f"quality_mode must be 'analytical' or 'uniform', "
                         f"got {quality_mode!r}")

    # env is the BinClearingGymEnv wrapper. underlying robosuite env is env.env
    sensing_ctrl.set_sensing_pose_direct()
    env.env._forward()

    src_bin_world = env.env.get_src_bin_world_pos()
    z_floor = float(src_bin_world[2])
    # ~3 cube heights of vertical range. clip in _analytical_quality handles overshoots
    z_ceiling = z_floor + 3.0 * (2.0 * _ITEM_HALF_HEIGHT)

    items = list(env._source_bin_items())  # [(name, world_pos)]
    if not items:
        meta = {
            "method": "ppo",
            "visibility_mode": visibility_mode,
            "quality_mode": quality_mode,
            "n_in_bin": 0,
            "n_visible": 0,
            "n_candidates": 0,
        }
        return [], meta

    item_names = [n for n, _ in items]
    item_positions = [np.asarray(p, dtype=float) for _, p in items]
    other_pos = {
        name: [(n2, p2) for (n2, p2) in zip(item_names, item_positions)
                if n2 != name]
        for name in item_names
    }
    item_xy = np.asarray([p[:2] for p in item_positions], dtype=np.float64)

    candidates = []
    n_visible = 0
    for name, pos_com in zip(item_names, item_positions):
        pos_com = np.asarray(pos_com, dtype=float)
        target_top_z = float(pos_com[2]) + _ITEM_HALF_HEIGHT

        if visibility_mode == "raycast":
            if not _is_visible_from_above(pos_com, target_top_z,
                                          other_pos[name]):
                continue
        n_visible += 1

        # world_pos = cube top centre, not COM. matches env snap path at execution time
        try:
            bid = env.env.sim.model.body_name2id(name)
        except Exception:
            bid = -1
        try:
            R_item = np.asarray(env.env.sim.data.body_xmat[bid],
                                dtype=float).reshape(3, 3)
        except Exception:
            R_item = np.eye(3)
        world_pos = pos_com + R_item @ np.array([0.0, 0.0, _ITEM_HALF_HEIGHT])

        # gripper Z down, jaws aligned with item's GT short horizontal axis
        world_quat = np.asarray(env._grasp_quat_for_item(name), dtype=float)

        angle_rad = _gripper_yaw_for_short_axis(world_quat)

        if quality_mode == "uniform":
            quality = 1.0
        else:
            # exclude self from clearance computation
            mask = np.ones(len(item_xy), dtype=bool)
            try:
                idx = item_names.index(name)
                mask[idx] = False
            except ValueError:
                pass
            quality = _analytical_quality(
                target_pos=pos_com,
                R_target=R_item,
                other_xy=item_xy[mask],
                bin_aabb_z=(z_floor, z_ceiling),
            )

        candidates.append({
            "px":             -1,
            "py":             -1,
            "depth_m":        float(z_ceiling - world_pos[2] + 0.30),
            "angle_rad":      float(angle_rad),
            "width_px":       float(_DEFAULT_WIDTH_PX),
            "width_m":        float(_DEFAULT_WIDTH_M),
            "quality":        float(quality),
            "world_pos":      np.asarray(world_pos, dtype=np.float64),
            "world_quat":     world_quat,
            "component_size": 1,
            "source_body_id": int(bid),
        })

    candidates.sort(key=lambda c: -float(c["quality"]))
    candidates = candidates[:K]

    meta = {
        "method":           "ppo",
        "visibility_mode":  visibility_mode,
        "quality_mode":     quality_mode,
        "n_in_bin":         len(items),
        "n_visible":        n_visible,
        "n_candidates":     len(candidates),
        "src_bin_world":    src_bin_world,
        "bin_aabb_z":       (z_floor, z_ceiling),
    }
    return candidates, meta
