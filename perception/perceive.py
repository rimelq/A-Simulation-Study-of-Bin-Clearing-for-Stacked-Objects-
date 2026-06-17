"""Overhead-depth -> GG-CNN -> top-K world-frame grasp candidates.

Each candidate dict carries px, py, depth_m, angle_rad, width_px, width_m,
quality, world_pos, world_quat, component_size. No GT-snap to sim items. no wrist-Z clamp, world_pos is the item-top point under the camera ray.
"""
import numpy as np
import cv2

from scipy import ndimage as _ndi
from scipy.spatial.transform import Rotation as _Rot

from sim.sensing_pose import BIN_HALF_SIZE
from sim.camera_setup import get_camera_intrinsics
from perception.ggcnn_infer import GGCNNInference  # noqa: F401 (re-export convenience)
from perception.candidate_extractor import extract_top_k_candidates
from perception.candidate_to_pose import candidate_to_world_frame, pixels_to_width_meters


_CAMERA_FOV = 60.0
_CAMERA_W   = 640
_CAMERA_H   = 480

# eye-in-hand overhead camera: 180 deg about X -> optical axis points world -Z
_CAM_QUAT_WORLD = np.array([0.0, 1.0, 0.0, 0.0])   # wxyz

_GGCNN_SIZE = 300

# a single rectangle is ~1000-3000 px in the 300x300 mask at our scale
_SINGLE_ITEM_MIN = 1000
_SINGLE_ITEM_MAX = 3000


def perceive_grasp_candidates(env, sensing_ctrl, ggcnn, K: int = 10):
    """Run overhead-depth -> GG-CNN -> top-K grasp candidate pipeline.

    Calls sensing_ctrl.set_sensing_pose_direct() so the camera sees the
    current layout. Returns (candidates, meta) with up to K candidates
    sorted by quality desc.
    """
    sensing_ctrl.set_sensing_pose_direct()
    env._forward()

    K_mat = get_camera_intrinsics(_CAMERA_FOV, _CAMERA_W, _CAMERA_H)
    src_bin_world = env.get_src_bin_world_pos()
    cam_pos = env.get_camera_world_pos()
    floor_depth = float(cam_pos[2] - src_bin_world[2])   # ~0.60 m

    depth = env.get_wrist_depth_meters()

    # mask gripper jaws (very-close pixels) as floor depth
    GRIPPER_MASK_DEPTH = floor_depth - 0.08
    depth_masked = depth.copy()
    depth_masked[depth_masked < GRIPPER_MASK_DEPTH] = floor_depth
    bad = ~np.isfinite(depth_masked) | (depth_masked <= 0)
    depth_masked[bad] = floor_depth

    # bin-tight crop -> 300x300
    fx = K_mat[0, 0]
    bin_half_full = int(np.ceil(fx * max(BIN_HALF_SIZE[:2]) / floor_depth))
    side = int(min(2 * bin_half_full, _CAMERA_W, _CAMERA_H))
    cx_full = int(round(K_mat[0, 2]))
    cy_full = int(round(K_mat[1, 2]))
    x_lo = max(0, cx_full - side // 2)
    x_hi = x_lo + side
    if x_hi > _CAMERA_W:
        x_hi = _CAMERA_W
        x_lo = x_hi - side
    y_lo = max(0, cy_full - side // 2)
    y_hi = y_lo + side
    if y_hi > _CAMERA_H:
        y_hi = _CAMERA_H
        y_lo = y_hi - side

    depth_crop_bin = depth_masked[y_lo:y_hi, x_lo:x_hi]
    depth_m_300 = cv2.resize(depth_crop_bin, (_GGCNN_SIZE, _GGCNN_SIZE),
                             interpolation=cv2.INTER_LINEAR)
    crop_to_orig_scale = side / float(_GGCNN_SIZE)

    # auto-normalise crop to [0, 1] for GG-CNN input
    auto_min = float(depth_m_300.min())
    auto_max = float(depth_m_300.max())
    auto_range = max(auto_max - auto_min, 0.001)
    depth_pp = np.clip((depth_m_300 - auto_min) / auto_range, 0.0, 1.0).astype(np.float32)

    preds = ggcnn.predict(depth_pp)
    quality = preds["quality"]
    angle   = preds["angle"]
    width   = preds["width"]

    raw_cands = extract_top_k_candidates(
        quality, angle, width,
        depth_m_crop=depth_m_300,
        K=K,
        min_quality=0.1,    # low threshold so we get at least one candidate when possible
        nms_size=11,
    )

    # connected-component object mask: depth_pp normalised so floor->1.0, objects->~0.0..0.5
    obj_px_mask = (depth_pp < 0.5)
    n_obj_px = int(obj_px_mask.sum())
    labels, n_components = _ndi.label(obj_px_mask)

    comp_info = {}
    for lbl in range(1, n_components + 1):
        m = (labels == lbl)
        sz = int(m.sum())
        ys_, xs_ = np.where(m)
        comp_info[lbl] = {
            "size": sz,
            "cx": float(xs_.mean()),
            "cy": float(ys_.mean()),
            "mask": m,
            "single": _SINGLE_ITEM_MIN <= sz <= _SINGLE_ITEM_MAX,
        }

    def _pca_short_axis_image(mask):
        """Image-frame short-axis angle (rad) of a pixel blob, or None if too small.

        Grasp axis (jaw line) is chosen ALONG the short axis so jaws close
        ACROSS the long axis.
        """
        ys_, xs_ = np.where(mask)
        if len(xs_) < 30:
            return None
        pts = np.column_stack([xs_ - xs_.mean(), ys_ - ys_.mean()]).astype(np.float64)
        cov = np.cov(pts.T)
        evals, evecs = np.linalg.eigh(cov)
        short_axis = evecs[:, 0]
        return float(np.arctan2(short_axis[1], short_axis[0]))

    candidates = []
    for c in raw_cands:
        cpx, cpy = int(c["px"]), int(c["py"])
        lbl = int(labels[cpy, cpx])
        comp = comp_info.get(lbl)
        # if peak sits on floor, snap to nearest object pixel within +-20 px
        if comp is None or comp["size"] < 30:
            win = 20
            ylo, yhi = max(0, cpy - win), min(_GGCNN_SIZE, cpy + win + 1)
            xlo, xhi = max(0, cpx - win), min(_GGCNN_SIZE, cpx + win + 1)
            sub = labels[ylo:yhi, xlo:xhi]
            nz = sub[sub > 0]
            if len(nz) > 0:
                lbl = int(np.bincount(nz).argmax())
                comp = comp_info.get(lbl)

        px_out, py_out = cpx, cpy
        ang_out = float(c["angle_rad"])
        comp_size = 0
        if comp is not None:
            comp_size = comp["size"]
            # snap pixel to component centroid (jaws need to be centred on item)
            px_out = int(round(comp["cx"]))
            py_out = int(round(comp["cy"]))
            # GG-CNN angle is unreliable for axis-aligned rects. override with PCA short axis
            ang_pca = _pca_short_axis_image(comp["mask"])
            if ang_pca is not None:
                ang_out = ang_pca

        px_out = int(np.clip(px_out, 0, _GGCNN_SIZE - 1))
        py_out = int(np.clip(py_out, 0, _GGCNN_SIZE - 1))
        depth_out = float(depth_m_300[py_out, px_out])
        # if centroid landed on floor depth, fall back to original peak depth on object
        if depth_out >= floor_depth - 0.002 and c["depth_m"] > 0:
            depth_out = float(c["depth_m"])

        cand_for_world = {
            "px": px_out, "py": py_out,
            "depth_m": depth_out,
            "angle_rad": ang_out,
            "width_px": float(c["width_px"]),
        }

        world_pos, world_quat = candidate_to_world_frame(
            candidate=cand_for_world,
            K_matrix=K_mat,
            crop_offset=(x_lo, y_lo),
            cam_pos_world=cam_pos,
            cam_quat_world=_CAM_QUAT_WORLD,
            crop_to_orig_scale=crop_to_orig_scale,
        )
        world_pos = np.asarray(world_pos, dtype=np.float64).copy()

        # world-quat override from PCA short axis: image-y flips world-y because cam is rotated 180 deg about X
        sx = float(np.cos(ang_out))
        sy = float(-np.sin(ang_out))
        gz = np.array([0.0, 0.0, -1.0])                  # gripper Z down
        gy = np.array([sx, sy, 0.0])                     # jaws close along short axis
        ny = np.linalg.norm(gy)
        if ny > 1e-9:
            gy = gy / ny
            gx = np.cross(gy, gz)
            gx = gx / max(np.linalg.norm(gx), 1e-9)
            R_world = np.column_stack([gx, gy, gz])
            xyzw = _Rot.from_matrix(R_world).as_quat()
            world_quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])  # wxyz

        width_m = pixels_to_width_meters(float(c["width_px"]), depth_out, fx)

        candidates.append({
            "px": px_out,
            "py": py_out,
            "depth_m": depth_out,
            "angle_rad": ang_out,
            "width_px": float(c["width_px"]),
            "width_m": float(width_m),
            "quality": float(c["quality"]),
            "world_pos": world_pos,
            "world_quat": np.asarray(world_quat, dtype=np.float64),
            "component_size": int(comp_size),
        })

    meta = {
        "cam_pos": cam_pos,
        "cam_quat": _CAM_QUAT_WORLD,
        "K": K_mat,
        "floor_depth": floor_depth,
        "src_bin_world": src_bin_world,
        "crop_window": (x_lo, x_hi, y_lo, y_hi),
        "crop_to_orig_scale": crop_to_orig_scale,
        "n_object_pixels": n_obj_px,
        "n_components": int(n_components),
        "n_candidates": len(candidates),
        "depth_pp": depth_pp,
        "quality_map": quality,
        "auto_norm_range": (auto_min, auto_max),
    }
    return candidates, meta
