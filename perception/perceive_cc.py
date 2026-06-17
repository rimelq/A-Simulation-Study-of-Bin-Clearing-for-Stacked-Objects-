"""Depth-only connected-component candidate generator (no GG-CNN).

Drop-in replacement for perceive.perceive_grasp_candidates: same input
(one overhead depth image at the sensing pose), same output dict schema.
Pipeline:
  1. Mask pixels above the bin floor (depth-normalised < 0.5).
  2. Label connected components.
  3. Per component, pick the pixel of MINIMUM depth (closest to camera =
     top of pile) as the candidate location.
  4. PCA short-axis of the component pixel mask -> grasp angle.
  5. Back-project (px, py, depth) to world frame.
  6. Elevation score replaces GG-CNN quality.
"""
import numpy as np
import cv2

from scipy import ndimage as _ndi
from scipy.spatial.transform import Rotation as _Rot

from sim.sensing_pose import BIN_HALF_SIZE
from sim.camera_setup import get_camera_intrinsics
from perception.candidate_to_pose import candidate_to_world_frame, pixels_to_width_meters


# constants mirrored from perceive.py so depth pre-processing is IDENTICAL
_CAMERA_FOV = 60.0
_CAMERA_W   = 640
_CAMERA_H   = 480

_CAM_QUAT_WORLD = np.array([0.0, 1.0, 0.0, 0.0])   # wxyz

_GGCNN_SIZE = 300

# perceive.py uses 1000-3000 as single-item band. relax to 200 so we don't drop
# small visible patches at the edge but still reject speckle noise
_MIN_COMPONENT_PX = 200

# uniform 5.9 cm cubes. the primitive overrides width anyway in snap mode
_DEFAULT_WIDTH_M = 0.08
_DEFAULT_WIDTH_PX = 28.0


def _pca_short_axis_image(mask: np.ndarray):
    """Short-axis angle (rad, image frame) of a pixel blob. None if too small.

    Grasp axis (jaw line) chosen ALONG the short axis so jaws close ACROSS the long axis.
    """
    ys_, xs_ = np.where(mask)
    if len(xs_) < 30:
        return None
    pts = np.column_stack([xs_ - xs_.mean(), ys_ - ys_.mean()]).astype(np.float64)
    cov = np.cov(pts.T)
    evals, evecs = np.linalg.eigh(cov)
    short_axis = evecs[:, 0]
    return float(np.arctan2(short_axis[1], short_axis[0]))


def perceive_cc_candidates(env, sensing_ctrl, ggcnn=None, K: int = 10):
    """Depth-only candidate generator.

    Drop-in for perception.perceive.perceive_grasp_candidates. ``ggcnn`` is
    accepted for API compatibility and IGNORED. Returns (candidates, meta)
    with up to K candidates sorted by world-z desc.
    """
    sensing_ctrl.set_sensing_pose_direct()
    env._forward()

    K_mat = get_camera_intrinsics(_CAMERA_FOV, _CAMERA_W, _CAMERA_H)
    src_bin_world = env.get_src_bin_world_pos()
    cam_pos = env.get_camera_world_pos()
    floor_depth = float(cam_pos[2] - src_bin_world[2])

    depth = env.get_wrist_depth_meters()

    GRIPPER_MASK_DEPTH = floor_depth - 0.08
    depth_masked = depth.copy()
    depth_masked[depth_masked < GRIPPER_MASK_DEPTH] = floor_depth
    bad = ~np.isfinite(depth_masked) | (depth_masked <= 0)
    depth_masked[bad] = floor_depth

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

    auto_min = float(depth_m_300.min())
    auto_max = float(depth_m_300.max())
    auto_range = max(auto_max - auto_min, 0.001)
    depth_pp = np.clip((depth_m_300 - auto_min) / auto_range, 0.0, 1.0).astype(np.float32)

    # same threshold as perceive.py: < 0.5 above floor, >= 0.5 floor
    obj_px_mask = (depth_pp < 0.5)
    n_obj_px = int(obj_px_mask.sum())
    labels, n_components = _ndi.label(obj_px_mask)

    candidates = []
    for lbl in range(1, n_components + 1):
        comp_mask = (labels == lbl)
        comp_size = int(comp_mask.sum())
        if comp_size < _MIN_COMPONENT_PX:
            continue

        # pixel of MINIMUM metric depth in this component (closest to camera = top of pile)
        comp_depth = np.where(comp_mask, depth_m_300, np.inf)
        flat_idx = int(np.argmin(comp_depth))
        py_out, px_out = int(flat_idx // _GGCNN_SIZE), int(flat_idx % _GGCNN_SIZE)
        depth_out = float(depth_m_300[py_out, px_out])

        # at-or-below floor depth -> noise component, skip
        if depth_out >= floor_depth - 0.002 or depth_out <= 0.0:
            continue

        ang_pca = _pca_short_axis_image(comp_mask)
        if ang_pca is None:
            # tiny component without stable PCA. env snap will correct this
            ang_pca = 0.0
        ang_out = float(ang_pca)

        cand_for_world = {
            "px": px_out, "py": py_out,
            "depth_m": depth_out,
            "angle_rad": ang_out,
            "width_px": _DEFAULT_WIDTH_PX,
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

        # gripper Z down, jaws Y aligned with PCA short axis (image-y flips world-y)
        sx = float(np.cos(ang_out))
        sy = float(-np.sin(ang_out))
        gz = np.array([0.0, 0.0, -1.0])
        gy = np.array([sx, sy, 0.0])
        ny = np.linalg.norm(gy)
        if ny > 1e-9:
            gy = gy / ny
            gx = np.cross(gy, gz)
            gx = gx / max(np.linalg.norm(gx), 1e-9)
            R_world = np.column_stack([gx, gy, gz])
            xyzw = _Rot.from_matrix(R_world).as_quat()
            world_quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])  # wxyz
        world_quat = np.asarray(world_quat, dtype=np.float64)

        width_m = _DEFAULT_WIDTH_M

        # elevation score: closer to camera -> higher. ~[0, 0.3] for multi-layer stacks. clipped to [0, 1].
        elevation = (floor_depth - depth_out) / max(floor_depth, 1e-6)
        quality = float(np.clip(elevation * 4.0, 0.0, 1.0))

        candidates.append({
            "px": px_out,
            "py": py_out,
            "depth_m": depth_out,
            "angle_rad": ang_out,
            "width_px": float(_DEFAULT_WIDTH_PX),
            "width_m": float(width_m),
            "quality": float(quality),
            "world_pos": world_pos,
            "world_quat": world_quat,
            "component_size": int(comp_size),
        })

    candidates.sort(key=lambda c: -float(c["world_pos"][2]))
    candidates = candidates[:K]

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
        "quality_map": None,
        "auto_norm_range": (auto_min, auto_max),
        "method": "cc",
    }
    return candidates, meta
