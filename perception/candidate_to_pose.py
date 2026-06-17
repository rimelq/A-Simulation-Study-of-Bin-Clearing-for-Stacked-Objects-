"""Convert GG-CNN pixel candidates to 3D world-frame grasp poses."""
import numpy as np
from scipy.spatial.transform import Rotation

from sim.camera_setup import backproject_pixel, camera_to_world


def pixels_to_width_meters(width_px: float, depth_m: float, fx: float) -> float:
    """Pinhole conversion of gripper width pixels to metres at given depth."""
    if fx < 1e-6 or depth_m < 1e-6:
        return 0.0
    return float(width_px * depth_m / fx)


def candidate_to_camera_frame(
    candidate: dict,
    K_matrix: np.ndarray,
    crop_offset: tuple,
    crop_to_orig_scale: float = 1.0,
):
    """Backproject (px, py, depth) from quality-map space to the camera frame.

    Pixel coords live in the cropped+resized quality-map space. scale back to
    cropped pixels, add crop offset to recover full-image pixels, then backproject.

    Returns (point_cam, grasp_angle_cam, width_m).
    """
    px_crop = candidate["px"] * crop_to_orig_scale
    py_crop = candidate["py"] * crop_to_orig_scale
    x0, y0  = crop_offset
    px_full = px_crop + x0
    py_full = py_crop + y0

    depth_m = candidate["depth_m"]
    point_cam = backproject_pixel(px_full, py_full, depth_m, K_matrix)

    fx = K_matrix[0, 0]
    width_m = pixels_to_width_meters(candidate["width_px"], depth_m, fx)

    return point_cam, candidate["angle_rad"], width_m


def candidate_to_world_frame(
    candidate: dict,
    K_matrix: np.ndarray,
    crop_offset: tuple,
    cam_pos_world: np.ndarray,
    cam_quat_world: np.ndarray,
    crop_to_orig_scale: float = 1.0,
):
    """Full pipeline: pixel candidate to world-frame (grasp_pos, grasp_quat wxyz)."""
    point_cam, grasp_angle_cam, width_m = candidate_to_camera_frame(
        candidate, K_matrix, crop_offset, crop_to_orig_scale
    )

    grasp_pos_world = camera_to_world(
        point_cam, cam_pos_world, cam_quat_world
    )

    # apply in-plane grasp rotation about camera Z, then compose with cam orientation
    cw, cx, cy, cz = cam_quat_world
    r_cam = Rotation.from_quat([cx, cy, cz, cw])
    r_grasp_cam = Rotation.from_euler("z", grasp_angle_cam)

    r_grasp_world = r_cam * r_grasp_cam
    xyzw = r_grasp_world.as_quat()
    grasp_quat_world = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])  # wxyz

    return grasp_pos_world, grasp_quat_world
