"""Camera utilities: depth conversion, intrinsics, backprojection, camera-to-world transform."""
import numpy as np
from scipy.spatial.transform import Rotation


def depth_to_meters(depth_obs: np.ndarray, near: float, far: float) -> np.ndarray:
    """Convert robosuite depth buffer (values in [0,1]) to metric depth in meters."""
    depth_obs = np.asarray(depth_obs, dtype=np.float32)
    denominator = far - (far - near) * depth_obs
    # guard against the far-plane singularity (depth_obs == 1)
    denominator = np.where(np.abs(denominator) < 1e-8, 1e-8, denominator)
    depth_m = (near * far) / denominator
    return depth_m.astype(np.float32)


def get_camera_intrinsics(fov_deg: float, width: int, height: int) -> np.ndarray:
    """Compute 3x3 pinhole intrinsic matrix K from vertical FOV and image size."""
    fov_rad = np.deg2rad(fov_deg)
    fy = (height / 2.0) / np.tan(fov_rad / 2.0)
    fx = fy  # square pixels
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return K


def backproject_pixel(
    px: float,
    py: float,
    depth_m: float,
    K: np.ndarray,
) -> np.ndarray:
    """Backproject a pixel + metric depth into a 3D point in camera frame (+Z forward, +X right, +Y down)."""
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    X = (px - cx) * depth_m / fx
    Y = (py - cy) * depth_m / fy
    Z = depth_m
    return np.array([X, Y, Z], dtype=np.float64)


def camera_to_world(
    point_cam: np.ndarray,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
) -> np.ndarray:
    """Transform a 3D point from camera frame to world frame. cam_quat is (w,x,y,z)."""
    w, x, y, z = cam_quat
    r = Rotation.from_quat([x, y, z, w])  # scipy expects xyzw
    point_world = cam_pos + r.apply(point_cam)
    return point_world


def get_camera_extrinsics_from_obs(obs: dict, camera_name: str):
    """Extract (cam_pos, cam_quat_wxyz) from a robosuite obs dict."""
    pos = obs.get(f"{camera_name}_pos", np.zeros(3))
    xyzw = obs.get(f"{camera_name}_quat", np.array([0.0, 0.0, 0.0, 1.0]))
    cam_quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])  # xyzw -> wxyz
    return pos, cam_quat
