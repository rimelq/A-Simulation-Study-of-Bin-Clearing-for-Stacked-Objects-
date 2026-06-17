"""Compute pre-grasp and lift poses offset from a target grasp pose."""
import numpy as np
from scipy.spatial.transform import Rotation


class PregraspPlanner:
    """Pre-grasp offset is along the gripper +Z (approach) axis."""

    def compute_pregrasp_pose(
        self,
        grasp_pos: np.ndarray,
        grasp_quat: np.ndarray,
        approach_dist: float = 0.10,
    ):
        """Return (pos, quat) offset approach_dist along the gripper approach axis."""
        grasp_pos  = np.asarray(grasp_pos,  dtype=np.float64)
        grasp_quat = np.asarray(grasp_quat, dtype=np.float64)

        # quat wxyz -> scipy xyzw
        w, x, y, z = grasp_quat
        r = Rotation.from_quat([x, y, z, w])

        # gripper +Z in world. for a top-down grasp this points down
        approach_axis_world = r.apply(np.array([0.0, 0.0, 1.0]))

        pregrasp_pos = grasp_pos - approach_axis_world * approach_dist
        pregrasp_quat = grasp_quat.copy()
        return pregrasp_pos, pregrasp_quat

    def compute_lift_pose(
        self,
        grasp_pos: np.ndarray,
        grasp_quat: np.ndarray,
        lift_height: float = 0.20,
    ):
        """Return (pos, quat) lift_height above grasp_pos along world Z."""
        grasp_pos = np.asarray(grasp_pos, dtype=np.float64)
        lift_pos  = grasp_pos + np.array([0.0, 0.0, lift_height])
        return lift_pos, grasp_quat.copy()
