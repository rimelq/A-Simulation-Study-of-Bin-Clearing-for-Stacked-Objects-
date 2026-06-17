"""Move the robot above the destination bin and release the gripper."""
import numpy as np

from sim.sensing_pose import DEST_BIN_POS


class DropExecutor:
    """Move to a drop position above the destination bin and open the gripper."""

    DROP_HEIGHT = 0.30  # m above bin bottom, >=15 cm above object during release

    def __init__(self, dest_bin_pos: np.ndarray = None, drop_height: float = None):
        if dest_bin_pos is None:
            dest_bin_pos = DEST_BIN_POS
        if drop_height is not None:
            self.DROP_HEIGHT = drop_height

        self.dest_pos = np.asarray(dest_bin_pos, dtype=np.float64)
        self.drop_pos = np.array([
            self.dest_pos[0],
            self.dest_pos[1],
            self.dest_pos[2] + self.DROP_HEIGHT,
        ])

    def execute(self, env, grasp_executor, kp: float = 5.0,
                max_steps: int = 80, n_release_steps: int = 30):
        """Move EEF to drop_pos and release gripper. Returns True if reached."""
        target_pos  = self.drop_pos.copy()
        target_quat = np.array([0.0, 1.0, 0.0, 0.0])  # gripper down (wxyz)
        tol = 0.03

        reached = False
        for _ in range(max_steps):
            current_pos = env.get_robot_eef_pos()
            delta_pos = target_pos - current_pos
            dist = np.linalg.norm(delta_pos)

            if dist < tol:
                reached = True
                break

            action_pos = np.clip(delta_pos * kp, -1.0, 1.0)
            action = np.concatenate([action_pos, np.zeros(3), [1.0]])  # keep closed
            env.step(action)

        grasp_executor.open_gripper(env, n_steps=n_release_steps)
        return reached

    def get_drop_position(self) -> np.ndarray:
        return self.drop_pos.copy()
