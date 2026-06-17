"""Move the robot to the fixed overhead sensing pose (teleport or OSC)."""
import numpy as np

try:
    import mujoco as _mujoco
    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False

from sim.sensing_pose import SENSING_JOINTS, SENSING_EEF_POS, SENSING_EEF_QUAT


class SensingPoseController:
    def __init__(self, env, n_steps: int = 100):
        self.env = env
        self.n_steps = n_steps

    def set_sensing_pose_direct(self):
        """Set robot joint qpos to SENSING_JOINTS. Must be called after env.reset()."""
        robot = self.env.robots[0]
        joint_names = robot.robot_joints

        for i, jnt_name in enumerate(joint_names):
            if i >= len(SENSING_JOINTS):
                break
            try:
                jid = self.env.sim.model.joint_name2id(jnt_name)
                qpos_adr = self.env.sim.model.jnt_qposadr[jid]
                self.env.sim.data.qpos[qpos_adr] = SENSING_JOINTS[i]
            except Exception as e:
                print(f"[SensingPoseController] Could not set joint '{jnt_name}': {e}")

        for i, jnt_name in enumerate(joint_names):
            if i >= len(SENSING_JOINTS):
                break
            try:
                jid = self.env.sim.model.joint_name2id(jnt_name)
                qvel_adr = self.env.sim.model.jnt_dofadr[jid]
                self.env.sim.data.qvel[qvel_adr] = 0.0
            except Exception:
                pass

        self._forward()

    def _forward(self):
        if _HAS_MUJOCO:
            raw_model = getattr(self.env.sim.model, "_model", self.env.sim.model)
            raw_data  = getattr(self.env.sim.data,  "_data",  self.env.sim.data)
            _mujoco.mj_forward(raw_model, raw_data)
        else:
            self.env.sim.forward()

    def move_to_sensing_pose_osc(self, n_steps: int = None) -> bool:
        """Cartesian OSC approach to SENSING_EEF_POS. Returns True if within tol."""
        if n_steps is None:
            n_steps = self.n_steps

        target_pos  = SENSING_EEF_POS.copy()
        target_quat = SENSING_EEF_QUAT.copy()  # wxyz

        kp = 5.0
        tol = 0.02

        for _ in range(n_steps):
            current_pos  = self.env.get_robot_eef_pos()
            delta_pos    = target_pos - current_pos
            dist = np.linalg.norm(delta_pos)

            if dist < tol:
                return True

            action_pos = np.clip(delta_pos * kp, -1.0, 1.0)
            action_ori = np.zeros(3)
            action_gripper = np.array([-1.0])

            action = np.concatenate([action_pos, action_ori, action_gripper])
            self.env.step(action)

        return False

    def move_to_sensing_pose(self) -> bool:
        return self.move_to_sensing_pose_osc(self.n_steps)
