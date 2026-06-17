"""Full pick-and-place execution coordinating IK, gripper and sub-controllers.

Phases:
  0. IK-teleport arm to pregrasp (bypasses sensing-pose singularity)
  1. Pregrasp short physics settle with arm locked
  2. Descend to grasp pose
  3. Close gripper
  4. Lift object
  5. Move to above destination bin
  6. Open gripper (release)
  7. Return to sensing pose
"""
import os
import numpy as np
from scipy.spatial.transform import Rotation

try:
    import mujoco as _mujoco
    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False

from control.pregrasp_planner import PregraspPlanner
from control.grasp_executor import GraspExecutor
from control.drop_executor import DropExecutor
from control.grasp_success_predicate import evaluate_grasp


# robosuite OSC default: action=1 -> 0.05 m delta per step
_OSC_POS_SCALE = 0.05

# Verbose IK / wrist-patch prints, off by default for cluster runs.
_VERBOSE_IK = os.environ.get("PPP_VERBOSE_IK", "0") == "1"


class PickPlacePrimitive:
    """Fixed pick-and-place execution primitive."""

    def __init__(self, env, dest_bin_pos: np.ndarray, sensing_controller=None):
        self.env = env
        self.dest_bin_pos = np.asarray(dest_bin_pos, dtype=np.float64)
        self.pregrasp_planner = PregraspPlanner()
        self.grasp_executor   = GraspExecutor()
        self.drop_executor    = DropExecutor(dest_bin_pos)
        self.sensing_ctrl     = sensing_controller
        # Set `contact_log_path` before execute() to enable per-step contact diagnostics.
        self.contact_log_path = None
        self._contact_log_file = None
        self._contact_geom2item = {}
        self._contact_gripper_gids = set()
        self._contact_prev_pos = {}
        self._contact_summary = {
            "gripper_touched": {},
            "target_touched":  {},
            "moved_no_contact": {},
        }

    def execute(
        self,
        grasp_pos: np.ndarray,
        grasp_quat: np.ndarray,
        target_item_name: str = None,
        pregrasp_dist: float = 0.12,
        lift_height: float   = 0.20,
        max_steps_per_phase: int = 80,
        geom_xy_tol: float   = 0.035,
        geom_z_tol:  float   = 0.04,
    ) -> dict:
        """Execute full pick-place. Returns dict with success / phase_reached / counts."""
        grasp_pos  = np.asarray(grasp_pos,  dtype=np.float64)
        grasp_quat = np.asarray(grasp_quat, dtype=np.float64)

        total_steps  = 0
        phase_reached = "start"
        phase_log = []

        def _log_motion_phase(name, target_pos, n_steps):
            eef_before = self.env.get_robot_eef_pos()
            steps = self._move_eef_to_target(
                target_pos, grasp_quat_local,
                tol=_tol_map.get(name, 0.015),
                max_steps=max_steps_per_phase,
                gripper_action=_grip_map.get(name, -1.0),
            )
            eef_after = self.env.get_robot_eef_pos()
            phase_log.append({
                "phase":       name,
                "target_z":    round(float(target_pos[2]), 4),
                "eef_before_z": round(float(eef_before[2]), 4),
                "eef_after_z": round(float(eef_after[2]), 4),
                "steps":       steps,
            })
            return steps

        _tol_map  = {"pregrasp": 0.015, "grasp_pose": 0.01,
                     "lifted": 0.02,   "above_dest": 0.03}
        _grip_map = {"pregrasp": -1.0, "grasp_pose": -1.0,
                     "lifted": 1.0,    "above_dest": 1.0}

        grasp_quat_local = grasp_quat

        # UR5e XML caps wrist actuators at +-28 Nm. gravity at the grasp workspace
        # saturates the OSC and drifts the arm upward. Patch once per execute().
        self._patch_wrist_limits()

        # Suppress the test script's env.step video capture during execute()
        # frames are captured only AFTER each re-pin / clamp pass so every frame
        # is a stable, snapped-back state (eliminates source-pile flicker).
        self._suppress_step_capture = True
        def _post_pin_capture():
            fh = getattr(self, "frame_hook", None)
            if fh is not None:
                try:
                    fh()
                except Exception:
                    pass

        self._contact_log_file = None
        if self.contact_log_path:
            self._contact_log_setup(target_item_name)

        # Contact regime: phases 0-2 ghost (palm body would detonate thin
        # rectangles), phase 3 grasp_target_only (pads physically grip target)
        # phases 4-6 ghost again (OSC can't hold pose under load at low Z
        # carry kinematically via existing attach-and-track).
        if _HAS_MUJOCO:
            self._apply_contact_mode("ghost")
        print("  [Contacts] mode=ghost (gripper touches nothing) for approach+descent")

        # Snapshot every item's freejoint qpos. Partition into:
        # source_snapshots, re-pinned during close + lift + transport so the
        # close-phase impulses can't fling them out of the bowl.
        # dest_snapshots, NEVER re-pinned. obey real gravity/friction so
        # they settle naturally instead of being frozen mid-air.
        # The grasped target is exempt from re-pinning in phases 4-6.
        _item_snapshots = self._snapshot_item_poses() if _HAS_MUJOCO else {}
        _src_bin_xy = (self.env.get_src_bin_world_pos()[0], self.env.get_src_bin_world_pos()[1]) if _HAS_MUJOCO else (0.0, 0.0)
        from sim.sensing_pose import BIN_HALF_SIZE as _BHS_EXEC
        _src_snapshots = {}
        _dst_snapshots = {}
        if _HAS_MUJOCO:
            for _nm, _qp in _item_snapshots.items():
                if (abs(float(_qp[0]) - _src_bin_xy[0]) < _BHS_EXEC[0]
                    and abs(float(_qp[1]) - _src_bin_xy[1]) < _BHS_EXEC[1]):
                    _src_snapshots[_nm] = _qp
                else:
                    _dst_snapshots[_nm] = _qp
        print(f"  [Snapshot] Captured {len(_item_snapshots)} item poses  "
              f"(source-bin={len(_src_snapshots)}, dest-bin/other={len(_dst_snapshots)})")

        # No tracked item until Phase 4.
        self._tracked_item   = None
        self._tracked_offset = None

        # Phase 0: IK teleport to pregrasp position
        # Sensing pose at z=1.40 hits a kinematic singularity around z=1.20 with OSC
        # Jacobian IK teleport bypasses this.
        pregrasp_pos, pregrasp_quat = self.pregrasp_planner.compute_pregrasp_pose(
            grasp_pos, grasp_quat, approach_dist=pregrasp_dist
        )
        grasp_quat_local = pregrasp_quat

        eef_before_ik = self.env.get_robot_eef_pos()
        print(f"  [Phase 0] Smooth approach: EEF start z={eef_before_ik[2]:.4f}m -> target z={pregrasp_pos[2]:.4f}m")
        # 6-DoF IK so jaws arrive aligned to the item short axis, descended in
        # interpolated waypoints for a smooth wrist-camera POV.
        self._smooth_teleport(pregrasp_pos, grasp_quat, n_steps=14)
        eef_after_ik = self.env.get_robot_eef_pos()
        ik_err = float(np.linalg.norm(pregrasp_pos - eef_after_ik))
        print(f"  [Phase 0] IK done: EEF z={eef_after_ik[2]:.4f}m  err={ik_err:.4f}m")
        phase_log.append({"phase": "ik_teleport",
                           "target_z": round(float(pregrasp_pos[2]), 4),
                           "eef_before_z": round(float(eef_before_ik[2]), 4),
                           "eef_after_z":  round(float(eef_after_ik[2]), 4),
                           "steps": 0})
        total_steps += 0
        phase_reached = "ik_teleport"

        # Phase 1: Pregrasp short physics settle with arm locked
        # OSC fine-tune at this height drove the arm UP (gravity > torque) and
        # rotated the gripper away from planned yaw. Replace with locked-arm
        # physics steps so contacts settle without OSC corrupting the IK pose.
        eef_b1 = self.env.get_robot_eef_pos()
        _PREGRASP_SETTLE = 5
        # Partial-open the gripper to the target's short width + 1.2 cm clearance.
        # Full open (~8 cm) makes the pads sweep ~5 cm before closing on a 3 cm
        # item, brushing neighbours. partial open enters the pile already roughly
        # at target size and travels only a few mm to grip.
        _open_amount = -1.0  # fully open fallback
        if _HAS_MUJOCO and target_item_name is not None:
            try:
                _tbid = self.env.sim.model.body_name2id(target_item_name)
                _raw_model_ow = getattr(self.env.sim.model, "_model", self.env.sim.model)
                # geom_aabb = (cx, cy, cz, hx, hy, hz). local frame matches body frame here.
                _short_half = None
                for _gid in range(_raw_model_ow.ngeom):
                    if self.env.sim.model.geom_bodyid[_gid] != _tbid:
                        continue
                    _aabb = _raw_model_ow.geom_aabb[_gid]
                    _hx, _hy = float(_aabb[3]), float(_aabb[4])
                    _short_half = min(_hx, _hy)
                    break
                if _short_half is not None and _short_half > 0:
                    # PandaGripper action in [-1, +1] ~ gap in [~0.08 m, 0.0 m]:
                    # gap_m ~ (1 - action) * 0.04 => action = 1 - gap/0.04.
                    _desired_gap = 2.0 * _short_half + 0.012
                    _desired_gap = float(np.clip(_desired_gap, 0.015, 0.080))
                    _open_amount = float(np.clip(1.0 - _desired_gap / 0.040, -1.0, 0.5))
                    print(f"  [Pregrasp] target half-width={_short_half*1000:.1f} mm  "
                          f"-> pad gap target={_desired_gap*1000:.1f} mm  "
                          f"-> gripper action={_open_amount:+.2f}")
            except Exception as _e:
                print(f"  [Pregrasp] couldn't size target ({_e}); using full open")
        _open_action = np.zeros(7)
        _open_action[-1] = _open_amount
        if _HAS_MUJOCO:
            raw_model_1 = getattr(self.env.sim.model, "_model", self.env.sim.model)
            raw_data_1  = getattr(self.env.sim.data,  "_data",  self.env.sim.data)
            qpos_idxs_1, qvel_idxs_1 = self._get_arm_indices()
            locked_qpos_1 = np.array([raw_data_1.qpos[idx] for idx in qpos_idxs_1])
            # Phase-1 only re-pin: sub-mm CSV overlaps would otherwise explode
            # in the first env.step. Re-pinning is dropped after Phase 3 close.
            for _stepi in range(_PREGRASP_SETTLE):
                self.env.step(_open_action)
                for k, idx in enumerate(qpos_idxs_1):
                    raw_data_1.qpos[idx] = locked_qpos_1[k]
                for idx in qvel_idxs_1:
                    raw_data_1.qvel[idx] = 0.0
                self._repin_items(_src_snapshots)
                _mujoco.mj_forward(raw_model_1, raw_data_1)
                self.env.robots[0].controller.update(force=True)
                self.env.robots[0].controller.reset_goal()
                # log AFTER mj_forward -> contact state matches the video frame
                self._log_contacts("pregrasp_settle", _stepi, target_item_name)
                _post_pin_capture()
        eef_a1 = self.env.get_robot_eef_pos()
        phase_log.append({"phase": "pregrasp",
                           "target_z": round(float(pregrasp_pos[2]), 4),
                           "eef_before_z": round(float(eef_b1[2]), 4),
                           "eef_after_z":  round(float(eef_a1[2]), 4),
                           "steps": _PREGRASP_SETTLE})
        total_steps  += _PREGRASP_SETTLE
        phase_reached = "pregrasp"

        # Phase 2: descent to grasp position (ghost contacts)
        eef_b2 = self.env.get_robot_eef_pos()
        print(f"  [Phase 2] Smooth descent: EEF z={eef_b2[2]:.4f}m -> target z={grasp_pos[2]:.4f}m")
        if _HAS_MUJOCO:
            self._smooth_teleport(grasp_pos, grasp_quat, n_steps=12)
            raw_data_2 = getattr(self.env.sim.data, "_data", self.env.sim.data)
            qpos_idxs_2, qvel_idxs_2 = self._get_arm_indices()
            for idx in qvel_idxs_2:
                raw_data_2.qvel[idx] = 0.0
            raw_model_2 = getattr(self.env.sim.model, "_model", self.env.sim.model)
            _mujoco.mj_forward(raw_model_2, raw_data_2)
            self.env.robots[0].controller.update(force=True)
            self.env.robots[0].controller.reset_goal()
            print(f"  [Phase 2] Descent done")
        else:
            self._teleport_arm_to_pos(grasp_pos, target_quat=grasp_quat)
        eef_a2 = self.env.get_robot_eef_pos()
        ik_err2 = float(np.linalg.norm(grasp_pos - eef_a2))
        print(f"  [Phase 2] Done: EEF z={eef_a2[2]:.4f}m  err={ik_err2:.4f}m")
        phase_log.append({"phase": "grasp_pose",
                           "target_z": round(float(grasp_pos[2]), 4),
                           "eef_before_z": round(float(eef_b2[2]), 4),
                           "eef_after_z":  round(float(eef_a2[2]), 4),
                           "steps": 0})
        phase_reached = "grasp_pose"

        # Phase 3: Close gripper with arm locked
        # At z~0.90 m, gravity > 150 Nm wrist patch. arm drifts 22 cm in 30 free
        # steps. Lock arm qpos every step so EEF stays put while gripper closes.
        # Mode: grasp_target_only, pads only collide with the target, so closing
        # pads pass cleanly through neighbours (eliminates outward-jolt glitch).
        if _HAS_MUJOCO:
            self._apply_contact_mode("grasp_target_only", target_item_name=target_item_name)
            print(f"  [Contacts] mode=grasp_target_only (pads <-> '{target_item_name}' only) for close")
        # 12 steps: pads contact target within a few steps, then freeze on contact
        # more steps would just be held frozen frames.
        _CLOSE_STEPS = 12
        close_action = np.zeros(7); close_action[-1] = 1.0
        eef_b3 = self.env.get_robot_eef_pos()
        print(f"    [GraspClose] EEF (wrist) before close: xyz={eef_b3}  z={eef_b3[2]:.4f}m")
        _NEAR_RADIUS_CLOSE = 0.08
        _V_LIN_CLOSE       = 0.4
        _V_ANG_CLOSE       = 8.0
        _MAX_SHOVE_CLOSE   = 0.025
        # Floor at z=TABLE_HEIGHT (=0.80). rim ~0.885. Cap centre Z at 0.85
        # so a bumped neighbour can't ride the bowl wall and escape.
        from sim.robosuite_env import _TABLE_HEIGHT as _SRC_FLOOR_Z, OBJECT_SCALE as _OBJECT_SCALE, _BIN_WALL_SCALE
        # Wall clamp: centre can't get closer to a wall than its rotation-worst-case
        # half-extent. Using half-diagonal (0.027 * sqrt(2) ~= 0.038) accounts for
        # any rotation, and we use the actual STL wall half-extent (0.127 m), NOT
        # the looser BIN_HALF_SIZE (0.13) perception uses.
        _SRC_FLOOR_Z = float(_SRC_FLOOR_Z)
        _Z_CAP_CLOSE = _SRC_FLOOR_Z + 0.050
        _ITEM_HALF_XY = 0.040
        _WALL_HALF    = 0.5087 * _OBJECT_SCALE   # 0.127 m
        _SRC_X = float(self.env.get_src_bin_world_pos()[0])
        _SRC_Y = float(self.env.get_src_bin_world_pos()[1])
        _WALL_X_MIN = _SRC_X - _WALL_HALF + _ITEM_HALF_XY
        _WALL_X_MAX = _SRC_X + _WALL_HALF - _ITEM_HALF_XY
        _WALL_Y_MIN = _SRC_Y - _WALL_HALF + _ITEM_HALF_XY
        _WALL_Y_MAX = _SRC_Y + _WALL_HALF - _ITEM_HALF_XY
        _FLOOR_Z_CLAMP = _SRC_FLOOR_Z + 0.027   # centre >= floor + side-half
        if _HAS_MUJOCO:
            raw_model_3 = getattr(self.env.sim.model, "_model", self.env.sim.model)
            raw_data_3  = getattr(self.env.sim.data,  "_data",  self.env.sim.data)
            qpos_idxs_3, qvel_idxs_3 = self._get_arm_indices()
            locked_qpos = np.array([raw_data_3.qpos[idx] for idx in qpos_idxs_3])
            # Re-pin EVERY non-target source item: in grasp_target_only mode the
            # gripper provably can't contact non-targets, so leaving them free
            # just lets gravity drift them a few mm per step (the visible glitch).
            far_snap = dict(_item_snapshots)
            print(f"  [Phase 3] close: ALL {len(far_snap)} source items re-pinned "
                  f"(incl. target '{target_item_name}'), no item left free, no drift")

            # Finger-freeze-on-contact: once pads hit the (re-pinned, infinitely-
            # rigid) target the actuator keeps pressing -> MuJoCo soft contact
            # oscillates the pad position -> visible glitch. Fix: 2 steps after
            # first pad<->target contact, freeze finger qpos for the rest of close.
            finger_qadr = []
            for jn in ("gripper0_finger_joint1", "gripper0_finger_joint2"):
                try:
                    jid = self.env.sim.model.joint_name2id(jn)
                    finger_qadr.append((int(self.env.sim.model.jnt_qposadr[jid]),
                                        int(self.env.sim.model.jnt_dofadr[jid])))
                except Exception:
                    pass
            grip_gids = set(self._gripper_geom_ids())
            target_gids = set()
            if target_item_name is not None:
                try:
                    _tbid = self.env.sim.model.body_name2id(target_item_name)
                    target_gids = {g for g in range(raw_model_3.ngeom)
                                   if self.env.sim.model.geom_bodyid[g] == _tbid}
                except Exception:
                    pass
            _contact_step  = None
            _frozen_finger = None

            for _stepi in range(_CLOSE_STEPS):
                self.env.step(close_action)
                for k, idx in enumerate(qpos_idxs_3):
                    raw_data_3.qpos[idx] = locked_qpos[k]
                for idx in qvel_idxs_3:
                    raw_data_3.qvel[idx] = 0.0
                if _frozen_finger is not None:
                    for (qa, va), val in zip(finger_qadr, _frozen_finger):
                        raw_data_3.qpos[qa] = val
                        raw_data_3.qvel[va] = 0.0
                self._repin_items(far_snap)
                _mujoco.mj_forward(raw_model_3, raw_data_3)
                self.env.robots[0].controller.update(force=True)
                self.env.robots[0].controller.reset_goal()

                # detect first gripper<->target contact (fresh after mj_forward)
                if _contact_step is None and grip_gids and target_gids:
                    for ci in range(int(raw_data_3.ncon)):
                        c = raw_data_3.contact[ci]
                        g1, g2 = int(c.geom1), int(c.geom2)
                        if ((g1 in grip_gids and g2 in target_gids) or
                                (g2 in grip_gids and g1 in target_gids)):
                            _contact_step = _stepi
                            break
                # capture finger qpos 2 steps after first contact
                if (_frozen_finger is None and _contact_step is not None
                        and _stepi >= _contact_step + 2):
                    _frozen_finger = [float(raw_data_3.qpos[qa]) for qa, _ in finger_qadr]

                self._log_contacts("close", _stepi, target_item_name)
                _post_pin_capture()
            if _contact_step is not None:
                print(f"  [Phase 3] fingers froze ~step {_contact_step + 3} "
                      f"(first gripper<->target contact at step {_contact_step}), no pad jitter after")
            else:
                print(f"  [Phase 3] no gripper<->target contact detected during close")
        else:
            for _ in range(_CLOSE_STEPS):
                self.env.step(close_action)
        eef_a3 = self.env.get_robot_eef_pos()
        print(f"    [GraspClose] EEF (wrist) after close:  xyz={eef_a3}  z={eef_a3[2]:.4f}m")
        # target was frozen during close -> clean per-pick boolean
        physics_grasp_success = self.grasp_executor.check_grasp_success(
            self.env, self.env.get_obj_names()
        )
        # Refresh snapshots to POST-close positions so the small close-phase
        # nudges to neighbours persist instead of being snapped back.
        if _HAS_MUJOCO:
            for _name in list(_item_snapshots.keys()):
                if _name == target_item_name:
                    continue
                try:
                    _bid = self.env.sim.model.body_name2id(_name)
                    for _j in range(raw_model_3.njnt):
                        if raw_model_3.jnt_bodyid[_j] == _bid:
                            _qa = int(raw_model_3.jnt_qposadr[_j])
                            _item_snapshots[_name] = np.array(raw_data_3.qpos[_qa:_qa + 7],
                                                              dtype=np.float64)
                            break
                except Exception:
                    pass
        # Lift/transport: GHOST. Hand body still overlaps thin tablets at this Z
        # (grasp_pose z=0.93, hand z~1.03, tablets z~0.82-0.85). switching to
        # 'normal' here would fire a big impulse on next env.step (close ran in
        # grasp_pads_only so the hand overlap was silent). Ghost mode lets the
        # closed gripper sweep up out of the pile carrying the item kinematically.
        if _HAS_MUJOCO:
            self._apply_contact_mode("ghost")
            print("  [Contacts] mode=ghost (clear of source pile) for lift/transport")

        # Fresh snapshot post-close so phases 4/5/6 re-pin at POST-CLOSE poses
        # not the original CSV state (which had sub-mm overlaps that would explode
        # without re-pin). Keeps small close-phase nudges persistent into next pick.
        if _HAS_MUJOCO:
            _src_post_close = self._snapshot_item_poses()
            _src_post_close = {nm: qp for nm, qp in _src_post_close.items()
                               if nm in _src_snapshots}
        else:
            _src_post_close = {}

        # Geometric grasp-success: the perception target is inside the jaw bbox
        # at close-time. Decouples delivery from MuJoCo thin-rectangle physics
        # clean deterministic signal for RL.
        geometric_grasp_success = False
        geom_dxy = float('nan')
        geom_dz  = float('nan')
        if target_item_name is not None and _HAS_MUJOCO:
            try:
                bid = self.env.sim.model.body_name2id(target_item_name)
                item_pos = np.array(self.env.sim.data.body_xpos[bid])
                pad_z = eef_a3[2] - 0.097
                geom_dxy = float(np.linalg.norm(item_pos[:2] - eef_a3[:2]))
                geom_dz  = float(abs(item_pos[2] - pad_z))
                geometric_grasp_success = (geom_dxy < geom_xy_tol) and (geom_dz < geom_z_tol)
                print(f"    [GeomCheck] target='{target_item_name}'  "
                      f"dxy={geom_dxy*1000:.1f}mm (tol {geom_xy_tol*1000:.0f}mm)  "
                      f"dz={geom_dz*1000:.1f}mm (tol {geom_z_tol*1000:.0f}mm)  "
                      f"-> geometric_success={geometric_grasp_success}")
            except Exception as e:
                print(f"    [GeomCheck] error: {e}")

        grasp_success = physics_grasp_success or geometric_grasp_success

        phase_log.append({"phase": "gripper_close",
                           "target_z": "-", "eef_before_z": "-", "eef_after_z": "-",
                           "steps": _CLOSE_STEPS})
        total_steps   += _CLOSE_STEPS
        phase_reached  = "gripper_closed"

        # Phase 4: IK teleport to lift position + object tracking
        # OSC can't maintain XY while lifting from z~0.90 (wrist torques saturate
        # even at 150 Nm. drift 16 cm in X in 50 steps). Teleport via IK, move
        # the grasped object to maintain its gripper-relative offset.
        _LIFT_SETTLE = 30
        lift_action  = np.zeros(7)
        lift_action[-1] = 1.0

        lift_pos, lift_quat = self.pregrasp_planner.compute_lift_pose(
            grasp_pos, grasp_quat, lift_height=lift_height
        )
        eef_b4 = self.env.get_robot_eef_pos()
        print(f"  [Phase 4] IK lift: EEF z={eef_b4[2]:.4f}m -> lift_pos z={lift_pos[2]:.4f}m")

        # Carry decision: the PERCEPTION TARGET is authoritative whenever the
        # geometric jaw-box check confirmed it. Contact-only detection picks the
        # closest brushed neighbour in clutter (bug #4).
        grasped_obj, obj_offset = None, None
        if geometric_grasp_success and target_item_name is not None and _HAS_MUJOCO:
            try:
                bid = self.env.sim.model.body_name2id(target_item_name)
                opos = np.array(self.env.sim.data.body_xpos[bid])
                grasped_obj = target_item_name
                obj_offset  = opos - eef_b4
                print(f"  [Phase 4] Carrying perception target: {grasped_obj}  "
                      f"EEF-offset dz={obj_offset[2]:.4f}m")
            except Exception as e:
                print(f"  [Phase 4] Could not look up target '{target_item_name}': {e}")
        if grasped_obj is None:
            grasped_obj, obj_offset = self._find_grasped_object_and_offset(eef_b4)
            if grasped_obj:
                print(f"  [Phase 4] Carrying contact-detected object: {grasped_obj}  "
                      f"EEF-offset dz={obj_offset[2]:.4f}m")

        if grasped_obj is None:
            print(f"  [Phase 4] No grasped object identified; "
                  f"relying on friction during settle")

        # Lift in interpolated waypoints, grasped item tracks gripper through
        # every step, so it rises with the jaws on the video.
        if grasped_obj is not None:
            self._tracked_item   = grasped_obj
            self._tracked_offset = obj_offset.copy()
        self._smooth_teleport(lift_pos, grasp_quat, n_steps=10)
        eef_a4_ik = self.env.get_robot_eef_pos()
        print(f"  [Phase 4] Lift done: EEF z={eef_a4_ik[2]:.4f}m")

        if grasped_obj is not None and _HAS_MUJOCO:
            new_obj_pos = eef_a4_ik + obj_offset
            self._move_object_qpos(grasped_obj, new_obj_pos)
            print(f"  [Phase 4] Object {grasped_obj} moved to z={new_obj_pos[2]:.4f}m")

        # Settle: gripper in ghost mode can't hold the item with friction at
        # this Z. keep target kinematically anchored to the gripper. ONLY cheat
        # remaining in lift/transport, actual RELEASE (Phase 6) is fully physical.
        target_lift_obj_pos = (eef_a4_ik + obj_offset) if grasped_obj is not None else None
        if _HAS_MUJOCO:
            raw_model_4 = getattr(self.env.sim.model, "_model", self.env.sim.model)
            raw_data_4  = getattr(self.env.sim.data,  "_data",  self.env.sim.data)
            qpos_idxs_4, qvel_idxs_4 = self._get_arm_indices()
            locked_qpos_4 = np.array([raw_data_4.qpos[idx] for idx in qpos_idxs_4])
            for _stepi in range(_LIFT_SETTLE):
                self.env.step(lift_action)
                for k, idx in enumerate(qpos_idxs_4):
                    raw_data_4.qpos[idx] = locked_qpos_4[k]
                for idx in qvel_idxs_4:
                    raw_data_4.qvel[idx] = 0.0
                # Hold fingers at the Phase-3 grip position (in ghost mode the
                # actuator would otherwise keep closing the pads while airborne).
                if _frozen_finger is not None:
                    for (qa, va), val in zip(finger_qadr, _frozen_finger):
                        raw_data_4.qpos[qa] = val
                        raw_data_4.qvel[va] = 0.0
                # Re-pin to POST-close snapshot, preserves close-phase nudges.
                self._repin_items(_src_post_close, except_name=grasped_obj)
                if grasped_obj is not None and target_lift_obj_pos is not None:
                    self._move_object_qpos(grasped_obj, target_lift_obj_pos)
                _mujoco.mj_forward(raw_model_4, raw_data_4)
                self.env.robots[0].controller.update(force=True)
                self.env.robots[0].controller.reset_goal()
                self._log_contacts("lift_settle", _stepi, target_item_name)
                _post_pin_capture()
        else:
            for _ in range(_LIFT_SETTLE):
                self.env.step(lift_action)

        eef_a4 = self.env.get_robot_eef_pos()
        print(f"  [Phase 4] Settled: EEF z={eef_a4[2]:.4f}m")
        phase_log.append({"phase": "lifted",
                           "target_z": round(float(lift_pos[2]), 4),
                           "eef_before_z": round(float(eef_b4[2]), 4),
                           "eef_after_z":  round(float(eef_a4[2]), 4),
                           "steps": _LIFT_SETTLE})
        total_steps  += _LIFT_SETTLE
        phase_reached = "lifted"

        # Phase 5: IK teleport to above dest bin + object tracking
        # No locked-arm settle here, Phase 6 does a *physical* release.

        drop_pos = self.drop_executor.get_drop_position()
        eef_b5   = self.env.get_robot_eef_pos()
        print(f"  [Phase 5] IK transport: EEF ({eef_b5[0]:.3f},{eef_b5[1]:.3f},"
              f"{eef_b5[2]:.4f}) -> drop_pos ({drop_pos[0]:.3f},{drop_pos[1]:.3f},"
              f"{drop_pos[2]:.4f})")

        # Reuse the Phase-4 grasped object directly (contact detection misses it:
        # 8 cm fallback threshold < 9.94 cm EEF-to-object distance).
        grasped_obj5  = grasped_obj
        obj_offset5   = None
        if grasped_obj5 is not None and _HAS_MUJOCO:
            try:
                bid  = self.env.sim.model.body_name2id(grasped_obj5)
                opos = np.array(self.env.sim.data.body_xpos[bid])
                dist = float(np.linalg.norm(opos - eef_b5))
                obj_offset5 = opos - eef_b5
                print(f"  [Phase 5] Object {grasped_obj5}: dist={dist:.4f}m  "
                      f"offset dz={obj_offset5[2]:.4f}m")
                if dist > 0.25:
                    print(f"  [Phase 5] Object too far ({dist:.4f}m), grasp likely lost")
                    grasped_obj5 = None
            except Exception as e:
                print(f"  [Phase 5] Object lookup failed: {e}")
                grasped_obj5 = None

        if grasped_obj5 is not None and obj_offset5 is not None:
            self._tracked_item   = grasped_obj5
            self._tracked_offset = obj_offset5.copy()

        # Transport across, weak null-space (large shoulder-pan change needed).
        self._smooth_teleport(drop_pos, grasp_quat, n_steps=16, null_gain=0.02)
        eef_a5_ik = self.env.get_robot_eef_pos()
        ik_err5   = float(np.linalg.norm(drop_pos - eef_a5_ik))
        print(f"  [Phase 5] Transport done above dest: EEF z={eef_a5_ik[2]:.4f}m  err={ik_err5:.4f}m")

        # Dest-bin slot from the execute-start partition (_dst_snapshots).
        # Inline body_xpos scan was returning 0 even with a placed item present
        # so the new item landed on top and knocked it out.
        dest_floor_z = self.dest_bin_pos[2] + 0.020
        n_in_dest_already = len(_dst_snapshots) if _HAS_MUJOCO else 0
        # 3x3 grid, 7 cm spacing, items ~5.4 cm wide, 5 cm slots overlapped and
        # knocked neighbours out. 7 cm leaves ~1.6 cm gap and outer ring at offset
        # 7 cm + half 2.7 cm = 9.7 cm stays inside 13 cm bin half.
        row, col = n_in_dest_already // 3, n_in_dest_already % 3
        off_x = (col - 1) * 0.07
        off_y = (row - 1) * 0.07
        slot_xy = np.array([self.dest_bin_pos[0] + off_x, self.dest_bin_pos[1] + off_y])
        print(f"  [Phase 5] {n_in_dest_already} item(s) already in dest bin "
              f"-> this item goes to slot #{n_in_dest_already} at "
              f"({slot_xy[0]:+.3f},{slot_xy[1]:+.3f})")

        # Lower above the slot but stop high enough that (a) the released item
        # falls freely a few cm for a realistic drop, AND (b) the still-held
        # item doesn't overlap in Z with already-placed items resting on the
        # bin floor (cubes are 5.4 cm tall). Wrist 22 cm above floor -> pads
        # ~12.3 cm above floor -> item bottom ~9 cm above floor.
        place_pos = np.array([slot_xy[0], slot_xy[1], self.dest_bin_pos[2] + 0.220])
        if grasped_obj5 is not None and obj_offset5 is not None:
            self._tracked_offset = obj_offset5.copy()
        self._smooth_teleport(place_pos, grasp_quat, n_steps=8, null_gain=0.02)
        eef_a5_ik = self.env.get_robot_eef_pos()
        print(f"  [Phase 5] Lowered above slot #{n_in_dest_already}: EEF z={eef_a5_ik[2]:.4f}m  "
              f"(item will fall ~{(eef_a5_ik[2] - 0.097 - 0.027 - self.dest_bin_pos[2])*100:.1f} cm under gravity)")

        # Stop tracking, Phase 6 will open jaws and let physics drop the item.
        # NO kinematic teleport to a slot, NO re-pin: dest-bin layout is physics.
        self._tracked_item   = None
        self._tracked_offset = None

        # Arm-locked settle above the slot. Grasped item is KINEMATICALLY anchored
        # at its current pose (between closed pads, 12 cm above dest-bin floor)
        # so it doesn't fall before Phase 6 releases. Source-bin items pinned
        # dest-bin items continue to settle under real gravity.
        held_obj_pos = None
        if grasped_obj5 is not None and _HAS_MUJOCO:
            try:
                bid = self.env.sim.model.body_name2id(grasped_obj5)
                held_obj_pos = np.array(self.env.sim.data.body_xpos[bid], dtype=np.float64)
            except Exception:
                held_obj_pos = None
        _DROP_SETTLE = 20
        drop_action  = np.zeros(7)
        drop_action[-1] = 1.0
        if _HAS_MUJOCO:
            raw_model_5 = getattr(self.env.sim.model, "_model", self.env.sim.model)
            raw_data_5  = getattr(self.env.sim.data,  "_data",  self.env.sim.data)
            qpos_idxs_5, qvel_idxs_5 = self._get_arm_indices()
            locked_qpos_5 = np.array([raw_data_5.qpos[idx] for idx in qpos_idxs_5])
            for _stepi in range(_DROP_SETTLE):
                self.env.step(drop_action)
                for k, idx in enumerate(qpos_idxs_5):
                    raw_data_5.qpos[idx] = locked_qpos_5[k]
                for idx in qvel_idxs_5:
                    raw_data_5.qvel[idx] = 0.0
                # carry Phase-3 grip pose unchanged
                if _frozen_finger is not None:
                    for (qa, va), val in zip(finger_qadr, _frozen_finger):
                        raw_data_5.qpos[qa] = val
                        raw_data_5.qvel[va] = 0.0
                self._repin_items(_src_post_close, except_name=grasped_obj5)
                if grasped_obj5 is not None and held_obj_pos is not None:
                    self._move_object_qpos(grasped_obj5, held_obj_pos)
                _mujoco.mj_forward(raw_model_5, raw_data_5)
                self.env.robots[0].controller.update(force=True)
                self.env.robots[0].controller.reset_goal()
                self._log_contacts("drop_settle", _stepi, target_item_name)
                _post_pin_capture()
        else:
            for _ in range(_DROP_SETTLE):
                self.env.step(drop_action)

        eef_a5 = self.env.get_robot_eef_pos()
        print(f"  [Phase 5] Settled: EEF z={eef_a5[2]:.4f}m")
        phase_log.append({"phase": "above_dest",
                           "target_z": round(float(drop_pos[2]), 4),
                           "eef_before_z": round(float(eef_b5[2]), 4),
                           "eef_after_z":  round(float(eef_a5[2]), 4),
                           "steps": _DROP_SETTLE})
        total_steps  += _DROP_SETTLE
        phase_reached = "above_dest"

        # Phase 6: Release
        # Stay in GHOST through release: pads still envelop the item. switching
        # to 'normal' fires a huge impulse from that overlap, launching the item.
        # In ghost mode the gripper has bitmask 4. items keep bitmask 1, so
        # gripper<->item contacts are off but item<->(bin/floor/items) ON, the
        # released item still falls and settles physically.
        if _HAS_MUJOCO:
            print("  [Contacts] mode=ghost (kept; item<->bin/floor on, gripper<->item off) for release")
        _RELEASE_STEPS = 35   # ~1.75 s @ 20 Hz, covers fall + bounce + roll-to-rest
        open_action = np.zeros(7)
        open_action[-1] = -1.0
        if _HAS_MUJOCO:
            raw_model_6 = getattr(self.env.sim.model, "_model", self.env.sim.model)
            raw_data_6  = getattr(self.env.sim.data,  "_data",  self.env.sim.data)
            qpos_idxs_6, qvel_idxs_6 = self._get_arm_indices()
            locked_qpos_6 = np.array([raw_data_6.qpos[idx] for idx in qpos_idxs_6])
            for _stepi in range(_RELEASE_STEPS):
                self.env.step(open_action)
                # Grasped item is NOT in _src_post_close any more (it left the
                # source bin), so it falls under gravity. Dest-bin items are free.
                for k, idx in enumerate(qpos_idxs_6):
                    raw_data_6.qpos[idx] = locked_qpos_6[k]
                for idx in qvel_idxs_6:
                    raw_data_6.qvel[idx] = 0.0
                self._repin_items(_src_post_close, except_name=grasped_obj)
                _mujoco.mj_forward(raw_model_6, raw_data_6)
                self.env.robots[0].controller.update(force=True)
                self.env.robots[0].controller.reset_goal()
                self._log_contacts("release", _stepi, target_item_name)
                _post_pin_capture()
        else:
            self.grasp_executor.open_gripper(self.env, n_steps=_RELEASE_STEPS)
        phase_log.append({"phase": "release",
                           "target_z": "-", "eef_before_z": "-", "eef_after_z": "-",
                           "steps": _RELEASE_STEPS})
        total_steps  += _RELEASE_STEPS
        phase_reached = "released"

        # Phase 7: Return to sensing pose
        # Use direct joint teleport. move_to_sensing_pose_osc() mixes a robot-local
        # target with a world-frame current position, producing a bogus delta that
        # drives the arm -Z through the dest bin, ejecting the object.
        if self.sensing_ctrl is not None:
            self.sensing_ctrl.set_sensing_pose_direct()
        phase_log.append({"phase": "sensing_pose",
                           "target_z": "-", "eef_before_z": "-", "eef_after_z": "-",
                           "steps": 0})
        phase_reached = "sensing_pose"

        delivered = False
        if grasped_obj is not None and _HAS_MUJOCO:
            try:
                bid = self.env.sim.model.body_name2id(grasped_obj)
                fpos = np.array(self.env.sim.data.body_xpos[bid])
                delivered = (
                    abs(fpos[0] - self.dest_bin_pos[0]) < 0.13
                    and abs(fpos[1] - self.dest_bin_pos[1]) < 0.13
                    and fpos[2] > self.dest_bin_pos[2] - 0.05
                )
            except Exception:
                pass

        if _HAS_MUJOCO:
            self._apply_contact_mode("normal")
            print("  [Contacts] mode=normal (original masks restored)")

        self._suppress_step_capture = False

        self._contact_log_teardown()

        return {
            "success":              delivered,
            "phase_reached":        phase_reached,
            "n_env_steps":          total_steps,
            "grasp_success":        grasp_success,
            "physics_grasp":        bool(physics_grasp_success),
            "geometric_grasp":      bool(geometric_grasp_success),
            "delivered":            bool(delivered),
            "grasped_object_name":  grasped_obj,
            "target_item_name":     target_item_name,
            "phase_log":            phase_log,
        }

    def attempt_grasp_physical(
        self,
        grasp_pos: np.ndarray,
        grasp_quat: np.ndarray,
        target_item_name: str,
        n_close: int = 36,
        near_radius: float = 0.14,
        disturb_floor_m: float = 0.005,
    ) -> dict:
        """Physical grasp attempt with real contact physics (used by RL reward).

        Unlike ``execute()``, target + near neighbours move freely while the jaws
        close, so the outcome reflects what the chosen pose actually does to the
        pile. The caller handles delivery on success.

        Returns dict with grasp_ok, grasp_quality, n_pads_engaged, grip forces,
        force_imbalance, target_between_pads, target_xy_offset_m,
        neighbour_disturbance_m, n_near_items, items_ejected, ik_err_m.
        """
        out = {
            "grasp_ok": False, "grasp_quality": 0.0, "n_pads_engaged": 0,
            "grip_force_left": 0.0, "grip_force_right": 0.0, "force_imbalance": 0.0,
            "target_between_pads": False, "target_xy_offset_m": float("nan"),
            "neighbour_disturbance_m": 0.0, "n_near_items": 0, "items_ejected": 0,
            "ik_err_m": float("nan"), "target_item_name": target_item_name,
        }
        if not _HAS_MUJOCO:
            return out

        grasp_pos  = np.asarray(grasp_pos,  dtype=np.float64)
        grasp_quat = np.asarray(grasp_quat, dtype=np.float64)
        model_raw  = getattr(self.env.sim.model, "_model", self.env.sim.model)
        data_raw   = getattr(self.env.sim.data,  "_data",  self.env.sim.data)

        try:
            tbid = self.env.sim.model.body_name2id(target_item_name)
        except Exception:
            return out

        # Items "near" the grasp XY can move. the rest are frozen.
        snap = self._snapshot_item_poses()
        item_pos0 = {}
        near_items, far_items = [], []
        for name in self.env.get_obj_names():
            try:
                bid = self.env.sim.model.body_name2id(name)
            except Exception:
                continue
            p = np.array(self.env.sim.data.body_xpos[bid], dtype=np.float64)
            item_pos0[name] = p
            if float(np.linalg.norm(p[:2] - grasp_pos[:2])) <= near_radius:
                near_items.append(name)
            else:
                far_items.append(name)
        if target_item_name not in near_items:
            near_items.append(target_item_name)
            if target_item_name in far_items:
                far_items.remove(target_item_name)
        far_snap = {n: snap[n] for n in far_items if n in snap}
        out["n_near_items"] = max(0, len(near_items) - 1)

        left_gids, right_gids = self._finger_pad_geom_ids()
        target_gids = {g for g in range(model_raw.ngeom)
                       if self.env.sim.model.geom_bodyid[g] == tbid}

        try:
            self._patch_wrist_limits()

            # 1) Single DLS IK from sensing pose, this isn't the video path
            # just place the gripper. Contacts off (ghost) so nothing moves yet.
            self._apply_contact_mode("ghost")
            self._teleport_arm_to_pos(grasp_pos, target_quat=grasp_quat,
                                      max_iter=300, tol=0.008)
            eef = self.env.get_robot_eef_pos()
            out["ik_err_m"] = float(np.linalg.norm(grasp_pos - eef))

            # 2) Physical close: pads collide with free target + near neighbours
            # palm ghosted. far items frozen.
            self._apply_contact_mode("grasp_pads_only")
            qpos_idxs, qvel_idxs = self._get_arm_indices()
            locked_qpos = np.array([data_raw.qpos[idx] for idx in qpos_idxs])
            # (qpos_adr, dof_adr, start_xy) per free item, clamps how far / fast
            # the closing pads can fling a thin rectangle.
            near_pin = []
            target_pin = None
            for name in near_items:
                try:
                    nbid = self.env.sim.model.body_name2id(name)
                    for j in range(model_raw.njnt):
                        if model_raw.jnt_bodyid[j] == nbid:
                            qa = int(model_raw.jnt_qposadr[j]); va = int(model_raw.jnt_dofadr[j])
                            xy0 = np.array(self.env.sim.data.body_xpos[nbid][:2], dtype=np.float64)
                            if name == target_item_name:
                                target_pin = (qa, va, xy0)
                            else:
                                near_pin.append((qa, va, xy0))
                            break
                except Exception:
                    pass
            close_action = np.zeros(7); close_action[-1] = 1.0
            _V_LIN, _V_ANG = 0.5, 10.0
            # Was 0.08 / 0.04, at n_objects in a 30 cm bin, 8 cm shove was larger
            # than the wall distance for many items, so the safety clamp itself
            # ejected them (40 % per-attempt ejections on ppo_n5_filtered).
            _MAX_SHOVE = 0.03
            _MAX_TARGET_DRIFT = 0.03
            # Source-bin XY extents: items must stay inside (2 cm wall margin) to
            # prevent deep-overlap impulse from launching them through a wall.
            from sim.sensing_pose import BIN_HALF_SIZE as _BHS
            _BIN_CENTER = self.env.get_src_bin_world_pos()
            _BIN_MARGIN = 0.02
            _BIN_LIMIT = (float(_BHS[0]) - _BIN_MARGIN, float(_BHS[1]) - _BIN_MARGIN)
            for _ in range(int(n_close)):
                self.env.step(close_action)
                for k, idx in enumerate(qpos_idxs):
                    data_raw.qpos[idx] = locked_qpos[k]
                for idx in qvel_idxs:
                    data_raw.qvel[idx] = 0.0
                self._repin_items(far_snap)
                for items, cap in ((near_pin, _MAX_SHOVE),
                                   ([] if target_pin is None else [target_pin], _MAX_TARGET_DRIFT)):
                    for qa, va, xy0 in items:
                        # bleed off deep-overlap impulses
                        v = data_raw.qvel[va:va + 3]; s = float(np.linalg.norm(v))
                        if s > _V_LIN:
                            data_raw.qvel[va:va + 3] = v * (_V_LIN / s)
                        w = data_raw.qvel[va + 3:va + 6]; sw = float(np.linalg.norm(w))
                        if sw > _V_ANG:
                            data_raw.qvel[va + 3:va + 6] = w * (_V_ANG / sw)
                        # cap XY displacement from start
                        d = data_raw.qpos[qa:qa + 2] - xy0; dn = float(np.linalg.norm(d))
                        if dn > cap:
                            data_raw.qpos[qa:qa + 2] = xy0 + d * (cap / dn)
                            data_raw.qvel[va:va + 2] = 0.0
                        # keep inside source-bin XY footprint (deep-overlap impulse
                        # + safety clamp can otherwise leave an item just outside)
                        x_now, y_now = data_raw.qpos[qa], data_raw.qpos[qa + 1]
                        x_clamped = float(np.clip(x_now, _BIN_CENTER[0] - _BIN_LIMIT[0],
                                                         _BIN_CENTER[0] + _BIN_LIMIT[0]))
                        y_clamped = float(np.clip(y_now, _BIN_CENTER[1] - _BIN_LIMIT[1],
                                                         _BIN_CENTER[1] + _BIN_LIMIT[1]))
                        if x_clamped != x_now or y_clamped != y_now:
                            data_raw.qpos[qa]     = x_clamped
                            data_raw.qpos[qa + 1] = y_clamped
                            data_raw.qvel[va:va + 2] = 0.0
                _mujoco.mj_forward(model_raw, data_raw)
                self.env.robots[0].controller.update(force=True)
                self.env.robots[0].controller.reset_goal()

            # 3) Read outcome
            eef_a = self.env.get_robot_eef_pos()
            # pads sit ~9.7 cm below the wrist for a top grasp
            pad_center = np.array([eef_a[0], eef_a[1], eef_a[2] - 0.097], dtype=np.float64)
            target_pos_a = np.array(self.env.sim.data.body_xpos[tbid], dtype=np.float64)
            out["target_xy_offset_m"] = float(np.linalg.norm(target_pos_a[:2] - pad_center[:2]))

            fL, fR = 0.0, 0.0
            wrench = np.zeros(6)
            for i in range(self.env.sim.data.ncon):
                c = self.env.sim.data.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                if g1 in target_gids:
                    pg = g2
                elif g2 in target_gids:
                    pg = g1
                else:
                    continue
                try:
                    _mujoco.mj_contactForce(model_raw, data_raw, i, wrench)
                    fn = abs(float(wrench[0]))
                except Exception:
                    fn = 1.0
                if pg in left_gids:
                    fL += fn
                elif pg in right_gids:
                    fR += fn
            n_pads = int(fL > 1e-6) + int(fR > 1e-6)
            tot = fL + fR
            imbalance = (abs(fL - fR) / tot) if tot > 1e-6 else 1.0
            off = out["target_xy_offset_m"]
            # would-hold: jaws ended firmly on target AND target well-centred.
            # n_pads==1 accepted iff target dead-centred (<=2 cm): a thin tablet
            # between closed pads is pinned even if solver only registered the
            # near pad's contact (items are only ~1.7 cm wide. the far pad often
            # just grazes the threshold). force <= 5 N -> brush, not a grip.
            grip_force = max(fL, fR)
            out["grip_force_left"]   = float(fL)
            out["grip_force_right"]  = float(fR)
            out["force_imbalance"]   = float(imbalance)
            out["n_pads_engaged"]    = int(n_pads)
            out["target_between_pads"] = bool(n_pads >= 1 and off < 0.020)
            out["grasp_ok"] = bool(
                grip_force > 5.0 and (
                    (n_pads == 2 and off < 0.030) or (n_pads == 1 and off < 0.020)
                )
            )
            # graded quality (0 if no contact at all)
            center_score = float(np.clip(1.0 - off / 0.03, 0.0, 1.0))
            out["grasp_quality"] = (0.0 if n_pads == 0 else
                                    float(np.clip(0.6 * center_score + 0.4 * (n_pads / 2.0),
                                                  0.0, 1.0)))

            # disturbance + ejections (per-item move capped at 0.20 m so a freak
            # spike can't blow up the reward)
            src = self.env.get_src_bin_world_pos()
            from sim.sensing_pose import BIN_HALF_SIZE
            disturb = 0.0
            ejected = 0
            for name in near_items:
                if name == target_item_name:
                    continue
                bid = self.env.sim.model.body_name2id(name)
                p1 = np.array(self.env.sim.data.body_xpos[bid], dtype=np.float64)
                p0 = item_pos0.get(name, p1)
                d = float(np.linalg.norm(p1[:2] - p0[:2]))
                disturb += min(0.20, max(0.0, d - disturb_floor_m))
                in_src = (abs(p1[0] - src[0]) < BIN_HALF_SIZE[0]
                          and abs(p1[1] - src[1]) < BIN_HALF_SIZE[1]
                          and p1[2] > src[2] - 0.05)
                was_in_src = (abs(p0[0] - src[0]) < BIN_HALF_SIZE[0]
                              and abs(p0[1] - src[1]) < BIN_HALF_SIZE[1]
                              and p0[2] > src[2] - 0.05)
                if was_in_src and not in_src:
                    ejected += 1
            out["neighbour_disturbance_m"] = float(disturb)
            out["items_ejected"] = int(ejected)
        finally:
            try:
                self._apply_contact_mode("normal")
            except Exception:
                pass
        return out

    def attempt_grasp_hybrid(
        self,
        grasp_pos: np.ndarray,
        grasp_quat: np.ndarray,
        target_item_name: str,
        n_close: int = 18,
        near_radius: float = 0.14,
        disturb_floor_m: float = 0.005,
        frame_hook=None,
        frame_hook_timing: str = "post_teleport",
        descent_order: str = "step_then_teleport",
        # v5: optional pose-refinement + snap-gating kwargs. The v5 gym env
        # already does the math BEFORE calling this primitive, so for the RL
        # path defaults are no-ops. Direct callers (smoke / ablation) can pass
        # non-zero (dx, dy, dyaw) and snap flags to do the same here.
        dx: float = 0.0,
        dy: float = 0.0,
        dyaw: float = 0.0,
        use_snap_xy: bool = False,
        use_snap_z: bool = False,
    ) -> dict:
        """HYBRID grasp engine (Option B). Used by reward_mode="hybrid".

        Splits the pick into:
          GRASP SUCCESS, deterministic geometric predicate (evaluate_grasp).
          DISTURBANCE  , real MuJoCo physics. Pads close with contacts ON. near non-target items are FREE, can be shoved /
                          cascade / ejected over the bin rim. Only safety
                          clamp is a velocity clamp.
          DELIVERY     , caller teleports target to dest bin if grasp_ok.

        Target is FROZEN during the physics close (fate already decided
        geometrically. freezing keeps the disturbance measurement clean
        = effect on OTHER items only).

        Returns the keys rl/reward.py consumes: grasp_ok, grasp_quality,
        neighbour_disturbance_m, items_ejected, n_near_items, target_item_name,
        picked_item, predicate_reason, ik_err_m. Plus diagnostics
        neighbour_disturbance_raw_m (no floor, no cap) and
        neighbour_disturbance_max_m (single largest, capped 0.30 m).
        """
        out = {
            "grasp_ok": False, "grasp_quality": 0.0,
            "neighbour_disturbance_m": 0.0, "n_near_items": 0, "items_ejected": 0,
            "neighbour_disturbance_raw_m": 0.0,
            "neighbour_disturbance_max_m": 0.0,
            "target_item_name": target_item_name, "picked_item": None,
            "predicate_reason": "", "ik_err_m": float("nan"),
        }
        grasp_pos  = np.asarray(grasp_pos,  dtype=np.float64)
        grasp_quat = np.asarray(grasp_quat, dtype=np.float64)

        # v5 optional pose refinement + snap gating (no-op when called from RL env).
        if float(dx) != 0.0 or float(dy) != 0.0 or float(dyaw) != 0.0:
            grasp_pos = grasp_pos + np.array(
                [float(dx), float(dy), 0.0], dtype=np.float64)
            if float(dyaw) != 0.0:
                # world-frame yaw delta: q_out = q_yaw * q_in (wxyz Hamilton, inlined)
                half = 0.5 * float(dyaw)
                qy_w, qy_z = float(np.cos(half)), float(np.sin(half))
                w1, x1, y1, z1 = qy_w, 0.0, 0.0, qy_z
                w2, x2, y2, z2 = (float(grasp_quat[0]), float(grasp_quat[1]),
                                  float(grasp_quat[2]), float(grasp_quat[3]))
                grasp_quat = np.array([
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                ], dtype=np.float64)
        if use_snap_xy or use_snap_z:
            # mirror BinClearingGymEnv.step snap behaviour. Constants pasted here
            # to avoid an rl <- control import cycle. keep in sync.
            _GRASP_DESCENT_OFFSET = 0.012
            _ITEM_HALF_HEIGHT     = 0.029
            try:
                from control.grasp_success_predicate import _FINGER_TO_WRIST
            except Exception:
                _FINGER_TO_WRIST = 0.10
            try:
                bid = self.env.sim.model.body_name2id(target_item_name)
                item_pos = np.array(self.env.sim.data.body_xpos[bid],
                                    dtype=np.float64)
                item_top = float(item_pos[2]) + _ITEM_HALF_HEIGHT
                if use_snap_xy:
                    grasp_pos = np.array([float(item_pos[0]),
                                          float(item_pos[1]),
                                          float(grasp_pos[2])],
                                         dtype=np.float64)
                if use_snap_z:
                    grasp_pos = np.array([float(grasp_pos[0]),
                                          float(grasp_pos[1]),
                                          item_top - _GRASP_DESCENT_OFFSET
                                          + _FINGER_TO_WRIST],
                                         dtype=np.float64)
            except Exception:
                pass

        # 1. GEOMETRIC GRASP SUCCESS (deterministic, pre-physics)
        try:
            predicate = evaluate_grasp(self.env, grasp_pos, grasp_quat, target_item_name)
        except Exception as e:
            predicate = {"success": False, "reason": f"predicate error: {e}",
                         "picked_item": None, "item_between_jaws": False,
                         "jaws_aligned": False, "mid_height": False,
                         "approach_clear": False}
        out["grasp_ok"]         = bool(predicate.get("success", False))
        out["picked_item"]      = predicate.get("picked_item")
        out["predicate_reason"] = predicate.get("reason", "")
        # weighted fraction of the four sub-criteria -> smooth signal
        out["grasp_quality"] = float(
            0.40 * bool(predicate.get("item_between_jaws", False)) +
            0.20 * bool(predicate.get("jaws_aligned",      False)) +
            0.20 * bool(predicate.get("mid_height",        False)) +
            0.20 * bool(predicate.get("approach_clear",    False))
        )

        if not _HAS_MUJOCO:
            return out

        model_raw = getattr(self.env.sim.model, "_model", self.env.sim.model)
        data_raw  = getattr(self.env.sim.data,  "_data",  self.env.sim.data)
        try:
            self.env.sim.model.body_name2id(target_item_name)
        except Exception:
            return out

        # 2. PHYSICS DISTURBANCE
        snap = self._snapshot_item_poses()
        item_pos0 = {}
        near_items, far_items = [], []
        for name in self.env.get_obj_names():
            try:
                bid = self.env.sim.model.body_name2id(name)
            except Exception:
                continue
            p = np.array(self.env.sim.data.body_xpos[bid], dtype=np.float64)
            item_pos0[name] = p
            if name == target_item_name:
                continue   # target frozen separately
            if float(np.linalg.norm(p[:2] - grasp_pos[:2])) <= near_radius:
                near_items.append(name)
            else:
                far_items.append(name)
        # Frozen set = far items + target. Near non-target items stay free.
        frozen_snap = {n: snap[n] for n in far_items if n in snap}
        if target_item_name in snap:
            frozen_snap[target_item_name] = snap[target_item_name]
        out["n_near_items"] = len(near_items)

        _saved_hook = getattr(self, "frame_hook", None)
        try:
            self._patch_wrist_limits()

            if frame_hook is not None:
                self.frame_hook = frame_hook   # _teleport_arm_to_pos captures frames
            qpos_idxs, qvel_idxs = self._get_arm_indices()
            near_dof = []
            for name in near_items:
                try:
                    nbid = self.env.sim.model.body_name2id(name)
                    for j in range(model_raw.njnt):
                        if model_raw.jnt_bodyid[j] == nbid:
                            near_dof.append(int(model_raw.jnt_dofadr[j]))
                            break
                except Exception:
                    pass

            # 2a. IK the two descent endpoints (ghost mode, no contacts)
            self._apply_contact_mode("ghost")
            above_pile = grasp_pos + np.array([0.0, 0.0, 0.12])   # 12 cm clear of items
            self._teleport_arm_to_pos(above_pile, target_quat=grasp_quat,
                                      max_iter=300, tol=0.010)
            qpos_above = np.array([data_raw.qpos[idx] for idx in qpos_idxs])
            self._teleport_arm_to_pos(grasp_pos, target_quat=grasp_quat,
                                      max_iter=300, tol=0.008)
            qpos_grasp = np.array([data_raw.qpos[idx] for idx in qpos_idxs])
            out["ik_err_m"] = float(np.linalg.norm(
                grasp_pos - self.env.get_robot_eef_pos()))
            # park back at above_pile for the physical descent
            for k, idx in enumerate(qpos_idxs):
                data_raw.qpos[idx] = qpos_above[k]
            for idx in qvel_idxs:
                data_raw.qvel[idx] = 0.0
            _mujoco.mj_forward(model_raw, data_raw)

            # 2b. GRADUAL physical descent + close (contacts ON).
            # Joint-interpolated waypoints -> each physics step resolves only
            # SHALLOW overlap -> small realistic push, never the explosive
            # impulse of a single deep-interpenetration teleport. Only clamp
            # is velocity (numerical safety). no displacement cap, no XY clamp.
            self._apply_contact_mode("grasp_pads_only")
            open_action  = np.zeros(7); open_action[-1]  = -1.0
            close_action = np.zeros(7); close_action[-1] =  1.0
            _V_LIN, _V_ANG = 0.5, 12.0

            def _lock_arm_to_waypoint(locked_qpos):
                """Snap arm to locked_qpos, zero arm vel, re-pin frozen items,
                clamp near-item velocities."""
                for k, idx in enumerate(qpos_idxs):
                    data_raw.qpos[idx] = locked_qpos[k]
                for idx in qvel_idxs:
                    data_raw.qvel[idx] = 0.0
                self._repin_items(frozen_snap)
                for va in near_dof:
                    v = data_raw.qvel[va:va + 3]; s = float(np.linalg.norm(v))
                    if s > _V_LIN:
                        data_raw.qvel[va:va + 3] = v * (_V_LIN / s)
                    w = data_raw.qvel[va + 3:va + 6]; sw = float(np.linalg.norm(w))
                    if sw > _V_ANG:
                        data_raw.qvel[va + 3:va + 6] = w * (_V_ANG / sw)
                _mujoco.mj_forward(model_raw, data_raw)
                self.env.robots[0].controller.update(force=True)
                self.env.robots[0].controller.reset_goal()

            def _step_and_pin(action, locked_qpos, pre_teleport_hook=None):
                """One physics step with the arm anchored at locked_qpos.

                Two ordering modes:
                  step_then_teleport (default, training-faithful): env.step
                    first (arm at PREVIOUS waypoint), then teleport+re-pin.
                    Visible arm lags cube positions by one iteration.
                  teleport_then_step: arm teleported FIRST, controller goal
                    reset, then env.step. Visually clean, but physics impulse
                    timing differs by one iteration.

                pre_teleport_hook (only meaningful in step_then_teleport) fires
                between env.step and the teleport so the visible arm pose
                matches the visible cube positions of the same env.step.
                """
                if descent_order == "teleport_then_step":
                    _lock_arm_to_waypoint(locked_qpos)
                    self.env.step(action)
                    _lock_arm_to_waypoint(locked_qpos)
                    if pre_teleport_hook is not None:
                        try:
                            pre_teleport_hook()
                        except Exception:
                            pass
                else:
                    # step_then_teleport (legacy / training-faithful)
                    self.env.step(action)
                    if pre_teleport_hook is not None:
                        try:
                            pre_teleport_hook()
                        except Exception:
                            pass
                    _lock_arm_to_waypoint(locked_qpos)

            # frame_hook firing:
            # post_teleport (default): after qpos override. visible arm = new
            # waypoint, visible cubes = PREVIOUS contact resolution (1-iter lag).
            # pre_teleport: between env.step and qpos override. visible arm
            # wherever OSC+physics left it, cubes from same env.step (no lag).
            _pre_hook = frame_hook if frame_hook_timing == "pre_teleport" else None
            _post_hook = frame_hook if frame_hook_timing != "pre_teleport" else None

            # descent: interpolate arm qpos from above_pile -> grasp pose
            n_descent = 8
            for i in range(1, n_descent + 1):
                frac = i / n_descent
                qpos_wp = qpos_above + frac * (qpos_grasp - qpos_above)
                _step_and_pin(open_action, qpos_wp,
                              pre_teleport_hook=_pre_hook)
                if _post_hook is not None:
                    try: _post_hook()
                    except Exception: pass

            # close: gripper closes while arm holds the grasp pose
            for _ in range(int(n_close)):
                _step_and_pin(close_action, qpos_grasp,
                              pre_teleport_hook=_pre_hook)
                if _post_hook is not None:
                    try: _post_hook()
                    except Exception: pass

            # 2c. measure disturbance + ejections
            src = self.env.get_src_bin_world_pos()
            from sim.sensing_pose import BIN_HALF_SIZE
            disturb = 0.0
            raw_disturb = 0.0
            max_disturb = 0.0
            ejected = 0
            for name in near_items:
                bid = self.env.sim.model.body_name2id(name)
                p1 = np.array(self.env.sim.data.body_xpos[bid], dtype=np.float64)
                p0 = item_pos0.get(name, p1)
                d = float(np.linalg.norm(p1[:2] - p0[:2]))
                # cap counted move at 0.30 m so a freak spike can't blow up
                # the reward, but 0.30 m is well past the bin, so genuine
                # ejections are fully counted.
                disturb += min(0.30, max(0.0, d - disturb_floor_m))
                raw_disturb += d                       # no floor, no cap
                if d > max_disturb:
                    max_disturb = d                    # single largest
                in_src = (abs(p1[0] - src[0]) < BIN_HALF_SIZE[0]
                          and abs(p1[1] - src[1]) < BIN_HALF_SIZE[1]
                          and p1[2] > src[2] - 0.05)
                was_in_src = (abs(p0[0] - src[0]) < BIN_HALF_SIZE[0]
                              and abs(p0[1] - src[1]) < BIN_HALF_SIZE[1]
                              and p0[2] > src[2] - 0.05)
                if was_in_src and not in_src:
                    ejected += 1
            out["neighbour_disturbance_m"] = float(disturb)
            out["neighbour_disturbance_raw_m"] = float(raw_disturb)
            # cap max for numerical safety (same 0.30 m sentinel)
            out["neighbour_disturbance_max_m"] = float(min(0.30, max_disturb))
            out["items_ejected"] = int(ejected)
        finally:
            try:
                self._apply_contact_mode("normal")
            except Exception:
                pass
            self.frame_hook = _saved_hook
        return out

    def _finger_pad_geom_ids(self):
        """Return (left_gids, right_gids), robosuite Panda names them
        gripper0_finger1*/finger2* (sometimes ..._joint1_tip / ..._pad_collision).
        The bulky hand/palm geoms (no 'finger' in the name) collide during close
        but don't count toward the per-pad readout."""
        model = self.env.sim.model
        left, right = set(), set()
        for gid in range(model.ngeom):
            bid = model.geom_bodyid[gid]
            bname = (model.body_id2name(bid) or "").lower()
            gname = (model.geom_id2name(gid) or "").lower()
            tag = bname + " " + gname
            if "finger" not in tag:
                continue
            if ("1" in tag) or ("left" in tag):
                left.add(gid)
            elif ("2" in tag) or ("right" in tag):
                right.add(gid)
        return left, right

    def _move_eef_to_target(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        tol: float = 0.01,
        max_steps: int = 80,
        kp: float = 5.0,
        ko: float = 1.0,
        gripper_action: float = -1.0,
    ) -> int:
        """OSC_POSE: [dx, dy, dz, drx, dry, drz, gripper]. pos delta [-1,1] -> 0.05 m/step."""
        target_pos  = np.asarray(target_pos,  dtype=np.float64)
        target_quat = np.asarray(target_quat, dtype=np.float64)

        tw, tx, ty, tz = target_quat
        r_target = Rotation.from_quat([tx, ty, tz, tw])

        n_steps = 0
        _log_interval = 50
        for step_i in range(max_steps):
            current_pos  = self.env.get_robot_eef_pos()
            current_quat = self.env.get_robot_eef_quat()  # wxyz

            delta_pos  = target_pos - current_pos
            dist = np.linalg.norm(delta_pos)

            if step_i % _log_interval == 0:
                print(f"      [step {step_i:4d}] EEF z={current_pos[2]:.4f}  "
                      f"target z={target_pos[2]:.4f}  dist={dist:.4f}m  "
                      f"delta=({delta_pos[0]:.3f},{delta_pos[1]:.3f},{delta_pos[2]:.3f})")

            if dist < tol:
                print(f"      [step {step_i:4d}] Converged (dist={dist:.4f}m < tol={tol}m)")
                break

            action_pos = np.clip(delta_pos * kp, -1.0, 1.0)

            # axis-angle orientation error
            cw, cx, cy, cz = current_quat
            r_current = Rotation.from_quat([cx, cy, cz, cw])
            r_delta   = r_target * r_current.inv()
            rotvec    = r_delta.as_rotvec()
            action_ori = np.clip(rotvec * ko, -1.0, 1.0)

            action = np.concatenate([action_pos, action_ori, [gripper_action]])
            self.env.step(action)
            n_steps += 1

        return n_steps

    def _patch_wrist_limits(self):
        """Raise wrist actuator (j4-j6) ctrlrange to +-150 Nm at runtime.

        XML caps these at +-28 Nm (continuous UR5e spec). gravity at the grasp
        workspace exceeds this, saturating actuators and blocking descent below
        z~1.20 m. Two clipping paths must be patched:
          1) raw_model.actuator_ctrlrange, used by single_arm.py torque_limits
          2) controller.actuator_min/max , cached in base_controller.__init__
        """
        if not _HAS_MUJOCO:
            return
        raw_model = getattr(self.env.sim.model, "_model", self.env.sim.model)
        controller = self.env.robots[0].controller
        wrist_names = ["torq_j4", "torq_j5", "torq_j6"]
        for name in wrist_names:
            try:
                act_id = self.env.sim.model.actuator_name2id(name)
                raw_model.actuator_ctrlrange[act_id, 0] = -150.0
                raw_model.actuator_ctrlrange[act_id, 1] = 150.0
                ref_idxs = self.env.robots[0]._ref_joint_actuator_indexes
                controller_idx = list(ref_idxs).index(act_id) if act_id in ref_idxs else None
                if controller_idx is not None:
                    controller.actuator_min[controller_idx] = -150.0
                    controller.actuator_max[controller_idx] = 150.0
            except Exception:
                pass
        if _VERBOSE_IK:
            print("  [wrist patch] ctrlrange for torq_j4/j5/j6 raised to +-150 Nm")

    def _get_arm_indices(self):
        """Return (qpos_idxs, qvel_idxs) for the 6 arm joints."""
        joint_names = self.env.robots[0].robot_joints[:6]
        qpos_idxs, qvel_idxs = [], []
        for jnt_name in joint_names:
            jid = self.env.sim.model.joint_name2id(jnt_name)
            qpos_idxs.append(int(self.env.sim.model.jnt_qposadr[jid]))
            qvel_idxs.append(int(self.env.sim.model.jnt_dofadr[jid]))
        return qpos_idxs, qvel_idxs

    def _gripper_geom_ids(self):
        """Geom IDs of all gripper bodies (hand + fingers + pads)."""
        model = self.env.sim.model
        out = []
        for gid in range(model.ngeom):
            bid = model.geom_bodyid[gid]
            bname = (model.body_id2name(bid) or "").lower()
            gname = (model.geom_id2name(gid) or "").lower()
            if ("gripper" in bname or "finger" in bname
                    or "gripper" in gname or "finger" in gname):
                out.append(gid)
        return out

    def _item_geom_ids(self):
        """Geom IDs of all spawned item bodies."""
        model = self.env.sim.model
        item_body_ids = set()
        for name in self.env.get_obj_names():
            try:
                item_body_ids.add(model.body_name2id(name))
            except Exception:
                pass
        return [gid for gid in range(model.ngeom)
                if model.geom_bodyid[gid] in item_body_ids]

    def _contact_log_setup(self, target_item_name):
        """Open contact-log file and pre-compute geom->item lookups."""
        if not _HAS_MUJOCO or not self.contact_log_path:
            self._contact_log_file = None
            return
        model = self.env.sim.model
        self._contact_geom2item = {}
        for name in self.env.get_obj_names():
            try:
                bid = model.body_name2id(name)
            except Exception:
                continue
            for gid in range(model.ngeom):
                if model.geom_bodyid[gid] == bid:
                    self._contact_geom2item[gid] = name
        self._contact_gripper_gids = set(self._gripper_geom_ids())
        self._contact_prev_pos = {}
        for name in self.env.get_obj_names():
            try:
                bid = model.body_name2id(name)
                self._contact_prev_pos[name] = np.array(
                    self.env.sim.data.body_xpos[bid], dtype=np.float64)
            except Exception:
                pass
        self._contact_summary = {"gripper_touched": {}, "target_touched": {},
                                 "moved_no_contact": {}}
        self._contact_log_file = open(self.contact_log_path, "w")
        f = self._contact_log_file
        f.write("CONTACT DIAGNOSTIC LOG\n")
        f.write("=" * 70 + "\n")
        f.write(f"Target item: {target_item_name}\n")
        f.write(
            "Per-step report of REAL MuJoCo contacts (from sim.data.contact),\n"
            "plus non-target items that moved with NO contact (= pure visual\n"
            "glitch from re-pin snap-back or LCP-solver noise, NOT a real\n"
            "gripper/target disturbance).\n")
        f.write("=" * 70 + "\n\n")

    def _log_contacts(self, phase_name, step_idx, target_item_name):
        """Per-step contact report. Distinguishes gripper<->target, gripper<->other,
        target<->other, other<->other, and moved-with-no-contact (visual glitch)."""
        if not _HAS_MUJOCO or self._contact_log_file is None:
            return
        model = self.env.sim.model
        data  = self.env.sim.data
        raw_model = getattr(model, "_model", model)
        raw_data  = getattr(data,  "_data",  data)

        g2i      = self._contact_geom2item
        grip_g   = self._contact_gripper_gids
        ncon     = int(data.ncon)
        wrench   = np.zeros(6)

        grip_target, grip_other, tgt_other, other_other = [], [], [], []
        items_with_contact = set()

        for i in range(ncon):
            c   = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            is_g1_grip = g1 in grip_g
            is_g2_grip = g2 in grip_g
            it1 = g2i.get(g1)
            it2 = g2i.get(g2)
            try:
                _mujoco.mj_contactForce(raw_model, raw_data, i, wrench)
                fN = abs(float(wrench[0]))
            except Exception:
                fN = float("nan")
            if (is_g1_grip and it2) or (is_g2_grip and it1):
                item = it2 if is_g1_grip else it1
                items_with_contact.add(item)
                if item == target_item_name:
                    grip_target.append((item, fN))
                else:
                    grip_other.append((item, fN))
                    prev = self._contact_summary["gripper_touched"].get(item, 0.0)
                    self._contact_summary["gripper_touched"][item] = max(prev, fN)
            elif it1 and it2:
                items_with_contact.add(it1)
                items_with_contact.add(it2)
                if target_item_name in (it1, it2):
                    other = it2 if it1 == target_item_name else it1
                    tgt_other.append((other, fN))
                    prev = self._contact_summary["target_touched"].get(other, 0.0)
                    self._contact_summary["target_touched"][other] = max(prev, fN)
                else:
                    other_other.append((it1, it2, fN))

        # On step 0 of a phase we only refresh the baseline (the gap since the
        # previous logged step spans a kinematic teleport, cross-phase motion
        # not per-step jitter).
        moved_no_contact = []
        is_phase_start = (step_idx == 0)
        for name, prev_xyz in list(self._contact_prev_pos.items()):
            try:
                bid = model.body_name2id(name)
                now = np.array(data.body_xpos[bid], dtype=np.float64)
            except Exception:
                continue
            delta = float(np.linalg.norm(now - prev_xyz))
            self._contact_prev_pos[name] = now
            if is_phase_start:
                continue
            if delta > 5e-4 and name not in items_with_contact and name != target_item_name:
                moved_no_contact.append((name, delta))
                prev = self._contact_summary["moved_no_contact"].get(name, 0.0)
                self._contact_summary["moved_no_contact"][name] = max(prev, delta)

        f = self._contact_log_file
        f.write(f"[{phase_name:<22s} step {step_idx:03d}] ncon={ncon}\n")
        if grip_target:
            s = " ; ".join(f"{n} F={fN:.2f}N" for n, fN in grip_target)
            f.write(f"   gripper<->TARGET   : {s}\n")
        else:
            f.write(f"   gripper<->TARGET   : (none)\n")
        if grip_other:
            s = " ; ".join(f"{n} F={fN:.2f}N" for n, fN in grip_other)
            f.write(f"   gripper<->OTHER    : {s}   <- REAL gripper disturbance\n")
        else:
            f.write(f"   gripper<->OTHER    : (none)\n")
        if tgt_other:
            s = " ; ".join(f"{n} F={fN:.2f}N" for n, fN in tgt_other)
            f.write(f"   TARGET<->OTHER     : {s}   <- REAL (target shoves neighbour)\n")
        else:
            f.write(f"   TARGET<->OTHER     : (none)\n")
        if other_other:
            s = " ; ".join(f"{a}-{b} F={fN:.2f}N" for a, b, fN in other_other)
            f.write(f"   OTHER<->OTHER      : {s}\n")
        else:
            f.write(f"   OTHER<->OTHER      : (none)\n")
        if moved_no_contact:
            s = " ; ".join(f"{n} d={d*1000:.2f}mm" for n, d in moved_no_contact)
            f.write(f"   MOVED-NO-CONTACT   : {s}   <- VISUAL GLITCH (no real contact)\n")
        f.write("\n")

    def _contact_log_teardown(self):
        """Write the summary block and close the contact-log file."""
        if self._contact_log_file is None:
            return
        f = self._contact_log_file
        s = self._contact_summary
        f.write("=" * 70 + "\n")
        f.write("SUMMARY (whole pick)\n")
        f.write("=" * 70 + "\n")
        gt = s["gripper_touched"]
        f.write("Non-target items the GRIPPER really contacted "
                "(REAL disturbance, matters to RL):\n")
        if gt:
            for n, fN in sorted(gt.items(), key=lambda kv: -kv[1]):
                f.write(f"   {n}   max F={fN:.2f} N\n")
        else:
            f.write("   (none, the gripper never touched a non-target item)\n")
        tt = s["target_touched"]
        f.write("Non-target items the TARGET item pushed into "
                "(REAL inter-item disturbance):\n")
        if tt:
            for n, fN in sorted(tt.items(), key=lambda kv: -kv[1]):
                f.write(f"   {n}   max F={fN:.2f} N\n")
        else:
            f.write("   (none)\n")
        mn = s["moved_no_contact"]
        f.write("Non-target items that MOVED with NO contact at all "
                "(pure VISUAL GLITCH, NOT real disturbance):\n")
        if mn:
            for n, d in sorted(mn.items(), key=lambda kv: -kv[1]):
                f.write(f"   {n}   max per-step Delta={d*1000:.2f} mm\n")
        else:
            f.write("   (none)\n")
        f.write("=" * 70 + "\n")
        f.close()
        self._contact_log_file = None

    def _apply_contact_mode(self, mode, target_item_name=None):
        """Set gripper/item collision bitmasks for one of several regimes.

        Bit layout: bit0 (1) = world (table/bin/items all carry it -> item<->bin/
        table/item contacts). bit1 (2) = gripper-can-touch. bit2 (4) = gripper-
        isolated (gripper alone, touches nothing).

        Modes:
          ghost            , gripper 4/4 -> touches NOTHING (approach + descent
                              + lift/transport so the hand can't detonate the pile).
          grasp_pads_only  , finger geoms 2/2, palm/hand ghosted 4/4, items
                              conaff |= 2 -> pads<->any item: YES, palm<->anything: NO.
                              Used by attempt_grasp_physical (RL reward).
          grasp            , gripper 2/2, items conaff |= 2.
          grasp_target_only, gripper 2/2, ONLY target_item_name gets conaff |= 2. every other item stays on bit0 only. Close phase
                              uses this so pads pass cleanly through neighbours.
          normal           , restore original masks (call once at the end).

        Original masks are cached on first non-normal call.
        """
        if not _HAS_MUJOCO:
            return
        raw_model   = getattr(self.env.sim.model, "_model", self.env.sim.model)
        gripper_ids = self._gripper_geom_ids()
        item_ids    = self._item_geom_ids()

        target_gids = set()
        if target_item_name is not None:
            try:
                tbid = self.env.sim.model.body_name2id(target_item_name)
                target_gids = {g for g in item_ids
                               if self.env.sim.model.geom_bodyid[g] == tbid}
            except Exception:
                pass

        if not getattr(self, "_orig_contact_masks", None):
            self._orig_contact_masks = {
                "g_ct": {g: int(raw_model.geom_contype[g])     for g in gripper_ids},
                "g_ca": {g: int(raw_model.geom_conaffinity[g]) for g in gripper_ids},
                "i_ct": {g: int(raw_model.geom_contype[g])     for g in item_ids},
                "i_ca": {g: int(raw_model.geom_conaffinity[g]) for g in item_ids},
            }

        if mode == "ghost":
            for g in gripper_ids:
                raw_model.geom_contype[g]     = 4
                raw_model.geom_conaffinity[g] = 4
            for g in item_ids:
                raw_model.geom_contype[g]     = 1
                raw_model.geom_conaffinity[g] = 1
        elif mode == "grasp_pads_only":
            try:
                left, right = self._finger_pad_geom_ids()
                pad_gids = set(left) | set(right)
            except Exception:
                pad_gids = set()
            for g in gripper_ids:
                if g in pad_gids:
                    raw_model.geom_contype[g]     = 2
                    raw_model.geom_conaffinity[g] = 2
                else:
                    raw_model.geom_contype[g]     = 4   # ghost palm/hand
                    raw_model.geom_conaffinity[g] = 4
            for g in item_ids:
                raw_model.geom_contype[g]     = 1
                raw_model.geom_conaffinity[g] = 3   # bit0 | bit1
        elif mode == "grasp":
            for g in gripper_ids:
                raw_model.geom_contype[g]     = 2
                raw_model.geom_conaffinity[g] = 2
            for g in item_ids:
                raw_model.geom_contype[g]     = 1
                raw_model.geom_conaffinity[g] = 3
        elif mode == "grasp_target_only":
            for g in gripper_ids:
                raw_model.geom_contype[g]     = 2
                raw_model.geom_conaffinity[g] = 2
            for g in item_ids:
                raw_model.geom_contype[g]     = 1
                raw_model.geom_conaffinity[g] = (3 if g in target_gids else 1)
        elif mode == "normal":
            m = self._orig_contact_masks or {}
            for g, v in m.get("g_ct", {}).items(): raw_model.geom_contype[g]     = v
            for g, v in m.get("g_ca", {}).items(): raw_model.geom_conaffinity[g] = v
            for g, v in m.get("i_ct", {}).items(): raw_model.geom_contype[g]     = v
            for g, v in m.get("i_ca", {}).items(): raw_model.geom_conaffinity[g] = v
            self._orig_contact_masks = None
        else:
            raise ValueError(f"unknown contact mode: {mode!r}")

    def _snapshot_item_poses(self):
        """{name: np.ndarray(7,)} freejoint qpos [x,y,z,qw,qx,qy,qz] for every item."""
        if not _HAS_MUJOCO:
            return {}
        raw_model = getattr(self.env.sim.model, "_model", self.env.sim.model)
        raw_data  = getattr(self.env.sim.data,  "_data",  self.env.sim.data)
        out = {}
        for name in self.env.get_obj_names():
            try:
                bid = self.env.sim.model.body_name2id(name)
                for j in range(raw_model.njnt):
                    if raw_model.jnt_bodyid[j] == bid:
                        qa = int(raw_model.jnt_qposadr[j])
                        out[name] = np.array(raw_data.qpos[qa:qa+7], dtype=np.float64)
                        break
            except Exception:
                pass
        return out

    def _repin_items(self, snapshots, except_name=None):
        """Reset every snapshotted item to captured qpos, zero its velocities.
        Call after each env.step() inside a physics phase so non-grasped items
        can never drift from inter-item contact resolution."""
        if not _HAS_MUJOCO or not snapshots:
            return
        raw_model = getattr(self.env.sim.model, "_model", self.env.sim.model)
        raw_data  = getattr(self.env.sim.data,  "_data",  self.env.sim.data)
        for name, qpos7 in snapshots.items():
            if name == except_name:
                continue
            try:
                bid = self.env.sim.model.body_name2id(name)
                for j in range(raw_model.njnt):
                    if raw_model.jnt_bodyid[j] == bid:
                        qa = int(raw_model.jnt_qposadr[j])
                        va = int(raw_model.jnt_dofadr[j])
                        raw_data.qpos[qa:qa+7] = qpos7
                        raw_data.qvel[va:va+6] = 0.0
                        break
            except Exception:
                pass

    def _smooth_teleport(self, target_pos, target_quat, n_steps=14, **ik_kwargs):
        """Move EEF to (target_pos, target_quat) along a straight Cartesian line,
        IK-ing to ~n_steps interpolated waypoints. Fires self.frame_hook (if set)
        once per waypoint and tracks self._tracked_item if set.

        A single converge-from-far IK takes a wobbly path that tilts the rigidly
        wrist-mounted camera all over the place, interpolated waypoints read as
        a smooth monotonic descent.
        """
        if not _HAS_MUJOCO:
            self._teleport_arm_to_pos(target_pos, target_quat=target_quat, **ik_kwargs)
            return
        start_pos = self.env.get_robot_eef_pos().copy()
        target_pos = np.asarray(target_pos, dtype=np.float64)
        n = max(2, int(n_steps))
        for i in range(1, n + 1):
            frac = i / n
            wp = start_pos + frac * (target_pos - start_pos)
            tol = 0.004 if i == n else 0.012
            self._teleport_arm_to_pos(wp, target_quat=target_quat,
                                      tol=tol, max_iter=400, **ik_kwargs)

    def _teleport_arm_to_pos(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray = None,
        max_iter: int = 1000,
        step_size: float = 0.01,
        rot_step: float = 0.3,
        tol: float = 0.004,
        rot_tol: float = 0.02,
        damping: float = 0.05,
        null_gain: float = 0.08,
        max_dq: float = 0.05,
    ) -> np.ndarray:
        """Damped-least-squares IK. Two regimes:

          target_quat is None    , pure position IK. null-space pull toward
                                    SENSING_JOINTS keeps the elbow consistent.
          target_quat is not None, task-priority IK: position primary, full
                                    3-axis orientation secondary in the position
                                    null-space. For a 6-DoF arm the null-space
                                    of a 3-DoF position task is 3-DoF, exactly
                                    fitting a 3-DoF orientation task, both
                                    reached simultaneously. Parallel-jaw grasps
                                    are 180-symmetric around gripper Z, so pick
                                    whichever of {target, target * R_z(pi)} is
                                    closer to current orientation per iteration.
        """
        if not _HAS_MUJOCO:
            print("  [IK teleport] mujoco not available, skipping teleport")
            return self.env.get_robot_eef_pos()

        from sim.sensing_pose import SENSING_JOINTS
        q_ref = np.array(SENSING_JOINTS, dtype=np.float64)

        robot = self.env.robots[0]
        joint_names = robot.robot_joints[:6]

        qpos_idxs, qvel_idxs = [], []
        for jnt_name in joint_names:
            jid = self.env.sim.model.joint_name2id(jnt_name)
            qpos_idxs.append(int(self.env.sim.model.jnt_qposadr[jid]))
            qvel_idxs.append(int(self.env.sim.model.jnt_dofadr[jid]))

        raw_model = getattr(self.env.sim.model, "_model", self.env.sim.model)
        raw_data  = getattr(self.env.sim.data,  "_data",  self.env.sim.data)

        eef_bid = self.env.sim.model.body_name2id("robot0_right_hand")
        nv = raw_model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv)) if target_quat is not None else None

        target_pos = np.asarray(target_pos, dtype=np.float64)
        r_target_options = None
        if target_quat is not None:
            tq = np.asarray(target_quat, dtype=np.float64)
            r_target = Rotation.from_quat([tq[1], tq[2], tq[3], tq[0]])  # wxyz->xyzw
            r_target_flip = r_target * Rotation.from_rotvec([0, 0, np.pi])
            r_target_options = (r_target, r_target_flip)

        # Optional video hook: invoke self.frame_hook every _FRAME_HOOK_EVERY
        # IK iterations (capped) so the wrist-camera video shows the teleport as
        # a smooth descent rather than an instantaneous jump.
        _frame_hook = getattr(self, "frame_hook", None)
        _FRAME_HOOK_EVERY = 3
        _FRAME_HOOK_MAX   = 24
        _frames_captured  = 0

        # Optional grasped-item tracking: while self._tracked_item is set, the
        # named item is teleported every IK iteration to keep its captured
        # offset relative to the EEF, makes lift/transport read as the gripper
        # carrying the item, not floating off and snapping back.
        _tracked_item   = getattr(self, "_tracked_item", None)
        _tracked_offset = getattr(self, "_tracked_offset", None)

        for i in range(max_iter):
            _mujoco.mj_forward(raw_model, raw_data)
            current_pos = np.array(self.env.sim.data.body_xpos[eef_bid])

            if _tracked_item is not None and _tracked_offset is not None:
                self._move_object_qpos(_tracked_item, current_pos + _tracked_offset)

            if (_frame_hook is not None and i % _FRAME_HOOK_EVERY == 0
                    and _frames_captured < _FRAME_HOOK_MAX):
                try:
                    _frame_hook()
                    _frames_captured += 1
                except Exception:
                    pass

            dp = target_pos - current_pos
            dist = float(np.linalg.norm(dp))

            do, rot_err = None, 0.0
            if target_quat is not None:
                R_current = np.array(raw_data.xmat[eef_bid]).reshape(3, 3)
                r_current = Rotation.from_matrix(R_current)
                # nearer of {target, target * R_z(pi)} (jaw symmetry)
                err_vecs = [
                    (rt * r_current.inv()).as_rotvec() for rt in r_target_options
                ]
                err_norms = [float(np.linalg.norm(v)) for v in err_vecs]
                k_best = int(np.argmin(err_norms))
                do      = err_vecs[k_best]
                rot_err = err_norms[k_best]
                converged = (dist < tol) and (rot_err < rot_tol)
                if _VERBOSE_IK and i % 50 == 0:
                    print(f"      [IK iter {i:4d}] EEF z={current_pos[2]:.4f}  "
                          f"dist={dist:.4f}m  rot_err={np.degrees(rot_err):.2f} deg")
            else:
                converged = dist < tol
                if _VERBOSE_IK and i % 50 == 0:
                    print(f"      [IK iter {i:4d}] EEF z={current_pos[2]:.4f}  dist={dist:.4f}m")

            if converged:
                if _VERBOSE_IK:
                    if target_quat is not None:
                        print(f"      [IK iter {i:4d}] Converged "
                              f"(dist={dist:.4f}m, rot_err={np.degrees(rot_err):.2f} deg)")
                    else:
                        print(f"      [IK iter {i:4d}] Converged (dist={dist:.4f}m)")
                break

            dp_scaled = dp * (step_size / dist) if dist > step_size else dp

            jacp[:] = 0.0
            body_pos = np.ascontiguousarray(
                raw_data.xpos[eef_bid].reshape(3, 1), dtype=np.float64
            )

            if target_quat is None:
                _mujoco.mj_jac(raw_model, raw_data, jacp, None, body_pos, eef_bid)
                J_p = jacp[:, qvel_idxs]   # (3, 6)

                JJT = J_p @ J_p.T + damping * np.eye(3)
                J_pinv = J_p.T @ np.linalg.solve(JJT, np.eye(3))   # (6, 3)
                dq_primary = J_pinv @ dp_scaled

                ng = (null_gain * 0.25) if dist < 0.02 else null_gain
                q_current = np.array([raw_data.qpos[idx] for idx in qpos_idxs])
                null_proj = np.eye(6) - J_pinv @ J_p
                dq_null = ng * (q_ref - q_current)
                dq = dq_primary + null_proj @ dq_null
            else:
                # task-priority IK: position primary, orientation secondary
                jacr[:] = 0.0
                _mujoco.mj_jac(raw_model, raw_data, jacp, jacr, body_pos, eef_bid)
                J_p = jacp[:, qvel_idxs]   # (3, 6)
                J_r = jacr[:, qvel_idxs]   # (3, 6)

                JJTp = J_p @ J_p.T + damping * np.eye(3)
                J_p_pinv = J_p.T @ np.linalg.solve(JJTp, np.eye(3))   # (6, 3)
                dq_pos = J_p_pinv @ dp_scaled

                N_p = np.eye(6) - J_p_pinv @ J_p   # (6, 6), rank 3

                # J_r @ (dq_pos + N_p @ dq_rot) = do_scaled
                # => (J_r N_p) @ dq_rot = do_scaled - J_r @ dq_pos
                do_scaled = do * (rot_step / rot_err) if rot_err > rot_step else do
                J_rN = J_r @ N_p   # (3, 6)
                JJTr = J_rN @ J_rN.T + damping * np.eye(3)
                J_rN_pinv = J_rN.T @ np.linalg.solve(JJTr, np.eye(3))   # (6, 3)
                dq_rot = J_rN_pinv @ (do_scaled - J_r @ dq_pos)

                dq = dq_pos + N_p @ dq_rot

            dq = np.clip(dq, -max_dq, max_dq)

            for k, idx in enumerate(qpos_idxs):
                raw_data.qpos[idx] += dq[k]

        # Zero joint vel so the robot doesn't jump on next env.step()
        for idx in qvel_idxs:
            raw_data.qvel[idx] = 0.0

        _mujoco.mj_forward(raw_model, raw_data)

        # Sync OSC controller's cached ee_pos / goal_pos to the IK position
        # otherwise stale goal (z=1.40 sensing pose) drives the arm back UP.
        controller = self.env.robots[0].controller
        controller.update(force=True)
        controller.reset_goal()

        return np.array(self.env.sim.data.body_xpos[eef_bid])

    def _find_grasped_object_and_offset(self, eef_pos: np.ndarray):
        """Find the object in contact with the gripper and return its offset
        relative to the EEF body centre. Returns (None, None) if nothing."""
        if not _HAS_MUJOCO:
            return None, None

        obj_names = set(self.env.get_obj_names())
        try:
            gripper_geom_names = set(self.env.robots[0].gripper.contact_geoms)
        except Exception:
            gripper_geom_names = set()
        gripper_geom_names.update({
            "gripper0_finger1", "gripper0_finger2",
            "robot0_leftfinger", "robot0_rightfinger",
        })

        for i in range(self.env.sim.data.ncon):
            contact = self.env.sim.data.contact[i]
            gid1, gid2 = contact.geom1, contact.geom2
            g1 = self.env.sim.model.geom_id2name(gid1) or ""
            g2 = self.env.sim.model.geom_id2name(gid2) or ""
            b1 = self.env.sim.model.geom_bodyid[gid1]
            b2 = self.env.sim.model.geom_bodyid[gid2]
            b1n = self.env.sim.model.body_id2name(b1) or ""
            b2n = self.env.sim.model.body_id2name(b2) or ""

            if g1 in gripper_geom_names and b2n in obj_names:
                bid = self.env.sim.model.body_name2id(b2n)
                obj_pos = np.array(self.env.sim.data.body_xpos[bid])
                return b2n, obj_pos - eef_pos
            if g2 in gripper_geom_names and b1n in obj_names:
                bid = self.env.sim.model.body_name2id(b1n)
                obj_pos = np.array(self.env.sim.data.body_xpos[bid])
                return b1n, obj_pos - eef_pos

        # Fallback: closest object within 13 cm of the wrist body. Wrist is
        # ~9.7 cm above the pads. a successfully gripped item is 9-11 cm from
        # the wrist body centre (the previous 8 cm threshold missed every grip).
        best_name, best_dist = None, float('inf')
        for name in obj_names:
            try:
                bid  = self.env.sim.model.body_name2id(name)
                opos = np.array(self.env.sim.data.body_xpos[bid])
                d    = float(np.linalg.norm(opos - eef_pos))
                if d < best_dist:
                    best_dist, best_name = d, name
            except Exception:
                pass
        if best_name is not None and best_dist < 0.13:
            bid  = self.env.sim.model.body_name2id(best_name)
            opos = np.array(self.env.sim.data.body_xpos[bid])
            return best_name, opos - eef_pos

        return None, None

    def _move_object_qpos(self, obj_name: str, target_pos: np.ndarray):
        """Teleport an object to target_pos via free-joint qpos. Orientation preserved."""
        if not _HAS_MUJOCO:
            return
        try:
            obj_bid  = self.env.sim.model.body_name2id(obj_name)
            raw_data  = getattr(self.env.sim.data,  "_data",  self.env.sim.data)
            raw_model = getattr(self.env.sim.model, "_model", self.env.sim.model)
            for j in range(raw_model.njnt):
                if raw_model.jnt_bodyid[j] == obj_bid:
                    qa = int(raw_model.jnt_qposadr[j])
                    va = int(raw_model.jnt_dofadr[j])
                    raw_data.qpos[qa]     = target_pos[0]
                    raw_data.qpos[qa + 1] = target_pos[1]
                    raw_data.qpos[qa + 2] = target_pos[2]
                    raw_data.qvel[va]     = 0.0
                    raw_data.qvel[va + 1] = 0.0
                    raw_data.qvel[va + 2] = 0.0
                    _mujoco.mj_forward(raw_model, raw_data)
                    return
        except Exception as e:
            print(f"  [_move_object_qpos] Error moving {obj_name}: {e}")
