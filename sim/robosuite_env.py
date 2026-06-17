"""BinClearingEnv: robosuite UR5e env that transfers objects from a source bin to a destination bin."""
import os
import sys
import numpy as np

import xml.etree.ElementTree as ET
import robosuite
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite import load_controller_config

try:
    import mujoco as _mujoco
    _HAS_MUJOCO_BINDINGS = True
except ImportError:
    _HAS_MUJOCO_BINDINGS = False

from sim.sensing_pose import (
    SOURCE_BIN_POS,
    DEST_BIN_POS,
    BIN_HALF_SIZE,
    OBJECT_SCALE,
    CAMERA_FOV_DEG,
    CAMERA_W,
    CAMERA_H,
    CAMERA_NEAR,
    CAMERA_FAR,
)
from sim.object_spawner import ObjectSpawner
from sim.reward_state import RewardState
from sim.camera_setup import depth_to_meters

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_GENERATED_XML_DIR = os.path.join(_PACKAGE_ROOT, "assets", "generated_xml")

# Mesh package resolved relative to repo root
_STL_DIR   = os.path.join(_PACKAGE_ROOT, "assets", "square_22_pkg", "stl_vis")
_POSES_CSV = os.path.join(_PACKAGE_ROOT, "assets", "square_22_pkg",
                          "poses_relative_to_container.csv")

_TABLE_HEIGHT = 0.8
_TABLE_FULL_SIZE = (1.0, 1.5)

# Robot base x-offset matches base_xpos_offset["table"](1.0)
_ROBOT_BASE_X = -0.66

# Indigo: darkened from 0.45 0.40 0.75 to prevent diffuse clipping ("white shadow")
# on faces aimed straight at a scene light. Item colour does not affect perception
# (GG-CNN consumes depth, not RGB).
_ITEM_RGBA = "0.32 0.28 0.52 1"

_CONTAINER_RGBA = "0.65 0.45 0.20 1"

# Container STL geometry (measured from the STL):
# The STL open face is at local -Y. After +90 deg around X (applied at spawn)
# the world Z extent comes from the STL's Y extent (half = 0.5087 unscaled).
_CONTAINER_Y_HALF_UNSCALED = 0.5087
# Wall-height shrink: 1/3 -> walls ~4 cm tall (vs ~13 cm original) for arm clearance
# during transport. XY footprint unchanged. Item placement is independent.
_BIN_WALL_SCALE = 1.0 / 3.0
_CONTAINER_BODY_Z = _TABLE_HEIGHT + _CONTAINER_Y_HALF_UNSCALED * OBJECT_SCALE * _BIN_WALL_SCALE

# Eye-in-hand camera: 20 cm below the EEF wrist at sensing height (z=1.40 -> cam z=1.20).
# Below-wrist mount keeps the arm body / fingers above the camera, invisible in the
# downward depth image. Bin (z=0.80) is 0.40 m below camera. bin fills ~75% of input
# width vs ~37% when camera was at the wrist.
_CAM_MOUNT_HEIGHT = 0.40


class BinClearingEnv(SingleArmEnv):
    """Bin-clearing manipulation env: UR5e + Panda gripper, two-bin transfer task."""

    def __init__(
        self,
        n_objects: int = 20,
        max_episode_steps: int = 500,
        has_renderer: bool = False,
        render_offscreen: bool = True,
        camera_height: int = 480,
        camera_width: int = 640,
        object_subset_seed: int = 42,
        stl_dir: str = None,
        poses_csv: str = None,
        generated_xml_dir: str = None,
        item_rgba: str = None,
        use_camera_obs: bool = True,
        **kwargs,
    ):
        self.n_objects = n_objects
        self.max_episode_steps = max_episode_steps
        self._object_subset_seed = object_subset_seed
        self._step_count = 0
        self._overview_renderer = None
        self._wrist_renderer = None
        self._item_rgba = item_rgba or _ITEM_RGBA

        controller_config = load_controller_config(default_controller="OSC_POSE")

        _stl_dir = stl_dir or _STL_DIR
        _poses_csv = poses_csv or _POSES_CSV
        _xml_dir = generated_xml_dir or _GENERATED_XML_DIR

        # Spawner must generate XML before super().__init__ loads the model
        self._spawner = ObjectSpawner(
            stl_dir=_stl_dir,
            poses_csv=_poses_csv,
            generated_xml_dir=_xml_dir,
            scale=OBJECT_SCALE,
            n_objects=n_objects,
            seed=object_subset_seed,
        )
        self._container_stl = os.path.join(_stl_dir, "container.stl")
        self._spawner.generate_xml_files()

        self._reward_state = RewardState(
            dest_bin_pos=DEST_BIN_POS,
            dest_bin_size=BIN_HALF_SIZE,
        )
        self._prev_n_in_dest = 0

        # use_camera_obs=False skips robosuite's built-in camera obs every step
        # (~0.3 s -> ~0.05 s/step). Custom mujoco.Renderer paths used by
        # perception/debug are unaffected.
        super().__init__(
            robots="UR5e",
            controller_configs=controller_config,
            gripper_types="PandaGripper",  # Panda pads at fingertip, no protruding knuckle
            has_renderer=has_renderer,
            has_offscreen_renderer=True,   # custom mujoco.Renderer needs a GL context
            render_camera="agentview",
            camera_names=(["robot0_eye_in_hand", "agentview"] if use_camera_obs else ["agentview"]),
            camera_heights=camera_height,
            camera_widths=camera_width,
            camera_depths=bool(use_camera_obs),
            use_camera_obs=bool(use_camera_obs),
            horizon=max_episode_steps,
            **kwargs,
        )

    def _load_model(self):
        super()._load_model()

        mujoco_arena = TableArena(
            table_full_size=(_TABLE_FULL_SIZE[0], _TABLE_FULL_SIZE[1], 0.05),
            table_friction=(1.0, 0.005, 0.0001),
            table_offset=(0.0, 0.0, _TABLE_HEIGHT),
        )
        mujoco_arena.set_origin([0, 0, 0])

        self.robots[0].robot_model.set_base_xpos(
            self.robots[0].robot_model.base_xpos_offset["table"](_TABLE_FULL_SIZE[0])
        )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[r.robot_model for r in self.robots],
            mujoco_objects=[],
        )

        src_world_x = _ROBOT_BASE_X + SOURCE_BIN_POS[0]
        src_world_y = SOURCE_BIN_POS[1]
        for cam in self.model.worldbody.iter("camera"):
            name = cam.get("name")
            if name == "birdview":
                # 0.4 m to +x of bin, 0.92 m above, tilted ~24 deg from vertical.
                # Clear of robot arm (base at x=-0.66). quat = (cos12, 0, sin12, 0).
                cam.set("pos", f"{src_world_x + 0.4:.3f} {src_world_y:.3f} 1.850")
                cam.set("quat", "0.978 0 0.208 0")
            elif name == "agentview":
                cam.set("pos", f"{_ROBOT_BASE_X:.3f} -0.900 1.600")
                cam.set("quat", "0.8375 0.4516 -0.1460 -0.2708")

        # Fixed XML camera for supervision video: a FREE MjvCamera leaked state from
        # the wrist-POV free cam (shared global fovy / scene state), producing
        # inconsistent frames. A FIXED camera has its pose baked into the model so
        # it cannot be affected by another camera's setup.
        _mid_y_world = 0.0
        _tf_pos = f"{src_world_x:.3f} {_mid_y_world + 1.60:.3f} {1.95:.3f}"
        # axis-angle from world -Z (default cam forward) to look-direction
        import numpy as _np
        _dir = _np.array([0.0, -1.60, -0.95]); _dir /= _np.linalg.norm(_dir)
        _neg_z = _np.array([0.0, 0.0, -1.0])
        _axis = _np.cross(_neg_z, _dir)
        _axis_norm = _np.linalg.norm(_axis)
        _angle = float(_np.arccos(float(_np.clip(_np.dot(_neg_z, _dir), -1.0, 1.0))))
        if _axis_norm > 1e-9 and _angle > 1e-9:
            _axis = _axis / _axis_norm
            _half = _angle * 0.5
            _qw, _qxyz = _np.cos(_half), _axis * _np.sin(_half)
            _tf_quat = f"{_qw:.5f} {_qxyz[0]:.5f} {_qxyz[1]:.5f} {_qxyz[2]:.5f}"
        else:
            _tf_quat = "1 0 0 0"
        self.model.worldbody.append(ET.fromstring(
            f'<camera name="top_front_fixed" mode="fixed" '
            f'pos="{_tf_pos}" quat="{_tf_quat}" fovy="55"/>'
        ))

        src_x = _ROBOT_BASE_X + SOURCE_BIN_POS[0]
        src_y = SOURCE_BIN_POS[1]
        dst_x = _ROBOT_BASE_X + DEST_BIN_POS[0]
        dst_y = DEST_BIN_POS[1]

        # Each bin = 5 thin box geoms (floor + 4 walls). Box geoms have exact
        # (non-hull) collision so the bin is genuinely hollow. The mesh bowl is
        # NOT used: MuJoCo collides a mesh as its convex hull, and the bowl's
        # hull is a solid block - items would interpenetrate by up to 16 cm.
        # Inner half-footprint 0.135 m sits 0.5 cm beyond the disturbance clamp
        # at BIN_HALF_SIZE = 0.13, guaranteeing items never touch the wall.
        _BIN_IN_HALF   = 0.135   # physical inner wall (clamp limit is 0.130)
        _BIN_WALL_T    = 0.012
        _BIN_WALL_H    = 0.100   # matches frozen eval/training
        _BIN_FLOOR_TOP = _TABLE_HEIGHT

        def _mk_box(name, px, py, pz, sx, sy, sz):
            return ET.Element("geom", {
                "name": name, "type": "box",
                "pos":  f"{px:.4f} {py:.4f} {pz:.4f}",
                "size": f"{sx:.4f} {sy:.4f} {sz:.4f}",
                "rgba": _CONTAINER_RGBA, "group": "1",
                "contype": "1", "conaffinity": "1",
                "condim": "4", "friction": "0.7 0.005 0.0001",
            })

        for tag, bx, by in [("source_bin", src_x, src_y), ("dest_bin", dst_x, dst_y)]:
            t, ih, wh, fz = _BIN_WALL_T, _BIN_IN_HALF, _BIN_WALL_H, _BIN_FLOOR_TOP
            binbody = ET.Element("body", {"name": tag, "pos": "0 0 0"})
            # Floor lifted 2 mm: its top face and the TableArena surface are both
            # at z=0.80 (coplanar -> z-fight, white blotch). 2 mm clears the tie.
            binbody.append(_mk_box(f"{tag}_floor", bx, by, fz - t / 2 + 0.002,
                                   ih + t, ih + t, t / 2))
            binbody.append(_mk_box(f"{tag}_wall_px", bx + ih + t / 2, by, fz + wh / 2,
                                   t / 2, ih + t, wh / 2))
            binbody.append(_mk_box(f"{tag}_wall_nx", bx - ih - t / 2, by, fz + wh / 2,
                                   t / 2, ih + t, wh / 2))
            binbody.append(_mk_box(f"{tag}_wall_py", bx, by + ih + t / 2, fz + wh / 2,
                                   ih + t, t / 2, wh / 2))
            binbody.append(_mk_box(f"{tag}_wall_ny", bx, by - ih - t / 2, fz + wh / 2,
                                   ih + t, t / 2, wh / 2))
            self.model.worldbody.append(binbody)

        self._obj_names = []
        obj_list = self._spawner.get_object_list()
        for obj in obj_list:
            name = obj["name"]
            stl_path = obj["abs_stl"]
            if not os.path.isfile(stl_path):
                print(f"[BinClearingEnv] STL missing: {stl_path}")
                continue
            try:
                self.model.asset.append(ET.Element("mesh", {
                    "name": f"{name}_mesh",
                    "file": stl_path,
                    "scale": "0.25 0.25 0.25",
                }))
                # group=1 makes geom visual-only for rendering. Group>0 geoms are
                # excluded from MuJoCo's inertia computation, so we supply explicit
                # <inertial> (50g disk, radius~2cm).
                # mass=0.5 kg (10x heavier than disc default) so items resist the
                # IK-teleport impulse when the gripper descends.
                body = ET.fromstring(
                    f'<body name="{name}" pos="0 0 10">'
                    f'  <freejoint name="{name}_jnt"/>'
                    f'  <inertial pos="0 0 0" mass="0.5"'
                    f'            diaginertia="2e-4 2e-4 2e-4"/>'
                    f'  <geom type="mesh" mesh="{name}_mesh"'
                    f'        rgba="{self._item_rgba}"'
                    f'        friction="1.2 0.01 0.001"'
                    f'        group="1"'
                    f'        condim="4"'
                    f'        contype="1" conaffinity="1"/>'
                    f'</body>'
                )
                self.model.worldbody.append(body)
                self._obj_names.append(name)
            except Exception as e:
                print(f"[BinClearingEnv] Warning: could not inject {name}: {e}")

        # Cosmetic camera shell on the wrist so the eye-in-hand camera is visible
        _cam_attached = False
        for body in self.model.worldbody.iter("body"):
            if body.get("name") == "robot0_right_hand":
                body.append(ET.fromstring(
                    '<geom type="box" size="0.020 0.015 0.012" '
                    '     pos="0.05 0.0 0.0" '
                    '     rgba="0.12 0.12 0.12 1.0" '
                    '     group="1" contype="0" conaffinity="0"/>'
                ))
                body.append(ET.fromstring(
                    '<geom type="cylinder" size="0.008 0.005" '
                    '     pos="0.05 0.0 0.013" '
                    '     rgba="0.02 0.02 0.02 1.0" '
                    '     group="1" contype="0" conaffinity="0"/>'
                ))
                _cam_attached = True
                break
        if not _cam_attached:
            print("[BinClearingEnv] Warning: robot0_right_hand body not found; "
                  "camera model not attached to arm.")

        self._table_z = _TABLE_HEIGHT

        # SOURCE_BIN_POS / DEST_BIN_POS are in robot-local frame (x forward, y left)
        self._src_bin_world = np.array([
            _ROBOT_BASE_X + SOURCE_BIN_POS[0],
            SOURCE_BIN_POS[1],
            _TABLE_HEIGHT,
        ])
        self._dst_bin_world = np.array([
            _ROBOT_BASE_X + DEST_BIN_POS[0],
            DEST_BIN_POS[1],
            _TABLE_HEIGHT,
        ])

        self._reward_state.dest_bin_pos = self._dst_bin_world.copy()

    def _reset_internal(self):
        super()._reset_internal()
        self._step_count = 0
        self._prev_n_in_dest = 0
        self._reward_state.reset()
        # Invalidate cached renderers: robosuite's reset() may swap the MjData
        # pointer, leaving a cached renderer attached to a stale pointer (renders
        # garbage / near-black frames).
        self._overview_renderer = None
        self._birdview_renderer = None
        self._wrist_renderer = None
        self._dest_renderer = None
        self._eih_renderer = None
        self._wpov_renderer = None
        self._tf_renderer = None

        # CSV z=0 is the container bottom, which sits on the table surface
        container_world_pos  = self._src_bin_world.copy()
        container_world_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # square_22 stack is verified non-overlapping at the bottom layer
        # (min 3D pairwise ~5.0-5.4 cm vs ~5.9 cm cubes). the flat-grid
        # workaround (_place_clean_layout) is kept below as legacy.
        world_entries = self._spawner.get_world_positions(
            container_world_pos, container_world_quat)
        obj_list = self._spawner.get_object_list()
        for obj, entry in zip(obj_list, world_entries):
            self._set_object_pose(obj["name"], entry["world_pos"],
                                  entry["world_quat"])

        self._forward()

        self._reward_state.update(self.get_object_positions())

    def _place_clean_layout(self):
        """Legacy: place selected items in a clean non-overlapping grid on the source-bin floor.

        Kept for the OLD overlapping stack data where genuine bottom-layer items
        had centres ~2 cm apart while items were ~5 cm long. Unused with square_22.
        """
        from scipy.spatial.transform import Rotation

        obj_list = self._spawner.get_object_list()
        names = [o["name"] for o in obj_list]
        n = len(names)

        src = self._src_bin_world
        # Item-centre clearance from wall = rotated footprint half-diagonal (~3.3 cm).
        # Wall half ~0.127 -> centres can reach +-0.094 from bin centre.
        half_range = 0.094
        n_cols, n_rows = 4, 3
        xs = np.linspace(src[0] - half_range, src[0] + half_range, n_cols)
        ys = np.linspace(src[1] - half_range, src[1] + half_range, n_rows)
        slots = [(float(x), float(y)) for y in ys for x in xs]

        rng = np.random.default_rng(self._object_subset_seed)
        # Rx(90)*Ry(270) lays the part on its largest face (3.6 cm tall, 5.3x3.7
        # footprint). Rz(90) aligns the long axis with the wider row spacing.
        r_flat = (Rotation.from_euler("z", 90, degrees=True)
                  * Rotation.from_euler("x", 90, degrees=True)
                  * Rotation.from_euler("y", 270, degrees=True))
        place_z = _TABLE_HEIGHT + 0.020   # item centre at rest (~3.6 cm tall part)

        for i, name in enumerate(names):
            if i >= len(slots):
                break
            cx, cy = slots[i]
            jx  = float(rng.uniform(-0.007, 0.007))
            jy  = float(rng.uniform(-0.007, 0.007))
            yaw = float(rng.uniform(-np.deg2rad(8), np.deg2rad(8)))
            r = Rotation.from_euler("z", yaw) * r_flat
            xyzw = r.as_quat()
            quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
            pos = np.array([cx + jx, cy + jy, place_z])
            self._set_object_pose(name, pos, quat)

    def _settle_source_pile(self, max_steps=1500, v_lin_clamp=0.4,
                            v_ang_clamp=8.0, damping=0.85, speed_tol=0.01):
        """Run mj_step until the source-bin pile reaches a genuine rest state.

        Mild velocity damping (qvel *= 0.85) lets overlapping items depenetrate
        without launching. Wall/floor clamps keep items inside the bin. Exit
        early when the fastest item's speed drops below speed_tol.
        """
        if not _HAS_MUJOCO_BINDINGS:
            return 0
        raw_model = getattr(self.sim.model, "_model", self.sim.model)
        raw_data  = getattr(self.sim.data,  "_data",  self.sim.data)

        # Lock the arm at its reset qpos so it doesn't drift while items settle
        try:
            arm_joint_names = list(self.robots[0].robot_joints[:6])
        except Exception:
            arm_joint_names = []
        arm_qpos_idx, arm_qvel_idx = [], []
        for jn in arm_joint_names:
            try:
                jid = self.sim.model.joint_name2id(jn)
                arm_qpos_idx.append(int(self.sim.model.jnt_qposadr[jid]))
                arm_qvel_idx.append(int(self.sim.model.jnt_dofadr[jid]))
            except Exception:
                pass
        locked_arm_qpos = np.array([raw_data.qpos[i] for i in arm_qpos_idx])

        item_names = [obj["name"] for obj in self._spawner.get_object_list()]
        item_qadr, item_dadr = {}, {}
        for name in item_names:
            try:
                bid = self.sim.model.body_name2id(name)
                for j in range(raw_model.njnt):
                    if raw_model.jnt_bodyid[j] == bid:
                        item_qadr[name] = int(raw_model.jnt_qposadr[j])
                        item_dadr[name] = int(raw_model.jnt_dofadr[j])
                        break
            except Exception:
                pass

        src_xy = self._src_bin_world[:2]
        wall_half = 0.5087 * OBJECT_SCALE                   # actual STL wall ~0.127
        item_half = 0.040                                    # rotation-worst-case half-XY
        x_min = src_xy[0] - wall_half + item_half
        x_max = src_xy[0] + wall_half - item_half
        y_min = src_xy[1] - wall_half + item_half
        y_max = src_xy[1] + wall_half - item_half
        floor_z = _TABLE_HEIGHT + 0.005                      # catch genuine fall-through only

        settled_at = max_steps
        for step in range(max_steps):
            _mujoco.mj_step(raw_model, raw_data)
            for k, idx in enumerate(arm_qpos_idx):
                raw_data.qpos[idx] = locked_arm_qpos[k]
            for idx in arm_qvel_idx:
                raw_data.qvel[idx] = 0.0
            max_speed = 0.0
            for name, qa in item_qadr.items():
                va = item_dadr[name]
                raw_data.qvel[va:va + 6] *= damping
                v = raw_data.qvel[va:va + 3]
                s = float(np.linalg.norm(v))
                if s > v_lin_clamp:
                    raw_data.qvel[va:va + 3] = v * (v_lin_clamp / s)
                    s = v_lin_clamp
                w = raw_data.qvel[va + 3:va + 6]
                sw = float(np.linalg.norm(w))
                if sw > v_ang_clamp:
                    raw_data.qvel[va + 3:va + 6] = w * (v_ang_clamp / sw)
                if raw_data.qpos[qa] < x_min:
                    raw_data.qpos[qa] = x_min; raw_data.qvel[va] = 0.0
                if raw_data.qpos[qa] > x_max:
                    raw_data.qpos[qa] = x_max; raw_data.qvel[va] = 0.0
                if raw_data.qpos[qa + 1] < y_min:
                    raw_data.qpos[qa + 1] = y_min; raw_data.qvel[va + 1] = 0.0
                if raw_data.qpos[qa + 1] > y_max:
                    raw_data.qpos[qa + 1] = y_max; raw_data.qvel[va + 1] = 0.0
                if raw_data.qpos[qa + 2] < floor_z:
                    raw_data.qpos[qa + 2] = floor_z; raw_data.qvel[va + 2] = 0.0
                max_speed = max(max_speed, s)
            _mujoco.mj_forward(raw_model, raw_data)
            if max_speed < speed_tol and step > 30:
                settled_at = step + 1
                break

        for name in item_qadr:
            va = item_dadr[name]
            raw_data.qvel[va:va + 6] = 0.0
        _mujoco.mj_forward(raw_model, raw_data)
        if settled_at < max_steps:
            print(f"[Settle] Pile reached rest after {settled_at} steps "
                  f"(max item speed < {speed_tol} m/s)")
        else:
            print(f"[Settle] Pile ran the full {max_steps} steps without fully converging")
        return settled_at

    def _set_object_pose(self, obj_name: str, pos: np.ndarray, quat: np.ndarray):
        """Set a freejoint position/orientation directly in sim.data.qpos."""
        jnt_name = f"{obj_name}_jnt"
        try:
            jid = self.sim.model.joint_name2id(jnt_name)
            qpos_adr = self.sim.model.jnt_qposadr[jid]
            state = np.concatenate([pos, quat])  # [x,y,z, w,x,y,z]
            self.sim.data.qpos[qpos_adr : qpos_adr + 7] = state
        except Exception as e:
            print(f"[BinClearingEnv] Could not set pose for {obj_name}: {e}")

    def _forward(self):
        if _HAS_MUJOCO_BINDINGS:
            raw_model = getattr(self.sim.model, "_model", self.sim.model)
            raw_data  = getattr(self.sim.data,  "_data",  self.sim.data)
            _mujoco.mj_forward(raw_model, raw_data)
        else:
            self.sim.forward()

    def reward(self, action=None):
        """Shaped: +1 per new object in dest bin, -0.01 step penalty."""
        obj_positions = self.get_object_positions()
        curr_n = self._reward_state.update(obj_positions)
        rew = self._reward_state.compute_reward(self._prev_n_in_dest, curr_n)
        self._prev_n_in_dest = curr_n
        return rew

    def _setup_observables(self):
        observables = super()._setup_observables()
        return observables

    def _check_success(self):
        return self._reward_state.get_n_transferred() >= len(self._obj_names)

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)
        self._step_count += 1
        info["n_transferred"] = self._reward_state.get_n_transferred()
        info["n_objects"] = len(self._obj_names)
        return reward, done, info

    @staticmethod
    def _scene_opt():
        """MjvOption that shows visual geoms (group 1) and hides collision geoms (group 0)."""
        opt = _mujoco.MjvOption()
        opt.geomgroup[0] = 0
        opt.geomgroup[1] = 1
        return opt

    def _get_overhead_cam(self):
        """Free MjvCamera directly above the source bin looking straight down."""
        cam = _mujoco.MjvCamera()
        cam.type      = _mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self._src_bin_world.copy()
        cam.distance  = _CAM_MOUNT_HEIGHT
        cam.azimuth   = 90.0
        cam.elevation = -90.0
        return cam

    def get_wrist_rgb(self, height=None, width=None) -> np.ndarray:
        """(H,W,3) uint8 RGB from an overhead free cam above the source bin.

        Shares the cached _wrist_renderer with the depth path on purpose: building a
        fresh renderer per call corrupts the shared GL context that GG-CNN depth
        depends on (silently degraded perception, changed episode outcomes). Cost:
        overhead RGB frames render somewhat dark in the video's secondary panel.
        """
        h = height or CAMERA_H
        w = width  or CAMERA_W
        if not _HAS_MUJOCO_BINDINGS:
            return np.zeros((h, w, 3), dtype=np.uint8)
        try:
            raw_model = getattr(self.sim.model, "_model", self.sim.model)
            raw_data  = getattr(self.sim.data,  "_data",  self.sim.data)
            if not hasattr(self, "_wrist_renderer") or self._wrist_renderer is None:
                self._wrist_renderer = _mujoco.Renderer(raw_model, height=h, width=w)
            cam = self._get_overhead_cam()
            self._wrist_renderer.update_scene(raw_data, camera=cam,
                                              scene_option=self._scene_opt())
            return self._wrist_renderer.render()
        except Exception:
            self._wrist_renderer = None
            return np.zeros((h, w, 3), dtype=np.uint8)

    def get_dest_bin_overhead_rgb(self, height=None, width=None) -> np.ndarray:
        """(H,W,3) uint8 RGB from straight above the destination bin."""
        h = height or CAMERA_H
        w = width  or CAMERA_W
        if not _HAS_MUJOCO_BINDINGS:
            return np.zeros((h, w, 3), dtype=np.uint8)
        try:
            raw_model = getattr(self.sim.model, "_model", self.sim.model)
            raw_data  = getattr(self.sim.data,  "_data",  self.sim.data)
            if not hasattr(self, "_dest_renderer") or self._dest_renderer is None:
                self._dest_renderer = _mujoco.Renderer(raw_model, height=h, width=w)
            cam = _mujoco.MjvCamera()
            cam.type      = _mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = self._dst_bin_world.copy()
            cam.distance  = _CAM_MOUNT_HEIGHT
            cam.azimuth   = 90.0
            cam.elevation = -90.0
            self._dest_renderer.update_scene(raw_data, camera=cam,
                                             scene_option=self._scene_opt())
            return self._dest_renderer.render()
        except Exception:
            self._dest_renderer = None
            return np.zeros((h, w, 3), dtype=np.uint8)

    def _get_overview_renderer(self, h, w):
        """Return (renderer, camera) for the overview.

        Renderer cached (GPU alloc), but MjvCamera is recreated every call: update_scene()
        mutates the camera object in place, so reusing it drifts the view on reset.
        """
        if not hasattr(self, "_overview_renderer") or self._overview_renderer is None:
            raw_model = getattr(self.sim.model, "_model", self.sim.model)
            self._overview_renderer = _mujoco.Renderer(raw_model, height=h, width=w)
        cam = _mujoco.MjvCamera()
        cam.type      = _mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self._src_bin_world + np.array([0, 0, 0.13])
        cam.distance  = 0.6
        cam.azimuth   = 90.0
        cam.elevation = -60.0
        return self._overview_renderer, cam

    def get_agentview_rgb(self, height: int = None, width: int = None) -> np.ndarray:
        """(H,W,3) uint8 overview aimed at the source bin. Falls back to robosuite agentview."""
        h = height or CAMERA_H
        w = width  or CAMERA_W
        if _HAS_MUJOCO_BINDINGS:
            try:
                raw_data = getattr(self.sim.data, "_data", self.sim.data)
                renderer, cam = self._get_overview_renderer(h, w)
                renderer.update_scene(raw_data, camera=cam,
                                      scene_option=self._scene_opt())
                return renderer.render()
            except Exception:
                self._overview_renderer = None  # force recreate next call
        obs = self._get_observations()
        key = "agentview_image"
        if key in obs:
            img = obs[key]
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            return img
        return np.zeros((h, w, 3), dtype=np.uint8)

    def get_birdview_rgb(self, height: int = None, width: int = None) -> np.ndarray:
        """(H,W,3) uint8 near-top-down RGB of the source bin.

        Free cam at azimuth=90 (camera on +x side, away from robot arm), elevation=-80.
        Uses the custom mujoco.Renderer to avoid robosuite render orientation issues.
        """
        h = height or CAMERA_H
        w = width  or CAMERA_W
        if not _HAS_MUJOCO_BINDINGS:
            return np.zeros((h, w, 3), dtype=np.uint8)
        try:
            raw_model = getattr(self.sim.model, "_model", self.sim.model)
            raw_data  = getattr(self.sim.data,  "_data",  self.sim.data)
            if not hasattr(self, "_birdview_renderer") or self._birdview_renderer is None:
                self._birdview_renderer = _mujoco.Renderer(raw_model, height=h, width=w)
            cam = _mujoco.MjvCamera()
            cam.type      = _mujoco.mjtCamera.mjCAMERA_FREE
            mid_y = (self._src_bin_world[1] + self._dst_bin_world[1]) / 2.0
            cam.lookat[:] = np.array([self._src_bin_world[0], mid_y, 0.95])
            cam.distance  = 1.5
            cam.azimuth   = 90.0
            cam.elevation = -55.0
            self._birdview_renderer.update_scene(raw_data, camera=cam,
                                                  scene_option=self._scene_opt())
            return self._birdview_renderer.render()
        except Exception:
            self._birdview_renderer = None
            return np.zeros((h, w, 3), dtype=np.uint8)

    def get_top_front_rgb(self, height: int = None, width: int = None) -> np.ndarray:
        """(H,W,3) uint8 supervision view: front+above 3/4 angle, sees both bins + arm."""
        h = height or CAMERA_H
        w = width  or CAMERA_W
        if not _HAS_MUJOCO_BINDINGS:
            return np.zeros((h, w, 3), dtype=np.uint8)
        try:
            raw_model = getattr(self.sim.model, "_model", self.sim.model)
            raw_data  = getattr(self.sim.data,  "_data",  self.sim.data)
            # Fresh renderer every call: a cached mujoco.Renderer's MjvScene retained
            # state from the wrist-POV free cam (rendered right before this in the
            # same _capture_all loop), causing the top-front frames to randomly
            # switch to a wrist-POV closeup. Throwing the renderer away each frame
            # makes all captured frames consistent (~50% more wall time, correct).
            self._tf_renderer = _mujoco.Renderer(raw_model, height=h, width=w)
            self._tf_renderer.update_scene(raw_data, camera="agentview",
                                           scene_option=self._scene_opt())
            return self._tf_renderer.render()
        except Exception:
            self._tf_renderer = None
            return np.zeros((h, w, 3), dtype=np.uint8)

    def get_wrist_depth_meters(self, height=None, width=None) -> np.ndarray:
        """(H,W) float32 metric depth from the overhead camera above the source bin."""
        h = height or CAMERA_H
        w = width  or CAMERA_W
        if not _HAS_MUJOCO_BINDINGS:
            return np.zeros((h, w), dtype=np.float32)
        try:
            raw_model = getattr(self.sim.model, "_model", self.sim.model)
            raw_data  = getattr(self.sim.data,  "_data",  self.sim.data)
            if not hasattr(self, "_wrist_renderer") or self._wrist_renderer is None:
                self._wrist_renderer = _mujoco.Renderer(raw_model, height=h, width=w)
            self._wrist_renderer.enable_depth_rendering()
            cam = self._get_overhead_cam()
            self._wrist_renderer.update_scene(raw_data, camera=cam,
                                              scene_option=self._scene_opt())
            depth_buf = self._wrist_renderer.render()
            self._wrist_renderer.disable_depth_rendering()
            # mujoco.Renderer returns metric depth directly. Do NOT apply
            # depth_to_meters() here (that is for robosuite's normalised [0,1] buffer).
            return depth_buf.astype(np.float32)
        except Exception:
            self._wrist_renderer = None
            return np.zeros((h, w), dtype=np.float32)

    def get_eye_in_hand_rgb(self, height=None, width=None) -> np.ndarray:
        """(H,W,3) uint8 RGB from the wrist-mounted robot0_eye_in_hand camera.

        robosuite mounts this camera with a fixed tilt, so when the wrist yaws the
        view swings toward the horizon. For a clean top-down picking POV use
        get_wrist_pov_rgb() instead.
        """
        h = height or CAMERA_H
        w = width  or CAMERA_W
        if not _HAS_MUJOCO_BINDINGS:
            return np.zeros((h, w, 3), dtype=np.uint8)
        try:
            raw_model = getattr(self.sim.model, "_model", self.sim.model)
            raw_data  = getattr(self.sim.data,  "_data",  self.sim.data)
            if not hasattr(self, "_eih_renderer") or self._eih_renderer is None:
                self._eih_renderer = _mujoco.Renderer(raw_model, height=h, width=w)
            self._eih_renderer.update_scene(raw_data, camera="robot0_eye_in_hand",
                                            scene_option=self._scene_opt())
            return self._eih_renderer.render()
        except Exception:
            self._eih_renderer = None
            return np.zeros((h, w, 3), dtype=np.uint8)

    def get_wrist_pov_rgb(self, height=None, width=None) -> np.ndarray:
        """(H,W,3) uint8 RGB from a virtual wrist POV cam that FOLLOWS the EEF XY but always looks straight down.

        Camera sits above the wrist (clamped to a sensible overhead band) with
        distance growing with wrist height, so the bin / item / fingers fill the
        frame regardless of wrist yaw.
        """
        h = height or CAMERA_H
        w = width  or CAMERA_W
        if not _HAS_MUJOCO_BINDINGS:
            return np.zeros((h, w, 3), dtype=np.uint8)
        try:
            raw_model = getattr(self.sim.model, "_model", self.sim.model)
            raw_data  = getattr(self.sim.data,  "_data",  self.sim.data)
            if not hasattr(self, "_wpov_renderer") or self._wpov_renderer is None:
                self._wpov_renderer = _mujoco.Renderer(raw_model, height=h, width=w)
            eef_pos = self.get_robot_eef_pos()
            _BIN_FLOOR_Z = 0.80
            distance = float(np.clip(eef_pos[2] + 0.50 - _BIN_FLOOR_Z, 0.50, 1.00))
            cam = _mujoco.MjvCamera()
            # Initialize defaults (up, frustum, ...) BEFORE overwriting specific fields
            _mujoco.mjv_defaultFreeCamera(raw_model, cam)
            cam.type      = _mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance  = distance
            cam.lookat[:] = np.array([eef_pos[0], eef_pos[1], _BIN_FLOOR_Z])
            cam.azimuth   = 90.0
            cam.elevation = -90.0
            # Do NOT mutate mjModel.vis.global_.fovy: it is a SHARED global. Mutating
            # it here AND in get_top_front_rgb causes one camera's fovy to leak into
            # the other when both are called from the same _capture_all loop
            # (top-front frames sometimes rendered at the wrist-POV's narrow ~28 deg).
            # Achieve the narrow-overhead effect by keeping distance close instead.
            _saved_fovy = None
            self._wpov_renderer.update_scene(raw_data, camera=cam,
                                             scene_option=self._scene_opt())
            img = self._wpov_renderer.render()
            if _saved_fovy is not None:
                try:
                    raw_model.vis.global_.fovy = _saved_fovy
                except Exception:
                    pass
            return img
        except Exception:
            self._wpov_renderer = None
            return np.zeros((h, w, 3), dtype=np.uint8)

    def get_robot_eef_pos(self) -> np.ndarray:
        """(3,) EEF position in world frame.

        Reads sim body positions so it works immediately after mj_forward
        (set_sensing_pose_direct), not only after env.step().
        """
        try:
            bid = self.sim.model.body_name2id("robot0_right_hand")
            return np.array(self.sim.data.body_xpos[bid])
        except Exception:
            return np.array(self.robots[0].recent_ee_pose.last[:3])

    def get_robot_eef_quat(self) -> np.ndarray:
        """(4,) EEF quaternion (w,x,y,z) in world frame."""
        pose = self.robots[0].recent_ee_pose.last
        if len(pose) >= 7:
            xyzw = pose[3:7]
            if np.linalg.norm(xyzw) > 0.5:  # skip zero quat before first step
                return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
        return np.array([1.0, 0.0, 0.0, 0.0])

    def get_object_positions(self) -> dict:
        """Dict name -> (3,) world-frame position."""
        positions = {}
        for name in self._obj_names:
            try:
                bid = self.sim.model.body_name2id(name)
                pos = self.sim.data.body_xpos[bid].copy()
                positions[name] = pos
            except Exception:
                pass
        return positions

    def n_objects_in_dest(self) -> int:
        return self._reward_state.get_n_transferred()

    def get_obj_names(self):
        return list(self._obj_names)

    def get_src_bin_world_pos(self) -> np.ndarray:
        return self._src_bin_world.copy()

    def get_dst_bin_world_pos(self) -> np.ndarray:
        return self._dst_bin_world.copy()

    def get_camera_world_pos(self) -> np.ndarray:
        """World-frame position of the eye-in-hand sensing camera.

        20 cm below the EEF wrist so the arm/fingers are above the camera and
        invisible in the downward depth image.
        """
        return self._src_bin_world + np.array([0.0, 0.0, _CAM_MOUNT_HEIGHT])
