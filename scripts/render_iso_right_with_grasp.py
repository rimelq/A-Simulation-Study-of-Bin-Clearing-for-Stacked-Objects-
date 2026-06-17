"""Render a single iso-style frame from the opposite diagonal (az=225).

Scene: same as the RL top-cube demo (n=17, jitter=0, seed=1000).
Moment captured: gripper holds item_0015 above the destination bin, just
before release. Output: results/env_views/iso_diagonal_right_grasp.png.
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import mujoco
from sb3_contrib import MaskablePPO

from rl.bin_clearing_env import (
    BinClearingGymEnv, _decode_action_v5,
    _quat_mul_wxyz, _yaw_quat_wxyz,
)
from control.grasp_success_predicate import _FINGER_TO_WRIST

_RL_MODEL = os.path.join(_ROOT, "rl", "models", "best_model.zip")


def _render_free_cam(env_inner, *, azimuth, elevation, distance, lookat,
                     height=1080, width=1920):
    model = env_inner.sim.model._model
    # bump the offscreen framebuffer if the model was compiled at the default 640x480
    if model.vis.global_.offwidth < width or model.vis.global_.offheight < height:
        model.vis.global_.offwidth = max(width, int(model.vis.global_.offwidth))
        model.vis.global_.offheight = max(height, int(model.vis.global_.offheight))
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env_inner.sim.model._model, cam)
    cam.lookat[:] = lookat
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)
    cam.distance = float(distance)
    scene_opt = (env_inner._scene_opt()
                 if hasattr(env_inner, "_scene_opt") else None)
    if scene_opt is not None:
        renderer.update_scene(env_inner.sim.data._data, camera=cam,
                              scene_option=scene_opt)
    else:
        renderer.update_scene(env_inner.sim.data._data, camera=cam)
    rgb = renderer.render()
    renderer.close()
    return rgb


def main():
    env = BinClearingGymEnv(
        n_objects=17, K=10, candidate_source="ppo",
        ppo_visibility_mode="raycast", ppo_quality_mode="analytical",
        filter_candidates=True, orientation_source="snap",
        use_snap_xy=False, use_snap_z=True,
        layout_jitter=0.0, failure_mask_size=3,
        reward_mode="hybrid_physics",
    )
    obs, info = env.reset(seed=1000)
    raw = env.env
    pp = env.pick_place

    print("[1] loading RL model + selecting action")
    model = MaskablePPO.load(_RL_MODEL)
    mask = np.asarray(env.action_masks(), dtype=bool)
    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    action = int(action)
    k, dx, dy, dyaw = _decode_action_v5(action)
    chosen = env._candidates[k]
    print(f"    action={action}  slot k={k}  body={chosen['source_body_id']}")

    target_name = env._associate_item(chosen["world_pos"])
    grasp_pos = np.asarray(env._wrist_target_from_candidate(chosen), dtype=float)
    grasp_pos[0] += dx
    grasp_pos[1] += dy
    item_top_z = float(raw.get_object_positions()[target_name][2]) + 0.029
    grasp_pos[2] = item_top_z + _FINGER_TO_WRIST + 0.0
    cand_world_quat = np.asarray(chosen["world_quat"], dtype=float)
    grasp_quat = _quat_mul_wxyz(_yaw_quat_wxyz(float(dyaw)), cand_world_quat)
    above_pile = grasp_pos + np.array([0.0, 0.0, 0.12])

    print("[2] smooth approach to above_pile (no capture)")
    pp.frame_hook = None
    pp._smooth_teleport(above_pile, grasp_quat, n_steps=20)

    print("[3] running attempt_grasp_hybrid (no capture)")
    grasp_result = pp.attempt_grasp_hybrid(
        grasp_pos=grasp_pos, grasp_quat=grasp_quat,
        target_item_name=target_name,
        frame_hook=None, frame_hook_timing="pre_teleport",
    )
    assert grasp_result.get("grasp_ok"), f"grasp failed: {grasp_result}"
    picked = grasp_result.get("picked_item_name", target_name)
    print(f"    grasp_ok=True  picked={picked}")

    print("[4] attaching cube to gripper kinematically")
    # Canonical carry offset: cube sits at the finger-pad midpoint, which is
    # _FINGER_TO_WRIST below the wrist with zero XY offset. This avoids
    # baking in any pre-grasp or post-retract pose drift from attempt_grasp_hybrid.
    pp._tracked_item = picked
    pp._tracked_offset = np.array([0.0, 0.0, -_FINGER_TO_WRIST], dtype=float)
    pp._apply_contact_mode("ghost")
    pp._patch_wrist_limits()
    # Snap the cube to the correct carry pose immediately so any subsequent
    # rendering or physics step starts with cube-between-fingers.
    eef_now = np.asarray(raw.get_robot_eef_pos(), dtype=float)
    pp._move_object_qpos(picked, eef_now + pp._tracked_offset)
    raw._forward()

    print("[5] cinematic lift + transport + lower (no capture)")
    pp.frame_hook = None
    dst_bin = np.asarray(raw.get_dst_bin_world_pos(), dtype=float)

    wp_lift = grasp_pos.copy()
    wp_lift[2] += 0.20
    pp._smooth_teleport(wp_lift, grasp_quat, n_steps=20)

    wp_transport = np.array([dst_bin[0], dst_bin[1], wp_lift[2]], dtype=float)
    pp._smooth_teleport(wp_transport, grasp_quat, n_steps=30, null_gain=0.02)

    wp_lower = np.array([dst_bin[0], dst_bin[1], dst_bin[2] + 0.12], dtype=float)
    pp._smooth_teleport(wp_lower, grasp_quat, n_steps=20, null_gain=0.02)

    raw._forward()

    print("[6] rendering iso right view (az=225, el=-40, dist=1.30)")
    # Match the original iso_diagonal.png POV exactly, just mirrored to the
    # opposite diagonal. lookat is the midpoint of the two bins, slightly
    # raised so the gripper is in the upper third of the frame.
    src_bin = np.asarray(raw.get_src_bin_world_pos(), dtype=float)
    mid = 0.5 * (src_bin + dst_bin) + np.array([0.0, 0.0, 0.05])

    rgb = _render_free_cam(
        raw, azimuth=225.0, elevation=-40.0, distance=1.30,
        lookat=mid, height=1080, width=1920,
    )

    import imageio
    out_dir = os.path.join(_ROOT, "results", "env_views")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "iso_diagonal_right_grasp.png")
    imageio.imwrite(out_path, rgb)
    print(f"\nSaved: {out_path}")
    print(f"       dims: {rgb.shape}")


if __name__ == "__main__":
    main()
