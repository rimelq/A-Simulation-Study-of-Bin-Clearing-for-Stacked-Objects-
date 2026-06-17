"""Multi-view scene renderer for the report.

Builds the same scene as the top-cube RL/GGCNN demos
(n_objects=17, layout_jitter=0.0, seed=1000, candidate_source='ppo'),
resets the env, puts the gripper at the sensing pose so the wrist is parked
above the source bin, and renders the scene from multiple free-cam viewpoints
using mujoco.MjvCamera. Outputs are saved as high-resolution PNGs to
results/env_views/.
"""

import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl.bin_clearing_env import BinClearingGymEnv


# Match the demo scene exactly.
_SCENE_KWARGS = dict(
    n_objects=17,
    K=10,
    candidate_source="ppo",
    ppo_visibility_mode="raycast",
    ppo_quality_mode="analytical",
    filter_candidates=True,
    orientation_source="snap",
    use_snap_xy=False,
    use_snap_z=True,
    layout_jitter=0.0,
    failure_mask_size=3,
    reward_mode="hybrid_physics",
)
_SEED = 1000

# Render resolution. 1920x1080 for nice report-quality images.
_RENDER_W = 1920
_RENDER_H = 1080

_OUT_DIR = os.path.join(_ROOT, "results", "env_views")


def _ensure_offscreen_buffer(env_inner, *, height, width):
    """MuJoCo's offscreen framebuffer is sized from model.vis.global_.offwidth /
    offheight at model load (defaults 640x480). Bump it up so high-resolution
    Renderers can be created."""
    raw_model = getattr(env_inner.sim.model, "_model", env_inner.sim.model)
    cur_w = int(raw_model.vis.global_.offwidth)
    cur_h = int(raw_model.vis.global_.offheight)
    if cur_w < width:
        raw_model.vis.global_.offwidth = int(width)
    if cur_h < height:
        raw_model.vis.global_.offheight = int(height)


def _render_free_cam(env_inner, *, azimuth, elevation, distance, lookat,
                     height, width):
    """Render the live MuJoCo scene from a free camera with the supplied
    parameters (degrees / metres / 3-vector in world frame)."""
    import mujoco as _mj
    raw_model = getattr(env_inner.sim.model, "_model", env_inner.sim.model)
    raw_data = getattr(env_inner.sim.data, "_data", env_inner.sim.data)
    renderer = _mj.Renderer(raw_model, height=height, width=width)
    cam = _mj.MjvCamera()
    cam.type = _mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.asarray(lookat, dtype=float)
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)
    cam.distance = float(distance)
    scene_opt = (env_inner._scene_opt()
                 if hasattr(env_inner, "_scene_opt") else None)
    if scene_opt is not None:
        renderer.update_scene(raw_data, camera=cam, scene_option=scene_opt)
    else:
        renderer.update_scene(raw_data, camera=cam)
    frame = renderer.render()
    # Free the GPU context promptly between views.
    try:
        renderer.close()
    except Exception:
        pass
    return frame


def _save_rgb_png(path, frame_rgb):
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def _viewpoints(src, dst):
    """Return the list of (filename, kwargs) tuples for the report views."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    mid = (src + dst) / 2.0
    return [
        ("wide_3q_front.png", dict(
            azimuth=180.0, elevation=-50.0, distance=1.05,
            lookat=mid + np.array([0.0, 0.0, 0.05]))),
        ("wide_side_right.png", dict(
            azimuth=90.0, elevation=-30.0, distance=1.30, lookat=mid)),
        ("wide_overhead.png", dict(
            azimuth=180.0, elevation=-85.0, distance=1.40, lookat=mid)),
        ("closeup_source_bin.png", dict(
            azimuth=180.0, elevation=-50.0, distance=0.50,
            lookat=src + np.array([0.0, 0.0, 0.05]))),
        ("wide_back.png", dict(
            azimuth=0.0, elevation=-30.0, distance=1.50, lookat=mid)),
        ("wide_low_angle.png", dict(
            azimuth=180.0, elevation=-12.0, distance=1.50,
            lookat=mid + np.array([0.0, 0.0, 0.10]))),
        ("iso_diagonal.png", dict(
            azimuth=135.0, elevation=-40.0, distance=1.30, lookat=mid)),
        ("closeup_dest_bin.png", dict(
            azimuth=180.0, elevation=-50.0, distance=0.50,
            lookat=dst + np.array([0.0, 0.0, 0.05]))),
    ]


def main():
    os.makedirs(_OUT_DIR, exist_ok=True)
    print(f"[setup] out_dir = {_OUT_DIR}")
    print(f"[setup] render = {_RENDER_W}x{_RENDER_H}")
    print(f"[setup] scene  = n_objects={_SCENE_KWARGS['n_objects']}, "
          f"seed={_SEED}, layout_jitter={_SCENE_KWARGS['layout_jitter']}, "
          f"candidate_source={_SCENE_KWARGS['candidate_source']}")

    env = BinClearingGymEnv(**_SCENE_KWARGS)
    obs, info = env.reset(seed=_SEED)
    raw_env = env.env
    print(f"[setup] reset done. n_items_remaining={info['n_items_remaining']}")

    # Park gripper at the sensing pose (above source bin), matching the demo.
    env.sensing_ctrl.set_sensing_pose_direct()
    raw_env._forward()

    # Bump the offscreen framebuffer so 1920x1080 renders are allowed.
    _ensure_offscreen_buffer(raw_env, height=_RENDER_H, width=_RENDER_W)

    src = np.asarray(raw_env.get_src_bin_world_pos(), dtype=float)
    dst = np.asarray(raw_env.get_dst_bin_world_pos(), dtype=float)
    print(f"[setup] src_bin = {src.tolist()}")
    print(f"[setup] dst_bin = {dst.tolist()}")

    views = _viewpoints(src, dst)
    print(f"[setup] rendering {len(views)} viewpoints ...")

    for fname, kw in views:
        out_path = os.path.join(_OUT_DIR, fname)
        try:
            frame = _render_free_cam(
                raw_env, height=_RENDER_H, width=_RENDER_W, **kw)
            _save_rgb_png(out_path, frame)
            h, w = frame.shape[:2]
            print(f"[save] {fname}  {w}x{h}  "
                  f"(az={kw['azimuth']:+.1f} el={kw['elevation']:+.1f} "
                  f"d={kw['distance']:.2f})")
        except Exception as e:
            print(f"[ERROR] {fname}: {type(e).__name__}: {e}")
            raise

    print(f"\n[done] {len(views)} views saved to {_OUT_DIR}")


if __name__ == "__main__":
    main()
