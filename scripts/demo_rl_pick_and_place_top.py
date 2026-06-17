"""
RL pick + place TOP-cube cinematic showcase.

MaskablePPO selects a candidate slot k and a (dx, dy, dyaw) refinement on the
n_objects=17, seed=1000, layout_jitter=0.0 scene. After
attempt_grasp_hybrid succeeds, the demo extends the video with a CINEMATIC
transport sequence (lift -> transport -> lower -> release) by kinematically
pinning the picked item to the EEF via PickPlacePrimitive._tracked_item.

outcome.json records placement result (in_target_bin, picked_item_final_xyz,
cinematic_transport_seconds, n_frames_total).
"""

import os
import sys
import csv
import json
import argparse
import datetime
import traceback
import numpy as np
import cv2
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sb3_contrib import MaskablePPO

from sim.camera_setup import get_camera_intrinsics
from sim.sensing_pose import BIN_HALF_SIZE
from rl.bin_clearing_env import (
    BinClearingGymEnv,
    _GRASP_DESCENT_OFFSET,
    _ITEM_HALF_HEIGHT,
    _decode_action_v5,
    _quat_mul_wxyz,
    _yaw_quat_wxyz,
    N_REFINEMENT,
)
from control.grasp_success_predicate import _FINGER_TO_WRIST, evaluate_grasp


_CAM_W = 640
_CAM_H = 480
_GGCNN_SIZE = 300
_RENDER_FOVY = 45.0   # FOV 45 deg matches the MuJoCo global render setting.
_PAD_Y = (_CAM_W - _CAM_H) // 2   # 80
_CAM_QUAT_WORLD = np.array([0.0, 1.0, 0.0, 0.0])   # wxyz

# Display-only crop margins.
_BIN_CROP_MARGIN_PX_640 = 60
_BIN_CROP_MARGIN_PX_300 = 14

# Default jaw-line length when the candidate source does not supply width_px.
_DEFAULT_WIDTH_PX_300 = 30.0

# Frozen MaskablePPO weights, resolved from the submission root.
_DEFAULT_RL_MODEL_PATH = os.path.join(
    _ROOT, "rl", "models", "best_model.zip")


def _bin_bounds_full(cam_pos, src_bin):
    """Project the source-bin floor rectangle (at z = src_bin[2]) into the
    640x480 overhead image and return (x0, y0, x1, y1) clipped to image, plus
    a cosmetic margin. Camera convention:
        Xc = X_world - cam_x . Yc = -(Y_world - cam_y) . Zc = cam_z - Z_world
        px = Xc*fx/Zc + cx . py = Yc*fy/Zc + cy
    """
    K_back = get_camera_intrinsics(_RENDER_FOVY, _CAM_W, _CAM_H)
    fx = float(K_back[0, 0])
    fy = float(K_back[1, 1])
    cx = float(K_back[0, 2])
    cy = float(K_back[1, 2])
    Zc = float(cam_pos[2] - src_bin[2])
    if Zc <= 1e-6:
        return 0, 0, _CAM_W, _CAM_H
    hx = float(BIN_HALF_SIZE[0])
    hy = float(BIN_HALF_SIZE[1])
    corners_world_xy = [
        (src_bin[0] - hx, src_bin[1] - hy),
        (src_bin[0] - hx, src_bin[1] + hy),
        (src_bin[0] + hx, src_bin[1] - hy),
        (src_bin[0] + hx, src_bin[1] + hy),
    ]
    xs, ys = [], []
    for Xw, Yw in corners_world_xy:
        Xc = Xw - cam_pos[0]
        Yc = -(Yw - cam_pos[1])
        px = Xc * fx / Zc + cx
        py = Yc * fy / Zc + cy
        xs.append(px)
        ys.append(py)
    x0 = int(np.floor(min(xs))) - _BIN_CROP_MARGIN_PX_640
    x1 = int(np.ceil (max(xs))) + _BIN_CROP_MARGIN_PX_640
    y0 = int(np.floor(min(ys))) - _BIN_CROP_MARGIN_PX_640
    y1 = int(np.ceil (max(ys))) + _BIN_CROP_MARGIN_PX_640
    x0 = int(np.clip(x0, 0, _CAM_W))
    x1 = int(np.clip(x1, 0, _CAM_W))
    y0 = int(np.clip(y0, 0, _CAM_H))
    y1 = int(np.clip(y1, 0, _CAM_H))
    if x1 <= x0:
        x0, x1 = 0, _CAM_W
    if y1 <= y0:
        y0, y1 = 0, _CAM_H
    return x0, y0, x1, y1


def _project_world_to_full_pixels(world_pos, cam_pos):
    K_back = get_camera_intrinsics(_RENDER_FOVY, _CAM_W, _CAM_H)
    fx = float(K_back[0, 0])
    fy = float(K_back[1, 1])
    cx = float(K_back[0, 2])
    cy = float(K_back[1, 2])
    Xw, Yw, Zw = float(world_pos[0]), float(world_pos[1]), float(world_pos[2])
    Xc = Xw - float(cam_pos[0])
    Yc = -(Yw - float(cam_pos[1]))
    Zc = float(cam_pos[2]) - Zw
    if Zc <= 1e-6:
        return float("nan"), float("nan")
    px = Xc * fx / Zc + cx
    py = Yc * fy / Zc + cy
    return float(px), float(py)


def capture_overhead_for_viz(env_inner, sensing_ctrl):
    """Set sensing pose, render wrist depth + meta. PPO candidates already
    come from env._candidates, this only produces viz images."""
    sensing_ctrl.set_sensing_pose_direct()
    env_inner._forward()

    cam_pos = env_inner.get_camera_world_pos()
    src_bin = env_inner.get_src_bin_world_pos()
    floor_depth = float(cam_pos[2] - src_bin[2])

    depth = env_inner.get_wrist_depth_meters(
        height=_CAM_H, width=_CAM_W).astype(np.float32)
    bad = ~np.isfinite(depth) | (depth <= 0)
    depth[bad] = floor_depth
    # Gripper-mask cutoff (relaxed): preserves stacked cube tops.
    depth[depth < floor_depth - 0.18] = floor_depth

    K_back = get_camera_intrinsics(_RENDER_FOVY, _CAM_W, _CAM_H)

    return {
        "cam_pos": cam_pos,
        "K_back": K_back,
        "floor_depth": floor_depth,
        "src_bin_world": src_bin,
        "depth_full": depth,
        "pad_y": _PAD_Y,
        "scale_300_to_640": _CAM_W / float(_GGCNN_SIZE),
    }


def _render_three_quarter(env_inner, height=480, width=640):
    import mujoco as _mj
    raw_model = getattr(env_inner.sim.model, "_model", env_inner.sim.model)
    raw_data = getattr(env_inner.sim.data, "_data", env_inner.sim.data)
    renderer = _mj.Renderer(raw_model, height=height, width=width)
    cam = _mj.MjvCamera()
    cam.type = _mj.mjtCamera.mjCAMERA_FREE
    src = env_inner.get_src_bin_world_pos()
    dst = env_inner.get_dst_bin_world_pos()
    midpoint = (np.asarray(src, dtype=float) + np.asarray(dst, dtype=float)) / 2.0
    cam.lookat[:] = midpoint + np.array([0.0, 0.0, 0.05])
    cam.azimuth = 180.0
    cam.elevation = -50.0
    cam.distance = 1.05
    scene_opt = env_inner._scene_opt() if hasattr(env_inner, "_scene_opt") else None
    if scene_opt is not None:
        renderer.update_scene(raw_data, camera=cam, scene_option=scene_opt)
    else:
        renderer.update_scene(raw_data, camera=cam)
    return renderer.render()


def _render_top_down(env_inner, height=480, width=640):
    import mujoco as _mj
    raw_model = getattr(env_inner.sim.model, "_model", env_inner.sim.model)
    raw_data = getattr(env_inner.sim.data, "_data", env_inner.sim.data)
    renderer = _mj.Renderer(raw_model, height=height, width=width)
    cam = _mj.MjvCamera()
    cam.type = _mj.mjtCamera.mjCAMERA_FREE
    src = env_inner.get_src_bin_world_pos()
    cam.lookat[:] = np.asarray(src, dtype=float) + np.array([0.0, 0.0, 0.0])
    cam.azimuth = 180.0
    cam.elevation = -89.5
    cam.distance = 0.75
    scene_opt = env_inner._scene_opt() if hasattr(env_inner, "_scene_opt") else None
    if scene_opt is not None:
        renderer.update_scene(raw_data, camera=cam, scene_option=scene_opt)
    else:
        renderer.update_scene(raw_data, camera=cam)
    return renderer.render()


def _save_rgb_png(path, frame_rgb, title=None):
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if title:
        cv2.putText(bgr, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
    cv2.imwrite(path, bgr)


def _annotate_frame(frame_rgb, overlay_lines, grasp_state="PENDING"):
    """Overlay text lines + a colour-coded grasp status line.
    grasp_state in {"PENDING" -> white, "SUCCESS" -> green, "FAIL" -> red}."""
    out = frame_rgb.copy()
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    pad = 6
    line_h = 18
    n = len(overlay_lines)
    box_h = 10 + (n + 1) * line_h + pad
    cv2.rectangle(bgr, (5, 5), (380, box_h), (0, 0, 0), -1)
    for i, line in enumerate(overlay_lines):
        cv2.putText(bgr, line, (10, 24 + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    color_map = {
        "PENDING": (255, 255, 255),
        "SUCCESS": (0, 255, 0),
        "FAIL":    (0, 0, 255),
    }
    grasp_color = color_map.get(grasp_state, (255, 255, 255))
    cv2.putText(bgr, f"grasp: {grasp_state}",
                (10, 24 + n * line_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, grasp_color, 2)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _save_video(path, frames, fps=20):
    if not frames:
        print(f"[video] No frames captured; skipping {path}")
        return
    imageio.mimsave(path, frames, fps=fps)
    print(f"[video] Saved {path}  ({len(frames)} frames @ {fps} fps "
          f"= {len(frames) / fps:.2f}s)")


# Style constants for the candidate overlay (red = non-chosen, green = chosen).
_RED_MARKER_S        = 80.0
_RED_MARKER_EDGE_LW  = 0.7
_RED_TEXT_FS         = 7.5
_RED_LINE_LW         = 1.6
_RED_LINE_ZORDER     = 2
_RED_MARKER_ZORDER   = 2
_RED_TEXT_ZORDER     = 3

_GREEN_MARKER_S      = 320.0
_GREEN_MARKER_EDGE_LW = 2.0
_GREEN_TEXT_FS       = 10.0
_GREEN_LINE_LW       = 3.0
_GREEN_LINE_ZORDER   = 10
_GREEN_MARKER_ZORDER = 10
_GREEN_TEXT_ZORDER   = 11

_GREEN_TEXT_BBOX = dict(facecolor="white", alpha=0.85,
                        edgecolor="green", boxstyle="round,pad=0.3")
_RED_TEXT_BBOX = dict(boxstyle="round,pad=0.12",
                      facecolor="black", alpha=0.55, edgecolor="none")


def _save_depth_full(path, depth_full, title, bin_bounds=None):
    """640x480 overhead depth, optionally bin-cropped, then rotated 90 deg CW."""
    valid = depth_full[np.isfinite(depth_full) & (depth_full > 0)]
    if len(valid) > 0:
        d_min = float(np.percentile(valid, 1))
        d_max = float(np.percentile(valid, 99))
    else:
        d_min, d_max = 0.4, 1.2
    if bin_bounds is not None:
        x0, y0, x1, y1 = bin_bounds
        depth_show = depth_full[y0:y1, x0:x1]
    else:
        depth_show = depth_full
    depth_disp = np.rot90(depth_show, k=-1)
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(depth_disp, cmap="viridis", vmin=d_min, vmax=d_max)
    ax.set_title("Overhead wrist depth", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.04, label="depth (m)")
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _draw_predictions_on_axis_rotated(ax, candidates, chosen_idx, H_orig,
                                       crop_x0=0, crop_y0=0,
                                       crop_x1=_CAM_W, crop_y1=_CAM_H,
                                       chosen_label_extra=""):
    """Annotate predictions on a CW-90 rotated 640x480 image. (px, py) shifted
    into the crop frame, then remapped via (px, py) -> (H-1-py, px)."""
    width_scale = (_CAM_W / float(_GGCNN_SIZE))

    def _disp_coords(px, py):
        px_c = px - crop_x0
        py_c = py - crop_y0
        return (H_orig - 1 - py_c, px_c)

    for i, c in enumerate(candidates):
        if i == chosen_idx:
            continue
        x = float(c["px_full"])
        y = float(c["py_full"])
        if not (crop_x0 <= x < crop_x1 and crop_y0 <= y < crop_y1):
            continue
        x_d, y_d = _disp_coords(x, y)
        ax.scatter([x_d], [y_d], s=_RED_MARKER_S,
                   facecolor="red", edgecolor="black",
                   linewidths=_RED_MARKER_EDGE_LW,
                   zorder=_RED_MARKER_ZORDER)
        ax.text(x_d + 8, y_d - 8,
                f"#{i} q={float(c['quality']):.2f}",
                color="red", fontsize=_RED_TEXT_FS, weight="bold",
                bbox=_RED_TEXT_BBOX,
                zorder=_RED_TEXT_ZORDER)

    # Chosen drawn LAST so it wins z-order.
    if 0 <= chosen_idx < len(candidates):
        c = candidates[chosen_idx]
        x = float(c["px_full"])
        y = float(c["py_full"])
        inside = (crop_x0 <= x < crop_x1 and crop_y0 <= y < crop_y1)
        if inside:
            x_d, y_d = _disp_coords(x, y)
            ax.scatter([x_d], [y_d],
                       s=_GREEN_MARKER_S * 1.6,
                       facecolor="white", edgecolor="white",
                       lw=0.0, zorder=_GREEN_MARKER_ZORDER)
            ax.scatter([x_d], [y_d], marker="*",
                       s=_GREEN_MARKER_S,
                       facecolor="lime", edgecolor="black",
                       linewidths=_GREEN_MARKER_EDGE_LW,
                       zorder=_GREEN_MARKER_ZORDER + 1)
            label = (f"[CHOSEN] #{chosen_idx} q={float(c['quality']):.3f}")
            if chosen_label_extra:
                label = label + "\n" + chosen_label_extra
            ax.text(x_d + 10, y_d - 10, label,
                    color="darkgreen", fontsize=_GREEN_TEXT_FS, weight="bold",
                    bbox=_GREEN_TEXT_BBOX,
                    zorder=_GREEN_TEXT_ZORDER)


def _save_overlay_on_image(path, image, candidates, chosen_idx,
                           is_depth, suptitle, caption, bin_bounds=None,
                           chosen_label_extra=""):
    """Overlay candidates on 480x640 RGB/depth, bin-cropped + CW-90 rotated."""
    if bin_bounds is not None:
        x0, y0, x1, y1 = bin_bounds
        image_show = image[y0:y1, x0:x1]
    else:
        x0, y0, x1, y1 = 0, 0, image.shape[1], image.shape[0]
        image_show = image
    H_orig = image_show.shape[0]
    W_orig = image_show.shape[1]
    image_disp = np.rot90(image_show, k=-1)
    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    if is_depth:
        valid = image[np.isfinite(image) & (image > 0)]
        if len(valid) > 0:
            d_min = float(np.percentile(valid, 1))
            d_max = float(np.percentile(valid, 99))
        else:
            d_min, d_max = 0.4, 1.2
        ax.imshow(image_disp, cmap="viridis", vmin=d_min, vmax=d_max)
    else:
        ax.imshow(image_disp)
    _draw_predictions_on_axis_rotated(
        ax, candidates, chosen_idx,
        H_orig=H_orig,
        crop_x0=x0, crop_y0=y0,
        crop_x1=x1, crop_y1=y1,
        chosen_label_extra=chosen_label_extra)
    ax.set_xlim(0, H_orig)
    ax.set_ylim(W_orig, 0)
    ax.set_aspect("equal")
    ax.set_title(suptitle, fontsize=11)
    ax.set_xlabel(caption, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _save_three_quarter_view(path, frame_rgb, title):
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(bgr, title, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(path, bgr)


def _verify_bin_wall_height():
    """Wall height must stay at 0.100 to match eval/training."""
    path = os.path.join(_ROOT, "sim", "robosuite_env.py")
    with open(path, "r") as fh:
        src = fh.read()
    needle = "_BIN_WALL_H    = 0.100"
    if needle not in src:
        raise SystemExit(
            "[FATAL] sim/robosuite_env.py does not have _BIN_WALL_H = 0.100. "
            "Wall height MUST stay at 0.100 to match eval/training. Aborting.")
    print(f"[wall-check] OK: _BIN_WALL_H = 0.100 in {path}")


def _associate_body(env_outer, world_xy):
    try:
        items = env_outer._source_bin_items()
    except Exception:
        return "?"
    if not items:
        return "?"
    name, _pos = min(items, key=lambda np_: float(np.linalg.norm(
        np.asarray(np_[1][:2]) - np.asarray(world_xy[:2]))))
    return name


def _snapshot_all_cube_xyz(env_inner):
    snap = {}
    for name in env_inner.get_obj_names():
        try:
            bid = env_inner.sim.model.body_name2id(name)
            snap[name] = np.array(env_inner.sim.data.body_xpos[bid],
                                  dtype=float).copy()
        except Exception:
            continue
    return snap


def _in_src_bin(p, src_xyz):
    return (abs(p[0] - src_xyz[0]) < BIN_HALF_SIZE[0]
            and abs(p[1] - src_xyz[1]) < BIN_HALF_SIZE[1]
            and p[2] > src_xyz[2] - 0.05)


def _augment_ppo_candidates_for_viz(candidates, cam_pos, env_inner=None):
    """DISPLAY-only: use the GT body COM (env_inner.sim.data.body_xpos[bid])
    instead of candidate.world_pos for dot placement.

    PPO computes world_pos = body_com + R_item @ [0, 0, _ITEM_HALF_HEIGHT] (the
    cube TOP face centre). This is correct only for upright cubes. when a cube
    lies on a side face, world_pos becomes the SIDE face centre (~2.9 cm off
    from COM) and the rendered dot lands next to the cube. We only relocate the
    dot here. the RL policy, analytical quality, and eval grasp_pos still use
    the original world_pos.
    """
    augmented = []
    for c in candidates:
        c2 = dict(c)
        if env_inner is not None and c2.get("source_body_id", -1) >= 0:
            bid = int(c2["source_body_id"])
            com_world = np.array(env_inner.sim.data.body_xpos[bid], dtype=float)
            disp_world = com_world
        else:
            disp_world = np.asarray(c2["world_pos"], dtype=float)
        px640, py480 = _project_world_to_full_pixels(disp_world, cam_pos)
        c2["px_full"] = float(px640)
        c2["py_full"] = float(py480)
        # PPO sets px=-1, py=-1. keep as sentinels.
        c2.setdefault("px300", -1)
        c2.setdefault("py300", -1)
        wpx = float(c2.get("width_px", 0.0) or 0.0)
        if wpx <= 0.0:
            c2["width_px"] = _DEFAULT_WIDTH_PX_300
        else:
            c2["width_px"] = wpx
        augmented.append(c2)
    return augmented


def run(args):
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "log.txt")
    log_fh = open(log_path, "w")

    def _log(msg):
        print(msg)
        log_fh.write(msg + "\n")
        log_fh.flush()

    _log("=" * 78)
    _log("RL PICK + PLACE TOP-CUBE SHOWCASE (MaskablePPO + GT ghost candidates)")
    _log("  layout_jitter = 0.0   (no jitter; cubes spawn at mesh-defined")
    _log("                          stack positions from poses_relative_to_")
    _log("                          container.csv via object_spawner.py)")
    _log(f"  n_objects     = {args.n_objects}   (matched value from n-sweep at")
    _log("                          seed=1000, layout_jitter=0.0)")
    _log("  Cinematic post-grasp transport: lift -> transport -> lower -> drop")
    _log("  Goal: showcase RL picking a TOP-LAYER cube (body_com_z >= 0.86)")
    _log("        AND placing it in the destination bin.")
    _log(f"datetime    : {datetime.datetime.now().isoformat(timespec='seconds')}")
    _log(f"out_dir     : {out_dir}")
    _log(f"rl_model    : {args.rl_model_path}")
    _log("=" * 78)

    _verify_bin_wall_height()

    _log("\n[1] Building BinClearingGymEnv (RL eval config, hybrid_physics)")
    _log(f"    n_objects          = {args.n_objects}")
    _log(f"    layout_jitter      = {args.layout_jitter}")
    _log(f"    candidate_source   = 'ppo' (Perfect-Perception Oracle ghost)")
    _log(f"    ppo_visibility_mode= 'raycast'")
    _log(f"    ppo_quality_mode   = 'analytical'")
    _log(f"    orientation_source = 'snap'")
    _log(f"    use_snap_xy        = False  (RL controls XY)")
    _log(f"    use_snap_z         = True   (GT-snapped Z)")
    _log(f"    failure_mask_size  = {args.failure_mask_size}")
    _log(f"    reward_mode        = 'hybrid_physics'")
    _log(f"    seed               = {args.seed}")

    env = BinClearingGymEnv(
        n_objects=args.n_objects,
        K=args.K,
        candidate_source="ppo",
        ppo_visibility_mode="raycast",
        ppo_quality_mode="analytical",
        filter_candidates=True,
        orientation_source="snap",
        use_snap_xy=False,
        use_snap_z=True,
        layout_jitter=args.layout_jitter,
        failure_mask_size=args.failure_mask_size,
        reward_mode="hybrid_physics",
    )

    obs, info = env.reset(seed=args.seed)
    raw_env = env.env
    _log(f"\n[1b] reset done. n_items_remaining={info['n_items_remaining']}")
    _log(f"    obs shape          = {obs.shape}  dtype = {obs.dtype}")
    _log(f"    action_space       = {env.action_space}")

    # PRE-CONTACT PILE-PIN FIX:
    # _reset_internal places the source pile via kinematic _set_object_pose +
    # _forward(), but never calls _settle_source_pile. The first mj_step
    # (inside attempt_grasp_hybrid stage 2b) therefore resolves residual
    # gravity/overlap stresses on cubes that haven't been touched by the
    # gripper yet, producing a visible "drift" before contact. Fix: snapshot
    # the freejoint qpos of every source-bin cube RIGHT NOW (the kinematic
    # rest state), and re-pin in _capture_one every render frame during the
    # pre-grasp smooth approach. attempt_grasp_hybrid's own pinning takes
    # over once it starts; we then re-snapshot for the post-grasp phase.
    pre_grasp_pile_snapshot = None
    try:
        pre_grasp_pile_snapshot = env.pick_place._snapshot_item_poses()
        _log(f"    pre-grasp pile snapshot: "
             f"{len(pre_grasp_pile_snapshot)} items pinned for pre-contact phase")
    except Exception as _e_presnap:
        _log(f"    [WARN] failed to take pre-grasp pile snapshot: {_e_presnap}")

    _log("\n[2] Capturing overhead depth + reading PPO ghost candidates ...")
    meta = capture_overhead_for_viz(env_inner=env.env,
                                    sensing_ctrl=env.sensing_ctrl)
    cam_pos = meta["cam_pos"]
    floor_depth = meta["floor_depth"]
    depth_full = meta["depth_full"]
    candidates_raw = list(env._candidates)
    candidates = _augment_ppo_candidates_for_viz(candidates_raw, cam_pos,
                                                 env_inner=env.env)
    _log(f"    cam_pos          : {cam_pos}")
    _log(f"    floor_depth      : {floor_depth:.4f} m")
    _log(f"    pad_y            : {meta['pad_y']}  (640x480 -> 640x640)")
    _log(f"    scale 300->640   : {meta['scale_300_to_640']:.4f}")
    _log(f"    n PPO candidates : {len(candidates)}")

    src_bin_world_for_crop = meta["src_bin_world"]
    bin_bounds_640 = _bin_bounds_full(cam_pos, src_bin_world_for_crop)
    _log(f"    bin_bounds 640   : x[{bin_bounds_640[0]},{bin_bounds_640[2]})  "
         f"y[{bin_bounds_640[1]},{bin_bounds_640[3]})  (display crop)")

    if not candidates:
        raise SystemExit("[FATAL] No PPO ghost candidates produced. Aborting.")

    _log("\n[3] Capturing overhead RGB ...")
    rgb_full = env.env.get_wrist_rgb(height=_CAM_H, width=_CAM_W)
    _log(f"    rgb_full shape   : {rgb_full.shape}  dtype: {rgb_full.dtype}")

    _log("\n[4] Loading MaskablePPO and selecting action ...")
    if not os.path.isfile(args.rl_model_path):
        raise SystemExit(
            f"[FATAL] RL model not found at {args.rl_model_path}. "
            f"Override with --rl_model_path.")
    model = MaskablePPO.load(args.rl_model_path)
    mask = np.asarray(env.action_masks(), dtype=bool)
    n_unmasked = int(mask.sum())
    action_pred, _ = model.predict(obs, action_masks=mask,
                                   deterministic=True)
    action = int(action_pred)
    chosen_idx, dx, dy, dyaw = _decode_action_v5(action)
    sub_action = int(action) % N_REFINEMENT
    _log(f"    n_unmasked actions : {n_unmasked} / {mask.size}")
    _log(f"    RL action          : {action}  -> (k={chosen_idx}, m={sub_action})")
    _log(f"    decoded refinement : dx={dx:+.4f} m  dy={dy:+.4f} m  "
         f"dyaw={dyaw:+.4f} rad")

    if not (0 <= chosen_idx < len(candidates)):
        raise SystemExit(
            f"[FATAL] RL selected slot k={chosen_idx} but only "
            f"{len(candidates)} candidates available. Action mask broken?")
    chosen = candidates[chosen_idx]
    _log(f"    chosen world_pos   : {chosen['world_pos']}")
    _log(f"    chosen quality     : {float(chosen['quality']):.4f}")
    _log(f"    chosen src_body_id : {chosen.get('source_body_id', -1)}")

    chosen_label_extra = ""

    depth_full_path = os.path.join(out_dir, "depth_full.png")
    _save_depth_full(
        depth_full_path, depth_full,
        title="Overhead wrist depth",
        bin_bounds=bin_bounds_640)
    _log(f"[save] {depth_full_path}")

    overlay_caption = ("red = ghost-oracle candidates,  green = RL chosen")
    rgb_overlay_path = os.path.join(out_dir, "overhead_rgb_with_predictions.png")
    _save_overlay_on_image(
        rgb_overlay_path, rgb_full, candidates, chosen_idx,
        is_depth=False,
        suptitle="RL grasp candidates",
        caption=overlay_caption,
        bin_bounds=bin_bounds_640,
        chosen_label_extra=chosen_label_extra)
    _log(f"[save] {rgb_overlay_path}")

    depth_overlay_path = os.path.join(out_dir, "depth_full_with_predictions.png")
    _save_overlay_on_image(
        depth_overlay_path, depth_full, candidates, chosen_idx,
        is_depth=True,
        suptitle="Depth with RL candidates",
        caption=overlay_caption,
        bin_bounds=bin_bounds_640,
        chosen_label_extra=chosen_label_extra)
    _log(f"[save] {depth_overlay_path}")

    _log("\n[5] Rendering 3/4 oblique view (UN-rotated) ...")
    three_quarter = _render_three_quarter(env.env,
                                          height=_CAM_H, width=_CAM_W)
    three_q_path = os.path.join(out_dir, "frontview_3q.png")
    _save_three_quarter_view(
        three_q_path, three_quarter,
        title="3/4 oblique view (scene)")
    _log(f"[save] {three_q_path}")

    candidates_csv = os.path.join(out_dir, "candidates.csv")
    info_txt = os.path.join(out_dir, "per_candidate_info.txt")
    with open(candidates_csv, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow([
            "idx", "body", "source_body_id", "quality",
            "pred_world_x", "pred_world_y", "pred_world_z",
            "pred_angle_rad", "pred_width_px",
            "world_quat_w", "world_quat_x", "world_quat_y", "world_quat_z",
            "chosen", "rl_action", "chosen_dx", "chosen_dy", "chosen_dyaw",
        ])
        with open(info_txt, "w") as txt:
            txt.write("Per-candidate RL eval output (candidate_source='ppo')\n")
            txt.write("=" * 78 + "\n")
            txt.write(f"datetime          : "
                      f"{datetime.datetime.now().isoformat(timespec='seconds')}\n")
            txt.write(f"rl_model_path     : {args.rl_model_path}\n")
            txt.write(f"n_objects         : {args.n_objects}\n")
            txt.write(f"layout_jitter     : {args.layout_jitter}\n")
            txt.write(f"seed              : {args.seed}\n")
            txt.write(f"floor_depth       : {floor_depth:.4f} m\n")
            txt.write(f"cam_pos           : {cam_pos}\n")
            txt.write(f"render_fovy       : {_RENDER_FOVY} deg "
                      f"(K_back at {_CAM_W}x{_CAM_H})\n")
            txt.write(f"chosen_idx (k)    : {chosen_idx}\n")
            txt.write(f"rl_action (raw)   : {action}\n")
            txt.write(f"sub_action (m)    : {sub_action}\n")
            txt.write(f"refinement        : dx={dx:+.4f} m  dy={dy:+.4f} m  "
                      f"dyaw={dyaw:+.4f} rad\n")
            txt.write(f"n_unmasked actions: {n_unmasked} / {mask.size}\n")
            txt.write("\nAction decoding (v5): action = k*27 + m, "
                      "m = dx_idx*9 + dy_idx*3 + dyaw_idx, idx in {0,1,2}, "
                      "dx,dy in {-0.015,0,+0.015} m, dyaw in {-0.1745,0,+0.1745} rad.\n")
            txt.write("=" * 78 + "\n\n")

            for i, c in enumerate(candidates):
                wp = np.asarray(c["world_pos"], dtype=float)
                wq = np.asarray(c["world_quat"], dtype=float)
                body_name = _associate_body(env, wp[:2])
                row = [
                    i, body_name,
                    int(c.get("source_body_id", -1)),
                    f"{float(c['quality']):.4f}",
                    f"{wp[0]:.4f}", f"{wp[1]:.4f}", f"{wp[2]:.4f}",
                    f"{float(c['angle_rad']):.4f}",
                    f"{float(c['width_px']):.2f}",
                    f"{wq[0]:.4f}", f"{wq[1]:.4f}",
                    f"{wq[2]:.4f}", f"{wq[3]:.4f}",
                    int(i == chosen_idx),
                    int(action) if i == chosen_idx else -1,
                    f"{dx:+.4f}" if i == chosen_idx else "",
                    f"{dy:+.4f}" if i == chosen_idx else "",
                    f"{dyaw:+.4f}" if i == chosen_idx else "",
                ]
                wr.writerow(row)
                chosen_tag = "  [CHOSEN by RL]" if i == chosen_idx else ""
                txt.write(f"[#{i:2d}] body~={body_name}  "
                          f"q={float(c['quality']):.4f}{chosen_tag}\n")
                txt.write(f"     source_body_id = "
                          f"{int(c.get('source_body_id', -1))}\n")
                txt.write(f"     px_full=({int(round(c['px_full'])):3d},"
                          f"{int(round(c['py_full'])):3d})  "
                          f"(projection-only, viz)\n")
                txt.write(f"     width_px={float(c['width_px']):6.2f}  "
                          f"angle_rad={float(c['angle_rad']):+.4f}  "
                          f"depth_m={float(c.get('depth_m', 0.0)):.4f}\n")
                txt.write(f"     world_xyz=({wp[0]:+.4f},{wp[1]:+.4f},"
                          f"{wp[2]:+.4f})\n")
                txt.write(f"     world_quat (wxyz) = "
                          f"({wq[0]:+.4f},{wq[1]:+.4f},"
                          f"{wq[2]:+.4f},{wq[3]:+.4f})\n")
                if i == chosen_idx:
                    txt.write(f"     rl_action={action}  sub_action(m)={sub_action}\n")
                    txt.write(f"     dx={dx:+.4f}  dy={dy:+.4f}  "
                              f"dyaw={dyaw:+.4f}\n")
                txt.write("\n")
    _log(f"[save] {candidates_csv}")
    _log(f"[save] {info_txt}")

    # Replicates env.step() for this action: base wrist target + (dx,dy,0)
    # GT-snapped Z, yaw composed via _quat_mul_wxyz(_yaw_quat(dyaw), cand_quat).
    grasp_pos_base = env._wrist_target_from_candidate(chosen)
    grasp_pos_refined = grasp_pos_base + np.array(
        [float(dx), float(dy), 0.0], dtype=float)
    assoc_name = env._associate_item(grasp_pos_refined)
    if assoc_name is None:
        raise SystemExit(
            f"[FATAL] RL-chosen candidate #{chosen_idx} (k={chosen_idx}, "
            f"m={sub_action}) could not be associated to any source-bin item.")
    item_pos = np.asarray(raw_env.get_object_positions()[assoc_name],
                          dtype=float)
    item_top = float(item_pos[2]) + _ITEM_HALF_HEIGHT
    grasp_pos = np.array([
        grasp_pos_refined[0], grasp_pos_refined[1],
        item_top - _GRASP_DESCENT_OFFSET + _FINGER_TO_WRIST,
    ], dtype=float)
    cand_world_quat = np.asarray(chosen["world_quat"], dtype=float)
    grasp_quat = _quat_mul_wxyz(_yaw_quat_wxyz(float(dyaw)), cand_world_quat)
    _log(f"\n[6] Eval grasp_pos / grasp_quat assembled (RL refinement applied)")
    _log(f"    target_item       = {assoc_name}")
    _log(f"    grasp_pos_base    = {grasp_pos_base}")
    _log(f"    grasp_pos_refined = {grasp_pos_refined}  (added dx,dy)")
    _log(f"    grasp_pos (final) = {grasp_pos}  (Z snapped to GT item top)")
    _log(f"    grasp_quat (wxyz) = {grasp_quat}  (yaw composed onto cand quat)")

    src_bin_world = raw_env.get_src_bin_world_pos()
    before_xyz = _snapshot_all_cube_xyz(raw_env)
    before_in_src = {n: _in_src_bin(p, src_bin_world)
                     for n, p in before_xyz.items()}

    before_top = _render_top_down(raw_env, height=480, width=640)
    before_3q = _render_three_quarter(raw_env, height=480, width=640)
    before_top_path = os.path.join(out_dir, "before_top.png")
    before_3q_path = os.path.join(out_dir, "before_3q.png")
    _save_rgb_png(before_top_path, before_top,
                  title=f"BEFORE  top-down  seed={args.seed} n={args.n_objects}")
    _save_rgb_png(before_3q_path, before_3q,
                  title=f"BEFORE  3/4 view  target={assoc_name}")
    _log(f"[save] {before_top_path}")
    _log(f"[save] {before_3q_path}")

    try:
        predicate_pre = evaluate_grasp(raw_env, grasp_pos, grasp_quat,
                                       assoc_name)
    except Exception as e:
        predicate_pre = {
            "success": False, "item_between_jaws": False, "jaws_aligned": False,
            "mid_height": False, "approach_clear": False,
            "reason": f"predicate error: {e}", "picked_item": None,
        }
    n_pred_succ = int(
        bool(predicate_pre.get("item_between_jaws", False)) +
        bool(predicate_pre.get("jaws_aligned", False)) +
        bool(predicate_pre.get("mid_height", False)) +
        bool(predicate_pre.get("approach_clear", False))
    )
    _log(f"\n[7] Predicate sub-checks (computed BEFORE physics):")
    _log(f"    item_between_jaws = {predicate_pre.get('item_between_jaws')}")
    _log(f"    jaws_aligned      = {predicate_pre.get('jaws_aligned')}")
    _log(f"    mid_height        = {predicate_pre.get('mid_height')}")
    _log(f"    approach_clear    = {predicate_pre.get('approach_clear')}")
    _log(f"    n succ sub-checks = {n_pred_succ}/4")

    video_frames = []
    disturb_state = {
        "sum_raw": 0.0,
        "ejected": 0,
        "grasp_state": "PENDING",
        # GLITCH FIX 2: once attempt_grasp_hybrid returns, freeze the overlay
        # value at grasp_result["neighbour_disturbance_raw_m"] (set elsewhere)
        # and stop the live recomputation below.
        "locked": False,
    }
    # GLITCH FIX 1: snapshot of source-pile freejoint qpos captured the moment
    # attempt_grasp_hybrid returns. Re-pinned every render frame in the place
    # phase so the source pile cannot drift after the grasp ends.
    # PRE-CONTACT PILE-PIN FIX: pre-load with the right-after-reset snapshot
    # so _capture_one re-pins the pile EVERY frame during the pre-grasp
    # smooth approach + ghost-IK descent. picked=None means every cube is
    # pinned (no exemption), nothing should move before the grasp starts.
    src_pile_snapshot = {"data": pre_grasp_pile_snapshot, "picked": None}

    def _capture_one():
        # GLITCH FIX 1: before rendering, re-pin every snapshotted source-pile
        # cube to its end-of-grasp pose (except the picked item, which the
        # cinematic block tracks separately). Zeroes velocities too.
        try:
            if src_pile_snapshot["data"] is not None:
                env.pick_place._repin_items(
                    src_pile_snapshot["data"],
                    except_name=src_pile_snapshot["picked"])
                raw_env._forward()
        except Exception:
            pass
        try:
            frame = _render_three_quarter(raw_env, height=480, width=640)
        except Exception:
            return
        # GLITCH FIX 2: gate the live recompute on the lock flag. Once the
        # grasp has ended, disturb_state["sum_raw"] is the value
        # attempt_grasp_hybrid returned and the overlay stops drifting.
        if not disturb_state.get("locked", False):
            try:
                sum_raw = 0.0
                ejected = 0
                for name, p0 in before_xyz.items():
                    try:
                        bid = raw_env.sim.model.body_name2id(name)
                        p1 = np.array(raw_env.sim.data.body_xpos[bid], dtype=float)
                    except Exception:
                        continue
                    if name == assoc_name:
                        continue
                    d = float(np.linalg.norm(p1[:2] - p0[:2]))
                    sum_raw += d
                    if before_in_src.get(name, False) and not _in_src_bin(p1, src_bin_world):
                        ejected += 1
                disturb_state["sum_raw"] = sum_raw
                disturb_state["ejected"] = ejected
            except Exception:
                pass
        overlay = [
            f"sum_disturb_raw_m: {disturb_state['sum_raw']:.4f}",
        ]
        video_frames.append(_annotate_frame(
            frame, overlay, grasp_state=disturb_state["grasp_state"]))

    orig_step = raw_env.step

    def _step_with_capture(action_):
        ret = orig_step(action_)
        _capture_one()
        return ret

    _log("[8] Smooth approach: sensing pose -> above_pile (visible descent)")
    # DISPLAY GLITCH FIX (does NOT touch physics / outcome):
    # attempt_grasp_hybrid internally does TWO descents:
    # (a) ghost IK descent sensing -> above_pile -> grasp_pos
    # (captured via _teleport_arm_to_pos's frame_hook every 3 iters)
    # (b) physical-descent loop above_pile -> grasp_pos
    # (captured via the user frame_hook + env.step monkey-patch)
    # Between (a) and (b) the arm is SNAPPED back up to qpos_above with NO
    # frames captured, the viewer sees: down -> SNAP UP -> down again.
    # Fix: (1) one captured smooth_teleport sensing -> above_pile
    # (2) install a NOOP frame_hook on pp so the ghost-IK descent
    # does not capture frames, (3) keep raw_env.step monkey-patched
    # so the physical descent + close still capture via env.step
    # (4) restore pp.frame_hook = _capture_one so the cinematic
    # transport block still captures.
    above_pile = grasp_pos + np.array([0.0, 0.0, 0.12], dtype=float)
    pp = env.pick_place
    pp.frame_hook = _capture_one
    pp._smooth_teleport(above_pile, grasp_quat, n_steps=20)

    _log("\n[9] Running pick_place.attempt_grasp_hybrid() with frame capture")
    _noop_hook = lambda: None
    pp.frame_hook = _noop_hook
    raw_env.step = _step_with_capture

    # PRE-CONTACT PILE-PIN FIX: clear the pre-grasp pile snapshot now that
    # the grasp routine is about to run. attempt_grasp_hybrid owns its own
    # frozen_snap pinning (far items + target) during stage 2b, and the
    # target cube needs to move with the gripper once grasped. Leaving
    # src_pile_snapshot["data"] populated here would cause _capture_one
    # (called inside _step_with_capture on every mj_step) to fight the
    # grasp's own physics on the target item.
    src_pile_snapshot["data"] = None
    src_pile_snapshot["picked"] = None

    grasp_result = None
    grasp_exc = None
    try:
        grasp_result = env.pick_place.attempt_grasp_hybrid(
            grasp_pos=grasp_pos,
            grasp_quat=grasp_quat,
            target_item_name=assoc_name,
            frame_hook=_noop_hook,
            frame_hook_timing="pre_teleport",
        )
    except Exception as e:
        grasp_exc = f"{type(e).__name__}: {e}"
        _log(f"[ERROR] attempt_grasp_hybrid raised: {grasp_exc}")
        _log(traceback.format_exc())
    finally:
        raw_env.step = orig_step
        pp.frame_hook = _capture_one

    if grasp_result is None:
        grasp_result = {
            "grasp_ok": False, "grasp_quality": 0.0,
            "neighbour_disturbance_m": 0.0,
            "neighbour_disturbance_raw_m": 0.0,
            "neighbour_disturbance_max_m": 0.0,
            "n_near_items": 0, "items_ejected": 0,
            "target_item_name": assoc_name, "picked_item": None,
            "predicate_reason": grasp_exc or "no result",
            "ik_err_m": float("nan"),
        }

    grasp_ok_bool = bool(grasp_result.get("grasp_ok", False))
    disturb_state["grasp_state"] = "SUCCESS" if grasp_ok_bool else "FAIL"
    _log(f"    grasp_state set to : {disturb_state['grasp_state']}")

    # GLITCH FIX 2: lock the overlay value to the canonical grasp-phase metric
    # returned by attempt_grasp_hybrid. After this point _capture_one stops
    # re-reading body_xpos and just renders the locked number.
    disturb_state["sum_raw"] = float(
        grasp_result.get("neighbour_disturbance_raw_m", 0.0))
    disturb_state["ejected"] = int(grasp_result.get("items_ejected", 0))
    disturb_state["locked"] = True
    _log(f"    sum_disturb_raw_m locked at: {disturb_state['sum_raw']:.4f}")

    # GLITCH FIX 1: snapshot the source pile NOW so every subsequent render
    # frame (cinematic transport, kinematic place, post-roll) re-pins them
    # back to this pose. attempt_grasp_hybrid restores contact mode to
    # "normal" in its finally, so post-grasp residual velocities + new
    # contacts would otherwise drift the pile during the place phase.
    picked_name_for_snapshot = grasp_result.get("picked_item") or assoc_name
    try:
        src_pile_snapshot["data"] = env.pick_place._snapshot_item_poses()
        src_pile_snapshot["picked"] = picked_name_for_snapshot
        _log(f"    src pile snapshot taken: "
             f"{len(src_pile_snapshot['data'])} items, "
             f"picked={picked_name_for_snapshot}")
    except Exception as _e_snap:
        _log(f"    [WARN] failed to snapshot source pile: {_e_snap}")

    # Cinematic transport runs only on grasp success.
    cinematic_info = {
        "ran": False,
        "n_frames_lift": 0,
        "n_frames_transport": 0,
        "n_frames_lower": 0,
        "n_frames_release": 0,
        "waypoints": {},
        "seconds": 0.0,
    }
    picked_name = grasp_result.get("picked_item") or assoc_name
    if grasp_ok_bool and picked_name is not None:
        cinematic_info["ran"] = True
        n_before = len(video_frames)
        pp = env.pick_place
        try:
            # Pin the picked item to the EEF via _tracked_item + _tracked_offset
            # so _teleport_arm_to_pos snaps it to (eef + offset) every IK iter.
            eef_now = raw_env.get_robot_eef_pos().copy()
            try:
                pbid = raw_env.sim.model.body_name2id(picked_name)
                item_xyz_now = np.array(raw_env.sim.data.body_xpos[pbid],
                                        dtype=float).copy()
            except Exception:
                item_xyz_now = eef_now.copy()
            grasp_offset = item_xyz_now - eef_now

            # Ghost-mode the gripper so carry impulses don't spike the solver.
            pp._apply_contact_mode("ghost")
            pp._patch_wrist_limits()
            pp._tracked_item = picked_name
            pp._tracked_offset = grasp_offset.copy()
            pp.frame_hook = _capture_one

            dst_bin = raw_env.get_dst_bin_world_pos()
            gz_lift = float(grasp_pos[2]) + 0.20
            wp_lift      = np.array([grasp_pos[0], grasp_pos[1], gz_lift],
                                    dtype=float)
            wp_transport = np.array([dst_bin[0], dst_bin[1], gz_lift],
                                    dtype=float)
            # SAFE-DROP FIX: place the carried cube COM ~18 cm above the bin
            # inner floor top. Bin wall height = 0.10 m, cube half-height
            # = 0.029 m, so cube bottom ends up ~0.18 - 0.029 = 0.151 m above
            # the floor top and clears the rim by ~0.05 m. The wrist target
            # must be ~_FINGER_TO_WRIST (0.10 m) above the desired cube COM
            # so that, after the IK pins (cube_pos = eef + grasp_offset, with
            # grasp_offset.z negative ~ -finger_to_wrist), the cube COM lands
            # at floor_top + 0.18 m. Empirically we use the measured
            # grasp_offset.z so we don't double-count the offset.
            _BIN_DROP_CUBE_Z = 0.18      # cube COM above bin inner floor top
            wp_lower_z = float(dst_bin[2]) + _BIN_DROP_CUBE_Z - float(grasp_offset[2])
            wp_lower   = np.array([dst_bin[0], dst_bin[1], wp_lower_z],
                                  dtype=float)
            # Geometry sanity log, makes the safety margin visible if the
            # bin dimensions ever change.
            expected_cube_com_z = wp_lower_z + float(grasp_offset[2])
            expected_cube_bottom_z = expected_cube_com_z - _ITEM_HALF_HEIGHT
            bin_wall_top_z = float(dst_bin[2]) + 0.100  # _BIN_WALL_H = 0.10
            _log(f"    SAFE-DROP geometry:")
            _log(f"      dst_bin[2] (floor top)   : {float(dst_bin[2]):.4f}")
            _log(f"      bin wall top z           : {bin_wall_top_z:.4f}")
            _log(f"      wp_lower.z (EEF target)  : {wp_lower_z:.4f}")
            _log(f"      grasp_offset.z           : {float(grasp_offset[2]):.4f}")
            _log(f"      expected cube COM z      : {expected_cube_com_z:.4f}")
            _log(f"      expected cube bottom z   : {expected_cube_bottom_z:.4f}")
            _log(f"      clearance over wall top  : "
                 f"{expected_cube_bottom_z - bin_wall_top_z:.4f}")
            cinematic_info["waypoints"] = {
                "lift":      [float(x) for x in wp_lift],
                "transport": [float(x) for x in wp_transport],
                "lower":     [float(x) for x in wp_lower],
            }

            _log("\n[10a] CINEMATIC pick-and-place sequence ...")
            _log(f"    grasp_offset (item - eef) : {grasp_offset}")
            _log(f"    waypoint lift             : {wp_lift}")
            _log(f"    waypoint transport        : {wp_transport}")
            _log(f"    waypoint lower            : {wp_lower}")

            n0 = len(video_frames)
            pp._smooth_teleport(wp_lift, grasp_quat, n_steps=20)
            cinematic_info["n_frames_lift"] = len(video_frames) - n0

            n0 = len(video_frames)
            pp._smooth_teleport(wp_transport, grasp_quat,
                                n_steps=30, null_gain=0.02)
            cinematic_info["n_frames_transport"] = len(video_frames) - n0

            n0 = len(video_frames)
            pp._smooth_teleport(wp_lower, grasp_quat,
                                n_steps=20, null_gain=0.02)
            cinematic_info["n_frames_lower"] = len(video_frames) - n0

            # PHYSICS-DROP RELEASE: let gravity drop the carried cube into
            # the destination bin. We KEEP contact mode at "ghost" so the
            # gripper geoms can't punch through the cube, items keep
            # contype=conaffinity=1 (bit0) in ghost mode, which means
            # cube<->bin-floor and cube<->bin-walls contacts are LIVE while
            # gripper<->anything is OFF. (Switching to "normal" here is the
            # source of the launch-impulse bug noted in pick_place_primitive
            # Phase 6.) The carried cube is unpinned so it integrates
            # naturally under gravity. Source pile is re-pinned per step.
            pp._tracked_item = None
            pp._tracked_offset = None
            pp.frame_hook = None
            # Contact mode stays "ghost" (item bit0 -> bin/floor contacts ON).

            _log("    releasing item PHYSICALLY (gravity-driven drop)")
            n0 = len(video_frames)
            try:
                import mujoco as _mj_drop
                _raw_model_drop = getattr(raw_env.sim.model, "_model",
                                          raw_env.sim.model)
                _raw_data_drop = getattr(raw_env.sim.data, "_data",
                                         raw_env.sim.data)
                # Lock the arm joints during the drop so the wrist doesn't
                # drift, mirrors pick_place_primitive Phase 6 release.
                qpos_idxs_d, qvel_idxs_d = pp._get_arm_indices()
                locked_qpos_d = np.array(
                    [_raw_data_drop.qpos[idx] for idx in qpos_idxs_d])
                open_action = np.zeros(7, dtype=float)
                open_action[-1] = -1.0   # gripper open
                _RELEASE_STEPS = 35
                for _stepi in range(_RELEASE_STEPS):
                    raw_env.step(open_action)
                    # Re-lock the arm joints.
                    for _k, _idx in enumerate(qpos_idxs_d):
                        _raw_data_drop.qpos[_idx] = locked_qpos_d[_k]
                    for _idx in qvel_idxs_d:
                        _raw_data_drop.qvel[_idx] = 0.0
                    # Re-pin every source-bin cube EXCEPT the dropped one
                    # (which falls freely under gravity).
                    if src_pile_snapshot["data"] is not None:
                        pp._repin_items(src_pile_snapshot["data"],
                                        except_name=picked_name)
                    _mj_drop.mj_forward(_raw_model_drop, _raw_data_drop)
                    try:
                        raw_env.robots[0].controller.update(force=True)
                        raw_env.robots[0].controller.reset_goal()
                    except Exception:
                        pass
                    _capture_one()
            except Exception as _e_drop:
                _log(f"    [WARN] physics-drop release failed: "
                     f"{type(_e_drop).__name__}: {_e_drop}")
            cinematic_info["n_frames_release"] = len(video_frames) - n0
        except Exception as e:
            _log(f"[ERROR] cinematic transport raised: "
                 f"{type(e).__name__}: {e}")
            _log(traceback.format_exc())
            try:
                pp._tracked_item = None
                pp._tracked_offset = None
                pp.frame_hook = None
                pp._apply_contact_mode("normal")
            except Exception:
                pass
        n_after = len(video_frames)
        cinematic_info["seconds"] = float(
            max(0, n_after - n_before)) / float(args.fps)
        _log(f"    cinematic frames           : {n_after - n_before}  "
             f"({cinematic_info['seconds']:.2f}s @ {args.fps} fps)")

    _log(f"\n[10b] Capturing post-roll (2.0 s at {args.fps} fps)")
    for _ in range(int(round(2.0 * args.fps))):
        _capture_one()

    after_xyz = _snapshot_all_cube_xyz(raw_env)
    after_top = _render_top_down(raw_env, height=480, width=640)
    after_3q = _render_three_quarter(raw_env, height=480, width=640)
    after_top_path = os.path.join(out_dir, "after_top.png")
    after_3q_path = os.path.join(out_dir, "after_3q.png")
    _save_rgb_png(after_top_path, after_top,
                  title=(f"AFTER  top-down  grasp_ok={grasp_result.get('grasp_ok')}"
                         f"  ejected={grasp_result.get('items_ejected')}"))
    _save_rgb_png(after_3q_path, after_3q,
                  title=(f"AFTER  3/4 view  sum_disturb_raw_m="
                         f"{grasp_result.get('neighbour_disturbance_raw_m', 0.0):.4f}"))
    _log(f"[save] {after_top_path}")
    _log(f"[save] {after_3q_path}")

    table_path = os.path.join(out_dir, "disturbance_table.csv")
    with open(table_path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow([
            "idx", "body",
            "before_x", "before_y", "before_z",
            "after_x", "after_y", "after_z",
            "dxy_m", "dxyz_m",
            "ejected",
            "is_target",
        ])
        names_sorted = sorted(before_xyz.keys())
        for i, name in enumerate(names_sorted):
            p0 = before_xyz[name]
            p1 = after_xyz.get(name, p0)
            dxy = float(np.linalg.norm(p1[:2] - p0[:2]))
            dxyz = float(np.linalg.norm(p1 - p0))
            ejected = bool(
                before_in_src.get(name, False)
                and not _in_src_bin(p1, src_bin_world)
            )
            wr.writerow([
                i, name,
                f"{p0[0]:.4f}", f"{p0[1]:.4f}", f"{p0[2]:.4f}",
                f"{p1[0]:.4f}", f"{p1[1]:.4f}", f"{p1[2]:.4f}",
                f"{dxy:.4f}", f"{dxyz:.4f}",
                int(ejected),
                int(name == assoc_name),
            ])
    _log(f"[save] {table_path}")

    dst_bin_world = raw_env.get_dst_bin_world_pos()
    picked_item_final_xyz = None
    in_target_bin = False
    picked_body_com_z_initial = None
    if picked_name is not None:
        try:
            pbid = raw_env.sim.model.body_name2id(picked_name)
            pxyz = np.array(raw_env.sim.data.body_xpos[pbid], dtype=float)
            picked_item_final_xyz = [float(pxyz[0]), float(pxyz[1]),
                                     float(pxyz[2])]
            in_target_bin = bool(
                abs(pxyz[0] - dst_bin_world[0]) < BIN_HALF_SIZE[0]
                and abs(pxyz[1] - dst_bin_world[1]) < BIN_HALF_SIZE[1]
                and pxyz[2] > dst_bin_world[2] - 0.05
            )
        except Exception:
            pass
        # body_com_z_initial >= 0.86 -> picked from the TOP layer.
        if picked_name in before_xyz:
            picked_body_com_z_initial = float(before_xyz[picked_name][2])

    _log("\n[11] Placement outcome:")
    _log(f"    picked_item        = {picked_name}")
    _log(f"    body_com_z_initial = {picked_body_com_z_initial}  "
         f"(>=0.86 means TOP layer)")
    _log(f"    picked_final_xyz   = {picked_item_final_xyz}")
    _log(f"    dst_bin_world      = {dst_bin_world.tolist()}")
    _log(f"    in_target_bin      = {in_target_bin}")

    outcome = {
        "policy": "rl_maskable_ppo",
        "rl_model_path": str(args.rl_model_path),
        "chosen_body": assoc_name,
        "chosen_idx": int(chosen_idx),
        "chosen_quality": float(chosen.get("quality", 0.0)),
        "chosen_source_body_id": int(chosen.get("source_body_id", -1)),
        "rl_action": int(action),
        "rl_sub_action_m": int(sub_action),
        "chosen_dx": float(dx),
        "chosen_dy": float(dy),
        "chosen_dyaw": float(dyaw),
        "n_unmasked_actions": int(n_unmasked),
        "n_total_actions": int(mask.size),
        "grasp_pos": [float(grasp_pos[0]), float(grasp_pos[1]),
                      float(grasp_pos[2])],
        "grasp_quat_wxyz": [float(grasp_quat[0]), float(grasp_quat[1]),
                            float(grasp_quat[2]), float(grasp_quat[3])],
        "grasp_pos_base": [float(grasp_pos_base[0]), float(grasp_pos_base[1]),
                           float(grasp_pos_base[2])],
        "candidate_world_quat_wxyz": [float(cand_world_quat[0]),
                                       float(cand_world_quat[1]),
                                       float(cand_world_quat[2]),
                                       float(cand_world_quat[3])],
        "grasp_success": bool(grasp_result.get("grasp_ok", False)),
        "grasp_ok": bool(grasp_result.get("grasp_ok", False)),
        "grasp_quality": float(grasp_result.get("grasp_quality", 0.0)),
        "items_ejected": int(grasp_result.get("items_ejected", 0)),
        "sum_disturb_raw_m": float(
            grasp_result.get("neighbour_disturbance_raw_m", 0.0)),
        "max_disturb_m": float(
            grasp_result.get("neighbour_disturbance_max_m", 0.0)),
        "neighbour_disturbance_m": float(
            grasp_result.get("neighbour_disturbance_m", 0.0)),
        "n_near_items": int(grasp_result.get("n_near_items", 0)),
        "n_predicate_succ_subchecks": int(n_pred_succ),
        "predicate_subchecks": {
            "item_between_jaws": bool(predicate_pre.get("item_between_jaws", False)),
            "jaws_aligned": bool(predicate_pre.get("jaws_aligned", False)),
            "mid_height": bool(predicate_pre.get("mid_height", False)),
            "approach_clear": bool(predicate_pre.get("approach_clear", False)),
        },
        "predicate_reason": str(grasp_result.get("predicate_reason", "")),
        "picked_item": grasp_result.get("picked_item"),
        "ik_err_m": float(grasp_result.get("ik_err_m", float("nan"))),
        "n_frames_captured": len(video_frames),
        "fps": int(args.fps),
        "seed": int(args.seed),
        "n_objects": int(args.n_objects),
        "n_candidates": int(len(candidates)),
        "exception": grasp_exc,
        "display_rotation_cw_deg": 90,
        "display_crop": "source_bin_square",
        "display_bin_bounds_640": list(bin_bounds_640),
        "env_kwargs": {
            "n_objects": int(args.n_objects),
            "K": int(args.K),
            "candidate_source": "ppo",
            "ppo_visibility_mode": "raycast",
            "ppo_quality_mode": "analytical",
            "filter_candidates": True,
            "orientation_source": "snap",
            "use_snap_xy": False,
            "use_snap_z": True,
            "layout_jitter": float(args.layout_jitter),
            "failure_mask_size": int(args.failure_mask_size),
            "reward_mode": "hybrid_physics",
        },
        "picked_body_com_z_initial": picked_body_com_z_initial,
        "is_top_layer_pick": bool(
            picked_body_com_z_initial is not None
            and picked_body_com_z_initial >= 0.86),
        "picked_item_final_xyz": picked_item_final_xyz,
        "dst_bin_world": [float(dst_bin_world[0]),
                          float(dst_bin_world[1]),
                          float(dst_bin_world[2])],
        "in_target_bin": bool(in_target_bin),
        "n_frames_total": int(len(video_frames)),
        "cinematic_transport_seconds": float(cinematic_info["seconds"]),
        "cinematic_transport": cinematic_info,
    }
    outcome_path = os.path.join(out_dir, "outcome.json")
    with open(outcome_path, "w") as fh:
        json.dump(outcome, fh, indent=2)
    _log(f"[save] {outcome_path}")

    video_path = os.path.join(out_dir, "pick_and_place_top.mp4")
    _save_video(video_path, video_frames, fps=args.fps)

    _log("\n" + "=" * 78)
    _log("RL PICK + PLACE  (TOP-cube cinematic showcase)  SUMMARY")
    _log("=" * 78)
    _log(f"out_dir              : {out_dir}")
    _log(f"rl_model             : {args.rl_model_path}")
    _log(f"seed                 : {args.seed}")
    _log(f"layout_jitter        : {args.layout_jitter}")
    _log(f"n_objects            : {args.n_objects}")
    _log(f"n PPO candidates     : {len(candidates)}  chosen=#{chosen_idx}  "
         f"q={float(chosen['quality']):.4f}")
    _log(f"RL action            : {action}  -> "
         f"(k={chosen_idx}, m={sub_action}) "
         f"dx={dx:+.3f} dy={dy:+.3f} dyaw={dyaw:+.3f}")
    _log(f"target_item          : {assoc_name}")
    _log(f"picked_body_com_z    : {picked_body_com_z_initial}  "
         f"(>=0.86 means TOP layer of pile)")
    _log(f"is_top_layer_pick    : "
         f"{picked_body_com_z_initial is not None and picked_body_com_z_initial >= 0.86}")
    _log(f"grasp_pos (wrist tgt): {grasp_pos}")
    _log(f"grasp_ok             : {grasp_result.get('grasp_ok')}")
    _log(f"items_ejected        : {grasp_result.get('items_ejected')}")
    _log(f"sum_disturb_raw_m    : "
         f"{grasp_result.get('neighbour_disturbance_raw_m', 0.0):.4f}")
    _log(f"max_disturb_m        : "
         f"{grasp_result.get('neighbour_disturbance_max_m', 0.0):.4f}")
    _log(f"predicate_subchecks  : {n_pred_succ}/4")
    _log(f"predicate_reason     : {grasp_result.get('predicate_reason')}")
    _log(f"cinematic ran        : {cinematic_info['ran']}")
    if cinematic_info["ran"]:
        _log(f"  waypoint lift      : {cinematic_info['waypoints'].get('lift')}")
        _log(f"  waypoint transport : {cinematic_info['waypoints'].get('transport')}")
        _log(f"  waypoint lower     : {cinematic_info['waypoints'].get('lower')}")
        _log(f"  n_frames_lift      : {cinematic_info['n_frames_lift']}")
        _log(f"  n_frames_transport : {cinematic_info['n_frames_transport']}")
        _log(f"  n_frames_lower     : {cinematic_info['n_frames_lower']}")
        _log(f"  n_frames_release   : {cinematic_info['n_frames_release']}")
        _log(f"  cinematic_seconds  : {cinematic_info['seconds']:.2f}s")
    _log(f"picked_final_xyz     : {picked_item_final_xyz}")
    _log(f"in_target_bin        : {in_target_bin}")
    _log(f"video frames captured: {len(video_frames)}  "
         f"(n_frames_total = {len(video_frames)})")
    _log("=" * 78)
    _log(f"\nFiles in {out_dir}:")
    for fn in sorted(os.listdir(out_dir)):
        _log(f"  {fn}")
    log_fh.close()


def main():
    parser = argparse.ArgumentParser(
        description=("RL pick + place TOP-cube cinematic showcase. MaskablePPO "
                     "+ PPO ghost candidates + hybrid_physics grasp, then a "
                     "cinematic kinematic-transport drop into the destination "
                     "bin. Default n_objects=17, seed=1000, layout_jitter=0.0 "
                     "is the matched config where the RL policy picks a TOP-"
                     "layer cube and all 4 predicate sub-checks pass."))
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(_ROOT, "demos", "rl_top_cube"))
    parser.add_argument("--rl_model_path", type=str,
                        default=_DEFAULT_RL_MODEL_PATH,
                        help="MaskablePPO .zip (default: rl/models/best_model.zip)")
    parser.add_argument("--n_objects", type=int, default=17)
    parser.add_argument("--layout_jitter", type=float, default=0.0,
                        help="0.0 = mesh-defined stack positions (deterministic)")
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--failure_mask_size", type=int, default=3)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
