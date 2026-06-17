"""
GG-CNN demo on the RL-success scene (n_objects=17, seed=1000, layout_jitter=0.0).

Parallel to demo_rl_pick_and_place_top.py: the same scene that succeeded under
RL is replayed here with the standard GG-CNN pipeline so the two methods can
be compared head-to-head on a matched scene. On this scene GG-CNN picks a
bin-edge candidate, predicate 0/4 sub-checks pass, and the grasp fails, that
outcome is reported honestly.
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

from scipy.spatial.transform import Rotation as _Rot

from sim.camera_setup import get_camera_intrinsics
from sim.sensing_pose import BIN_HALF_SIZE
from perception.candidate_extractor import extract_top_k_candidates
from perception.candidate_to_pose import pixels_to_width_meters
from rl.bin_clearing_env import (
    BinClearingGymEnv,
    _GRASP_DESCENT_OFFSET,
    _ITEM_HALF_HEIGHT,
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


def _bin_bounds_full(cam_pos, src_bin):
    """Project source-bin floor rect (at z = src_bin[2]) into the 640x480 image.
    Camera looks straight down so Zc is constant across the bin floor. the
    projection reduces to an affine mapping in world XY."""
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


def _bin_bounds_300(cam_pos, src_bin):
    """640x480 bin bounds mapped into the 300x300 GG-CNN input frame
    (Y-padded to 640x640, resized 640x640 -> 300x300)."""
    x0f, y0f, x1f, y1f = _bin_bounds_full(cam_pos, src_bin)
    y0p = y0f + _PAD_Y
    y1p = y1f + _PAD_Y
    s = _GGCNN_SIZE / float(_CAM_W)
    x0 = int(np.floor(x0f * s)) - _BIN_CROP_MARGIN_PX_300
    x1 = int(np.ceil (x1f * s)) + _BIN_CROP_MARGIN_PX_300
    y0 = int(np.floor(y0p * s)) - _BIN_CROP_MARGIN_PX_300
    y1 = int(np.ceil (y1p * s)) + _BIN_CROP_MARGIN_PX_300
    x0 = int(np.clip(x0, 0, _GGCNN_SIZE))
    x1 = int(np.clip(x1, 0, _GGCNN_SIZE))
    y0 = int(np.clip(y0, 0, _GGCNN_SIZE))
    y1 = int(np.clip(y1, 0, _GGCNN_SIZE))
    if x1 <= x0:
        x0, x1 = 0, _GGCNN_SIZE
    if y1 <= y0:
        y0, y1 = 0, _GGCNN_SIZE
    return x0, y0, x1, y1


def perceive_full_image(env_inner, sensing_ctrl, ggcnn, K=10):
    """Run GG-CNN on the FULL 640x480 depth (Y-padded to 640x640, resized to
    300x300). Returns (candidates, meta)."""
    sensing_ctrl.set_sensing_pose_direct()
    env_inner._forward()

    cam_pos = env_inner.get_camera_world_pos()
    src_bin = env_inner.get_src_bin_world_pos()
    floor_depth = float(cam_pos[2] - src_bin[2])

    depth = env_inner.get_wrist_depth_meters(height=_CAM_H, width=_CAM_W).astype(np.float32)
    bad = ~np.isfinite(depth) | (depth <= 0)
    depth[bad] = floor_depth
    # Gripper-mask cutoff at floor_depth - 0.18 (relaxed from 0.08): masks the
    # gripper region (depth ~0.10-0.20 m at sensing pose) while preserving
    # stacked-cube tops (depth 0.28-0.32 m).
    depth[depth < floor_depth - 0.18] = floor_depth

    depth_sq = np.full((_CAM_W, _CAM_W), floor_depth, dtype=np.float32)
    depth_sq[_PAD_Y:_PAD_Y + _CAM_H, :] = depth

    depth_300 = cv2.resize(depth_sq, (_GGCNN_SIZE, _GGCNN_SIZE),
                           interpolation=cv2.INTER_LINEAR)

    lo = float(depth_300.min())
    hi = float(depth_300.max())
    rng = max(hi - lo, 0.001)
    depth_pp = np.clip((depth_300 - lo) / rng, 0.0, 1.0).astype(np.float32)

    preds = ggcnn.predict(depth_pp)
    quality = preds["quality"]
    angle = preds["angle"]
    width = preds["width"]

    raw = extract_top_k_candidates(quality, angle, width,
                                   depth_m_crop=depth_300, K=K,
                                   min_quality=0.1, nms_size=11)

    K_back = get_camera_intrinsics(_RENDER_FOVY, _CAM_W, _CAM_H)
    fx = float(K_back[0, 0])
    fy = float(K_back[1, 1])
    cx = float(K_back[0, 2])
    cy = float(K_back[1, 2])
    scale = _CAM_W / float(_GGCNN_SIZE)

    candidates = []
    for c in raw:
        px_pad = float(c["px"]) * scale
        py_pad = float(c["py"]) * scale
        px_640 = px_pad
        py_480 = py_pad - _PAD_Y
        if not (0.0 <= py_480 < _CAM_H):
            continue

        ix = int(round(px_640))
        iy = int(round(py_480))
        ix = int(np.clip(ix, 0, _CAM_W - 1))
        iy = int(np.clip(iy, 0, _CAM_H - 1))
        z = float(depth[iy, ix])
        if z <= 0 or z >= floor_depth - 0.002:
            z_fallback = float(c["depth_m"])
            if z_fallback > 0 and z_fallback < floor_depth - 0.002:
                z = z_fallback
            else:
                continue

        Xc = (px_640 - cx) * z / fx
        Yc = (py_480 - cy) * z / fy
        Zc = z
        world_pos = cam_pos + np.array([Xc, -Yc, -Zc], dtype=float)

        ang = float(c["angle_rad"])
        sx = float(np.cos(ang))
        sy = float(-np.sin(ang))
        gz = np.array([0.0, 0.0, -1.0])
        gy = np.array([sx, sy, 0.0])
        ny = float(np.linalg.norm(gy))
        if ny > 1e-9:
            gy = gy / ny
            gx = np.cross(gy, gz)
            gx = gx / max(np.linalg.norm(gx), 1e-9)
            R_w = np.column_stack([gx, gy, gz])
            xyzw = _Rot.from_matrix(R_w).as_quat()
            world_quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]],
                                  dtype=float)
        else:
            world_quat = _CAM_QUAT_WORLD.copy()

        width_m = pixels_to_width_meters(float(c["width_px"]), z, fx)

        candidates.append({
            "px300": int(c["px"]),
            "py300": int(c["py"]),
            "px_full": float(px_640),
            "py_full": float(py_480),
            "angle_rad": ang,
            "quality": float(c["quality"]),
            "width_px": float(c["width_px"]),
            "width_m": float(width_m),
            "depth_m": float(z),
            "world_pos": world_pos.astype(np.float64),
            "world_quat": world_quat.astype(np.float64),
        })

    candidates.sort(key=lambda d: -d["quality"])
    meta = {
        "cam_pos": cam_pos,
        "K_back": K_back,
        "floor_depth": floor_depth,
        "src_bin_world": src_bin,
        "depth_full": depth,
        "depth_sq": depth_sq,
        "depth_300": depth_300,
        "depth_pp": depth_pp,
        "quality_map": quality,
        "angle_map": angle,
        "width_map": width,
        "auto_norm_range": (lo, hi),
        "pad_y": _PAD_Y,
        "scale_300_to_640": scale,
        "n_candidates": len(candidates),
    }
    return candidates, meta


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


# Style constants: red = non-chosen, green = chosen.
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


# Display-only crop+rotate (CW 90 deg) of perception PNGs. GG-CNN still sees
# the full 300x300 input. the candidate dicts and CSV pixel columns are kept
# in the original un-rotated frame. Angle remap for the jaw-line:
# (dx, dy) in the original frame -> (-dy, dx) under CW-90, equivalent to
# replacing (cos a, sin a) with (cos a', sin a') where a' = a + pi/2.

def _save_depth_full(path, depth_full, title, bin_bounds=None):
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


def _save_ggcnn_input(path, depth_pp, auto_norm_range, title, bin_bounds=None):
    """300x300 normalised GG-CNN input. Network still consumes the full
    300x300. crop is display-only."""
    if bin_bounds is not None:
        x0, y0, x1, y1 = bin_bounds
        input_show = depth_pp[y0:y1, x0:x1]
    else:
        input_show = depth_pp
    input_disp = np.rot90(input_show, k=-1)
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(input_disp, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("GG-CNN depth input", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.04,
                 label="normalised depth")
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _save_quality_heatmap(path, quality, candidates, chosen_idx, title,
                          bin_bounds=None):
    """Quality heatmap (300x300), bin-cropped + CW-90 rotated, chosen-on-top."""
    if bin_bounds is not None:
        x0, y0, x1, y1 = bin_bounds
        heatmap_show = quality[y0:y1, x0:x1]
    else:
        x0, y0, x1, y1 = 0, 0, quality.shape[1], quality.shape[0]
        heatmap_show = quality
    H_orig = heatmap_show.shape[0]
    W_orig = heatmap_show.shape[1]
    heatmap_disp = np.rot90(heatmap_show, k=-1)
    fig, ax = plt.subplots(figsize=(7.4, 7))
    im = ax.imshow(heatmap_disp, cmap="viridis", vmin=0.0, vmax=1.0)

    def _disp_coords(px, py):
        px_c = px - x0
        py_c = py - y0
        return (H_orig - 1 - py_c, px_c)

    for i, c in enumerate(candidates):
        if i == chosen_idx:
            continue
        if not (x0 <= c["px300"] < x1 and y0 <= c["py300"] < y1):
            continue
        ang = float(c["angle_rad"])
        ang_disp = ang + np.pi / 2.0
        half_w = 0.5 * float(c["width_px"])
        dx = half_w * np.cos(ang_disp)
        dy = half_w * np.sin(ang_disp)
        x_d, y_d = _disp_coords(float(c["px300"]), float(c["py300"]))
        ax.plot([x_d - dx, x_d + dx], [y_d - dy, y_d + dy],
                "-", color="red", lw=_RED_LINE_LW, zorder=_RED_LINE_ZORDER)
        ax.scatter([x_d], [y_d],
                   s=_RED_MARKER_S,
                   facecolor="red", edgecolor="black",
                   lw=_RED_MARKER_EDGE_LW,
                   zorder=_RED_MARKER_ZORDER)
        ax.text(x_d + 5, y_d - 5,
                f"#{i} q={float(c['quality']):.2f}",
                color="red", fontsize=_RED_TEXT_FS, weight="bold",
                zorder=_RED_TEXT_ZORDER)

    if 0 <= chosen_idx < len(candidates):
        c = candidates[chosen_idx]
        inside = (x0 <= c["px300"] < x1 and y0 <= c["py300"] < y1)
        if inside:
            ang = float(c["angle_rad"])
            ang_disp = ang + np.pi / 2.0
            half_w = 0.5 * float(c["width_px"])
            dx = half_w * np.cos(ang_disp)
            dy = half_w * np.sin(ang_disp)
            x_d, y_d = _disp_coords(float(c["px300"]), float(c["py300"]))
            ax.plot([x_d - dx, x_d + dx], [y_d - dy, y_d + dy],
                    "-", color="lime", lw=_GREEN_LINE_LW,
                    solid_capstyle="round", zorder=_GREEN_LINE_ZORDER)
            ax.scatter([x_d], [y_d],
                       s=_GREEN_MARKER_S * 1.6,
                       facecolor="white", edgecolor="white",
                       lw=0.0, zorder=_GREEN_MARKER_ZORDER)
            ax.scatter([x_d], [y_d], marker="*",
                       s=_GREEN_MARKER_S,
                       facecolor="lime", edgecolor="black",
                       lw=_GREEN_MARKER_EDGE_LW,
                       zorder=_GREEN_MARKER_ZORDER + 1)
            ax.text(x_d + 7, y_d - 7,
                    f"[CHOSEN] #{chosen_idx} q={float(c['quality']):.3f}",
                    color="darkgreen", fontsize=_GREEN_TEXT_FS, weight="bold",
                    bbox=_GREEN_TEXT_BBOX,
                    zorder=_GREEN_TEXT_ZORDER)

    ax.set_xlim(0, H_orig)
    ax.set_ylim(W_orig, 0)
    ax.set_aspect("equal")
    ax.set_title("GG-CNN grasp-quality heatmap", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.04,
                 label="grasp quality")
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _draw_predictions_on_axis_rotated(ax, candidates, chosen_idx, H_orig,
                                       crop_x0=0, crop_y0=0,
                                       crop_x1=_CAM_W, crop_y1=_CAM_H):
    """Annotate predictions on a CW-90 rotated 640x480 image."""
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
        half_w = 0.5 * float(c["width_px"]) * width_scale
        ang = float(c["angle_rad"])
        ang_disp = ang + np.pi / 2.0
        dx = half_w * np.cos(ang_disp)
        dy = half_w * np.sin(ang_disp)
        x_d, y_d = _disp_coords(x, y)
        ax.plot([x_d - dx, x_d + dx], [y_d - dy, y_d + dy],
                "-", color="red", lw=_RED_LINE_LW,
                solid_capstyle="round", zorder=_RED_LINE_ZORDER)
        ax.scatter([x_d], [y_d], s=_RED_MARKER_S,
                   facecolor="red", edgecolor="black",
                   linewidths=_RED_MARKER_EDGE_LW,
                   zorder=_RED_MARKER_ZORDER)
        ax.text(x_d + 8, y_d - 8,
                f"#{i} q={float(c['quality']):.2f}",
                color="red", fontsize=_RED_TEXT_FS, weight="bold",
                bbox=_RED_TEXT_BBOX,
                zorder=_RED_TEXT_ZORDER)

    if 0 <= chosen_idx < len(candidates):
        c = candidates[chosen_idx]
        x = float(c["px_full"])
        y = float(c["py_full"])
        inside = (crop_x0 <= x < crop_x1 and crop_y0 <= y < crop_y1)
        if inside:
            half_w = 0.5 * float(c["width_px"]) * width_scale
            ang = float(c["angle_rad"])
            ang_disp = ang + np.pi / 2.0
            dx = half_w * np.cos(ang_disp)
            dy = half_w * np.sin(ang_disp)
            x_d, y_d = _disp_coords(x, y)
            ax.plot([x_d - dx, x_d + dx], [y_d - dy, y_d + dy],
                    "-", color="lime", lw=_GREEN_LINE_LW,
                    solid_capstyle="round", zorder=_GREEN_LINE_ZORDER)
            ax.scatter([x_d], [y_d],
                       s=_GREEN_MARKER_S * 1.6,
                       facecolor="white", edgecolor="white",
                       lw=0.0, zorder=_GREEN_MARKER_ZORDER)
            ax.scatter([x_d], [y_d], marker="*",
                       s=_GREEN_MARKER_S,
                       facecolor="lime", edgecolor="black",
                       linewidths=_GREEN_MARKER_EDGE_LW,
                       zorder=_GREEN_MARKER_ZORDER + 1)
            ax.text(x_d + 10, y_d - 10,
                    f"[CHOSEN] #{chosen_idx} q={float(c['quality']):.3f}",
                    color="darkgreen", fontsize=_GREEN_TEXT_FS, weight="bold",
                    bbox=_GREEN_TEXT_BBOX,
                    zorder=_GREEN_TEXT_ZORDER)


def _save_overlay_on_image(path, image, candidates, chosen_idx,
                           is_depth, suptitle, caption, bin_bounds=None):
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
    _draw_predictions_on_axis_rotated(ax, candidates, chosen_idx,
                                      H_orig=H_orig,
                                      crop_x0=x0, crop_y0=y0,
                                      crop_x1=x1, crop_y1=y1)
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
    _log("GG-CNN demo on the RL-success scene "
         "(n_objects=17, layout_jitter=0.0, seed=1000)")
    _log(f"datetime : {datetime.datetime.now().isoformat(timespec='seconds')}")
    _log(f"out_dir  : {out_dir}")
    _log("=" * 78)

    _verify_bin_wall_height()

    _log("\n[1] Building BinClearingGymEnv (hybrid_physics eval config)")
    _log(f"    n_objects          = {args.n_objects}")
    _log(f"    layout_jitter      = {args.layout_jitter}")
    _log(f"    candidate_source   = 'ggcnn'  (env perception, then OVERRIDDEN)")
    _log(f"    orientation_source = 'snap'")
    _log(f"    use_snap_xy        = False  (predicted XY)")
    _log(f"    use_snap_z         = True   (GT-snapped Z)")
    _log(f"    failure_mask_size  = {args.failure_mask_size}")
    _log(f"    reward_mode        = 'hybrid_physics'")
    _log(f"    seed               = {args.seed}")

    env = BinClearingGymEnv(
        n_objects=args.n_objects,
        K=args.K,
        candidate_source="ggcnn",
        filter_candidates=False,
        orientation_source="snap",
        use_snap_xy=False,
        use_snap_z=True,
        layout_jitter=args.layout_jitter,
        failure_mask_size=args.failure_mask_size,
        reward_mode="hybrid_physics",
        ggcnn_device=args.ggcnn_device,
    )

    _obs, info = env.reset(seed=args.seed)
    raw_env = env.env
    _log(f"\n[1b] reset done. n_items_remaining={info['n_items_remaining']}")

    _log("\n[2] Running full-image GG-CNN perception ...")
    candidates, meta = perceive_full_image(
        env_inner=env.env,
        sensing_ctrl=env.sensing_ctrl,
        ggcnn=env.ggcnn,
        K=args.K,
    )
    cam_pos = meta["cam_pos"]
    floor_depth = meta["floor_depth"]
    depth_full = meta["depth_full"]
    depth_pp = meta["depth_pp"]
    quality_map = meta["quality_map"]
    auto_norm_range = meta["auto_norm_range"]
    _log(f"    cam_pos          : {cam_pos}")
    _log(f"    floor_depth      : {floor_depth:.4f} m")
    _log(f"    pad_y            : {meta['pad_y']}  (640x480 -> 640x640)")
    _log(f"    scale 300->640   : {meta['scale_300_to_640']:.4f}")
    _log(f"    auto_norm_range  : "
         f"[{auto_norm_range[0]:.4f}, {auto_norm_range[1]:.4f}] m")
    _log(f"    quality.max      : {quality_map.max():.4f}")
    _log(f"    n candidates     : {len(candidates)}")

    src_bin_world_for_crop = meta["src_bin_world"]
    bin_bounds_640 = _bin_bounds_full(cam_pos, src_bin_world_for_crop)
    bin_bounds_300 = _bin_bounds_300(cam_pos, src_bin_world_for_crop)
    _log(f"    bin_bounds 640   : x[{bin_bounds_640[0]},{bin_bounds_640[2]})  "
         f"y[{bin_bounds_640[1]},{bin_bounds_640[3]})  (display crop)")
    _log(f"    bin_bounds 300   : x[{bin_bounds_300[0]},{bin_bounds_300[2]})  "
         f"y[{bin_bounds_300[1]},{bin_bounds_300[3]})  (display crop)")

    if not candidates:
        raise SystemExit("[FATAL] No GG-CNN candidates produced. Aborting.")

    _log("\n[3] Capturing overhead RGB ...")
    rgb_full = env.env.get_wrist_rgb(height=_CAM_H, width=_CAM_W)
    _log(f"    rgb_full shape   : {rgb_full.shape}  dtype: {rgb_full.dtype}")

    qualities = np.array([float(c["quality"]) for c in candidates])
    chosen_idx = int(np.argmax(qualities))
    chosen = candidates[chosen_idx]
    _log(f"\n[4] Greedy chooses candidate #{chosen_idx}  "
         f"q={float(chosen['quality']):.4f}")
    _log(f"    world_pos        : {chosen['world_pos']}")

    depth_full_path = os.path.join(out_dir, "depth_full.png")
    _save_depth_full(
        depth_full_path, depth_full,
        title="Overhead wrist depth",
        bin_bounds=bin_bounds_640)
    _log(f"[save] {depth_full_path}")

    input_path = os.path.join(out_dir, "depth_ggcnn_input.png")
    _save_ggcnn_input(
        input_path, depth_pp, auto_norm_range,
        title="GG-CNN depth input",
        bin_bounds=bin_bounds_300)
    _log(f"[save] {input_path}")

    qheat_path = os.path.join(out_dir, "quality_heatmap.png")
    _save_quality_heatmap(
        qheat_path, quality_map, candidates, chosen_idx,
        title="GG-CNN grasp-quality heatmap",
        bin_bounds=bin_bounds_300)
    _log(f"[save] {qheat_path}")

    overlay_caption = ("red = predicted candidates,  "
                       "green = chosen (highest quality)")
    rgb_overlay_path = os.path.join(out_dir, "overhead_rgb_with_predictions.png")
    _save_overlay_on_image(
        rgb_overlay_path, rgb_full, candidates, chosen_idx,
        is_depth=False,
        suptitle="GG-CNN grasp candidates",
        caption=overlay_caption,
        bin_bounds=bin_bounds_640)
    _log(f"[save] {rgb_overlay_path}")

    depth_overlay_path = os.path.join(out_dir, "depth_full_with_predictions.png")
    _save_overlay_on_image(
        depth_overlay_path, depth_full, candidates, chosen_idx,
        is_depth=True,
        suptitle="Depth with GG-CNN candidates",
        caption=overlay_caption,
        bin_bounds=bin_bounds_640)
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
        wr.writerow(["idx", "body", "quality",
                     "pred_px", "pred_py",
                     "pred_world_x", "pred_world_y", "pred_world_z",
                     "pred_angle_rad", "pred_width_px"])
        with open(info_txt, "w") as txt:
            txt.write("Per-candidate GG-CNN output (full-image demo)\n")
            txt.write("=" * 78 + "\n")
            txt.write(f"datetime          : "
                      f"{datetime.datetime.now().isoformat(timespec='seconds')}\n")
            txt.write(f"n_objects         : {args.n_objects}\n")
            txt.write(f"layout_jitter     : {args.layout_jitter}\n")
            txt.write(f"seed              : {args.seed}\n")
            txt.write(f"floor_depth       : {floor_depth:.4f} m\n")
            txt.write(f"cam_pos           : {cam_pos}\n")
            txt.write(f"pad_y             : {meta['pad_y']}  "
                      f"(480 -> 640 with floor_depth)\n")
            txt.write(f"scale 300->640    : {meta['scale_300_to_640']:.4f}\n")
            txt.write(f"render_fovy       : {_RENDER_FOVY} deg  "
                      f"(K_back built at {_CAM_W}x{_CAM_H})\n")
            txt.write(f"auto_norm_range   : {auto_norm_range} m\n")
            txt.write(f"chosen_idx        : {chosen_idx}\n")
            txt.write("\nExecution semantics:\n")
            txt.write(" - Predicted XY  : IS used at execution (use_snap_xy=False).\n")
            txt.write(" - Predicted Z   : NOT used at execution; Z is GT-snapped.\n")
            txt.write(" - Predicted yaw : NOT used at execution; yaw is GT-snapped.\n")
            txt.write(" - Width (px)    : NOT used at execution; gripper closes\n"
                      "                   to a fixed pad gap target.\n")
            txt.write("=" * 78 + "\n\n")

            for i, c in enumerate(candidates):
                wp = np.asarray(c["world_pos"], dtype=float)
                body_name = _associate_body(env, wp[:2])
                row = [
                    i, body_name, f"{float(c['quality']):.4f}",
                    int(round(c["px_full"])), int(round(c["py_full"])),
                    f"{wp[0]:.4f}", f"{wp[1]:.4f}", f"{wp[2]:.4f}",
                    f"{float(c['angle_rad']):.4f}",
                    f"{float(c['width_px']):.2f}",
                ]
                wr.writerow(row)
                txt.write(f"[#{i:2d}] body~={body_name}  "
                          f"q={float(c['quality']):.4f}\n")
                txt.write(f"     px300=({c['px300']:3d},{c['py300']:3d})  "
                          f"px_full=({int(round(c['px_full'])):3d},"
                          f"{int(round(c['py_full'])):3d})\n")
                txt.write(f"     width_px={float(c['width_px']):6.2f}  "
                          f"angle_rad={float(c['angle_rad']):+.4f}  "
                          f"depth_m={float(c['depth_m']):.4f}\n")
                txt.write(f"     world_xyz=({wp[0]:+.4f},{wp[1]:+.4f},"
                          f"{wp[2]:+.4f})\n\n")
    _log(f"[save] {candidates_csv}")
    _log(f"[save] {info_txt}")

    env._candidates = [chosen]
    assoc_name = env._associate_item(np.asarray(chosen["world_pos"]))
    if assoc_name is None:
        raise SystemExit(
            f"[FATAL] Could not associate candidate #{chosen_idx} "
            f"to any source-bin item.")
    grasp_pos_base = env._wrist_target_from_candidate(chosen)
    item_pos = np.asarray(raw_env.get_object_positions()[assoc_name],
                          dtype=float)
    item_top = float(item_pos[2]) + _ITEM_HALF_HEIGHT
    grasp_pos = np.array([
        grasp_pos_base[0], grasp_pos_base[1],
        item_top - _GRASP_DESCENT_OFFSET + _FINGER_TO_WRIST,
    ], dtype=float)
    grasp_quat = env._grasp_quat_for_item(assoc_name)
    _log(f"\n[6] Eval grasp_pos / grasp_quat assembled")
    _log(f"    target_item       = {assoc_name}")
    _log(f"    grasp_pos         = {grasp_pos}")
    _log(f"    grasp_quat (wxyz) = {grasp_quat}")

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
    }

    def _capture_one():
        try:
            frame = _render_three_quarter(raw_env, height=480, width=640)
        except Exception:
            return
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

    def _step_with_capture(action):
        ret = orig_step(action)
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
    # (4) restore pp.frame_hook = _capture_one for the cinematic block.
    above_pile = grasp_pos + np.array([0.0, 0.0, 0.12], dtype=float)
    pp = env.pick_place
    pp.frame_hook = _capture_one
    pp._smooth_teleport(above_pile, grasp_quat, n_steps=20)

    _log("\n[9] Running pick_place.attempt_grasp_hybrid() with frame capture")
    _noop_hook = lambda: None
    pp.frame_hook = _noop_hook
    raw_env.step = _step_with_capture

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

    # Cinematic transport runs only on grasp success. On the n=17 scene the
    # GG-CNN pick fails (predicate 0/4) so this block is SKIPPED. the code is
    # kept in place for future seeds where GG-CNN might succeed.
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
            # Pin the picked item to the EEF via _tracked_item + _tracked_offset.
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
            wp_lower     = np.array([dst_bin[0], dst_bin[1],
                                     float(dst_bin[2]) + 0.12],
                                    dtype=float)
            cinematic_info["waypoints"] = {
                "lift":      [float(x) for x in wp_lift],
                "transport": [float(x) for x in wp_transport],
                "lower":     [float(x) for x in wp_lower],
            }

            _log("\n[9b] CINEMATIC pick-and-place sequence ...")
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

            pp._tracked_item = None
            pp._tracked_offset = None
            pp.frame_hook = None
            pp._apply_contact_mode("normal")

            _log("    releasing item (open gripper) + physics drop ...")
            open_action = np.zeros(7, dtype=np.float32)
            open_action[-1] = -1.0
            n0 = len(video_frames)
            raw_env.step = _step_with_capture
            try:
                for _ in range(30):
                    raw_env.step(open_action)
            finally:
                raw_env.step = orig_step
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

    _log(f"\n[10] Capturing post-roll (2.0 s at {args.fps} fps)")
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
        "chosen_body": assoc_name,
        "chosen_idx": chosen_idx,
        "chosen_quality": float(chosen.get("quality", 0.0)),
        "grasp_pos": [float(grasp_pos[0]), float(grasp_pos[1]),
                      float(grasp_pos[2])],
        "grasp_quat_wxyz": [float(grasp_quat[0]), float(grasp_quat[1]),
                            float(grasp_quat[2]), float(grasp_quat[3])],
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
        "exception": grasp_exc,
        "display_rotation_cw_deg": 90,
        "display_crop": "source_bin_square",
        "display_bin_bounds_640": list(bin_bounds_640),
        "display_bin_bounds_300": list(bin_bounds_300),
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
    _log("GG-CNN PICK + PLACE (TOP-cube head-to-head)  SUMMARY")
    _log("=" * 78)
    _log(f"out_dir              : {out_dir}")
    _log(f"n_objects            : {args.n_objects}  "
         f"layout_jitter={args.layout_jitter}  seed={args.seed}")
    _log(f"n candidates         : {len(candidates)}  "
         f"chosen=#{chosen_idx}  q={float(chosen['quality']):.4f}")
    _log(f"target_item          : {assoc_name}")
    _log(f"grasp_pos (wrist tgt): {grasp_pos}")
    _log(f"grasp_ok             : {grasp_result.get('grasp_ok')}")
    _log(f"items_ejected        : {grasp_result.get('items_ejected')}")
    _log(f"sum_disturb_raw_m    : "
         f"{grasp_result.get('neighbour_disturbance_raw_m', 0.0):.4f}")
    _log(f"max_disturb_m        : "
         f"{grasp_result.get('neighbour_disturbance_max_m', 0.0):.4f}")
    _log(f"predicate_subchecks  : {n_pred_succ}/4")
    _log(f"predicate_reason     : {grasp_result.get('predicate_reason')}")
    _log(f"picked_body_com_z    : {picked_body_com_z_initial}  "
         f"(>=0.86 means TOP layer of pile)")
    _log(f"is_top_layer_pick    : "
         f"{picked_body_com_z_initial is not None and picked_body_com_z_initial >= 0.86}")
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
        description=("GG-CNN demo on the RL-success scene (n_objects=17, "
                     "seed=1000, layout_jitter=0.0). Standard unmask pipeline; "
                     "5 perception PNGs cropped to source-bin square and "
                     "rotated 90 deg CW at display time only (GG-CNN still "
                     "sees the full 300x300 input)."))
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(_ROOT, "demos", "ggcnn_top_cube"))
    parser.add_argument("--n_objects", type=int, default=17)
    parser.add_argument("--layout_jitter", type=float, default=0.0)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--failure_mask_size", type=int, default=3)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--ggcnn_device", type=str, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
