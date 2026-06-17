"""Render the bin-cropped overhead RGB without any prediction overlays.

Same scene (n=17, jitter=0, seed=1000) and the same crop + rotation as
demos/*/overhead_rgb_with_predictions.png, just no candidate annotations.
Saved to results/env_views/overhead_rgb_plain.png.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl.bin_clearing_env import BinClearingGymEnv
from scripts.demo_rl_pick_and_place_top import (
    capture_overhead_for_viz, _bin_bounds_full,
)


def main():
    env = BinClearingGymEnv(
        n_objects=17, K=10, candidate_source="ppo",
        ppo_visibility_mode="raycast", ppo_quality_mode="analytical",
        filter_candidates=True, orientation_source="snap",
        use_snap_xy=False, use_snap_z=True,
        layout_jitter=0.0, failure_mask_size=3,
        reward_mode="hybrid_physics",
    )
    obs, _ = env.reset(seed=1000)
    raw = env.env

    meta = capture_overhead_for_viz(env_inner=raw, sensing_ctrl=env.sensing_ctrl)
    cam_pos = meta["cam_pos"]
    src_bin = raw.get_src_bin_world_pos()
    rgb_full = raw.get_wrist_rgb(height=480, width=640)
    bin_bounds = _bin_bounds_full(cam_pos, src_bin)

    x0, y0, x1, y1 = bin_bounds
    image_show = rgb_full[y0:y1, x0:x1]
    H_orig, W_orig = image_show.shape[:2]
    image_disp = np.rot90(image_show, k=-1)

    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    ax.imshow(image_disp)
    ax.set_xlim(0, H_orig)
    ax.set_ylim(W_orig, 0)
    ax.set_aspect("equal")
    ax.set_title("Overhead RGB (input to perception)", fontsize=11)

    out_dir = os.path.join(_ROOT, "results", "env_views")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "overhead_rgb_plain.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
