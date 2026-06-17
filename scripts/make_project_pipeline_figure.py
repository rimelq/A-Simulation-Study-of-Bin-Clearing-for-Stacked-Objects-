"""
make_project_pipeline_figure.py

Render a single dense, paper-quality pipeline schema for the cube-clearing project.

Layout:
  LEFT   : PROBLEM panel  (task + env thumbnail + per-step API)
  MIDDLE : 5 policy boxes (random, topdown, greedy_ggcnn, heuristic_augmented, rl)
           Each box has INPUT / METHOD / OUTPUT rows.
  RIGHT  : RESULTS panel  (bar chart of items delivered @ N=20)

Output:
  results/project_pipeline.png  (300 dpi)
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


# Constants

ROOT = Path(__file__).resolve().parent.parent
OUT_PNG = ROOT / "results" / "project_pipeline.png"
ENV_THUMB = ROOT / "results" / "env_views" / "wide_3q_front.png"


POLICIES = [
    {
        "key": "random",
        "name": "random",
        "color": "#888888",
        "input":  "Action mask\nover K=10 candidates x 27 refinements",
        "method": "Uniform sample\nover unmasked joint actions",
        "output": "action = k*27 + m\n(any m allowed)",
    },
    {
        "key": "topdown",
        "name": "topdown",
        "color": "#1f77b4",  # C0
        "input":  "PPO-oracle candidates\n+ world-z heights + mask",
        "method": "argmax world-z\n(tie-break lower body id)",
        "output": "action = k*27 + 13\n(zero refinement)",
    },
    {
        "key": "greedy_ggcnn",
        "name": "greedy_ggcnn",
        "color": "#ff7f0e",  # C1
        "input":  "GG-CNN candidates\n(quality scores) + mask",
        "method": "argmax quality\n(tie-break lower body id)",
        "output": "action = k*27 + 13\n(zero refinement)",
    },
    {
        "key": "heuristic_augmented",
        "name": "heuristic_augmented",
        "color": "#9467bd",  # C4
        "input":  "Per-candidate L4 features:\nquality, clear, dx/dy, exposure",
        "method": "Drop keystones -> prefer\nexposed + descent-clear ->\nargmax quality; refine dx/dy/dyaw",
        "output": "action = k*27 + m\n(non-trivial refinement)",
    },
    {
        "key": "rl",
        "name": "RL (MaskablePPO)",
        "color": "#2ca02c",  # C2
        "input":  "Full obs vector\n+ action mask",
        "method": "MaskablePPO.predict\n(deterministic, best_model.zip)",
        "output": "action = k*27 + m\n(learned slot + refinement)",
    },
]


RESULTS = {
    "random":              {"items": 10.53, "items_n20": 12.62, "cleared": 0.719, "disturb": 0.411},
    "topdown":             {"items":  9.08, "items_n20": 11.78, "cleared": 0.608, "disturb": 0.256},
    "greedy_ggcnn":        {"items":  2.89, "items_n20":  3.73, "cleared": 0.195, "disturb": 0.469},
    "heuristic_augmented": {"items":  8.58, "items_n20": 10.98, "cleared": 0.574, "disturb": 0.262},
    "rl":                  {"items": 10.59, "items_n20": 13.68, "cleared": 0.709, "disturb": 0.318},
}


# Drawing helpers

def draw_policy_box(ax, x, y, w, h, policy, fontsize_header=9, fontsize_body=7.2):
    """Draw one policy card with a colored header bar and INPUT/METHOD/OUTPUT rows."""
    color = policy["color"]

    # Outer rounded rectangle (white background w/ subtle border).
    outer = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.002,rounding_size=0.012",
        linewidth=1.0, edgecolor="#444444", facecolor="white", zorder=2,
    )
    ax.add_patch(outer)

    # Header bar (colored).
    header_h = h * 0.22
    header = Rectangle(
        (x, y + h - header_h), w, header_h,
        linewidth=0, facecolor=color, alpha=0.92, zorder=3,
    )
    ax.add_patch(header)
    ax.text(
        x + w / 2, y + h - header_h / 2,
        policy["name"],
        ha="center", va="center",
        fontsize=fontsize_header, fontweight="bold", color="white", zorder=4,
    )

    # Three rows: INPUT / METHOD / OUTPUT
    body_top = y + h - header_h
    body_h   = h - header_h
    row_h    = body_h / 3.0

    rows = [
        ("INPUT",  policy["input"]),
        ("METHOD", policy["method"]),
        ("OUTPUT", policy["output"]),
    ]
    for i, (label, content) in enumerate(rows):
        row_y = body_top - (i + 1) * row_h
        # Light divider line above each row except the first
        if i > 0:
            ax.plot(
                [x + 0.004, x + w - 0.004], [row_y + row_h, row_y + row_h],
                color="#cccccc", linewidth=0.6, zorder=3,
            )
        # Label (left, small caps, light color)
        ax.text(
            x + 0.008, row_y + row_h - 0.012,
            label,
            ha="left", va="top",
            fontsize=fontsize_body - 0.6, fontweight="bold",
            color=color, zorder=4,
        )
        # Content (below label, normal weight)
        ax.text(
            x + 0.008, row_y + row_h - 0.030,
            content,
            ha="left", va="top",
            fontsize=fontsize_body, color="#222222", zorder=4,
        )


def add_arrow(ax, start_xy, end_xy, color="#999999", lw=1.0, alpha=0.7):
    arr = FancyArrowPatch(
        start_xy, end_xy,
        arrowstyle="->,head_length=4,head_width=3",
        color=color, linewidth=lw, alpha=alpha,
        connectionstyle="arc3,rad=0.0",
        zorder=1,
    )
    ax.add_patch(arr)


# Main figure

def main():
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    # One big axes used as a normalized 0..1 canvas for layout.
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # Title
    ax.text(
        0.5, 0.965,
        "Project pipeline: 5 candidate-selection policies for stacked-cube clearing",
        ha="center", va="center",
        fontsize=16, fontweight="bold", color="#111111",
    )
    ax.text(
        0.5, 0.935,
        "Hybrid-physics MuJoCo evaluation  |  K=10 candidate slots x 27 refinement cells  |  frozen-eval n=60 per (policy, n_objects)",
        ha="center", va="center",
        fontsize=10, color="#555555", style="italic",
    )

    # Column anchors
    col_left_x,   col_left_w   = 0.025, 0.245
    col_mid_x,    col_mid_w    = 0.305, 0.405
    col_right_x,  col_right_w  = 0.735, 0.245

    panel_top    = 0.905
    panel_bottom = 0.045

    # LEFT: PROBLEM panel
    left_outer = FancyBboxPatch(
        (col_left_x, panel_bottom),
        col_left_w, panel_top - panel_bottom,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        linewidth=1.0, edgecolor="#333333", facecolor="#fafafa", zorder=1,
    )
    ax.add_patch(left_outer)

    # Header
    ax.text(
        col_left_x + col_left_w / 2, panel_top - 0.025,
        "PROBLEM",
        ha="center", va="center",
        fontsize=12, fontweight="bold", color="#222222",
    )

    # Task description
    task_text = (
        "Clear cubes from the source bin\n"
        "and place them in the destination bin\n"
        "under hybrid-physics MuJoCo simulation."
    )
    ax.text(
        col_left_x + 0.012, panel_top - 0.060,
        task_text,
        ha="left", va="top",
        fontsize=9.0, color="#222222",
    )

    # Env thumbnail
    thumb_x = col_left_x + 0.018
    thumb_y = panel_bottom + 0.275
    thumb_w = col_left_w - 0.036
    thumb_h = 0.38
    if ENV_THUMB.exists():
        img = plt.imread(str(ENV_THUMB))
        # Place image as inset
        ax_img = fig.add_axes(
            [thumb_x, thumb_y, thumb_w, thumb_h]
        )
        ax_img.imshow(img)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for spine in ax_img.spines.values():
            spine.set_edgecolor("#333333")
            spine.set_linewidth(0.8)
        ax_img.set_title("env (wide_3q_front)", fontsize=8, color="#555555", pad=2)
    else:
        ax.text(
            col_left_x + col_left_w / 2, thumb_y + thumb_h / 2,
            "[env thumbnail missing]",
            ha="center", va="center", fontsize=9, color="#aa0000",
        )

    # Per-step API box (below thumbnail)
    api_top = panel_bottom + 0.245
    api_box = FancyBboxPatch(
        (col_left_x + 0.012, panel_bottom + 0.025),
        col_left_w - 0.024, api_top - panel_bottom - 0.025,
        boxstyle="round,pad=0.003,rounding_size=0.008",
        linewidth=0.8, edgecolor="#888888", facecolor="white", zorder=2,
    )
    ax.add_patch(api_box)
    ax.text(
        col_left_x + col_left_w / 2, api_top - 0.018,
        "Per-step API",
        ha="center", va="center",
        fontsize=9, fontweight="bold", color="#222222",
    )
    api_text = (
        "env emits K=10 candidates\n"
        "  -> policy picks (slot k, refinement m)\n"
        "  -> attempt_grasp_hybrid runs physics\n"
        "  -> reward, terminated, truncated, info"
    )
    ax.text(
        col_left_x + 0.020, api_top - 0.035,
        api_text,
        ha="left", va="top",
        fontsize=7.6, color="#222222", family="monospace",
    )

    # MIDDLE: 5 POLICY BOXES (stacked vertically)
    mid_outer = FancyBboxPatch(
        (col_mid_x, panel_bottom),
        col_mid_w, panel_top - panel_bottom,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        linewidth=1.0, edgecolor="#333333", facecolor="#fafafa", zorder=1,
    )
    ax.add_patch(mid_outer)
    ax.text(
        col_mid_x + col_mid_w / 2, panel_top - 0.025,
        "POLICIES  (candidate-selection rule)",
        ha="center", va="center",
        fontsize=12, fontweight="bold", color="#222222",
    )

    n_pol = len(POLICIES)
    # Vertical region for the cards
    cards_top    = panel_top - 0.055
    cards_bottom = panel_bottom + 0.020
    total_h      = cards_top - cards_bottom
    gap          = 0.010
    card_h       = (total_h - gap * (n_pol - 1)) / n_pol
    card_x       = col_mid_x + 0.012
    card_w       = col_mid_w - 0.024

    # Track each card center-left and center-right for arrow targets
    card_centers = []
    for i, pol in enumerate(POLICIES):
        card_y = cards_top - (i + 1) * card_h - i * gap
        draw_policy_box(ax, card_x, card_y, card_w, card_h, pol)
        card_centers.append({
            "key": pol["key"],
            "left":  (card_x, card_y + card_h / 2),
            "right": (card_x + card_w, card_y + card_h / 2),
            "color": pol["color"],
        })

    # RIGHT: RESULTS panel
    right_outer = FancyBboxPatch(
        (col_right_x, panel_bottom),
        col_right_w, panel_top - panel_bottom,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        linewidth=1.0, edgecolor="#333333", facecolor="#fafafa", zorder=1,
    )
    ax.add_patch(right_outer)
    ax.text(
        col_right_x + col_right_w / 2, panel_top - 0.025,
        "RESULTS  (frozen eval, n=60)",
        ha="center", va="center",
        fontsize=12, fontweight="bold", color="#222222",
    )

    # Primary bar chart: items delivered @ N=20
    bar_x = col_right_x + 0.045
    bar_y = panel_bottom + 0.31
    bar_w = col_right_w - 0.060
    bar_h = 0.46
    ax_bar = fig.add_axes([bar_x, bar_y, bar_w, bar_h])

    names  = [p["name"]  for p in POLICIES]
    keys   = [p["key"]   for p in POLICIES]
    colors = [p["color"] for p in POLICIES]
    vals   = [RESULTS[k]["items_n20"] for k in keys]

    # Horizontal bars so the ordering matches the middle column top-to-bottom.
    y_pos = np.arange(n_pol)[::-1]  # invert so first policy is at top
    bars = ax_bar.barh(y_pos, vals, color=colors, edgecolor="#333333", linewidth=0.6)
    for bar, v in zip(bars, vals):
        ax_bar.text(
            v + 0.25, bar.get_y() + bar.get_height() / 2,
            f"{v:.2f}",
            va="center", ha="left",
            fontsize=8.5, color="#222222", fontweight="bold",
        )
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(names, fontsize=8.5)
    ax_bar.set_xlabel("items delivered  (mean, N=20, n=60)", fontsize=8.5)
    ax_bar.set_xlim(0, max(vals) * 1.20)
    ax_bar.tick_params(axis="x", labelsize=8)
    ax_bar.set_title("Clearing performance @ N=20", fontsize=9.5, pad=4)
    for spine in ("top", "right"):
        ax_bar.spines[spine].set_visible(False)
    ax_bar.grid(axis="x", linestyle=":", linewidth=0.5, color="#cccccc")
    ax_bar.set_axisbelow(True)

    # Secondary: small disturbance bars + small numeric table
    dist_x = col_right_x + 0.045
    dist_y = panel_bottom + 0.085
    dist_w = col_right_w - 0.060
    dist_h = 0.18
    ax_dist = fig.add_axes([dist_x, dist_y, dist_w, dist_h])
    dvals = [RESULTS[k]["disturb"] for k in keys]
    dbars = ax_dist.barh(y_pos, dvals, color=colors, edgecolor="#333333", linewidth=0.5, alpha=0.85)
    for bar, v in zip(dbars, dvals):
        ax_dist.text(
            v + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}",
            va="center", ha="left",
            fontsize=7.5, color="#222222",
        )
    ax_dist.set_yticks(y_pos)
    ax_dist.set_yticklabels(names, fontsize=7.5)
    ax_dist.set_xlabel("sum disturbance (m, lower = gentler)", fontsize=7.5)
    ax_dist.set_xlim(0, max(dvals) * 1.25)
    ax_dist.tick_params(axis="x", labelsize=7)
    ax_dist.set_title("Disturbance (secondary)", fontsize=8.5, pad=3)
    for spine in ("top", "right"):
        ax_dist.spines[spine].set_visible(False)
    ax_dist.grid(axis="x", linestyle=":", linewidth=0.4, color="#dddddd")
    ax_dist.set_axisbelow(True)

    # Headline note at the bottom of the RESULTS panel
    ax.text(
        col_right_x + col_right_w / 2, panel_bottom + 0.045,
        "RL edges random by ~1 item @ N=20.\n"
        "greedy_ggcnn collapses in clutter.\n"
        "topdown / heuristic are gentlest.",
        ha="center", va="center",
        fontsize=8.0, color="#333333", style="italic",
    )

    # ARROWS: PROBLEM -> each policy box (left side)
    #         each policy box -> matching bar (right side)
    left_anchor_x = col_left_x + col_left_w     # right edge of LEFT panel
    left_anchor_y = (panel_top + panel_bottom) / 2

    # The horizontal x at which the bar starts (data x=0 in ax_bar).
    # ax_bar occupies [bar_x .. bar_x + bar_w] in figure coords.
    # Bar starts at the left edge of the axes (data x=0).
    bar_left_fig_x = bar_x  # zero of the data axis aligns with axes left edge

    # For each policy card, draw two arrows.
    # Inputs: from PROBLEM panel right edge to card left.
    for cc in card_centers:
        add_arrow(
            ax,
            (left_anchor_x + 0.002, cc["left"][1]),
            (cc["left"][0] - 0.002, cc["left"][1]),
            color="#999999", lw=1.0, alpha=0.65,
        )

    # Outputs: from card right edge to corresponding bar (right column).
    # We need each card's y-mapped to the bar's y position in figure coords.
    # The bar axes occupies fig-y in [bar_y .. bar_y + bar_h]; rows are evenly
    # spaced. Row i (top) corresponds to fig-y = bar_y + bar_h * (1 - (i+0.5)/n).
    for i, cc in enumerate(card_centers):
        target_fig_y = bar_y + bar_h * (1.0 - (i + 0.5) / n_pol)
        # Slight color tint per policy for output arrow
        add_arrow(
            ax,
            (cc["right"][0] + 0.002, cc["right"][1]),
            (bar_left_fig_x - 0.002, target_fig_y),
            color=cc["color"], lw=1.0, alpha=0.65,
        )

    # Save
    fig.savefig(OUT_PNG, dpi=300, bbox_inches=None, facecolor="white")
    plt.close(fig)
    print(f"[ok] wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
