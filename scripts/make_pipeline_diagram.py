"""
Pipeline diagram for the 5-policy grasp candidate selection pipeline.

Renders a paper-style figure showing two parallel input streams:
  - Wrist depth -> bin-tight crop -> GG-CNN (4 heads) -> NMS + CC centroid snap +
    PCA yaw -> 20 raw -> top-K=10 candidates -> greedy_ggcnn
  - GT poses -> PPO ghost (XY-footprint occlusion + analytical q + yaw snap) ->
    K=10 candidates -> {random, topdown, heuristic_augmented, RL}
All 5 policies merge into env.step (reward_mode=hybrid_physics) ->
attempt_grasp_hybrid (no transport); _deliver_item teleports cube on success.

Pure matplotlib. Output: results/pipeline_diagram.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# Palette
COL_INPUT       = "#ECECEC"   # light gray
COL_MODEL       = "#FBD3D3"   # light pink/red
COL_CANDIDATES  = "#FFF3B0"   # light yellow
COL_MERGE       = "#BBDEFB"   # light blue
COL_FINAL       = "#C8E6C9"   # light green for "items delivered" tag

# Policy header colors
COL_RANDOM   = "#D9D9D9"
COL_TOPDOWN  = "#BBDEFB"
COL_GGCNN    = "#FFD8A8"
COL_HEUR     = "#D7C4E8"
COL_RL       = "#C8E6C9"

BORDER_LW = 1.2
ARROW_LW  = 1.4
BOX_STYLE = "round,pad=0.4,rounding_size=0.4"


# Helpers
def draw_box(ax, x, y, w, h, text, facecolor, fontsize=11, weight="normal"):
    """Centered rounded box with text. x,y is the box CENTER."""
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=BOX_STYLE,
        linewidth=BORDER_LW,
        edgecolor="black",
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, color="black", wrap=True)


def draw_policy_box(ax, cx, cy, w, h, header, header_color,
                    method_line, output_line):
    """
    Policy box with a colored header bar and white body.
    cx, cy is center; box spans w x h.
    """
    x0 = cx - w / 2
    y0 = cy - h / 2

    # White body
    body = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle=BOX_STYLE,
        linewidth=BORDER_LW,
        edgecolor="black",
        facecolor="white",
    )
    ax.add_patch(body)

    # Header bar overlay (a thinner colored rounded patch at top)
    hdr_h = h * 0.32
    hdr = FancyBboxPatch(
        (x0 + 0.05, y0 + h - hdr_h - 0.05), w - 0.10, hdr_h,
        boxstyle="round,pad=0.02,rounding_size=0.25",
        linewidth=BORDER_LW,
        edgecolor="black",
        facecolor=header_color,
    )
    ax.add_patch(hdr)
    ax.text(cx, y0 + h - hdr_h / 2 - 0.05, header,
            ha="center", va="center", fontsize=11.5, weight="bold")

    # Body text
    body_top = y0 + h - hdr_h - 0.15
    ax.text(cx, body_top - 0.30, "METHOD",
            ha="center", va="center", fontsize=8.5, weight="bold", color="#444")
    ax.text(cx, body_top - 0.60, method_line,
            ha="center", va="center", fontsize=9)
    ax.text(cx, body_top - 1.05, "OUTPUT",
            ha="center", va="center", fontsize=8.5, weight="bold", color="#444")
    ax.text(cx, body_top - 1.35, output_line,
            ha="center", va="center", fontsize=9)


def arrow(ax, x0, y0, x1, y1, lw=ARROW_LW, color="black",
          style="-|>", connectionstyle="arc3,rad=0"):
    a = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style,
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(a)


# Build figure
def build_figure():
    fig, ax = plt.subplots(figsize=(16, 11), dpi=300)
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 16)
    ax.set_aspect("equal")
    ax.axis("off")

    # X coordinates for the two streams
    X_LEFT  = 5.0
    X_RIGHT = 15.0

    # Row 1: Inputs
    Y_R1 = 15.0
    draw_box(ax, X_LEFT,  Y_R1, 6.5, 1.1,
             "Wrist depth -> bin-tight crop\n-> resize 300x300 -> per-image min-max norm",
             COL_INPUT, fontsize=10.5, weight="bold")
    draw_box(ax, X_RIGHT, Y_R1, 6.5, 1.1,
             "GT object poses\n(perfect-perception oracle)",
             COL_INPUT, fontsize=11.5, weight="bold")

    # Row 2: Generators
    Y_R2 = 13.0
    draw_box(ax, X_LEFT,  Y_R2, 6.5, 1.5,
             "GG-CNN inference (4 heads: pos, cos, sin, width)\n"
             "-> NMS peaks (size=11, min_q=0.1), raw K=20\n"
             "-> CC centroid snap + PCA short-axis yaw override",
             COL_MODEL, fontsize=9.5)
    draw_box(ax, X_RIGHT, Y_R2, 6.5, 1.5,
             "PPO ghost generator (one cand. per visible source-bin cube)\n"
             "XY-footprint occlusion check (NOT raycast)\n"
             "q = 0.4*clearance + 0.3*height + 0.3*tilt; yaw snapped to GT",
             COL_MODEL, fontsize=9.0)

    # Row 3: Candidate lists
    Y_R3 = 10.9
    draw_box(ax, X_LEFT,  Y_R3, 6.5, 1.3,
             "20 raw -> top-K=10 candidates\n"
             "(px, py, depth, world_pos, world_quat,\n"
             "quality, angle, width; source_body_id back-filled)",
             COL_CANDIDATES, fontsize=9.5, weight="bold")
    draw_box(ax, X_RIGHT, Y_R3, 6.5, 1.3,
             "K = 10 candidates (sorted by analytical q)\n"
             "(world_pos, world_quat, quality, angle, width,\n"
             "source_body_id; px=py=-1, no pixel coords)",
             COL_CANDIDATES, fontsize=9.5, weight="bold")

    # Row 4: Policies
    Y_R4 = 7.6
    POL_W = 3.6
    POL_H = 3.0

    # Left side: greedy_ggcnn alone, centered under left candidates
    draw_policy_box(
        ax, X_LEFT, Y_R4, POL_W, POL_H,
        header="greedy_ggcnn",
        header_color=COL_GGCNN,
        method_line="argmax GG-CNN quality\n(tie-break: lower body_id)",
        output_line="slot k + zero refinement (m=13)",
    )

    # Right side: 4 policies in a row under right candidates
    # Spread them around X_RIGHT
    pol_cx = [X_RIGHT - 6.0, X_RIGHT - 2.0, X_RIGHT + 2.0, X_RIGHT + 6.0]
    POL_W_R = 3.5

    draw_policy_box(
        ax, pol_cx[0], Y_R4, POL_W_R, POL_H,
        header="random",
        header_color=COL_RANDOM,
        method_line="uniform over masked\njoint Discrete(K*27)",
        output_line="slot k + refinement m\n(both sampled)",
    )
    draw_policy_box(
        ax, pol_cx[1], Y_R4, POL_W_R, POL_H,
        header="topdown",
        header_color=COL_TOPDOWN,
        method_line="argmax world-z\n(tie-break: lower body_id)",
        output_line="slot k + zero refinement (m=13)",
    )
    draw_policy_box(
        ax, pol_cx[2], Y_R4, POL_W_R, POL_H,
        header="heuristic_augmented",
        header_color=COL_HEUR,
        method_line="drop keystones (>0.66);\nargmax exposure (>0.2)\nelse argmax q on clear pool",
        output_line="dyaw from pred_angle;\ndx/dy probe only if\napproach_clear<0.5",
    )
    draw_policy_box(
        ax, pol_cx[3], Y_R4, POL_W_R, POL_H,
        header="RL (MaskablePPO)",
        header_color=COL_RL,
        method_line="best_model.zip; predict(\ndeterministic=True,\naction_masks=K*27 mask)",
        output_line="slot k + learned refinement m",
    )

    # Row 5: Merge box
    Y_R5 = 3.2
    X_MID = 11.0
    draw_box(
        ax, X_MID, Y_R5, 17.0, 1.6,
        "Encoded action a = k*27 + m  ->  env.step (reward_mode=hybrid_physics)\n"
        "->  attempt_grasp_hybrid  (real-physics descent + close, no transport)\n"
        "returns: grasp_ok, grasp_quality, disturbance, ejections; "
        "on success _deliver_item TELEPORTS cube to dest-bin slot",
        COL_MERGE, fontsize=10.5, weight="bold",
    )

    # Final tag
    Y_R6 = 1.2
    draw_box(ax, X_MID, Y_R6, 4.0, 0.9,
             "Items delivered", COL_FINAL,
             fontsize=11, weight="bold")

    # Arrows
    # Row 1 -> Row 2 (vertical)
    arrow(ax, X_LEFT,  Y_R1 - 0.55, X_LEFT,  Y_R2 + 0.75)
    arrow(ax, X_RIGHT, Y_R1 - 0.55, X_RIGHT, Y_R2 + 0.75)

    # Row 2 -> Row 3
    arrow(ax, X_LEFT,  Y_R2 - 0.75, X_LEFT,  Y_R3 + 0.65)
    arrow(ax, X_RIGHT, Y_R2 - 0.75, X_RIGHT, Y_R3 + 0.65)

    # Row 3 -> Row 4
    # Left: single arrow to greedy_ggcnn
    arrow(ax, X_LEFT, Y_R3 - 0.65, X_LEFT, Y_R4 + POL_H / 2)

    # Right: fan out from right candidates to 4 policies
    fan_origin_y = Y_R3 - 0.65
    bus_y = fan_origin_y - 0.55
    # vertical drop from box to bus
    arrow(ax, X_RIGHT, fan_origin_y, X_RIGHT, bus_y + 0.05,
          style="-", lw=ARROW_LW)
    # horizontal bus
    ax.plot([pol_cx[0], pol_cx[-1]], [bus_y, bus_y],
            color="black", linewidth=ARROW_LW)
    # drop arrows from bus to each policy
    for cx in pol_cx:
        arrow(ax, cx, bus_y, cx, Y_R4 + POL_H / 2)

    # Row 4 -> Row 5 (5 converging arrows to merge box top)
    merge_top_y = Y_R5 + 0.8
    all_pol_cx = [X_LEFT] + pol_cx
    # Use a horizontal "collection" bus a bit above the merge box
    collect_y = Y_R5 + 1.6
    # drops from each policy down to collection bus
    for cx in all_pol_cx:
        ax.plot([cx, cx], [Y_R4 - POL_H / 2, collect_y],
                color="black", linewidth=ARROW_LW)
    # horizontal bus spanning all 5
    ax.plot([min(all_pol_cx), max(all_pol_cx)],
            [collect_y, collect_y],
            color="black", linewidth=ARROW_LW)
    # single arrow from bus center down into merge box
    arrow(ax, X_MID, collect_y, X_MID, merge_top_y)

    # Row 5 -> final tag
    arrow(ax, X_MID, Y_R5 - 0.8, X_MID, Y_R6 + 0.45)

    # Side labels on each stream
    ax.text(X_LEFT, Y_R1 + 0.85,
            "Stream A: neural perception",
            ha="center", va="bottom",
            fontsize=10, style="italic", color="#555")
    ax.text(X_RIGHT, Y_R1 + 0.85,
            "Stream B: oracle (GT poses)",
            ha="center", va="bottom",
            fontsize=10, style="italic", color="#555")

    # Title
    ax.text(11.0, 15.95,
            "Grasp candidate selection pipeline (5 policies, 2 input streams)",
            ha="center", va="bottom",
            fontsize=14, weight="bold")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return fig


def main():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pipeline_diagram.png")

    fig = build_figure()
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
