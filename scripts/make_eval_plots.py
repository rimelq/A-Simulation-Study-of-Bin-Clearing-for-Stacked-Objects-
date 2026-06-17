"""Regenerate the 8 evaluation plots from the frozen per_episode.csv data.

Plots produced (matching the shipped PNGs in results/):
  1. plot_headline_items_delivered.png  - grouped bar chart with std error bars
  2. plot_items_delivered_fraction.png  - same but as fraction (items / n_objects)
  3. plot_pairwise_rl_vs_others.png     - paired-diff CIs for RL vs each policy
  4. plot_distribution_boxplots.png     - per-cell distribution boxplots
  5. plot_cleared_fraction.png          - simple bar chart of cleared%
  6. plot_disturbance.png               - mean disturbance per policy x n_obj
  7. plot_effect_size_heatmap.png       - Cohen-d matrix across all policy pairs
  8. plot_5policy_summary.png           - 3-panel stacked summary

Usage (from the repo root):
  python scripts/make_eval_plots.py
  python scripts/make_eval_plots.py \
      --data_dir results/data --out_dir results/
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

POLICIES = ["greedy_ggcnn", "topdown", "heuristic_augmented", "random", "rl"]
N_OBJECTS = [10, 15, 20]

COLORS = {
    "greedy_ggcnn":        "#999999",
    "topdown":             "#5A9BD4",
    "heuristic_augmented": "#70AD47",
    "random":              "#FFC000",
    "rl":                  "#C00000",
}

LABELS = {
    "greedy_ggcnn":        "greedy_ggcnn (depth -> GG-CNN)",
    "topdown":             "topdown (argmax z)",
    "heuristic_augmented": "heuristic_augmented (L4 rule)",
    "random":              "random (info-blind)",
    "rl":                  "RL (trained)",
}


def annotate_sig(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def plot_headline_items_delivered(df, out_dir, fraction=False):
    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.16
    x = np.arange(len(N_OBJECTS))

    for i, policy in enumerate(POLICIES):
        means, stds = [], []
        for n in N_OBJECTS:
            d = df[(df.policy == policy) & (df.n_objects == n)]["n_delivered"].values
            if fraction:
                means.append(np.mean(d) / n)
                stds.append(np.std(d) / n)
            else:
                means.append(np.mean(d))
                stds.append(np.std(d))
        offset = (i - 2) * width
        ax.bar(
            x + offset, means, width,
            yerr=stds, capsize=4,
            label=LABELS[policy], color=COLORS[policy],
            edgecolor="black", linewidth=0.5,
            error_kw={"linewidth": 1.0, "ecolor": "#333333"},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"n_objects = {n}" for n in N_OBJECTS], fontsize=11)
    if fraction:
        ax.set_ylabel("Mean fraction of items delivered\n(items_delivered / n_objects)", fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        title = "Fraction of items delivered per episode  (60 paired episodes per cell)"
        outname = "plot_items_delivered_fraction.png"
    else:
        ax.set_ylabel("Mean items delivered per episode", fontsize=11)
        ymax = max(N_OBJECTS) + 1
        ax.set_ylim(0, ymax)
        for j, n in enumerate(N_OBJECTS):
            ax.axhline(n, xmin=(j - 0.5 + 0.5) / len(N_OBJECTS),
                       xmax=(j + 0.5 + 0.5) / len(N_OBJECTS),
                       color="gray", linestyle=":", linewidth=0.6, alpha=0.4)
        title = "Items delivered per episode  (60 paired episodes per cell)"
        outname = "plot_headline_items_delivered.png"

    ax.set_title(title, fontsize=12, pad=10)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_dir / outname, dpi=140)
    plt.close(fig)
    return outname


def plot_pairwise_rl_vs_others(df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    competitors = ["random", "heuristic_augmented", "topdown", "greedy_ggcnn"]

    for ax_i, n in enumerate(N_OBJECTS):
        ax = axes[ax_i]
        means, lows, highs, sigs, names = [], [], [], [], []
        rl_d = df[(df.policy == "rl") & (df.n_objects == n)].sort_values("seed")["n_delivered"].values

        for comp in competitors:
            comp_d = df[(df.policy == comp) & (df.n_objects == n)].sort_values("seed")["n_delivered"].values
            diff = rl_d - comp_d
            m = np.mean(diff)
            se = np.std(diff, ddof=1) / np.sqrt(len(diff))
            ci_lo = m - 1.96 * se
            ci_hi = m + 1.96 * se
            t, p = stats.ttest_rel(rl_d, comp_d)
            means.append(m)
            lows.append(m - ci_lo)
            highs.append(ci_hi - m)
            sigs.append(annotate_sig(p))
            names.append(comp.replace("_", "\n"))

        ypos = np.arange(len(competitors))
        colors = ["#C00000" if m > 0 else "#5A9BD4" for m in means]
        ax.barh(ypos, means, xerr=[lows, highs], capsize=4,
                color=colors, alpha=0.85, edgecolor="black", linewidth=0.5,
                error_kw={"linewidth": 1.0, "ecolor": "#333333"})
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(ypos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_title(f"n_objects = {n}", fontsize=12, pad=8)
        ax.set_xlabel("Mean delta. RL minus competitor (items)", fontsize=10)
        ax.grid(axis="x", alpha=0.3, linestyle=":")
        ax.set_axisbelow(True)

        for j, (m, s) in enumerate(zip(means, sigs)):
            if m >= 0:
                xpos = m + highs[j] + 0.3
                ha = "left"
            else:
                xpos = 0.3
                ha = "left"
            ax.text(xpos, j, f"{m:+.2f} {s}", va="center", ha=ha,
                    fontsize=9, color="black")

    axes[0].invert_yaxis()
    fig.suptitle("RL vs other policies, paired difference in items delivered  (60 paired seeds, 95% CI)",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "plot_pairwise_rl_vs_others.png", dpi=140)
    plt.close(fig)
    return "plot_pairwise_rl_vs_others.png"


def plot_distribution_boxplots(df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax_i, n in enumerate(N_OBJECTS):
        ax = axes[ax_i]
        data_to_plot = []
        for p in POLICIES:
            d = df[(df.policy == p) & (df.n_objects == n)]["n_delivered"].values
            data_to_plot.append(d)
        bp = ax.boxplot(
            data_to_plot, patch_artist=True,
            tick_labels=[p.replace("_", "\n") for p in POLICIES],
            widths=0.6, showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="white",
                           markeredgecolor="black", markersize=5),
            medianprops=dict(color="black", linewidth=1.2),
        )
        for patch, p in zip(bp["boxes"], POLICIES):
            patch.set_facecolor(COLORS[p])
            patch.set_alpha(0.7)
            patch.set_edgecolor("black")

        ax.axhline(n, color="gray", linestyle=":", linewidth=0.8, alpha=0.6,
                   label=f"max delivery = {n}")
        ax.set_ylim(0, n + 2)
        ax.set_ylabel("Items delivered per episode")
        ax.set_title(f"n_objects = {n}", fontsize=12, pad=8)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle("Distribution of items delivered per episode (60 episodes per cell)",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "plot_distribution_boxplots.png", dpi=140)
    plt.close(fig)
    return "plot_distribution_boxplots.png"


def plot_cleared_fraction(df, out_dir):
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.16
    x = np.arange(len(N_OBJECTS))

    for i, policy in enumerate(POLICIES):
        fracs = []
        for n in N_OBJECTS:
            d = df[(df.policy == policy) & (df.n_objects == n)]["cleared"].values
            fracs.append(np.mean(d))
        offset = (i - 2) * width
        bars = ax.bar(x + offset, fracs, width,
                      label=LABELS[policy], color=COLORS[policy],
                      edgecolor="black", linewidth=0.5)
        for bar, f in zip(bars, fracs):
            if f > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, f + 0.005,
                        f"{f * 100:.1f}%",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"n_objects = {n}" for n in N_OBJECTS], fontsize=11)
    ax.set_ylabel("Cleared fraction (full bin emptied within budget)", fontsize=11)
    ax.set_ylim(0, 0.20)
    ax.set_title("Cleared fraction (binary. bin fully emptied within 3 * n_objects steps)\n"
                 "Only random clears at n=10. cleared_fraction is not the discriminating metric",
                 fontsize=11, pad=10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_cleared_fraction.png", dpi=140)
    plt.close(fig)
    return "plot_cleared_fraction.png"


def plot_disturbance(df, out_dir):
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.16
    x = np.arange(len(N_OBJECTS))

    for i, policy in enumerate(POLICIES):
        means, stds = [], []
        for n in N_OBJECTS:
            d = df[(df.policy == policy) & (df.n_objects == n)]["sum_disturbance_m"].values
            means.append(np.mean(d))
            stds.append(np.std(d))
        offset = (i - 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               label=LABELS[policy], color=COLORS[policy],
               edgecolor="black", linewidth=0.5,
               error_kw={"linewidth": 0.8, "ecolor": "#333333"})

    ax.set_xticks(x)
    ax.set_xticklabels([f"n_objects = {n}" for n in N_OBJECTS], fontsize=11)
    ax.set_ylabel("Mean neighbour disturbance per episode (m)", fontsize=11)
    ax.set_title("Neighbour disturbance per episode  (lower = less pile damage)",
                 fontsize=12, pad=10)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_disturbance.png", dpi=140)
    plt.close(fig)
    return "plot_disturbance.png"


def plot_effect_size_heatmap(df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax_i, n in enumerate(N_OBJECTS):
        ax = axes[ax_i]
        d_mat = np.zeros((len(POLICIES), len(POLICIES)))
        for i, pi in enumerate(POLICIES):
            di = df[(df.policy == pi) & (df.n_objects == n)].sort_values("seed")["n_delivered"].values
            for j, pj in enumerate(POLICIES):
                dj = df[(df.policy == pj) & (df.n_objects == n)].sort_values("seed")["n_delivered"].values
                diff = di - dj
                if np.std(diff) > 0:
                    d_mat[i, j] = np.mean(diff) / np.std(diff)
                else:
                    d_mat[i, j] = 0.0

        im = ax.imshow(d_mat, cmap="RdBu_r", vmin=-4, vmax=4, aspect="equal")
        ax.set_xticks(range(len(POLICIES)))
        ax.set_yticks(range(len(POLICIES)))
        ax.set_xticklabels([p.replace("_", "\n") for p in POLICIES], fontsize=7, rotation=0)
        ax.set_yticklabels([p.replace("_", "\n") for p in POLICIES], fontsize=7)
        ax.set_title(f"n_objects = {n}", fontsize=11, pad=8)

        for i in range(len(POLICIES)):
            for j in range(len(POLICIES)):
                if i == j:
                    txt = "0"
                else:
                    txt = f"{d_mat[i, j]:+.2f}"
                color = "white" if abs(d_mat[i, j]) > 2 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.04, pad=0.03)
    cbar.set_label("Cohen's d (row vs column)", fontsize=10)
    fig.suptitle("Effect size (Cohen's d) for paired difference in items delivered  "
                 "(row policy minus column policy)",
                 fontsize=12, y=0.97)
    fig.savefig(out_dir / "plot_effect_size_heatmap.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return "plot_effect_size_heatmap.png"


def plot_5policy_summary(df, out_dir):
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Panel 1: items delivered grouped bar
    ax = axes[0]
    width = 0.16
    x = np.arange(len(N_OBJECTS))
    for i, policy in enumerate(POLICIES):
        means, stds = [], []
        for n in N_OBJECTS:
            d = df[(df.policy == policy) & (df.n_objects == n)]["n_delivered"].values
            means.append(np.mean(d))
            stds.append(np.std(d))
        offset = (i - 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               label=LABELS[policy], color=COLORS[policy],
               edgecolor="black", linewidth=0.5,
               error_kw={"linewidth": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in N_OBJECTS])
    ax.set_ylabel("Items delivered")
    ax.set_title("Items delivered per episode (mean +/- std, 60 paired episodes)",
                 fontsize=11, pad=6)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)

    # Panel 2: items as fraction
    ax = axes[1]
    for i, policy in enumerate(POLICIES):
        fracs, fstds = [], []
        for n in N_OBJECTS:
            d = df[(df.policy == policy) & (df.n_objects == n)]["n_delivered"].values
            fracs.append(np.mean(d) / n)
            fstds.append(np.std(d) / n)
        offset = (i - 2) * width
        ax.bar(x + offset, fracs, width, yerr=fstds, capsize=3,
               color=COLORS[policy], edgecolor="black", linewidth=0.5,
               error_kw={"linewidth": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in N_OBJECTS])
    ax.set_ylabel("Fraction delivered (items / n_objects)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Fraction of items delivered (normalised for clutter level)",
                 fontsize=11, pad=6)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)

    # Panel 3: cleared fraction
    ax = axes[2]
    for i, policy in enumerate(POLICIES):
        fracs = []
        for n in N_OBJECTS:
            d = df[(df.policy == policy) & (df.n_objects == n)]["cleared"].values
            fracs.append(np.mean(d))
        offset = (i - 2) * width
        bars = ax.bar(x + offset, fracs, width,
                      color=COLORS[policy], edgecolor="black", linewidth=0.5)
        for bar, f in zip(bars, fracs):
            if f > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, f + 0.005,
                        f"{f * 100:.1f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in N_OBJECTS])
    ax.set_ylabel("Cleared fraction")
    ax.set_ylim(0, 0.18)
    ax.set_title("Cleared fraction (binary. bin fully emptied within step budget)",
                 fontsize=11, pad=6)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)

    fig.suptitle("5-policy comparison summary  (final eval, layout_jitter=0.02, 900 episodes total)",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "plot_5policy_summary.png", dpi=140)
    plt.close(fig)
    return "plot_5policy_summary.png"


def plot_items_and_disturbance_summary(df, out_dir):
    # Combined 3-panel summary: items delivered (count + fraction) + disturbance.
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    width = 0.16
    x = np.arange(len(N_OBJECTS))

    # Panel 1: items delivered (count, grouped bar)
    ax = axes[0]
    for i, policy in enumerate(POLICIES):
        means, stds = [], []
        for n in N_OBJECTS:
            d = df[(df.policy == policy) & (df.n_objects == n)]["n_delivered"].values
            means.append(np.mean(d))
            stds.append(np.std(d))
        offset = (i - 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               label=LABELS[policy], color=COLORS[policy],
               edgecolor="black", linewidth=0.5,
               error_kw={"linewidth": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in N_OBJECTS])
    ax.set_ylabel("Items delivered")
    ax.set_title("Items delivered per episode (mean +/- std, 60 paired episodes)",
                 fontsize=11, pad=6)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)

    # Panel 2: items as fraction
    ax = axes[1]
    for i, policy in enumerate(POLICIES):
        fracs, fstds = [], []
        for n in N_OBJECTS:
            d = df[(df.policy == policy) & (df.n_objects == n)]["n_delivered"].values
            fracs.append(np.mean(d) / n)
            fstds.append(np.std(d) / n)
        offset = (i - 2) * width
        ax.bar(x + offset, fracs, width, yerr=fstds, capsize=3,
               color=COLORS[policy], edgecolor="black", linewidth=0.5,
               error_kw={"linewidth": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in N_OBJECTS])
    ax.set_ylabel("Fraction delivered (items / n_objects)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Fraction of items delivered (normalised for clutter level)",
                 fontsize=11, pad=6)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)

    # Panel 3: neighbour disturbance per episode (lower = less pile damage)
    ax = axes[2]
    for i, policy in enumerate(POLICIES):
        means, stds = [], []
        for n in N_OBJECTS:
            d = df[(df.policy == policy) & (df.n_objects == n)]["sum_disturbance_m"].values
            means.append(np.mean(d))
            stds.append(np.std(d))
        offset = (i - 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               color=COLORS[policy], edgecolor="black", linewidth=0.5,
               error_kw={"linewidth": 0.8, "ecolor": "#333333"})
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in N_OBJECTS])
    ax.set_ylabel("Mean neighbour disturbance per episode (m)")
    ax.set_title("Neighbour disturbance per episode (lower = less pile damage)",
                 fontsize=11, pad=6)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)

    fig.suptitle("Items delivered + disturbance summary  (final eval, layout_jitter=0.02, 900 episodes total)",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "plot_items_and_disturbance_summary.png", dpi=140)
    plt.close(fig)
    return "plot_items_and_disturbance_summary.png"


def main():
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default=str(repo_root / "results" / "data"),
                    help="Directory containing per_episode.csv")
    ap.add_argument("--out_dir", default=str(repo_root / "results"),
                    help="Directory to write the 8 PNGs into")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_ep_csv = data_dir / "per_episode.csv"
    if not per_ep_csv.exists():
        raise FileNotFoundError(f"per_episode.csv not found at {per_ep_csv}")

    df = pd.read_csv(per_ep_csv)
    print(f"Loaded {len(df)} per-episode rows from {per_ep_csv}")
    print(f"Cells: {df.groupby(['policy','n_objects']).size().to_dict()}")

    produced = []
    produced.append(plot_headline_items_delivered(df, out_dir, fraction=False))
    produced.append(plot_headline_items_delivered(df, out_dir, fraction=True))
    produced.append(plot_pairwise_rl_vs_others(df, out_dir))
    produced.append(plot_distribution_boxplots(df, out_dir))
    produced.append(plot_cleared_fraction(df, out_dir))
    produced.append(plot_disturbance(df, out_dir))
    produced.append(plot_effect_size_heatmap(df, out_dir))
    produced.append(plot_5policy_summary(df, out_dir))
    produced.append(plot_items_and_disturbance_summary(df, out_dir))

    print()
    print("Produced plots:")
    for p in produced:
        path = out_dir / p
        print(f"  {path}  ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
