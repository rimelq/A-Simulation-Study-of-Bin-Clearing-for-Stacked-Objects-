"""Regenerate the training-side plots from the frozen RL training logs.

Reads from rl/models/ by default. Produces three plots in the output dir:
  1. plot_ppo_learning_curves.png    - sb3 progress.csv (entropy, EV, PG-loss, V-loss)
  2. plot_monitor_trajectory.png     - per-episode return / length / items delivered
  3. plot_bin_clearing_trajectory.png- mean items delivered per eval checkpoint

Also writes a small companion CSV table_ppo_iterations.csv next to the PPO plot.

Usage (from the repo root):
  python scripts/make_training_plots.py
  python scripts/make_training_plots.py \
      --data_dir rl/models --out_dir results/training/
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD = "RL policy training"
W = 200  # rolling window (episodes) for monitor smoothing


# Plot 1: sb3 progress.csv -> PPO learning curves (entropy / EV / PG / V loss)
def make_ppo_learning_curves(progress_csv: Path, out_dir: Path):
    df = pd.read_csv(progress_csv)
    train = df[df["train/entropy_loss"].notna()].copy()
    print(f"[ppo] train rows: {len(train)}")

    cols_keep = {
        "time/total_timesteps": "total_timesteps",
        "train/entropy_loss": "entropy_loss",
        "train/explained_variance": "explained_variance",
        "train/policy_gradient_loss": "policy_gradient_loss",
        "train/value_loss": "value_loss",
        "train/approx_kl": "approx_kl",
        "train/clip_fraction": "clip_fraction",
        "train/n_updates": "n_updates",
    }
    table = train[list(cols_keep.keys())].rename(columns=cols_keep).reset_index(drop=True)
    table = table.sort_values("total_timesteps").reset_index(drop=True)
    table_path = out_dir / "table_ppo_iterations.csv"
    table.to_csv(table_path, index=False)
    print(f"[ppo] saved table: {table_path}")

    x = table["total_timesteps"].to_numpy()

    def pad_ylim(ax, ymin, ymax, frac=0.08):
        span = ymax - ymin
        if span == 0:
            span = max(abs(ymax), 1.0)
        pad = span * frac
        ax.set_ylim(ymin - pad, ymax + pad)

    fig, axes = plt.subplots(4, 1, figsize=(8.5, 11), sharex=True)

    # Panel 1: entropy_loss
    ax = axes[0]
    y = table["entropy_loss"].to_numpy()
    ax.plot(x, y, marker="o", color="C0", label=METHOD)
    ax.set_ylabel("entropy_loss (nats)")
    ax.set_title("Policy entropy loss (lower magnitude = sharper policy)")
    ax.grid(True, alpha=0.3)
    pad_ylim(ax, y.min(), y.max())
    ax.legend(loc="best", fontsize=9)

    # Panel 2: explained_variance
    ax = axes[1]
    y = table["explained_variance"].to_numpy()
    ax.plot(x, y, marker="o", color="C1", label=METHOD)
    ax.set_ylabel("explained_variance")
    ax.set_title("Value head explained variance (higher = better fit)")
    ax.grid(True, alpha=0.3)
    pad_ylim(ax, y.min(), y.max())
    ax.legend(loc="best", fontsize=9)

    # Panel 3: policy_gradient_loss
    ax = axes[2]
    y = table["policy_gradient_loss"].to_numpy()
    ax.plot(x, y, marker="o", color="C2", label=METHOD)
    ax.set_ylabel("policy_gradient_loss")
    ax.set_title("Policy gradient loss magnitude (non-zero = real learning signal)")
    ax.grid(True, alpha=0.3)
    pad_ylim(ax, y.min(), y.max())
    ax.legend(loc="best", fontsize=9)

    # Panel 4: value_loss
    ax = axes[3]
    y = table["value_loss"].to_numpy()
    ax.plot(x, y, marker="o", color="C3", label=METHOD)
    ax.set_ylabel("value_loss")
    ax.set_title("Value loss")
    ax.set_xlabel("Training step")
    ax.grid(True, alpha=0.3)
    pad_ylim(ax, y.min(), y.max())
    ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        f"{METHOD}, training diagnostics over the full training run",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(top=0.93)

    out_png = out_dir / "plot_ppo_learning_curves.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[ppo] saved plot: {out_png}")
    return out_png


# Plot 2: per-episode monitor trajectory (return / length / items delivered)
EXPECTED_COLS = [
    "r", "l", "t", "n_delivered", "n_invalid", "n_empty_grab",
    "n_physics_attempts", "sum_grasp_quality", "sum_disturb_m",
    "sum_disturb_raw_m", "max_disturb_m", "n_ejected",
    "n_predicate_succ", "n_objects_initial",
]


def make_monitor_trajectory(monitor_csv: Path, out_dir: Path):
    mon = pd.read_csv(monitor_csv, skiprows=1)
    missing = [c for c in EXPECTED_COLS if c not in mon.columns]
    if missing:
        raise RuntimeError(f"Missing expected columns in {monitor_csv}: {missing}")

    n = len(mon)
    idx = mon.index.to_numpy()

    l_min = int(mon["l"].min())
    l_max = int(mon["l"].max())
    n_dips = int((mon["l"] < 60).sum())
    print(f"[monitor] n_episodes={n}")
    print(f"[monitor] episode length, min={l_min}, max={l_max}")
    print(f"[monitor] episodes with l<60: {n_dips}")

    argmin_l = int(mon["l"].idxmin())
    delivered_smooth = mon["n_delivered"].rolling(W, min_periods=1).mean()
    delivered_max = float(delivered_smooth.max())
    argmax_d = int(delivered_smooth.idxmax())

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11), sharex=True)

    # Panel 1: episode return (rolling mean)
    ax = axes[0]
    ax.plot(idx, mon["r"].rolling(W, min_periods=1).mean(),
            color="C0", label=METHOD)
    ax.set_ylabel("episode return")
    ax.set_title(f"Episode return (rolling mean, W={W})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # Panel 2: episode length (raw + min-length star)
    ax = axes[1]
    ax.plot(idx, mon["l"].values, color="C3", label=METHOD)
    ax.scatter([argmin_l], [l_min], marker="*", s=220,
               color="#FFD700", edgecolor="black", linewidth=1.0,
               zorder=5, label=f"earliest clearance: {l_min} steps")
    ax.annotate(
        f"min={l_min}, episode {argmin_l}",
        xy=(argmin_l, l_min),
        xytext=(argmin_l - 0.23 * n, l_min + 8),
        fontsize=9, ha="left", color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )
    ax.set_ylim(35, 62)
    ax.set_ylabel("episode length (steps)")
    ax.set_title("Episode length, lower = faster clear (star = earliest)")
    ax.legend(loc="lower left", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: items delivered (rolling mean only) + max-rolling-mean star
    ax = axes[2]
    ax.plot(idx, delivered_smooth, color="C2", label=METHOD)
    ax.scatter([argmax_d], [delivered_max], marker="*", s=220,
               color="#FFD700", edgecolor="black", linewidth=1.0,
               zorder=5, label=f"peak rolling mean: {delivered_max:.2f} delivered")
    ax.annotate(
        f"max={delivered_max:.2f}, episode {argmax_d}",
        xy=(argmax_d, delivered_max),
        xytext=(argmax_d - 0.23 * n, delivered_max - 0.8),
        fontsize=9, ha="left", color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )
    ax.set_xlim(0, n)
    ax.set_ylabel("items delivered / 20")
    ax.set_xlabel("episode index")
    ax.set_title("Items delivered per episode, higher = better (star = peak)")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{METHOD}, per-episode trajectory across {n:,} training episodes",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(top=0.93)

    out_png = out_dir / "plot_monitor_trajectory.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[monitor] wrote {out_png}")
    return out_png


def make_monitor_trajectory_2panel(monitor_csv: Path, out_dir: Path):
    # Episode length + items delivered only. No episode-return panel.
    mon = pd.read_csv(monitor_csv, skiprows=1)
    missing = [c for c in EXPECTED_COLS if c not in mon.columns]
    if missing:
        raise RuntimeError(f"Missing expected columns in {monitor_csv}: {missing}")

    n = len(mon)
    idx = mon.index.to_numpy()

    l_min = int(mon["l"].min())
    argmin_l = int(mon["l"].idxmin())
    delivered_smooth = mon["n_delivered"].rolling(W, min_periods=1).mean()
    delivered_max = float(delivered_smooth.max())
    argmax_d = int(delivered_smooth.idxmax())

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.0), sharex=True)

    # Panel 1: episode length (raw + earliest-clearance star)
    ax = axes[0]
    ax.plot(idx, mon["l"].values, color="C3", label=METHOD)
    ax.scatter([argmin_l], [l_min], marker="*", s=220,
               color="#FFD700", edgecolor="black", linewidth=1.0,
               zorder=5, label=f"earliest clearance: {l_min} steps")
    ax.annotate(
        f"min={l_min}, episode {argmin_l}",
        xy=(argmin_l, l_min),
        xytext=(argmin_l - 0.23 * n, l_min + 8),
        fontsize=9, ha="left", color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )
    ax.set_ylim(35, 62)
    ax.set_ylabel("episode length (steps)")
    ax.set_title("Episode length, lower = faster clear (star = earliest)")
    ax.legend(loc="lower left", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: items delivered (rolling mean) + peak star
    ax = axes[1]
    ax.plot(idx, delivered_smooth, color="C2", label=METHOD)
    ax.scatter([argmax_d], [delivered_max], marker="*", s=220,
               color="#FFD700", edgecolor="black", linewidth=1.0,
               zorder=5, label=f"peak rolling mean: {delivered_max:.2f} delivered")
    ax.annotate(
        f"max={delivered_max:.2f}, episode {argmax_d}",
        xy=(argmax_d, delivered_max),
        xytext=(argmax_d - 0.23 * n, delivered_max - 0.8),
        fontsize=9, ha="left", color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )
    ax.set_xlim(0, n)
    ax.set_ylabel("items delivered / 20")
    ax.set_xlabel("episode index")
    ax.set_title("Items delivered per episode, higher = better (star = peak)")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{METHOD}, per-episode trajectory across {n:,} training episodes",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.subplots_adjust(top=0.92)

    out_png = out_dir / "plot_monitor_trajectory_2panel.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[monitor-2panel] wrote {out_png}")
    return out_png


# Plot 3: bin clearing trajectory (mean items delivered per eval checkpoint)
def make_bin_clearing_trajectory(eval_monitor_csv: Path, eval_npz: Path,
                                 out_dir: Path, max_delivery: int = 20,
                                 eps_per_checkpoint: int = 30):
    df = pd.read_csv(eval_monitor_csv, skiprows=1)
    npz = np.load(eval_npz)
    x = npz["timesteps"]
    n_checkpoints = int(x.shape[0])

    n_keep = n_checkpoints * eps_per_checkpoint
    df = df.iloc[:n_keep].reset_index(drop=True)

    delivered = df["n_delivered"].to_numpy().reshape(n_checkpoints, eps_per_checkpoint)
    y = delivered.mean(axis=1)

    best_idx = int(y.argmax())
    best_x = float(x[best_idx])
    best_y = float(y[best_idx])

    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    ax.plot(
        x, y,
        color="#1f77b4", linewidth=1.6, marker="o", markersize=4.5,
        markerfacecolor="#1f77b4", markeredgecolor="white", markeredgewidth=0.6,
        label=f"Mean items delivered ({eps_per_checkpoint} eval episodes per checkpoint)",
        zorder=3,
    )
    ax.axhline(
        max_delivery, color="gray", linestyle=":", linewidth=0.9, alpha=0.7,
        label=f"Max delivery (full clearance = {max_delivery})", zorder=1,
    )
    ax.axhline(
        best_y, color="goldenrod", linestyle="--", linewidth=0.9, alpha=0.7,
        label=f"Peak = {best_y:.2f}", zorder=2,
    )
    ax.plot(
        best_x, best_y, marker="*", markersize=18,
        markerfacecolor="gold", markeredgecolor="black", markeredgewidth=0.8,
        linestyle="None",
        label=f"Best mean ({best_y:.2f} @ step {int(best_x):,})", zorder=4,
    )
    ax.annotate(
        f"best mean: {best_y:.2f} / {max_delivery}\nstep {int(best_x):,}",
        xy=(best_x, best_y), xytext=(-90, -38), textcoords="offset points",
        fontsize=9, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="goldenrod", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="goldenrod", lw=0.9),
    )

    ax.set_xlabel("Training step")
    ax.set_ylabel(f"Mean items delivered per episode (max {max_delivery})")
    ax.set_title(
        f"{METHOD}, mean items delivered over training "
        f"(averaged across {eps_per_checkpoint} deterministic eval episodes per checkpoint)",
        pad=10,
    )

    y_lo = min(10.0, float(y.min()) - 0.5)
    y_hi = max(16.0, float(y.max()) + 1.0)
    y_hi = max(y_hi, max_delivery + 0.5)
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.ticklabel_format(axis="x", style="plain")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _pos: f"{int(v):,}"))

    fig.tight_layout()
    out_png = out_dir / "plot_bin_clearing_trajectory.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[clearing] saved: {out_png}")
    print(f"[clearing]   first eval = {y[0]:.2f} delivered @ step {int(x[0]):,}")
    print(f"[clearing]   last eval  = {y[-1]:.2f} delivered @ step {int(x[-1]):,}")
    print(f"[clearing]   peak       = {best_y:.2f} delivered @ step {int(best_x):,}")
    return out_png


def main():
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default=str(repo_root / "rl" / "models"),
                    help="Directory containing training_progress.csv, "
                         "training_monitor.csv, training_eval_monitor.csv, "
                         "training_evaluations.npz")
    ap.add_argument("--out_dir", default=str(repo_root / "results" / "training"),
                    help="Directory to write the training plots into")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    progress_csv = data_dir / "training_progress.csv"
    monitor_csv = data_dir / "training_monitor.csv"
    eval_monitor_csv = data_dir / "training_eval_monitor.csv"
    eval_npz = data_dir / "training_evaluations.npz"

    for path in (progress_csv, monitor_csv, eval_monitor_csv, eval_npz):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    make_ppo_learning_curves(progress_csv, out_dir)
    make_monitor_trajectory(monitor_csv, out_dir)
    make_monitor_trajectory_2panel(monitor_csv, out_dir)
    make_bin_clearing_trajectory(eval_monitor_csv, eval_npz, out_dir)

    print()
    print(f"Done. Plots written to {out_dir}")


if __name__ == "__main__":
    main()
