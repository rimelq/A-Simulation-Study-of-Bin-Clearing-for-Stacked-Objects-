"""
Live training-curve plotting that survives, resume.

The default SB3 Monitor opens its CSV in write mode, so every time the env is
rebuilt (i.e. every resume) the previous run's per-episode lines get truncated.
This module fixes that by:

  1. Reading **every** ``monitor.monitor.csv*`` file in a run dir (so resumes
     that rename the prior CSV to ``monitor.monitor.csv.prev<N>`` keep their raw
     per-episode data alongside the current run's).
  2. **Optionally** also reading a ``history.json`` sidecar, a list of
     pre-existing run *aggregates* recovered from TensorBoard event files
     (per-rollout ep_rew_mean / ep_len_mean / total_timesteps).
     Use this when an old monitor.csv was already truncated and the only
     surviving source for that run is its TB event file.
  3. Stitching all of those into one continuous plot, x-axis = cumulative
     episode index across runs, with the rolling mean tracked across the
     run boundaries.

Public API:
  recover_history_from_tfevents(run_dir, out_json) -> int  # rollouts written
  read_monitor_csv(path) -> list[dict]
  read_history_json(path) -> list[dict]
  make_combined_curves(run_dir, out_path, title=None) -> str
"""
from __future__ import annotations

import os
import csv as _csv
import glob
import json
import math
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Data readers

def read_monitor_csv(path: str) -> list[dict]:
    """Read one SB3 Monitor / VecMonitor CSV and return its rows as a list of
    dicts with keys r, l, t, n_delivered (n_delivered is NaN if absent)."""
    rows = []
    try:
        with open(path, "r") as f:
            lines = [ln for ln in f if not ln.startswith("#")]
        rd = _csv.DictReader(lines)
        for row in rd:
            try:
                r = float(row["r"]); l = int(float(row["l"]))
            except Exception:
                continue
            try:
                t = float(row["t"])
            except Exception:
                t = float("nan")
            try:
                nd = float(row.get("n_delivered", "nan"))
            except Exception:
                nd = float("nan")
            rows.append({"r": r, "l": l, "t": t, "n_delivered": nd})
    except FileNotFoundError:
        pass
    return rows


def recover_history_from_tfevents(run_dir: str, out_json: str = None,
                                  exclude_paths: list = None) -> int:
    """Parse every ``events.out.tfevents.*`` file in ``run_dir`` (except those in
    ``exclude_paths``) and write a JSON list of per-rollout aggregate points to
    ``out_json`` (default: ``<run_dir>/history.json``). Use this once after a
    run got truncated to preserve its rolling-mean curves.

    JSON schema (a list of run-segments, in mtime order)::
      [{"source": "tfevents", "tfevents_path": "...", "steps": [...],
        "ep_rew_mean": [...], "ep_len_mean": [...]}, ...]
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    exclude_paths = set(os.path.realpath(p) for p in (exclude_paths or []))
    tf_paths = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")),
                      key=lambda p: os.path.getmtime(p))
    tf_paths = [p for p in tf_paths if os.path.realpath(p) not in exclude_paths]
    if not tf_paths:
        return 0
    out_json = out_json or os.path.join(run_dir, "history.json")
    history = []
    n_total = 0
    for tp in tf_paths:
        try:
            ea = EventAccumulator(tp, size_guidance={"scalars": 0})
            ea.Reload()
            tags = set(ea.Tags().get("scalars", []))
            if "rollout/ep_rew_mean" not in tags:
                continue
            rew = ea.Scalars("rollout/ep_rew_mean")
            length = ea.Scalars("rollout/ep_len_mean") if "rollout/ep_len_mean" in tags else []
            len_by_step = {p.step: p.value for p in length}
            seg = {
                "source": "tfevents",
                "tfevents_path": tp,
                "steps":         [p.step for p in rew],
                "ep_rew_mean":   [p.value for p in rew],
                "ep_len_mean":   [len_by_step.get(p.step, float("nan")) for p in rew],
            }
            history.append(seg)
            n_total += len(rew)
        except Exception as e:
            print(f"[curves] failed to read {tp}: {e}")
    with open(out_json, "w") as f:
        json.dump(history, f, indent=1)
    return n_total


def read_history_json(path: str) -> list[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


# Plotting

def _rolling_mean(arr, w):
    if len(arr) < 2:
        return np.asarray(arr, dtype=float)
    w = max(1, min(int(w), len(arr)))
    return np.convolve(np.asarray(arr, dtype=float), np.ones(w) / w, mode="valid")


def _sorted_monitor_csvs(run_dir: str) -> list[str]:
    """Return [monitor.monitor.csv.prev0, .prev1, ..., monitor.monitor.csv] in order.
    Older renamed-on-resume files first. the live (un-suffixed) one last."""
    paths = glob.glob(os.path.join(run_dir, "monitor.monitor.csv*"))
    def _key(p: str):
        base = os.path.basename(p)
        m = re.match(r"^monitor\.monitor\.csv\.prev(\d+)$", base)
        if m:
            return (0, int(m.group(1)))
        if base == "monitor.monitor.csv":
            return (1, 0)
        return (2, base)   # any other extras go last
    return sorted(paths, key=_key)


def make_combined_curves(run_dir: str, out_path: str = None,
                         title: str = None,
                         history_json: str = None) -> str:
    """
    Plot a 3-panel curves.png for ``run_dir`` combining:
      * a ``history.json`` sidecar (recovered TB aggregates for old runs whose
        monitor.csv was lost)
      * every ``monitor.monitor.csv*`` in the run dir, in resume order

    The plot's x-axis is cumulative-episode-index across runs. the boundary
    between runs is marked with a dashed vertical line.

    Returns the output path it wrote.
    """
    out_path = out_path or os.path.join(run_dir, "curves.png")
    history_json = history_json or os.path.join(run_dir, "history.json")
    history = read_history_json(history_json)
    csvs = _sorted_monitor_csvs(run_dir)

    # Build a unified list of "segments". Each segment exposes:
    # x = numpy array of cumulative episode indices (in the combined timeline)
    # r_raw = per-episode rewards (None for aggregate-only segments)
    # r_curve = aggregate / rolling-mean reward curve (always)
    # l_curve, d_curve = same for length and n_delivered (None where unavailable)
    # label = "Run #i [history]" / "Run #i (live)" / ...
    segments = []
    cum_ep = 0

    for seg in history:
        steps = np.asarray(seg.get("steps", []), dtype=float)
        rew   = np.asarray(seg.get("ep_rew_mean", []), dtype=float)
        ln    = np.asarray(seg.get("ep_len_mean", []), dtype=float)
        if rew.size == 0:
            continue
        # estimate cumulative episode index at each rollout: total_steps / mean_ep_len
        mean_ln = np.where(np.isfinite(ln) & (ln > 0), ln, np.nanmean(ln[np.isfinite(ln)]) or 20.0)
        est_eps = steps / np.where(mean_ln > 0, mean_ln, 1.0)
        n_eps_in_segment = float(est_eps[-1] - est_eps[0]) if est_eps.size >= 2 else 1.0
        x = cum_ep + (est_eps - est_eps[0])
        segments.append(dict(
            label=f"history (TB events: {os.path.basename(seg.get('tfevents_path','?'))})",
            kind="aggregate",
            x=x, r_raw=None, r_curve=rew, l_curve=ln, d_curve=None,
            n_eps=int(round(n_eps_in_segment)),
        ))
        cum_ep += n_eps_in_segment

    for i, csv_path in enumerate(csvs):
        rows = read_monitor_csv(csv_path)
        if not rows:
            continue
        r = np.array([row["r"] for row in rows], dtype=float)
        l = np.array([row["l"] for row in rows], dtype=float)
        d = np.array([row["n_delivered"] for row in rows], dtype=float)
        x = cum_ep + np.arange(1, len(r) + 1)
        # rolling mean (window ~5% of data, min 10, max 200)
        w = int(np.clip(len(r) // 20, 10, 200))
        r_curve = _rolling_mean(r, w)
        x_curve = x[w - 1:] if len(r) >= 2 else x
        segments.append(dict(
            label=f"{os.path.basename(csv_path)} (n={len(rows)} episodes)",
            kind="per_episode",
            x=x, r_raw=r, r_curve=r_curve, r_curve_x=x_curve,
            l_curve=l, d_curve=d, n_eps=len(rows),
        ))
        cum_ep += len(rows)

    if not segments:
        # nothing to plot, write an empty placeholder
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "no data yet", ha="center", va="center")
        fig.savefig(out_path, dpi=110); plt.close(fig)
        return out_path

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3), sharex=True)
    # color cycle so prior segments are visually distinct from the live one
    cmap = plt.get_cmap("tab10")
    boundary_eps = []
    running_x = 0

    for idx, seg in enumerate(segments):
        c = cmap(idx % 10)
        # panel 0: episode return
        if seg["kind"] == "per_episode" and seg["r_raw"] is not None:
            ax[0].plot(seg["x"], seg["r_raw"], ".", ms=2.5, alpha=0.25, color=c)
            ax[0].plot(seg["r_curve_x"], seg["r_curve"], "-", color=c, lw=2,
                       label=seg["label"])
        else:
            ax[0].plot(seg["x"], seg["r_curve"], "-", color=c, lw=2,
                       label=seg["label"])
        # panel 1: episode length
        ax[1].plot(seg["x"], seg["l_curve"], "-", color=c, lw=1.5,
                   label=seg["label"])
        # panel 2: items delivered / episode
        if seg.get("d_curve") is not None and np.any(np.isfinite(seg["d_curve"])):
            d = seg["d_curve"]
            ax[2].plot(seg["x"], d, ".", ms=2.5, alpha=0.25, color=c)
            if seg["kind"] == "per_episode" and len(d) >= 2:
                w = int(np.clip(len(d) // 20, 10, 200))
                ax[2].plot(seg["x"][w - 1:], _rolling_mean(d, w), "-", color=c, lw=2)
        running_x += seg["n_eps"]
        boundary_eps.append(running_x)

    # vertical dashed lines at run boundaries (except the very last)
    for x in boundary_eps[:-1]:
        for a in ax:
            a.axvline(x, color="gray", ls="--", lw=0.8, alpha=0.6)

    ax[0].set_xlabel("episode (cumulative across runs)"); ax[0].set_ylabel("episode return"); ax[0].set_title("episode return"); ax[0].grid(alpha=0.3); ax[0].legend(loc="best", fontsize=7)
    ax[1].set_xlabel("episode (cumulative across runs)"); ax[1].set_ylabel("steps"); ax[1].set_title("episode length"); ax[1].grid(alpha=0.3)
    ax[2].set_xlabel("episode (cumulative across runs)"); ax[2].set_ylabel("items delivered / episode"); ax[2].set_title("bin clearing"); ax[2].grid(alpha=0.3)
    if title is None:
        title = f"{os.path.basename(os.path.abspath(run_dir))}, cumulative {boundary_eps[-1]} episodes across {len(segments)} run(s)"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Make a combined curves.png for one run dir.")
    ap.add_argument("--run", required=True, help="path to rl/runs/<run_name>")
    ap.add_argument("--out", default=None, help="output PNG (default: <run>/curves.png)")
    ap.add_argument("--recover_history", action="store_true",
                    help="First parse any TB event files in --run into history.json")
    ap.add_argument("--history_json", default=None, help="path to history.json (default: <run>/history.json)")
    args = ap.parse_args()

    if args.recover_history:
        n = recover_history_from_tfevents(args.run, out_json=args.history_json)
        print(f"[curves] recovered {n} rollout points to {args.history_json or os.path.join(args.run, 'history.json')}")
    out = make_combined_curves(args.run, out_path=args.out, history_json=args.history_json)
    print(f"[curves] wrote {out}")
