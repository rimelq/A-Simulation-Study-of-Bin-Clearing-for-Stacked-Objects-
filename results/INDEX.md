# v5 Final Eval — Report Plot Catalog

**Source data**: `4method_jitter002_final_FROZEN/per_episode.csv` (900 rows = 60 paired episodes × 5 policies × 3 n_objects).

**Eval provenance**: SLURM 2967846, layout_jitter=0.02, hybrid_physics, seed_offset=1000, completed 10h 19m on 2026-06-07.

**RL model**: `rl_v5_l1_l4_action_augmented_n20/eval/best/best_model.zip` (step 389,844, training eval reward 76.93).

## Headline numbers (mean ± std items delivered, n=60 paired)

| Policy | n=10 / 10 | n=15 / 15 | n=20 / 20 |
|---|---|---|---|
| greedy_ggcnn | 2.08 ± 1.10 | 2.87 ± 1.37 | 3.73 ± 1.57 |
| topdown | 6.10 ± 1.20 | 9.35 ± 1.78 | 11.78 ± 3.09 |
| heuristic_augmented | 5.62 ± 1.78 | 9.15 ± 2.14 | 10.98 ± 2.79 |
| random | **7.87 ± 1.50** | **11.10 ± 1.85** | 12.62 ± 2.31 |
| RL_action_augmented | 7.13 ± 1.23 | 10.95 ± 1.45 | **13.68 ± 2.28** |

## Critical pairwise tests (paired by seed, n=60)

| Pair @ n_objects | Δ items | Cohen-d | p (t-test) | Verdict |
|---|---|---|---|---|
| RL vs random @ n=20 | **+1.07** | +0.31 | **0.019** | RL beats random (p < 0.05) |
| RL vs heuristic_augmented @ n=20 | **+2.70** | +0.81 | **5e-8** | RL beats heuristic decisively |
| random vs greedy_ggcnn @ n=20 | +8.88 | +2.87 | 4e-30 | PPO ghost crushes GG-CNN |
| RL vs random @ n=10 | −0.73 | −0.44 | 0.001 | Random beats RL (low clutter) |
| RL vs random @ n=15 | −0.15 | −0.06 | 0.64 | Tie |

## Plot catalog

| Filename | Type | What it shows | Report section |
|---|---|---|---|
| `plot_headline_items_delivered.png` | Grouped bar chart with std error bars. 5 policies × 3 clutter levels. | **The single most important figure**: shows clear ordering and how RL pulls ahead at n=20. | Headline figure (Results) |
| `plot_items_delivered_fraction.png` | Same data as above, normalized as fraction (items / n_objects). | Lets the reader compare clutter levels on the same scale. | Results — normalized view |
| `plot_pairwise_rl_vs_others.png` | Three-panel horizontal bar chart: RL minus each competitor at each n_objects, with 95% CI and significance stars (`***`, `**`, `*`, `n.s.`). | **The headline statistical figure**: shows RL beats heuristic_augmented at every n; RL beats random only at n=20; RL crushes greedy at all levels. | Results — pairwise significance |
| `plot_distribution_boxplots.png` | Three-panel boxplots (one per n_objects), 5 policies each, showing distribution of items_delivered per episode. Median, IQR, whiskers, outliers, mean diamonds. | Reveals tail behavior and variance — RL's distribution is tighter than topdown's. | Results — variance and tail analysis |
| `plot_cleared_fraction.png` | Grouped bar chart of cleared_fraction (binary). Annotates non-zero cells. | Shows that only random clears any bins (13.3% at n=10); RL clears 1.7% at n=10. **Cleared is NOT the discriminating metric.** | Results — cleared metric (with caveat) |
| `plot_disturbance.png` | Grouped bar chart of mean disturbance per episode (metres). | Shows greedy/random cause the most pile damage; RL/topdown/heuristic are conservative. RL's lower-than-random disturbance evidences the keystone-avoidance behaviour. | Results — secondary metric |
| `plot_effect_size_heatmap.png` | Three-panel matrix heatmap of Cohen's d between every policy pair at each n_objects (5×5 each). Red = row better than column; blue = column better than row. | **Single-glance summary** of all 30 pairwise comparisons. RL row is most red at n=20; random row most red at n=10/15. Tells the "story shifts with clutter" narrative immediately. | Results — comprehensive effect-size view |
| `plot_5policy_summary.png` | Three-panel composite: items delivered, fraction delivered, cleared fraction — all stacked. | One-figure overview, useful as a back-up or supplementary slide. | Supplementary |

## Existing analyzer outputs (also in `4method_jitter002_final_FROZEN/analysis/`)

The analyzer (`analyze_4method.py`) also produced these — most are redundant with the curated plots above, but they are preserved with the frozen archive:

| Filename | Notes |
|---|---|
| `plot_items_delivered.png` | Analyzer's basic version of the headline figure |
| `plot_cleared_fraction.png` | Analyzer's basic version |
| `plot_paired_diff_ci.png` | Analyzer's paired-diff plot vs greedy reference (less informative than RL-centred pairwise) |
| `plot_disturbance_vs_clearing.png` | Scatter of disturbance vs cleared fraction |
| `plot_failure_predicate.png` | Per-policy breakdown of which predicate sub-check fails most |
| `plot_per_step_outcome_breakdown.png` | Per-step outcome categories |
| `plot_floor_sanity.png` | Disturbance distribution KS test |

## Tables (raw CSV) preserved in frozen archive

- `table_items_delivered.csv` — mean/std per cell
- `table_items_delivered_paired.csv` — full paired statistics (Cohen-d, Wilcoxon, t-test, Bonferroni, Holm) vs greedy reference
- `table_cleared_fraction.csv`
- `table_disturbance.csv` (post-floor) and `table_disturbance_raw.csv` (raw)
- `table_failure_mode_breakdown.csv`
- `table_floor_sanity.csv` — KS test on raw disturbance distributions
- `table_paired_diff.csv` — analyzer's paired-diff vs greedy on cleared_fraction
- `stats_summary.txt` — human-readable summary

## The story (one paragraph)

The headline finding is that **at high clutter (n=20), RL_action_augmented delivers 13.68 items per episode, beating the next-best policy (random, 12.62) by +1.07 items (p=0.019) and the hand-crafted L4 heuristic_augmented by +2.70 items (p=5×10⁻⁸)**. At low/medium clutter (n=10/15), the picture is different: the information-blind `random` baseline leads because deterministic policies fixate via the shallow 3-deep failure mask, while random's uniform sampling side-steps fixation by construction (this confirms the scope §6.5 selection-saturation prediction). The v5 L1+L4 redesign delivers exactly where it was designed to: in the high-clutter regime where structural features (`exposure_value`, `is_keystone`) and learned XY/yaw refinement carry real signal. Across all clutter levels, every PPO-ghost-based policy beats GG-CNN end-to-end by 3–4×, confirming perception is the dominant bottleneck.

## Frozen archives

- `4method_layoutjitter0_FROZEN/` — the original layout_jitter=0.0 confounded run (SLURM 2967568). Diagnostic value only.
- `4method_jitter002_final_FROZEN/` — **the source of every number in this report**.
