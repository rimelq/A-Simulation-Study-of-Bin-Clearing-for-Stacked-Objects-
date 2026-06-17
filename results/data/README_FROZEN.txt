FROZEN ARCHIVE — FINAL 5-POLICY EVAL RESULTS
============================================
Date frozen: 2026-06-07
Source job : SLURM 2967846
Wall time  : 10h 19m (COMPLETED, exit 0)
Config     : layout_jitter=0.02, 5 policies, n_objects in {10,15,20},
             60 paired episodes per cell, seed_offset=1000
RL model   : rl/runs/rl_v5_l1_l4_action_augmented_n20/eval/best/best_model.zip
             (step 389,844, eval mean_reward 76.93)

KEY RESULTS (items delivered, mean ± std):
  At n=20: RL=13.68±2.28, random=12.62±2.31, topdown=11.78±3.09,
           heuristic_augmented=10.98±2.79, greedy_ggcnn=3.73±1.57.

PAIRWISE AT n=20 (paired t-test, 60 seeds):
  RL vs random:                +1.07 items (p=0.019, Cohen-d=0.31)
  RL vs heuristic_augmented:   +2.70 items (p=5e-8, Cohen-d=0.81)
  random vs greedy_ggcnn:      +8.88 items (p=4e-30, Cohen-d=2.87)

THIS IS THE FINAL DATASET for the report.
The earlier layout_jitter=0.0 run is at 4method_layoutjitter0_FROZEN/.

DO NOT MODIFY THIS DIRECTORY.
