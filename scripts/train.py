"""
Train a MaskablePPO policy on BinClearingGymEnv.

See README.md for example commands and the exact flags.
"""
import os
import sys
import json
import time
import argparse
import datetime

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl.bin_clearing_env import BinClearingGymEnv


# info-dict keys logged by (Vec)Monitor as extra columns per episode.
_MONITOR_INFO_KEYS = (
    "n_delivered",
    "n_invalid",
    "n_empty_grab",
    "n_physics_attempts",
    "sum_grasp_quality",
    "sum_disturb_m",
    "sum_disturb_raw_m",
    "max_disturb_m",
    "n_ejected",
    "n_predicate_succ",
    "n_objects_initial",
)


def _raw_env_thunk(n_objects, seed, reward_mode="hybrid",
                   reward_weights=None, max_steps_per_object=3.0,
                   layout_jitter=0.02, ggcnn_device=None,
                   orientation_source="snap", filter_candidates=True,
                   candidate_source="ggcnn", ppo_visibility_mode="raycast",
                   ppo_quality_mode="analytical", failure_mask_size=3, K=10):
    def _thunk():
        e = BinClearingGymEnv(
            n_objects=n_objects, K=K,
            reward_mode=reward_mode, render_mode=None,
            max_steps_per_object=max_steps_per_object,
            layout_jitter=layout_jitter,
            reward_weights=reward_weights,
            ggcnn_device=ggcnn_device,
            orientation_source=orientation_source,
            filter_candidates=bool(filter_candidates),
            candidate_source=candidate_source,
            ppo_visibility_mode=ppo_visibility_mode,
            ppo_quality_mode=ppo_quality_mode,
            failure_mask_size=failure_mask_size,
        )
        if seed is not None:
            e.reset(seed=int(seed))
        return e
    return _thunk


def make_train_env(n_objects, base_seed, n_envs, monitor_path, reward_mode="hybrid",
                   reward_weights=None, max_steps_per_object=3.0,
                   layout_jitter=0.02, ggcnn_device=None,
                   orientation_source="snap", filter_candidates=True,
                   candidate_source="ggcnn", ppo_visibility_mode="raycast",
                   ppo_quality_mode="analytical", failure_mask_size=3, K=10):
    """Build the (vectorized) Monitor-wrapped training env."""
    env_kwargs = dict(reward_mode=reward_mode, reward_weights=reward_weights,
                      max_steps_per_object=max_steps_per_object,
                      layout_jitter=layout_jitter, ggcnn_device=ggcnn_device,
                      orientation_source=orientation_source,
                      filter_candidates=filter_candidates,
                      candidate_source=candidate_source,
                      ppo_visibility_mode=ppo_visibility_mode,
                      ppo_quality_mode=ppo_quality_mode,
                      failure_mask_size=failure_mask_size,
                      K=K)
    if n_envs <= 1:
        from stable_baselines3.common.monitor import Monitor
        env = Monitor(_raw_env_thunk(n_objects, base_seed, **env_kwargs)(),
                      filename=monitor_path, info_keywords=_MONITOR_INFO_KEYS)
        return env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    thunks = [_raw_env_thunk(n_objects, base_seed + i, **env_kwargs) for i in range(n_envs)]
    venv = SubprocVecEnv(thunks, start_method="spawn")
    venv = VecMonitor(venv, filename=monitor_path, info_keywords=_MONITOR_INFO_KEYS)
    return venv


def make_eval_env(n_objects, seed, monitor_path, reward_mode="hybrid",
                  reward_weights=None, max_steps_per_object=3.0,
                  layout_jitter=0.02, ggcnn_device=None,
                  orientation_source="snap", filter_candidates=True,
                  candidate_source="ggcnn", ppo_visibility_mode="raycast",
                  ppo_quality_mode="analytical", failure_mask_size=3, K=10):
    """Single Monitor-wrapped env for EvalCallback."""
    from stable_baselines3.common.monitor import Monitor
    return Monitor(_raw_env_thunk(n_objects, seed,
                                  reward_mode=reward_mode,
                                  reward_weights=reward_weights,
                                  max_steps_per_object=max_steps_per_object,
                                  layout_jitter=layout_jitter,
                                  ggcnn_device=ggcnn_device,
                                  orientation_source=orientation_source,
                                  filter_candidates=filter_candidates,
                                  candidate_source=candidate_source,
                                  ppo_visibility_mode=ppo_visibility_mode,
                                  ppo_quality_mode=ppo_quality_mode,
                                  failure_mask_size=failure_mask_size,
                                  K=K)(),
                   filename=monitor_path, info_keywords=_MONITOR_INFO_KEYS)


def _make_plot_callback(run_dir, csv_path, every_steps):
    """Live training-curve callback. Stitches prior monitor.csv.prev<N> +
    live monitor.csv into a single cross-resume curves.png."""
    from stable_baselines3.common.callbacks import BaseCallback
    from rl.curves import make_combined_curves

    class _PlotCurves(BaseCallback):
        def __init__(self):
            super().__init__()
            self._last = 0

        def _plot(self):
            try:
                n_ts = getattr(self.model, "num_timesteps", None)
                title = f"{os.path.basename(run_dir)}" + (f"  -  {n_ts} steps" if n_ts else "")
                make_combined_curves(run_dir, title=title)
            except Exception as e:
                print(f"[train] curves.png plot failed: {e}")

        def _on_step(self):
            if self.num_timesteps - self._last >= every_steps:
                self._last = self.num_timesteps
                self._plot()
            return True

        def _on_training_end(self):
            self._plot()

    return _PlotCurves()


def _make_save_last_callback(ckpt_dir, every_steps):
    """Keep checkpoints/last.zip current so SLURM wall-clock kills stay resumable."""
    from stable_baselines3.common.callbacks import BaseCallback

    class _SaveLastZip(BaseCallback):
        def __init__(self):
            super().__init__()
            self._last = 0

        def _save(self):
            try:
                self.model.save(os.path.join(ckpt_dir, "last"))
            except Exception as e:
                print(f"[train] failed to refresh last.zip: {e}")

        def _on_step(self):
            if self.num_timesteps - self._last >= every_steps:
                self._last = self.num_timesteps
                self._save()
            return True

        def _on_training_end(self):
            self._save()

    return _SaveLastZip()


def archive_previous_run_artifacts(run_dir: str) -> int:
    """Rename prior monitor.csv / progress.csv to .prev<N> so the new Monitor
    doesn't truncate them. Returns the N used."""
    n = 0
    while os.path.exists(os.path.join(run_dir, f"monitor.monitor.csv.prev{n}")) or\
          os.path.exists(os.path.join(run_dir, f"progress.csv.prev{n}")):
        n += 1
    for src, ext in (("monitor.monitor.csv", f".prev{n}"),
                     ("progress.csv",        f".prev{n}")):
        p = os.path.join(run_dir, src)
        if os.path.exists(p):
            try:
                os.rename(p, p + ext)
                print(f"[train] preserved {src} -> {src}{ext}")
            except Exception as e:
                print(f"[train] could not preserve {src}: {e}")
    return n


def main():
    ap = argparse.ArgumentParser(
        description="RL training on BinClearingGymEnv (candidate selection)")
    ap.add_argument("--algo", choices=["ppo", "dqn", "maskable_ppo"], default="maskable_ppo")
    ap.add_argument("--reward_mode",
                    choices=["hybrid", "hybrid_physics", "physics", "geometric"],
                    default="hybrid")
    ap.add_argument("--total_timesteps", type=int, default=1_000_000)
    ap.add_argument("--n_objects", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_envs", type=int, default=1)
    ap.add_argument("--run_name", type=str, default=None,
                    help="subdir under rl/runs/ (default: <algo>_<timestamp>)")
    ap.add_argument("--runs_root", type=str, default=os.path.join(_ROOT, "rl", "runs"))
    ap.add_argument("--resume", type=str, default=None,
                    help="path to a saved .zip model to continue training from")
    ap.add_argument("--checkpoint_freq", type=int, default=50_000)
    ap.add_argument("--eval_freq", type=int, default=50_000)
    ap.add_argument("--eval_episodes", type=int, default=10)
    ap.add_argument("--eval_seed", type=int, default=12345)
    ap.add_argument("--curves_freq", type=int, default=500)
    ap.add_argument("--learning_rate", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    # ent_coef lowered from 0.01 so a 10-action softmax actually concentrates
    # (prior runs stayed at ~96% of ln(10) entropy for 845k steps).
    ap.add_argument("--ent_coef", type=float, default=0.003)
    ap.add_argument("--n_steps", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--n_epochs", type=int, default=10)
    ap.add_argument("--buffer_size", type=int, default=100_000)
    ap.add_argument("--exploration_fraction", type=float, default=0.3)
    ap.add_argument("--policy_kwargs", type=str, default='{"net_arch":[256,128]}')
    ap.add_argument("--max_steps_per_object", type=float, default=3.0,
                    help="episode step cap = round(max_steps_per_object * n_objects)")
    ap.add_argument("--max_steps", type=int, default=None,
                    help="absolute episode step cap (overrides --max_steps_per_object)")
    ap.add_argument("--layout_jitter", type=float, default=0.02)
    ap.add_argument("--ggcnn_device", type=str, default=None)
    ap.add_argument("--orientation_source", choices=["snap", "raw"], default="snap")
    ap.add_argument("--filter_candidates", type=int, default=1, choices=[0, 1])
    ap.add_argument("--candidate_source", choices=["ggcnn", "cc", "ppo"], default="ggcnn")
    ap.add_argument("--ppo_visibility_mode", choices=["raycast", "omniscient"], default="raycast")
    ap.add_argument("--ppo_quality_mode", choices=["analytical", "uniform"], default="analytical")
    ap.add_argument("--failure_mask_size", type=int, default=3)
    ap.add_argument("--K", type=int, default=10,
                    help="candidate slots; action-space = Discrete(K*27) for the v5 joint cand+refinement space")
    ap.add_argument("--eval_n_episodes", type=int, default=None,
                    help="alias for --eval_episodes used by the 4-method slurm")
    # Retuned reward weights (v2): see _DEFAULT_REWARD_WEIGHTS in rl/bin_clearing_env.py.
    ap.add_argument("--step_penalty", type=float, default=-0.02)
    ap.add_argument("--deliver_reward", type=float, default=3.0)
    ap.add_argument("--grasp_quality_coef", type=float, default=0.30)
    ap.add_argument("--empty_grab_penalty", type=float, default=-0.50)
    ap.add_argument("--invalid_penalty", type=float, default=-1.0,
                    help="unused under maskable_ppo (action masking eliminates invalid slots)")
    ap.add_argument("--disturb_coef", type=float, default=1.0)
    ap.add_argument("--eject_penalty", type=float, default=1.0)
    args = ap.parse_args()

    if getattr(args, "eval_n_episodes", None) is not None:
        args.eval_episodes = int(args.eval_n_episodes)

    print(f"[train.py] candidate_source={args.candidate_source} "
          f"ppo_visibility={args.ppo_visibility_mode} "
          f"ppo_quality={args.ppo_quality_mode} "
          f"failure_mask={args.failure_mask_size} K={args.K}")

    try:
        import torch
        torch.manual_seed(args.seed)
    except Exception:
        pass
    np.random.seed(args.seed)
    try:
        from stable_baselines3.common.utils import set_random_seed
        set_random_seed(args.seed)
    except Exception:
        pass

    run_name = args.run_name or f"{args.algo}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    run_dir   = os.path.join(args.runs_root, run_name)
    ckpt_dir  = os.path.join(run_dir, "checkpoints")
    mon_path  = os.path.join(run_dir, "monitor")
    eval_dir  = os.path.join(run_dir, "eval")
    for d in (run_dir, ckpt_dir, eval_dir):
        os.makedirs(d, exist_ok=True)

    if args.resume:
        prev_n = archive_previous_run_artifacts(run_dir)
        old_cfg = os.path.join(run_dir, "config.json")
        if os.path.exists(old_cfg):
            try:
                os.rename(old_cfg, os.path.join(run_dir, f"config.json.prev{prev_n}"))
                print(f"[train] preserved config.json -> config.json.prev{prev_n}")
            except Exception as e:
                print(f"[train] could not preserve config.json: {e}")

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "run_name": run_name,
                   "started": datetime.datetime.now().isoformat()}, f, indent=2)
    print(f"[train] run dir : {run_dir}")
    print(f"[train] config  : {vars(args)}")

    n_envs = max(1, int(args.n_envs))
    reward_weights = dict(
        step_penalty=args.step_penalty,
        deliver_reward=args.deliver_reward,
        grasp_quality_coef=args.grasp_quality_coef,
        empty_grab_penalty=args.empty_grab_penalty,
        invalid_penalty=args.invalid_penalty,
        disturb_coef=args.disturb_coef,
        eject_penalty=args.eject_penalty,
    )
    env_kwargs = dict(reward_mode=args.reward_mode, reward_weights=reward_weights,
                      max_steps_per_object=args.max_steps_per_object,
                      layout_jitter=args.layout_jitter,
                      ggcnn_device=args.ggcnn_device,
                      orientation_source=args.orientation_source,
                      filter_candidates=bool(int(args.filter_candidates)),
                      candidate_source=args.candidate_source,
                      ppo_visibility_mode=args.ppo_visibility_mode,
                      ppo_quality_mode=args.ppo_quality_mode,
                      failure_mask_size=args.failure_mask_size,
                      K=args.K)
    print(f"[train] building train env (reward_mode={args.reward_mode}, "
          f"n_objects={args.n_objects}, seed={args.seed}, "
          f"n_envs={n_envs}{' [SubprocVecEnv]' if n_envs > 1 else ''}) ...")
    print(f"[train] reward weights : {reward_weights}")
    print(f"[train] env params     : max_steps_per_object={args.max_steps_per_object} "
          f"layout_jitter={args.layout_jitter} ggcnn_device={args.ggcnn_device or 'auto'}")
    env = make_train_env(args.n_objects, args.seed, n_envs, monitor_path=mon_path,
                         **env_kwargs)

    try:
        _act_n = int(getattr(env.action_space, "n", -1))
        _expected_v5 = int(args.K) * 27
        _shape_str = (f"Discrete({_act_n}) "
                      f"[v5 expected Discrete({_expected_v5})="
                      f"{'OK' if _act_n == _expected_v5 else 'MISMATCH'}]")
        print(f"[train] action_space     : {_shape_str}")
        print(f"[train] observation_space: {env.observation_space}")
    except Exception as _e:
        print(f"[train] (could not introspect action_space: {_e})")

    eval_env = None
    if args.eval_freq and args.eval_episodes:
        print(f"[train] building eval env  (seed={args.eval_seed}) ...")
        eval_env = make_eval_env(args.n_objects, args.eval_seed,
                                 monitor_path=os.path.join(eval_dir, "monitor"),
                                 **env_kwargs)

    # SB3 *_freq counts _on_step calls. with a VecEnv each advances n_envs env-steps.
    ck_freq   = max(1, args.checkpoint_freq // n_envs)
    ev_freq   = max(1, args.eval_freq // n_envs) if args.eval_freq else 0

    from stable_baselines3.common.callbacks import CheckpointCallback
    callbacks = [
        CheckpointCallback(save_freq=ck_freq, save_path=ckpt_dir,
                           name_prefix="ckpt", save_replay_buffer=(args.algo == "dqn")),
        _make_save_last_callback(ckpt_dir, every_steps=args.checkpoint_freq),
    ]
    if eval_env is not None:
        # MaskablePPO needs MaskableEvalCallback so predict gets action_masks.
        if args.algo == "maskable_ppo":
            from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback as _EvalCB
        else:
            from stable_baselines3.common.callbacks import EvalCallback as _EvalCB
        callbacks.append(_EvalCB(
            eval_env, best_model_save_path=os.path.join(eval_dir, "best"),
            log_path=eval_dir, eval_freq=ev_freq,
            n_eval_episodes=args.eval_episodes, deterministic=True))
    if args.curves_freq and args.curves_freq > 0:
        callbacks.append(_make_plot_callback(run_dir, mon_path + ".monitor.csv",
                                             args.curves_freq))

    try:
        policy_kwargs = json.loads(args.policy_kwargs)
    except Exception:
        policy_kwargs = None

    try:
        import tensorboard  # noqa: F401
        _have_tb = True
        print(f"[train] tensorboard logging -> {run_dir}  (tensorboard --logdir {run_dir})")
    except Exception:
        _have_tb = False
        print("[train] tensorboard not installed -> logging to progress.csv + stdout only")

    if args.algo == "ppo":
        from stable_baselines3 import PPO
        Algo = PPO
        policy_name = "MlpPolicy"
        algo_kwargs = dict(
            n_steps=max(8, args.n_steps),
            batch_size=max(8, args.batch_size),
            n_epochs=args.n_epochs, gamma=args.gamma,
            gae_lambda=0.95, ent_coef=args.ent_coef,
            learning_rate=args.learning_rate,
        )
    elif args.algo == "maskable_ppo":
        from sb3_contrib import MaskablePPO
        Algo = MaskablePPO
        # MaskablePPO auto-resolves "MlpPolicy" to MaskableActorCriticPolicy.
        policy_name = "MlpPolicy"
        algo_kwargs = dict(
            n_steps=max(8, args.n_steps),
            batch_size=max(8, args.batch_size),
            n_epochs=args.n_epochs, gamma=args.gamma,
            gae_lambda=0.95, ent_coef=args.ent_coef,
            learning_rate=args.learning_rate,
        )
    else:
        from stable_baselines3 import DQN
        Algo = DQN
        policy_name = "MlpPolicy"
        algo_kwargs = dict(
            buffer_size=args.buffer_size, batch_size=max(8, args.batch_size),
            gamma=args.gamma, learning_rate=args.learning_rate,
            exploration_fraction=args.exploration_fraction,
            learning_starts=1000, target_update_interval=1000,
        )

    reset_num_timesteps = True
    if args.resume:
        print(f"[train] resuming from {args.resume}")
        model = Algo.load(args.resume, env=env)
        if args.algo == "dqn":
            rb = args.resume.replace(".zip", "_replay_buffer.pkl")
            if os.path.exists(rb):
                model.load_replay_buffer(rb)
                print(f"[train]   loaded replay buffer {rb}")
        reset_num_timesteps = False
    else:
        model = Algo(policy_name, env, seed=args.seed, verbose=1,
                     policy_kwargs=policy_kwargs, **algo_kwargs)

    from stable_baselines3.common.logger import configure as _sb3_configure
    _fmts = ["stdout", "csv"] + (["tensorboard"] if _have_tb else [])
    model.set_logger(_sb3_configure(run_dir, _fmts))

    # try/finally + periodic _SaveLastZip safety-net = any non-SIGKILL exit
    # leaves a resumable last.zip.
    t0 = time.time()
    last_path  = os.path.join(ckpt_dir, "last")
    final_path = os.path.join(ckpt_dir, "final")
    print(f"[train] learn() total_timesteps={args.total_timesteps} "
          f"(reset_num_timesteps={reset_num_timesteps}) ...")
    learn_ok = False
    try:
        model.learn(total_timesteps=args.total_timesteps,
                    callback=callbacks, reset_num_timesteps=reset_num_timesteps,
                    tb_log_name="run")
        learn_ok = True
    finally:
        try:
            model.save(last_path)
            if args.algo == "dqn":
                model.save_replay_buffer(os.path.join(ckpt_dir, "last_replay_buffer.pkl"))
            print(f"[train] (try/finally) refreshed {last_path}.zip "
                  f"at num_timesteps={getattr(model, 'num_timesteps', '?')} "
                  f"(learn_ok={learn_ok})")
        except Exception as _e:
            print(f"[train] (try/finally) failed to save last.zip: {_e}")
    dt = time.time() - t0

    model.save(final_path)
    if args.algo == "dqn":
        model.save_replay_buffer(os.path.join(ckpt_dir, "last_replay_buffer.pkl"))
    print(f"[train] saved {final_path}.zip and {last_path}.zip")

    ep_r, ep_l = [], []
    try:
        import csv as _csv
        with open(mon_path + ".monitor.csv", "r") as f:
            lines = [ln for ln in f if not ln.startswith("#")]
        for row in _csv.DictReader(lines):
            try:
                ep_r.append(float(row["r"])); ep_l.append(int(float(row["l"])))
            except Exception:
                pass
    except Exception:
        pass
    summary = {
        "wall_time_s": round(dt, 1),
        "timesteps": int(model.num_timesteps),
        "episodes": len(ep_r),
        "mean_episode_reward": float(np.mean(ep_r)) if ep_r else None,
        "last20_mean_episode_reward": float(np.mean(ep_r[-20:])) if ep_r else None,
        "mean_episode_length": float(np.mean(ep_l)) if ep_l else None,
        "finished": datetime.datetime.now().isoformat(),
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[train] summary : {summary}")

    env.close()
    if eval_env is not None:
        eval_env.close()
    print("[train] done.")


if __name__ == "__main__":
    main()
