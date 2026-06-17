"""Evaluate a trained policy (or a baseline) on BinClearingGymEnv.

Reports mean episode return, length, and bin-cleared fraction over N episodes.
Baselines (random, greedy top-q) need no checkpoint.

    See README.md for example commands and the exact flags.
"""
import os
import sys
import argparse

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl.bin_clearing_env import BinClearingGymEnv


def _pick_action(policy, model, obs, env):
    """Return a candidate index for the chosen policy on this observation."""
    if policy == "model":
        a, _ = model.predict(obs, deterministic=True)
        return int(a)
    if policy == "random":
        return int(env.action_space.sample())
    if policy == "greedy":
        # obs layout: K blocks of [valid_flag, q, cos, sin, width, gx, gy, gz, dist]
        K = env.action_space.n
        per = (len(obs) - 4) // K
        best_i, best_q = 0, -1.0
        for i in range(K):
            valid = obs[i * per + 0]
            q     = obs[i * per + 1]
            if valid > 0.5 and q > best_q:
                best_q, best_i = q, i
        return best_i
    raise ValueError(policy)


def main():
    ap = argparse.ArgumentParser(description="Evaluate a policy on BinClearingGymEnv")
    ap.add_argument("--policy", choices=["model", "random", "greedy"], default="model")
    ap.add_argument("--model", type=str, default=None,
                    help="path to a saved SB3 .zip (required if --policy model)")
    ap.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--n_objects", type=int, default=10)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--reward_mode", choices=["physics", "geometric"], default="physics",
                    help="forced to 'geometric' when --video is set")
    ap.add_argument("--video", type=str, default=None,
                    help="if set, render a wrist-POV mp4 of the first episode (slow)")
    args = ap.parse_args()

    model = None
    if args.policy == "model":
        if not args.model:
            ap.error("--policy model requires --model <path-to-.zip>")
        from stable_baselines3 import PPO, DQN
        model = (PPO if args.algo == "ppo" else DQN).load(args.model)

    slow = args.video is not None
    # video path needs the full primitive
    reward_mode = "geometric" if slow else args.reward_mode
    env = BinClearingGymEnv(n_objects=args.n_objects, K=10,
                            reward_mode=reward_mode, slow_physics=slow,
                            render_mode=("rgb_array" if slow else None))

    rng = np.random.default_rng(args.seed)
    returns, lengths, cleared_frac, deliveries = [], [], [], []
    frames = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        ep_r, ep_l, done = 0.0, 0, False
        n_spawn = info.get("n_items_remaining", args.n_objects)
        while not done:
            a = _pick_action(args.policy, model, obs, env)
            obs, r, term, trunc, info = env.step(a)
            ep_r += r
            ep_l += 1
            done = term or trunc
            if args.video and ep == 0:
                fr = info.get("frame")
                if fr is None and env.render_mode == "rgb_array":
                    fr = env.render()
                if fr is not None:
                    frames.append(np.asarray(fr))
        n_deliv = info.get("n_delivered", n_spawn - info.get("n_items_remaining", 0))
        returns.append(ep_r)
        lengths.append(ep_l)
        deliveries.append(n_deliv)
        cleared_frac.append(n_deliv / max(n_spawn, 1))
        print(f"  episode {ep + 1:3d}/{args.episodes}: return={ep_r:+7.3f}  "
              f"len={ep_l:3d}  delivered={n_deliv}/{n_spawn}")

    print("\n=== EVALUATION SUMMARY ===")
    print(f"  policy             : {args.policy}" +
          (f"  ({args.model})" if args.policy == 'model' else ""))
    print(f"  episodes           : {args.episodes}  (n_objects={args.n_objects})")
    print(f"  mean episode return: {np.mean(returns):+.3f}  +/- {np.std(returns):.3f}")
    print(f"  mean episode length: {np.mean(lengths):.1f}")
    print(f"  mean delivered     : {np.mean(deliveries):.2f} / {args.n_objects}")
    print(f"  mean bin-cleared % : {100 * np.mean(cleared_frac):.1f}%")
    print("==========================")

    if args.video and frames:
        import imageio
        imageio.mimsave(args.video, frames, fps=24)
        print(f"  wrist-POV video    : {args.video}  ({len(frames)} frames)")

    env.close()


if __name__ == "__main__":
    main()
