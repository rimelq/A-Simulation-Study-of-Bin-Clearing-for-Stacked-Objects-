"""RL-ready bin-clearing pipeline. Main entry: ``rl.bin_clearing_env.BinClearingGymEnv``."""

__all__ = ["BinClearingGymEnv", "OBS_DIM", "build_observation", "compute_reward"]


def __getattr__(name):
    if name == "BinClearingGymEnv":
        from rl.bin_clearing_env import BinClearingGymEnv
        return BinClearingGymEnv
    if name in ("OBS_DIM", "build_observation"):
        from rl import observation_builder
        return getattr(observation_builder, name)
    if name == "compute_reward":
        from rl.reward import compute_reward
        return compute_reward
    raise AttributeError(f"module 'rl' has no attribute {name!r}")
