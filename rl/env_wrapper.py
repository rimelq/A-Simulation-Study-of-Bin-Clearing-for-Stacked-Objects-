"""Backward-compatibility shim. the env lives in ``rl/bin_clearing_env.py``."""
from rl.bin_clearing_env import BinClearingGymEnv  # noqa: F401

__all__ = ["BinClearingGymEnv"]
