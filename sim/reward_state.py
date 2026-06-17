"""RewardState: tracks objects transferred to the destination bin and computes shaped rewards."""
import numpy as np


class RewardState:
    """Tracks objects in the destination bin AABB and computes shaped rewards."""

    def __init__(self, dest_bin_pos: np.ndarray, dest_bin_size: np.ndarray):
        self.dest_bin_pos = np.asarray(dest_bin_pos, dtype=np.float64)
        self.dest_bin_size = np.asarray(dest_bin_size, dtype=np.float64)
        self._transferred_names: set = set()
        self._n_transferred: int = 0

    def reset(self):
        self._transferred_names = set()
        self._n_transferred = 0

    def check_object_in_dest(self, obj_pos: np.ndarray) -> bool:
        """True if obj_pos lies within the destination bin AABB."""
        obj_pos = np.asarray(obj_pos, dtype=np.float64)
        diff = np.abs(obj_pos - self.dest_bin_pos)
        return bool(np.all(diff <= self.dest_bin_size))

    def get_n_transferred(self) -> int:
        return self._n_transferred

    def compute_reward(
        self,
        prev_in_dest: int,
        curr_in_dest: int,
        step_penalty: float = -0.01,
        transfer_bonus: float = 1.0,
    ) -> float:
        """Shaped per-step reward: +transfer_bonus per newly delivered object, plus step_penalty."""
        new_transfers = max(0, curr_in_dest - prev_in_dest)
        reward = new_transfers * transfer_bonus + step_penalty
        return float(reward)

    def update(self, object_positions: dict) -> int:
        """Scan all object positions. update the transferred set and return current count."""
        count = 0
        for name, pos in object_positions.items():
            if self.check_object_in_dest(pos):
                self._transferred_names.add(name)
                count += 1
        self._n_transferred = count
        return count
