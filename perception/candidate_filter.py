"""Border / depth / quality / workspace filtering of grasp candidates."""
import numpy as np


def filter_candidates(
    candidates: list,
    K: int,
    workspace_mask: np.ndarray = None,
    border_px: int = 10,
    min_depth: float = 0.05,
    max_depth: float = 1.0,
    min_quality: float = 0.3,
    image_size: int = 300,
) -> list:
    """Drop candidates outside border / depth / quality / mask bounds.

    Returns the filtered list sorted by quality desc, truncated to K.
    """
    filtered = []
    for c in candidates:
        px = c["px"]
        py = c["py"]
        q  = c["quality"]
        d  = c["depth_m"]

        if px < border_px or py < border_px:
            continue
        if px >= image_size - border_px or py >= image_size - border_px:
            continue

        if d < min_depth or d > max_depth:
            continue

        if q < min_quality:
            continue

        if workspace_mask is not None:
            if py < workspace_mask.shape[0] and px < workspace_mask.shape[1]:
                if workspace_mask[py, px] == 0:
                    continue

        filtered.append(c)

    filtered.sort(key=lambda c: -c["quality"])
    return filtered[:K]
