"""Top-K grasp candidate extraction from GG-CNN output maps via NMS."""
import numpy as np
from scipy.ndimage import maximum_filter


def extract_top_k_candidates(
    quality: np.ndarray,
    angle: np.ndarray,
    width: np.ndarray,
    depth_m_crop: np.ndarray,
    K: int = 10,
    min_quality: float = 0.3,
    nms_size: int = 11,
) -> list:
    """Return the top-K grasp candidates from GG-CNN output maps.

    Candidates are local quality maxima found via scipy maximum_filter NMS.
    Each dict contains px, py, quality, angle_rad, width_px, depth_m.
    """
    quality    = np.asarray(quality,      dtype=np.float32)
    angle      = np.asarray(angle,        dtype=np.float32)
    width      = np.asarray(width,        dtype=np.float32)
    depth_crop = np.asarray(depth_m_crop, dtype=np.float32)

    H, W = quality.shape

    local_max = maximum_filter(quality, size=nms_size)
    is_local_max = (quality == local_max) & (quality >= min_quality)

    ys, xs = np.where(is_local_max)
    if len(ys) == 0:
        return []

    scores = quality[ys, xs]
    order  = np.argsort(-scores)
    ys, xs, scores = ys[order], xs[order], scores[order]

    candidates = []
    for py, px, q in zip(ys[:K * 3], xs[:K * 3], scores[:K * 3]):
        d = float(depth_crop[py, px]) if depth_crop.shape == quality.shape else 0.0
        a = float(angle[py, px])
        w = float(width[py, px])

        candidates.append({
            "px":        int(px),
            "py":        int(py),
            "quality":   float(q),
            "angle_rad": a,
            "width_px":  w,
            "depth_m":   d,
        })
        if len(candidates) >= K:
            break

    return candidates
