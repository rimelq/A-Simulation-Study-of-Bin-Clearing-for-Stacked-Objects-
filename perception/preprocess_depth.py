"""Depth image preprocessing for GG-CNN inference."""
import numpy as np
import cv2


def crop_center_square(img: np.ndarray) -> np.ndarray:
    """Return the largest centre-aligned square crop of img."""
    h, w = img.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return img[y0:y0 + s, x0:x0 + s]


def inpaint_depth(depth: np.ndarray) -> np.ndarray:
    """Fill zero/nan/inf pixels via Telea inpainting + nearest-neighbour fallback."""
    depth = np.asarray(depth, dtype=np.float32)
    invalid_mask = (~np.isfinite(depth)) | (depth <= 0)

    if not np.any(invalid_mask):
        return depth

    mask_u8 = invalid_mask.astype(np.uint8) * 255

    valid = depth[~invalid_mask]
    if len(valid) == 0:
        return np.zeros_like(depth)

    d_min, d_max = float(valid.min()), float(valid.max())
    if d_max - d_min < 1e-6:
        depth[invalid_mask] = d_min
        return depth

    # cv2.inpaint requires uint8/uint16. round-trip through normalised uint16
    depth_norm = (depth - d_min) / (d_max - d_min)
    depth_norm = np.clip(depth_norm, 0, 1)
    depth_u16  = (depth_norm * 65535).astype(np.uint16)
    depth_u16[invalid_mask] = 0

    inpainted_u16 = cv2.inpaint(depth_u16, mask_u8, inpaintRadius=3,
                                 flags=cv2.INPAINT_TELEA)

    inpainted_norm  = inpainted_u16.astype(np.float32) / 65535.0
    inpainted_depth = inpainted_norm * (d_max - d_min) + d_min

    remaining = (inpainted_depth <= 0)
    if np.any(remaining):
        inpainted_depth[remaining] = float(valid.mean())

    return inpainted_depth


def preprocess_for_ggcnn(
    depth_m: np.ndarray,
    target_size: int = 300,
    inpaint_missing: bool = True,
) -> np.ndarray:
    """Center-crop, inpaint, resize to target_size, clip and normalise to [0, 1]."""
    depth_m = np.asarray(depth_m, dtype=np.float32)

    depth_crop = crop_center_square(depth_m)

    if inpaint_missing:
        depth_crop = inpaint_depth(depth_crop)

    depth_resized = cv2.resize(
        depth_crop,
        (target_size, target_size),
        interpolation=cv2.INTER_LINEAR,
    )

    clip_max = 1.2  # metres. depth beyond this is clipped to floor before normalising
    depth_clipped = np.clip(depth_resized, 0.0, clip_max)
    depth_norm    = depth_clipped / clip_max

    return depth_norm.astype(np.float32)


def get_crop_offset(original_h: int, original_w: int):
    """Return (x0, y0) of the centre-square crop within the original image."""
    s  = min(original_h, original_w)
    y0 = (original_h - s) // 2
    x0 = (original_w - s) // 2
    return x0, y0
