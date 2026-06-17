"""GGCNNInference: thin wrapper around the pre-trained GG-CNN model."""
import sys
import os
import numpy as np

import torch
import torch.nn.functional as F

# Vendored GG-CNN sources live in third_party/ggcnn/ relative to the repo
# root. Power users can still override via env vars:
#   GGCNN_REPO        -> directory containing a ``models/ggcnn.py``
#   GGCNN_WEIGHTS     -> path to a Cornell statedict (.pt)
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_DEFAULT_GGCNN_REPO = os.path.join(_REPO_ROOT, "third_party", "ggcnn")
_DEFAULT_WEIGHTS = os.path.join(
    _DEFAULT_GGCNN_REPO, "weights", "ggcnn_epoch_23_cornell_statedict.pt"
)

GGCNN_REPO = os.environ.get("GGCNN_REPO", _DEFAULT_GGCNN_REPO)
WEIGHTS_PATH = os.environ.get("GGCNN_WEIGHTS", _DEFAULT_WEIGHTS)

if GGCNN_REPO not in sys.path:
    sys.path.insert(0, GGCNN_REPO)


def _load_ggcnn_class():
    try:
        from models.ggcnn import GGCNN  # noqa: F401
        return GGCNN
    except ImportError as e:
        raise ImportError(
            f"Cannot import GGCNN from {GGCNN_REPO}/models/ggcnn.py. "
            f"Original error: {e}"
        )


class GGCNNInference:
    """Pre-trained GG-CNN forward pass: depth -> (quality, angle, width)."""

    def __init__(self, weights_path: str = WEIGHTS_PATH, device: str = None,
                 cpu_threads: int = 4):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # at default torch CPU thread count (= n_cpus) GG-CNN forward is ~9 s
        # on a 40-core node due to thread contention. cap to 4 to get ~0.1 s.
        if self.device.type == "cpu" and cpu_threads is not None and cpu_threads > 0:
            try:
                if torch.get_num_threads() > cpu_threads:
                    torch.set_num_threads(int(cpu_threads))
            except Exception:
                pass

        GGCNN = _load_ggcnn_class()
        self.model = GGCNN()
        self.model.to(self.device)

        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"GG-CNN weights not found at: {weights_path}"
            )

        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self._weights_path = weights_path

    def predict(self, depth_preprocessed: np.ndarray) -> dict:
        """Forward pass on a (H, W) float32 depth normalised to [0, 1].

        Returns {quality: (H,W), angle: (H,W) rad, width: (H,W) px}.
        """
        depth = np.asarray(depth_preprocessed, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(
                f"Expected 2-D depth array, got shape {depth.shape}"
            )

        x = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pos_out, cos_out, sin_out, width_out = self.model(x)

        pos_np   = pos_out.squeeze().cpu().numpy().astype(np.float32)
        cos_np   = cos_out.squeeze().cpu().numpy().astype(np.float32)
        sin_np   = sin_out.squeeze().cpu().numpy().astype(np.float32)
        width_np = width_out.squeeze().cpu().numpy().astype(np.float32)

        quality = 1.0 / (1.0 + np.exp(-pos_np))
        angle = 0.5 * np.arctan2(sin_np, cos_np)  # rad in (-pi/2, pi/2)
        width = np.clip(width_np, 0.0, None)

        return {
            "quality": quality,
            "angle":   angle,
            "width":   width,
        }

    def __repr__(self):
        return (
            f"GGCNNInference(weights='{self._weights_path}', "
            f"device='{self.device}')"
        )
