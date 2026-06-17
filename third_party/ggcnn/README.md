# Vendored GG-CNN

Minimal vendored subset of GG-CNN (https://github.com/dougsm/ggcnn) used for
the greedy_ggcnn baseline in this project.

Files copied verbatim from the upstream repository:

- `models/ggcnn.py` -- the `GGCNN` nn.Module (only file imported at inference).
- `LICENSE` -- BSD 3-Clause License (Copyright 2018, Douglas Morrison, ACRV/QUT).
- `weights/ggcnn_epoch_23_cornell_statedict.pt` -- pre-trained Cornell weights
  (released alongside the upstream repo, ~247 KB).

`models/__init__.py` is a stripped-down version of the upstream init that drops
the unused `ggcnn2` branch. No other upstream files are used at inference time.

The BSD-3 license is retained per its redistribution clause.
