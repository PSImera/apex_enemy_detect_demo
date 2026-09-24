"""Tracker setup shared by every backend."""

from __future__ import annotations

import os
import tempfile

import numpy as np

# Ultralytics defaults, with global motion compensation turned off:
# sparseOptFlow measured at 75% of the whole track() call (~44 ms/frame, eight
# times the inference), and gameplay footage tracks fine without it.
TRACKER_CFG = "botsort.yaml"
TRACKER_OVERRIDES = {"gmc_method": "none"}

_yaml_path = None


def tracker_yaml() -> str:
    """Path to a tracker YAML carrying our overrides.

    Ultralytics' `model.track()` only accepts a file path, so the patched config
    is written once to a temp file and reused.
    """
    global _yaml_path
    if _yaml_path is None:
        from ultralytics.utils import YAML
        from ultralytics.utils.checks import check_yaml

        params = YAML.load(check_yaml(TRACKER_CFG))
        params.update(TRACKER_OVERRIDES)
        fd, path = tempfile.mkstemp(prefix="botsort_", suffix=".yaml")
        os.close(fd)
        YAML.save(path, params)
        _yaml_path = path
    return _yaml_path


def build_tracker(frame_rate: int = 60, cfg: str | None = None):
    """Create a BoT-SORT instance from a tracker YAML."""
    from ultralytics.trackers import BOTSORT
    from ultralytics.utils import IterableSimpleNamespace, YAML
    from ultralytics.utils.checks import check_yaml

    params = YAML.load(check_yaml(cfg or TRACKER_CFG))
    params.update(TRACKER_OVERRIDES)
    return BOTSORT(IterableSimpleNamespace(**params), frame_rate=frame_rate)


class DetResults:
    """Adapter presenting raw detections the way Ultralytics trackers expect."""

    __slots__ = ("conf", "cls", "xywh", "xyxy")

    def __init__(self, conf, cls, xywh, xyxy):
        self.conf, self.cls, self.xywh, self.xyxy = conf, cls, xywh, xyxy

    @classmethod
    def from_detections(cls, det):
        """Build from an (N, 6) tensor of [x1, y1, x2, y2, conf, cls]."""
        xyxy = det[:, :4].cpu().numpy()
        xywh = np.stack(
            [
                (xyxy[:, 0] + xyxy[:, 2]) / 2,
                (xyxy[:, 1] + xyxy[:, 3]) / 2,
                xyxy[:, 2] - xyxy[:, 0],
                xyxy[:, 3] - xyxy[:, 1],
            ],
            axis=1,
        )
        return cls(det[:, 4].cpu().numpy(), det[:, 5].cpu().numpy(), xywh, xyxy)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, i):
        return DetResults(self.conf[i], self.cls[i], self.xywh[i], self.xyxy[i])
