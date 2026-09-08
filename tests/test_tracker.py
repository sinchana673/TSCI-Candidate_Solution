"""
Unit tests for Kalman Tracker and Association
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from tracker import RobustKalmanTracker, KalmanBoxTracker, TrackState, compute_iou


def test_compute_iou():
    box1 = [100, 100, 50, 50]
    box2 = [100, 100, 50, 50]
    assert compute_iou(box1, box2) == 1.0

    box3 = [200, 200, 50, 50]
    assert compute_iou(box1, box3) == 0.0


def test_kalman_box_tracker():
    tracker = KalmanBoxTracker([100, 100, 50, 50])
    pred_box = tracker.predict()
    assert len(pred_box) == 4

    tracker.update([105, 102, 50, 50], 0.95)
    cx, cy = tracker.get_centroid()
    assert 120 <= cx <= 140
    assert 120 <= cy <= 140


def test_robust_tracker_coasting():
    tracker = RobustKalmanTracker(max_coasting_frames=5, min_hits_to_confirm=1)

    # Frame 1: Detection at (100, 100)
    det1 = [{
        "bbox": [100, 100, 50, 50],
        "centroid": [125, 125],
        "confidence": 0.9,
        "object_type": "package",
        "color_feat": np.zeros(24, dtype=np.float32),
    }]
    tracks1 = tracker.update(det1)
    assert len(tracks1) == 1
    tid = list(tracks1.keys())[0]

    # Frame 2: Missing detection (occlusion) -> tracker should coast
    tracks2 = tracker.update([])
    assert tid in tracks2
    assert tracks2[tid]["is_coasting"] is True

    # Frame 3: Object reappears nearby -> re-associated to same tid
    det2 = [{
        "bbox": [104, 102, 50, 50],
        "centroid": [129, 127],
        "confidence": 0.9,
        "object_type": "package",
        "color_feat": np.zeros(24, dtype=np.float32),
    }]
    tracks3 = tracker.update(det2)
    assert tid in tracks3
    assert tracks3[tid]["is_coasting"] is False
