"""
Unit tests for Detector Module
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from detector import AdaptiveDetector, create_detector


def test_adaptive_detector_init():
    det = AdaptiveDetector(min_area=1000, max_area=20000)
    assert det.min_area == 1000
    assert det.max_area == 20000


def test_adaptive_detector_synthetic_object():
    det = AdaptiveDetector(min_area=500, max_area=10000)
    # Create dark frame with colored synthetic package
    frame = np.full((720, 1280, 3), 28, dtype=np.uint8)
    # Draw colored rectangle: BGR (70, 170, 255) size 80x60 at (200, 200)
    frame[200:260, 200:280] = (70, 170, 255)

    detections = det.detect(frame, frame_idx=1)
    assert len(detections) >= 1
    d = detections[0]
    assert "bbox" in d
    assert "centroid" in d
    assert "confidence" in d
    assert "color_feat" in d
    # Centroid should be around (240, 230)
    cx, cy = d["centroid"]
    assert 220 <= cx <= 260
    assert 210 <= cy <= 250


def test_create_detector_factory():
    cfg = {"detector_type": "adaptive", "min_area": 800}
    det = create_detector(cfg)
    assert isinstance(det, AdaptiveDetector)
