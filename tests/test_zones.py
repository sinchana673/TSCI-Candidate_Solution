"""
Unit tests for Zones and Spatial Geometry
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from zones import Zone, ZoneManager


def test_rect_zone():
    zone = Zone("A", [100, 100, 300, 300])
    assert zone.contains_point((200, 200)) is True
    assert zone.contains_point((50, 50)) is False
    assert zone.contains_point((350, 200)) is False


def test_polygon_zone():
    poly = [[100, 100], [300, 100], [300, 300], [100, 300]]
    zone = Zone("B", poly)
    assert zone.contains_point((200, 200)) is True
    assert zone.contains_point((400, 200)) is False


def test_zone_manager():
    zones_cfg = {
        "A": [50, 50, 200, 200],
        "B": [300, 50, 500, 200],
    }
    zm = ZoneManager(zones_cfg)
    assert zm.locate_point((100, 100)) == "A"
    assert zm.locate_point((400, 100)) == "B"
    assert zm.locate_point((250, 100)) is None
