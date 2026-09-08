"""
NOP Vision Intelligence Pipeline - Zones & Spatial Geometry Module
Supports arbitrary polygons, bounding boxes, normalized coordinates,
and directional tripwires with high-performance spatial testing and visualization.
"""

from typing import Dict, List, Tuple, Union, Optional, Any
import cv2
import numpy as np


class Zone:
    """Represents a discrete spatial zone (polygon or bounding box)."""

    def __init__(self, name: str, geometry: Union[List[int], List[List[int]], List[Tuple[int, int]]], color: Tuple[int, int, int] = (120, 120, 120)):
        self.name = name
        self.color = color
        self.polygon: np.ndarray = self._parse_geometry(geometry)
        self.bbox = self._compute_bbox()

    def _parse_geometry(self, geometry: Any) -> np.ndarray:
        # Check if rect format [x1, y1, x2, y2]
        if isinstance(geometry, (list, tuple)) and len(geometry) == 4 and isinstance(geometry[0], (int, float)):
            x1, y1, x2, y2 = geometry
            return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        # Otherwise list of points [[x1, y1], [x2, y2], ...]
        return np.array(geometry, dtype=np.int32)

    def _compute_bbox(self) -> Tuple[int, int, int, int]:
        x1 = int(np.min(self.polygon[:, 0]))
        y1 = int(np.min(self.polygon[:, 1]))
        x2 = int(np.max(self.polygon[:, 0]))
        y2 = int(np.max(self.polygon[:, 1]))
        return x1, y1, x2, y2

    def contains_point(self, point: Tuple[Union[int, float], Union[int, float]]) -> bool:
        """Tests if a 2D point (x, y) is strictly inside or on boundary of zone."""
        px, py = float(point[0]), float(point[1])
        x1, y1, x2, y2 = self.bbox
        if not (x1 <= px <= x2 and y1 <= py <= y2):
            return False
        return cv2.pointPolygonTest(self.polygon, (px, py), False) >= 0

    def intersects_bbox(self, bbox: List[int]) -> bool:
        """Tests if an object's bounding box [x, y, w, h] overlaps with the zone."""
        bx1, by1, bw, bh = bbox
        bx2, by2 = bx1 + bw, by1 + bh
        x1, y1, x2, y2 = self.bbox
        # Fast bounding box intersection check
        if bx2 < x1 or bx1 > x2 or by2 < y1 or by1 > y2:
            return False
        # Centroid check
        cx = bx1 + bw / 2.0
        cy = by1 + bh / 2.0
        return self.contains_point((cx, cy))


class ZoneManager:
    """Manages multi-zone configurations and spatial membership."""

    DEFAULT_COLORS = {
        "A": (60, 160, 240),      # Amber/Orange
        "B": (80, 200, 120),      # Green
        "QUEUE": (200, 160, 60),  # Blue/Cyan
        "DWELL": (180, 90, 200),  # Purple
        "RESTRICTED": (50, 50, 230) # Red
    }

    def __init__(self, zones_config: Dict[str, Any]):
        self.zones: Dict[str, Zone] = {}
        for name, geom in zones_config.items():
            color = self.DEFAULT_COLORS.get(name.upper(), (130, 130, 130))
            self.zones[name] = Zone(name, geom, color)

    def locate_point(self, point: Tuple[Union[int, float], Union[int, float]]) -> Optional[str]:
        """Returns the name of the first zone containing the point."""
        for name, zone in self.zones.items():
            if zone.contains_point(point):
                return name
        return None

    def locate_bbox(self, bbox: List[int]) -> Optional[str]:
        """Returns the name of the zone containing the centroid, or overlapping."""
        cx = bbox[0] + bbox[2] / 2.0
        cy = bbox[1] + bbox[3] / 2.0
        zone = self.locate_point((cx, cy))
        if zone is not None:
            return zone
        for name, z in self.zones.items():
            if z.intersects_bbox(bbox):
                return name
        return None

    def draw(self, frame: np.ndarray, alpha: float = 0.15) -> np.ndarray:
        """Draws translucent zone overlays with crisp borders and labels."""
        overlay = frame.copy()
        for name, zone in self.zones.items():
            cv2.fillPoly(overlay, [zone.polygon], zone.color)
            cv2.polylines(frame, [zone.polygon], True, zone.color, 2)

            # Label banner
            x1, y1 = zone.polygon[0]
            cv2.rectangle(frame, (x1, y1 - 24), (x1 + 100, y1), zone.color, -1)
            cv2.putText(
                frame,
                f"ZONE {name}",
                (x1 + 6, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
        return frame
