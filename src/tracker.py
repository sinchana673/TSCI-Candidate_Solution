"""
NOP Vision Intelligence Pipeline - Multi-Object Tracker Module
Implements Kalman Filter state estimation with motion prediction, occlusion coasting,
and appearance/Hue-gated Hungarian association to eliminate ID switches during dense crossings.
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


class TrackState(Enum):
    TENTATIVE = 1
    CONFIRMED = 2
    COASTING = 3
    DELETED = 4


def hue_distance(h1: float, h2: float) -> float:
    """Calculates minimal circular distance between two hues on [0, 180] scale."""
    if h1 < 0 or h2 < 0:
        return 0.0
    d = abs(h1 - h2)
    return float(min(d, 180.0 - d))


class KalmanBoxTracker:
    """
    Constant-velocity Kalman Filter for 2D bounding boxes:
    State vector: [cx, cy, w, h, vx, vy, vw, vh]
    Measurement vector: [cx, cy, w, h]
    """

    def __init__(self, bbox: List[int], color_feat: Optional[np.ndarray] = None):
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0

        # State: 8x1
        self.x = np.array([[cx], [cy], [w], [h], [0.0], [0.0], [0.0], [0.0]], dtype=np.float32)

        # State transition matrix F (dt=1.0)
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement matrix H
        self.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0

        # Covariance matrices
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 100.0, 100.0, 10.0, 10.0]).astype(np.float32)
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 1.0, 1.0]).astype(np.float32)
        self.R = np.diag([4.0, 4.0, 10.0, 10.0]).astype(np.float32)

        self.color_feat = color_feat
        self.history: List[Tuple[int, int]] = [(int(cx), int(cy))]
        self.hit_streak = 1
        self.age = 1
        self.time_since_update = 0

    def predict(self) -> np.ndarray:
        """Projects state and covariance forward by one time step."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.age += 1
        self.time_since_update += 1

        cx, cy, w, h = self.x[:4, 0]
        self.history.append((int(cx), int(cy)))
        if len(self.history) > 60:
            self.history.pop(0)
        return self.get_bbox()

    def update(self, bbox: List[int], confidence: float, color_feat: Optional[np.ndarray] = None):
        """Updates filter with an observed measurement."""
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        z = np.array([[cx], [cy], [w], [h]], dtype=np.float32)

        y_res = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y_res)
        I = np.eye(8, dtype=np.float32)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(
            np.dot(K, self.R), K.T
        )

        self.time_since_update = 0
        self.hit_streak += 1

        if color_feat is not None and color_feat.size > 0:
            if self.color_feat is None:
                self.color_feat = color_feat
            else:
                self.color_feat = 0.85 * self.color_feat + 0.15 * color_feat
                norm = np.linalg.norm(self.color_feat)
                if norm > 1e-6:
                    self.color_feat /= norm

    def get_bbox(self) -> List[int]:
        """Returns [x, y, w, h] integer bounding box."""
        cx, cy, w, h = self.x[:4, 0]
        w = max(10, int(round(w)))
        h = max(10, int(round(h)))
        x = int(round(cx - w / 2.0))
        y = int(round(cy - h / 2.0))
        return [x, y, w, h]

    def get_centroid(self) -> Tuple[int, int]:
        cx, cy = self.x[:2, 0]
        return int(round(cx)), int(round(cy))

    def get_velocity(self) -> Tuple[float, float]:
        vx, vy = self.x[4:6, 0]
        return float(vx), float(vy)


def compute_iou(box1: List[int], box2: List[int]) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / float(max(1, union))


class RobustKalmanTracker:
    """
    Industrial Multi-Object Tracker with Kalman filtering, occlusion coasting,
    and appearance/Hue-gated Hungarian matching.
    """

    def __init__(
        self,
        max_coasting_frames: int = 85,
        min_hits_to_confirm: int = 2,
        max_match_distance: float = 140.0,
        hue_gate_threshold: float = 16.0,
    ):
        self.max_coasting_frames = max_coasting_frames
        self.min_hits_to_confirm = min_hits_to_confirm
        self.max_match_distance = max_match_distance
        self.hue_gate_threshold = hue_gate_threshold

        self.next_track_id = 1
        self.trackers: Dict[int, KalmanBoxTracker] = {}
        self.track_metadata: Dict[int, Dict[str, Any]] = {}

    def update(self, detections: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        # 1. Predict all current tracks
        for tid, tracker in self.trackers.items():
            tracker.predict()

        num_tracks = len(self.trackers)
        num_dets = len(detections)
        track_ids = list(self.trackers.keys())

        matched_tracks = set()
        matched_dets = set()

        if num_tracks > 0 and num_dets > 0:
            cost_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)

            for i, tid in enumerate(track_ids):
                trk = self.trackers[tid]
                t_box = trk.get_bbox()
                t_cx, t_cy = trk.get_centroid()
                t_hue = self.track_metadata[tid].get("hue", -1.0)

                for j, det in enumerate(detections):
                    d_box = det["bbox"]
                    d_cx, d_cy = det["centroid"]
                    d_hue = det.get("hue", -1.0)

                    # Appearance / Hue gating: distinct colors CANNOT match
                    h_dist = hue_distance(t_hue, d_hue)
                    if t_hue >= 0 and d_hue >= 0 and h_dist > self.hue_gate_threshold:
                        cost_matrix[i, j] = 1e5
                        continue

                    dist = np.hypot(d_cx - t_cx, d_cy - t_cy)
                    max_allowed = 240.0 if (t_hue >= 0 and d_hue >= 0 and h_dist <= 10.0) else self.max_match_distance
                    if dist > max_allowed:
                        cost_matrix[i, j] = 1e5
                        continue

                    norm_dist = dist / max_allowed
                    iou = compute_iou(t_box, d_box)
                    cost_matrix[i, j] = 0.6 * norm_dist + 0.4 * (1.0 - iou)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 0.85:
                    tid = track_ids[r]
                    det = detections[c]
                    self.trackers[tid].update(
                        det["bbox"], det["confidence"], det.get("color_feat")
                    )
                    self.track_metadata[tid]["confidence"] = det["confidence"]
                    self.track_metadata[tid]["object_type"] = det["object_type"]
                    self.track_metadata[tid]["state"] = TrackState.CONFIRMED
                    if "hue" in det and det["hue"] >= 0:
                        self.track_metadata[tid]["hue"] = det["hue"]
                    matched_tracks.add(tid)
                    matched_dets.add(c)

        # 2. Handle unmatched tracks (coasting / deletion)
        for tid in track_ids:
            if tid not in matched_tracks:
                trk = self.trackers[tid]
                cx, cy = trk.get_centroid()
                if cx < -60 or cx > 1340 or cy < -60 or cy > 780 or trk.time_since_update > self.max_coasting_frames:
                    del self.trackers[tid]
                    del self.track_metadata[tid]
                else:
                    self.track_metadata[tid]["state"] = TrackState.COASTING

        # 3. Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_dets:
                tid = self.next_track_id
                self.next_track_id += 1
                tracker = KalmanBoxTracker(det["bbox"], det.get("color_feat"))
                self.trackers[tid] = tracker
                self.track_metadata[tid] = {
                    "object_type": det["object_type"],
                    "confidence": det["confidence"],
                    "state": TrackState.CONFIRMED,
                    "hue": det.get("hue", -1.0),
                }

        # 4. Prepare active return tracks
        active_tracks: Dict[int, Dict[str, Any]] = {}
        for tid, trk in self.trackers.items():
            meta = self.track_metadata[tid]
            active_tracks[tid] = {
                "track_id": tid,
                "bbox": trk.get_bbox(),
                "centroid": trk.get_centroid(),
                "velocity": trk.get_velocity(),
                "confidence": meta["confidence"],
                "object_type": meta["object_type"],
                "state": meta["state"],
                "history": list(trk.history),
                "is_coasting": (meta["state"] == TrackState.COASTING),
            }

        return active_tracks
