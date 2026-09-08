"""
NOP Vision Intelligence Pipeline - Advanced VMS Analytics Module
Provides Queue Occupancy profiling, Dwell Duration statistics, and Flow Velocity metrics.
"""

from typing import Dict, List, Any, Tuple, Optional, Union
import json
from pathlib import Path
import numpy as np

Optional_Path = Optional[Union[str, Path]]


class VMSAnalytics:
    """
    Computes warehouse/VMS operational metrics:
    - Queue occupancy timeline and peak queue length
    - Dwell time distributions per zone
    - Average object speeds and journey transit durations
    """

    def __init__(self, scenario_id: str, fps: float = 25.0):
        self.scenario_id = scenario_id
        self.fps = fps
        self.dwell_records: List[Dict[str, Any]] = []
        self.occupancy_timeline: List[Tuple[float, int]] = []
        self.speed_records: Dict[int, List[float]] = {}

    def log_occupancy(self, timestamp_s: float, count: int):
        self.occupancy_timeline.append((round(timestamp_s, 2), count))

    def log_dwell(self, track_id: int, zone: str, dwell_seconds: float):
        self.dwell_records.append({
            "track_id": track_id,
            "zone": zone,
            "dwell_seconds": round(dwell_seconds, 2)
        })

    def log_velocity(self, track_id: int, vx: float, vy: float):
        speed_px_per_sec = np.hypot(vx, vy) * self.fps
        if track_id not in self.speed_records:
            self.speed_records[track_id] = []
        self.speed_records[track_id].append(speed_px_per_sec)

    def generate_report(self, output_path: Optional_Path = None) -> Dict[str, Any]:
        queue_counts = [cnt for _, cnt in self.occupancy_timeline]
        peak_queue = max(queue_counts) if queue_counts else 0
        mean_queue = float(np.mean(queue_counts)) if queue_counts else 0.0

        dwell_by_zone: Dict[str, List[float]] = {}
        for rec in self.dwell_records:
            z = rec["zone"]
            if z not in dwell_by_zone:
                dwell_by_zone[z] = []
            dwell_by_zone[z].append(rec["dwell_seconds"])

        zone_stats = {}
        for z, vals in dwell_by_zone.items():
            zone_stats[z] = {
                "count": len(vals),
                "min_dwell_s": round(float(np.min(vals)), 2),
                "max_dwell_s": round(float(np.max(vals)), 2),
                "mean_dwell_s": round(float(np.mean(vals)), 2),
            }

        avg_speeds = {}
        for tid, speeds in self.speed_records.items():
            if speeds:
                avg_speeds[tid] = round(float(np.mean(speeds)), 1)

        summary = {
            "scenario_id": self.scenario_id,
            "fps": self.fps,
            "queue_analytics": {
                "peak_occupancy": peak_queue,
                "mean_occupancy": round(mean_queue, 2),
                "samples": len(self.occupancy_timeline),
            },
            "dwell_analytics": zone_stats,
            "track_speeds_px_per_sec": avg_speeds,
        }

        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

        return summary
