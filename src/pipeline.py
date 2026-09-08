"""
NOP Vision Intelligence Pipeline - Main Processing Engine
Integrates Detection, Tracking, Spatial Zones, Stateful Event Engine,
Evidence Generation, Video Annotation, and VMS Analytics.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image

try:
    from .detector import create_detector
    from .tracker import RobustKalmanTracker, TrackState
    from .zones import ZoneManager
    from .event_engine import EventEngine
    from .evidence import EvidenceWriter
    from .analytics import VMSAnalytics
except (ImportError, ValueError):
    from detector import create_detector
    from tracker import RobustKalmanTracker, TrackState
    from zones import ZoneManager
    from event_engine import EventEngine
    from evidence import EvidenceWriter
    from analytics import VMSAnalytics


class VisionPipeline:
    """
    End-to-End Vision Intelligence Pipeline for NOP Pro+ Challenge.
    """

    def __init__(self, config: Dict[str, Any], output_dir: str, scenario_id: Optional[str] = None):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scenario_id = scenario_id or config.get("scenario_id", "S01_BASIC_GOODS")
        self.camera_id = config.get("camera_id", "challenge_cam_01")

        # Initialize Subsystems
        self.detector = create_detector(config)
        self.tracker = RobustKalmanTracker(
            max_coasting_frames=config.get("max_coasting_frames", 65),
            min_hits_to_confirm=config.get("min_hits_to_confirm", 2),
            max_match_distance=config.get("max_match_distance", 120.0),
        )

        zones_cfg = config.get("zones", {})
        self.zone_manager = ZoneManager(zones_cfg)
        self.event_engine = EventEngine(
            dwell_threshold_seconds=config.get("dwell_threshold_seconds", 5.0),
            terminal_zones=set(config.get("terminal_zones", ["A", "B"])),
        )
        self.evidence_writer = EvidenceWriter(
            output_dir=self.output_dir,
            scenario_id=self.scenario_id,
            camera_id=self.camera_id,
            model_name=config.get("model_name", "NOP-KalmanVision-v2.0"),
            model_version=config.get("model_version", "2.0.0"),
        )
        self.analytics = VMSAnalytics(self.scenario_id)

    def process_video(
        self,
        video_path: str,
        display: bool = False,
        save_video: bool = True,
    ) -> Dict[str, Any]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video input: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.analytics.fps = fps

        video_writer = None
        webp_preview_frames = []
        if save_video:
            out_vid_path = self.output_dir / "annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(out_vid_path), fourcc, fps, (width, height))

        frame_idx = 0
        recent_banner = "SYSTEM READY"

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            frame_time_s = frame_idx / fps

            # 1. Perception (Detection)
            detections = self.detector.detect(frame, frame_idx=frame_idx)

            # 2. Tracking (Kalman + Appearance + Coasting)
            tracks = self.tracker.update(detections)

            # 3. Spatial reasoning & Events
            active_events = []
            for tid, trk in tracks.items():
                cx, cy = trk["centroid"]
                zone = self.zone_manager.locate_point((cx, cy))
                trk["zone"] = zone

                # Log velocity
                vx, vy = trk["velocity"]
                self.analytics.log_velocity(tid, vx, vy)

                # Process event engine
                evts = self.event_engine.update(
                    track_id=tid,
                    current_zone=zone,
                    frame_time_s=frame_time_s,
                    frame_idx=frame_idx,
                )

                for evt in evts:
                    active_events.append(evt)
                    recent_banner = f"{evt['event_type']} (Track {tid})"

                    # Record NOP compliant evidence
                    self.evidence_writer.record_event(
                        frame=frame,
                        track=trk,
                        event=evt,
                        frame_time_s=frame_time_s,
                        frame_idx=frame_idx,
                    )

            # Log queue occupancy
            self.analytics.log_occupancy(
                frame_time_s, self.event_engine.active_queue_occupancy
            )

            # 4. Visualization & Annotation
            annotated_frame = frame.copy()
            self.zone_manager.draw(annotated_frame, alpha=0.15)

            # Draw tracks
            for tid, trk in tracks.items():
                x, y, w, h = trk["bbox"]
                cx, cy = trk["centroid"]
                zone = trk["zone"]
                is_coasting = trk.get("is_coasting", False)

                # Color: Green for confirmed active, Yellow for occluded coasting
                box_color = (0, 215, 255) if is_coasting else (50, 230, 80)
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), box_color, 2)
                cv2.circle(annotated_frame, (cx, cy), 4, (0, 255, 255), -1)

                # Motion trail
                history = trk.get("history", [])
                if len(history) > 1:
                    pts = np.array(history, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [pts], False, box_color, 1)

                label = f"ID {tid} {trk['object_type']}"
                if zone:
                    label += f" [{zone}]"
                if is_coasting:
                    label += " (COASTING)"

                cv2.rectangle(
                    annotated_frame, (x, max(0, y - 20)), (x + len(label) * 9, y), (20, 20, 20), -1
                )
                cv2.putText(
                    annotated_frame,
                    label,
                    (x + 2, max(14, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                )

            # Top HUD Diagnostics Bar
            self._draw_hud(annotated_frame, frame_time_s, frame_idx, recent_banner)

            if video_writer:
                video_writer.write(annotated_frame)
                if frame_idx % 2 == 0:
                    small_rgb = cv2.cvtColor(
                        cv2.resize(annotated_frame, (640, int(640 * height / width))),
                        cv2.COLOR_BGR2RGB,
                    )
                    webp_preview_frames.append(Image.fromarray(small_rgb))

            if display:
                cv2.imshow("NOP Vision Intelligence Review", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if video_writer:
            video_writer.release()
            if webp_preview_frames:
                try:
                    webp_out = self.output_dir / "preview.webp"
                    webp_preview_frames[0].save(
                        str(webp_out),
                        save_all=True,
                        append_images=webp_preview_frames[1:],
                        duration=80,
                        loop=0,
                    )
                except Exception:
                    pass
        if display:
            cv2.destroyAllWindows()

        # Generate analytics report
        analytics_file = self.output_dir / "analytics_summary.json"
        summary = self.analytics.generate_report(analytics_file)

        return {
            "scenario_id": self.scenario_id,
            "processed_frames": frame_idx,
            "counts": self.evidence_writer.counts,
            "analytics": summary,
            "events_count": len(self.evidence_writer.events_log),
        }

    def _draw_hud(self, frame: np.ndarray, time_s: float, frame_idx: int, banner: str):
        # Draw sleek dark HUD overlay on top 70px
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 70), (18, 18, 24), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Title & Time
        cv2.putText(
            frame,
            f"NOP Pro+ Vision | {self.scenario_id}",
            (16, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (240, 240, 240),
            2,
        )
        cv2.putText(
            frame,
            f"t = {time_s:05.2f}s | Frame {frame_idx:04d}",
            (16, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (180, 180, 180),
            1,
        )

        # Live Event Counts
        a_to_b = self.evidence_writer.counts.get("A_TO_B", 0)
        b_to_a = self.evidence_writer.counts.get("B_TO_A", 0)
        queue_cnt = self.event_engine.active_queue_occupancy

        hud_stats = f"A->B: {a_to_b}  |  B->A: {b_to_a}  |  Queue: {queue_cnt}"
        cv2.putText(
            frame,
            hud_stats,
            (420, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (80, 220, 140),
            2,
        )

        # Status Ticker
        cv2.putText(
            frame,
            f"STATUS: {banner}",
            (420, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (220, 220, 100),
            1,
        )
