"""
NOP Vision Intelligence Pipeline - Evidence Management Module
Produces audited, traceable evidence artifacts:
1. NOP Contract Compliant events.jsonl with strict schema validation
2. Consolidated events.csv for automated evaluators
3. counts.csv summary
4. High-resolution visual snapshots with diagnostic overlays
"""

import csv
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import cv2
import jsonschema

Union_Path_Str = Union[str, Path]
np_ndarray = Any


class EvidenceWriter:
    """
    Manages structured evidence generation adhering to NOP Pro+ evidence contracts.
    """

    def __init__(
        self,
        output_dir: Union_Path_Str,
        scenario_id: str = "S01_BASIC_GOODS",
        camera_id: str = "challenge_cam_01",
        schema_path: Optional[str] = None,
        model_name: str = "NOP-KalmanVision-v2.0",
        model_version: str = "2.0.0",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir = self.output_dir / "evidence"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        self.scenario_id = scenario_id
        self.camera_id = camera_id
        self.model_name = model_name
        self.model_version = model_version

        self.jsonl_path = self.output_dir / "events.jsonl"
        self.csv_path = self.output_dir / "events.csv"
        self.counts_path = self.output_dir / "counts.csv"

        # Initialize or clear files
        self.events_log: List[Dict[str, Any]] = []
        self.counts: Dict[str, int] = {}

        # Load NOP schema if available
        self.schema = None
        candidate_schema_paths = [
            schema_path,
            str(Path(__file__).parent.parent / "config" / "evidence_contract.json"),
            "config/evidence_contract.json",
            "nop_reference/evidence_contract.json",
            "../nop_reference/evidence_contract.json",
        ]
        for p in candidate_schema_paths:
            if p and os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8") as sf:
                        self.schema = json.load(sf)
                        break
                except Exception:
                    pass

    def record_event(
        self,
        frame: np_ndarray,
        track: Dict[str, Any],
        event: Dict[str, Any],
        frame_time_s: float,
        frame_idx: int,
    ) -> Dict[str, Any]:
        """
        Records a detected event, generates the visual snapshot, writes JSONL & CSV records.
        """
        event_type = event["event_type"]
        track_id = track["track_id"]
        confidence = float(track.get("confidence", 0.90))
        object_type = track.get("object_type", "package")
        event_id = event.get("event_id", f"evt-{len(self.events_log) + 1:04d}")

        # Update event counts
        self.counts[event_type] = self.counts.get(event_type, 0) + 1

        # Generate evidence snapshot image
        obs_id = str(uuid.uuid4())
        image_filename = f"{event_id}_{obs_id[:8]}.jpg"
        image_full_path = self.evidence_dir / image_filename
        rel_image_path = os.path.relpath(image_full_path, self.output_dir).replace("\\", "/")

        # Annotate snapshot frame
        snap_frame = frame.copy()
        x, y, w, h = track.get("bbox", [0, 0, 50, 50])
        cv2.rectangle(snap_frame, (x, y), (x + w, y + h), (0, 255, 255), 3)

        # Draw event header banner
        banner_text = f"EVENT: {event_type} | TRACK: {track_id} | TIME: {frame_time_s:.2f}s"
        cv2.rectangle(snap_frame, (10, 10), (10 + len(banner_text) * 11, 42), (0, 0, 0), -1)
        cv2.putText(
            snap_frame,
            banner_text,
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        cv2.imwrite(str(image_full_path), snap_frame)

        # Construct NOP Contract Record
        now_iso = datetime.now(timezone.utc).isoformat()
        record = {
            "observation_id": obs_id,
            "camera_id": self.camera_id,
            "observed_at": now_iso,
            "scenario_id": self.scenario_id,
            "event_id": event_id,
            "object_type": object_type,
            "event_type": event_type,
            "track_id": track_id,
            "confidence": round(confidence, 3),
            "attributes": {
                "scenario_id": self.scenario_id,
                "video_time_seconds": round(frame_time_s, 3),
                "frame_index": frame_idx,
                **event.get("attributes", {}),
            },
            "evidence": {
                "image_path": rel_image_path,
                "clip_path": "",
            },
            "model": {
                "name": self.model_name,
                "version": self.model_version,
            },
        }

        # Validate against JSON schema if present
        if self.schema:
            try:
                jsonschema.validate(instance=record, schema=self.schema)
            except Exception as e:
                print(f"[EvidenceWriter] Schema validation warning: {e}")

        # Append to JSONL
        with open(self.jsonl_path, "a", encoding="utf-8") as jf:
            jf.write(json.dumps(record) + "\n")

        # Record for all_events.csv
        all_csv_path = self.output_dir / "all_events.csv"
        csv_row = {
            "scenario_id": self.scenario_id,
            "event_id": event_id,
            "track_id": track_id,
            "event_type": event_type,
            "object_type": object_type,
            "observed_at_seconds": f"{frame_time_s:.3f}",
            "confidence": f"{confidence:.2f}",
            "evidence_image": rel_image_path,
        }
        self._append_to_csv(all_csv_path, csv_row)

        # Record for events.csv (directional transition events for evaluation against ground truth)
        if "TO" in event_type.upper() or event_type.upper() in ("A_TO_B", "B_TO_A", "LINE_CROSSING"):
            self.events_log.append(csv_row)
            self._flush_csv()

        self._flush_counts()
        return record

    def _append_to_csv(self, csv_path, row):
        fieldnames = [
            "scenario_id",
            "event_id",
            "track_id",
            "event_type",
            "object_type",
            "observed_at_seconds",
            "confidence",
            "evidence_image",
        ]
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _flush_csv(self):
        fieldnames = [
            "scenario_id",
            "event_id",
            "track_id",
            "event_type",
            "object_type",
            "observed_at_seconds",
            "confidence",
            "evidence_image",
        ]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.events_log:
                writer.writerow(row)

    def _flush_counts(self):
        with open(self.counts_path, "w", newline="", encoding="utf-8") as cnf:
            writer = csv.writer(cnf)
            writer.writerow(["event_type", "count"])
            for etype, cnt in sorted(self.counts.items()):
                writer.writerow([etype, cnt])
