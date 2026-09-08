"""
Unit tests for NOP Evidence Contract Compliance
"""

import json
import os
import shutil
import sys
from pathlib import Path
import jsonschema
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from evidence import EvidenceWriter


def test_evidence_contract_schema_validation(tmp_path):
    local_cfg = Path(__file__).parent.parent / "config" / "evidence_contract.json"
    if local_cfg.is_file():
        schema_path = local_cfg
    elif Path("nop_reference/evidence_contract.json").is_file():
        schema_path = Path("nop_reference/evidence_contract.json")
    elif Path("../nop_reference/evidence_contract.json").is_file():
        schema_path = Path("../nop_reference/evidence_contract.json")
    else:
        schema_path = Path("config/evidence_contract.json")

    assert schema_path.is_file(), "NOP evidence contract JSON schema must exist"
    with open(schema_path, "r", encoding="utf-8") as sf:
        schema = json.load(sf)

    writer = EvidenceWriter(
        output_dir=tmp_path,
        scenario_id="S01_BASIC_GOODS",
        camera_id="cam_test_01",
        schema_path=str(schema_path),
    )

    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    dummy_track = {
        "track_id": 42,
        "bbox": [100, 100, 60, 40],
        "confidence": 0.95,
        "object_type": "package",
    }
    dummy_event = {
        "event_id": "evt-0042",
        "event_type": "A_TO_B",
        "attributes": {"source_zone": "A", "destination_zone": "B"},
    }

    record = writer.record_event(
        frame=dummy_frame,
        track=dummy_track,
        event=dummy_event,
        frame_time_s=12.34,
        frame_idx=308,
    )

    # Validate against JSON schema directly
    jsonschema.validate(instance=record, schema=schema)

    # Check generated files
    assert (tmp_path / "events.jsonl").is_file()
    assert (tmp_path / "events.csv").is_file()
    assert (tmp_path / "counts.csv").is_file()

    # Verify JSONL content
    with open(tmp_path / "events.jsonl", "r", encoding="utf-8") as jf:
        lines = jf.readlines()
        assert len(lines) == 1
        loaded_rec = json.loads(lines[0])
        assert loaded_rec["track_id"] == 42
        assert loaded_rec["event_type"] == "A_TO_B"
        assert loaded_rec["confidence"] == 0.95
        assert "evidence" in loaded_rec
        assert (tmp_path / loaded_rec["evidence"]["image_path"]).is_file()
