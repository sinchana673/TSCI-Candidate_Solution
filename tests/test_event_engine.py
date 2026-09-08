"""
Unit tests for Event Engine and Dwell Analytics
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from event_engine import EventEngine


def test_a_to_b_transition():
    engine = EventEngine()

    # Track 1 starts in A
    evts1 = engine.update(track_id=1, current_zone="A", frame_time_s=1.0, frame_idx=25)
    assert len(evts1) == 0

    # Moves to None (transitioning)
    evts2 = engine.update(track_id=1, current_zone=None, frame_time_s=2.0, frame_idx=50)
    assert len(evts2) == 0

    # Reaches B
    evts3 = engine.update(track_id=1, current_zone="B", frame_time_s=3.0, frame_idx=75)
    assert len(evts3) == 1
    assert evts3[0]["event_type"] == "A_TO_B"
    assert evts3[0]["track_id"] == 1


def test_reversal_deduplication():
    engine = EventEngine()

    # Track 2 starts in B
    engine.update(track_id=2, current_zone="B", frame_time_s=1.0, frame_idx=25)
    # Moves out of B
    engine.update(track_id=2, current_zone=None, frame_time_s=2.0, frame_idx=50)
    # Reverses toward B (re-enters B)
    engine.update(track_id=2, current_zone="B", frame_time_s=3.0, frame_idx=75)
    # Turns around again, moves toward A
    engine.update(track_id=2, current_zone=None, frame_time_s=4.0, frame_idx=100)
    # Finally reaches A
    evts = engine.update(track_id=2, current_zone="A", frame_time_s=5.0, frame_idx=125)

    assert len(evts) == 1
    assert evts[0]["event_type"] == "B_TO_A"

    # Staying in A should not emit duplicate
    evts_dup = engine.update(track_id=2, current_zone="A", frame_time_s=6.0, frame_idx=150)
    assert len(evts_dup) == 0


def test_queue_dwell():
    engine = EventEngine(dwell_threshold_seconds=2.0)

    # Track 3 starts in A, then enters QUEUE
    engine.update(track_id=3, current_zone="A", frame_time_s=0.5, frame_idx=12)
    engine.update(track_id=3, current_zone="QUEUE", frame_time_s=1.0, frame_idx=25)

    # 1 second in QUEUE (total 1.0s < 2.0s) -> no dwell event yet
    evts1 = engine.update(track_id=3, current_zone="QUEUE", frame_time_s=2.0, frame_idx=50)
    assert len(evts1) == 0

    # 2.5 seconds in QUEUE (total 2.5s >= 2.0s) -> DWELL_THRESHOLD event
    evts2 = engine.update(track_id=3, current_zone="QUEUE", frame_time_s=3.5, frame_idx=87)
    assert len(evts2) == 1
    assert evts2[0]["event_type"] == "DWELL_THRESHOLD"

    # Moving from QUEUE to B should complete A_TO_B macro transition!
    evts3 = engine.update(track_id=3, current_zone="B", frame_time_s=5.0, frame_idx=125)
    assert any(e["event_type"] == "A_TO_B" for e in evts3)
