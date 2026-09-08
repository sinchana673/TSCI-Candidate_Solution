"""
NOP Vision Intelligence Pipeline - Stateful Event Engine
Tracks trajectory journeys, origin-destination transitions, dwell times, and queue occupancy
with strict debouncing and duplicate suppression.
"""

from typing import Dict, List, Set, Optional, Any, Tuple
import uuid


class TrackJourney:
    """Maintains the temporal spatial journey history for a single object track."""

    def __init__(self, track_id: int, initial_zone: Optional[str], initial_time_s: float):
        self.track_id = track_id
        self.origin_zone: Optional[str] = initial_zone
        self.current_zone: Optional[str] = initial_zone
        self.last_terminal_zone: Optional[str] = initial_zone if initial_zone in ("A", "B") else None
        self.visited_zones: List[Tuple[str, float]] = []
        if initial_zone:
            self.visited_zones.append((initial_zone, initial_time_s))

        # Zone dwell timers: {zone_name: {"entered_at": t, "total_dwell": s, "completed": bool}}
        self.dwell_states: Dict[str, Dict[str, Any]] = {}
        if initial_zone:
            self.dwell_states[initial_zone] = {
                "entered_at": initial_time_s,
                "total_dwell": 0.0,
                "dwell_event_fired": False,
            }

        self.completed_transits: Set[str] = set()
        self.last_event_time: float = -1.0


class EventEngine:
    """
    Stateful event reasoning engine.
    Detects macro-transitions (A->B, B->A) even through intermediate zones (e.g., QUEUE),
    tracks dwell metrics, and suppresses false duplicates.
    """

    def __init__(
        self,
        min_transit_frames: int = 10,
        dwell_threshold_seconds: float = 5.0,
        terminal_zones: Optional[Set[str]] = None,
    ):
        self.min_transit_frames = min_transit_frames
        self.dwell_threshold_seconds = dwell_threshold_seconds
        self.terminal_zones = terminal_zones or {"A", "B"}
        self.journeys: Dict[int, TrackJourney] = {}
        self.event_counter = 1
        self.active_queue_occupancy: int = 0
        self.queue_history: List[Tuple[float, int]] = []

    def update(
        self,
        track_id: int,
        current_zone: Optional[str],
        frame_time_s: float,
        frame_idx: int,
    ) -> List[Dict[str, Any]]:
        events = []

        if track_id not in self.journeys:
            self.journeys[track_id] = TrackJourney(track_id, current_zone, frame_time_s)

        journey = self.journeys[track_id]
        prev_zone = journey.current_zone

        # 1. Update zone transition state
        if current_zone != prev_zone:
            # Handle exit of prev_zone
            if prev_zone and prev_zone in journey.dwell_states:
                d_state = journey.dwell_states[prev_zone]
                d_state["total_dwell"] += max(0.0, frame_time_s - d_state["entered_at"])

            # Handle entry of current_zone
            if current_zone:
                journey.visited_zones.append((current_zone, frame_time_s))
                if current_zone not in journey.dwell_states:
                    journey.dwell_states[current_zone] = {
                        "entered_at": frame_time_s,
                        "total_dwell": 0.0,
                        "dwell_event_fired": False,
                    }
                else:
                    journey.dwell_states[current_zone]["entered_at"] = frame_time_s

                # If track never had a terminal origin, assign first visited terminal zone
                if journey.last_terminal_zone is None and current_zone in self.terminal_zones:
                    journey.last_terminal_zone = current_zone
                    if journey.origin_zone is None:
                        journey.origin_zone = current_zone

            journey.current_zone = current_zone

        # 2. Check Dwell Thresholds for active zones (e.g. QUEUE)
        if current_zone and current_zone in journey.dwell_states:
            d_state = journey.dwell_states[current_zone]
            current_dwell = d_state["total_dwell"] + (frame_time_s - d_state["entered_at"])
            if (
                not d_state["dwell_event_fired"]
                and current_dwell >= self.dwell_threshold_seconds
                and current_zone.upper() in ("QUEUE", "DWELL", "WAITING")
            ):
                d_state["dwell_event_fired"] = True
                evt_id = f"evt-{self.event_counter:04d}"
                self.event_counter += 1
                events.append({
                    "event_id": evt_id,
                    "event_type": "DWELL_THRESHOLD",
                    "track_id": track_id,
                    "zone": current_zone,
                    "dwell_seconds": round(current_dwell, 2),
                    "frame_time_s": frame_time_s,
                    "attributes": {
                        "zone": current_zone,
                        "dwell_duration": round(current_dwell, 2),
                        "threshold": self.dwell_threshold_seconds,
                    },
                })

        # 3. Check Macro Terminal Transitions (A_TO_B / B_TO_A)
        if current_zone in self.terminal_zones:
            origin = journey.last_terminal_zone
            if origin and origin != current_zone:
                event_type = f"{origin}_TO_{current_zone}"
                if event_type not in journey.completed_transits:
                    journey.completed_transits.add(event_type)
                    journey.last_terminal_zone = current_zone
                    journey.last_event_time = frame_time_s

                    # Calculate dwell in intermediate zones if any
                    queue_dwell = 0.0
                    for z_name, d_info in journey.dwell_states.items():
                        if "QUEUE" in z_name.upper() or "DWELL" in z_name.upper():
                            queue_dwell += d_info.get("total_dwell", 0.0)

                    evt_id = f"evt-{self.event_counter:04d}"
                    self.event_counter += 1
                    events.append({
                        "event_id": evt_id,
                        "event_type": event_type,
                        "track_id": track_id,
                        "source_zone": origin,
                        "destination_zone": current_zone,
                        "frame_time_s": frame_time_s,
                        "attributes": {
                            "source_zone": origin,
                            "destination_zone": current_zone,
                            "intermediate_queue_dwell_s": round(queue_dwell, 2),
                            "visited_zones": [z[0] for z in journey.visited_zones],
                        },
                    })

        # 4. Update Queue Occupancy
        active_in_queue = sum(
            1
            for j in self.journeys.values()
            if j.current_zone and "QUEUE" in j.current_zone.upper()
        )
        self.active_queue_occupancy = active_in_queue
        self.queue_history.append((frame_time_s, active_in_queue))

        return events

    def get_dwell_summary(self) -> Dict[str, Any]:
        """Calculates global dwell analytics across all tracks."""
        summary = {}
        for tid, j in self.journeys.items():
            for z, d in j.dwell_states.items():
                dwell = d["total_dwell"]
                if z not in summary:
                    summary[z] = []
                summary[z].append({"track_id": tid, "dwell_seconds": round(dwell, 2)})
        return summary
