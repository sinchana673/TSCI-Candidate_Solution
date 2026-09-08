"""
NOP Vision Intelligence Package - Candidate Solution
"""

from .detector import AdaptiveDetector, PretrainedDnnDetector, create_detector
from .tracker import RobustKalmanTracker, TrackState
from .zones import Zone, ZoneManager
from .event_engine import EventEngine, TrackJourney
from .evidence import EvidenceWriter
from .analytics import VMSAnalytics
from .pipeline import VisionPipeline

__all__ = [
    "AdaptiveDetector",
    "PretrainedDnnDetector",
    "create_detector",
    "RobustKalmanTracker",
    "TrackState",
    "Zone",
    "ZoneManager",
    "EventEngine",
    "TrackJourney",
    "EvidenceWriter",
    "VMSAnalytics",
    "VisionPipeline",
]
