"""
Event Logic Layer (Task 10: Temporal Awareness + Task 19: Logging Upgrade).

Interprets severity, triggers events, controls output frequency.
Includes cooldown deduplication, severity escalation, temporal context,
and enhanced logging with trend/repeat tracking.
"""

import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Literal, Optional

from core.perception import PerceptionResult
from core.reasoning import DecisionResult
from config import TEMPORAL_WINDOW_SIZE

logger = logging.getLogger(__name__)


@dataclass
class InspectionEvent:
    timestamp: float
    defect: str
    severity: str
    priority: str
    action: str
    rationale: str
    visual_evidence: str
    confidence: str
    trend: str = "stable"           # Task 10: stable/worsening/intermittent
    repeat_count: int = 0           # Task 19: how many times this defect was seen
    time_delta: Optional[float] = None  # Task 19: seconds since last event
    frame_path: Optional[str] = None


class EventManager:
    """
    Application-level event logic.
    - Cooldown: suppresses duplicate alerts within a window
    - Escalation: maps severity to signal level
    - Throttling: limits UI updates
    - Task 10: Temporal awareness — trend analysis over recent detections
    - Task 19: Enhanced logging — repeat counts, time deltas
    """

    def __init__(self, cooldown_seconds: float = 30.0, max_history: int = 50):
        self.cooldown_seconds = cooldown_seconds
        self.max_history = max_history
        self._history: deque[InspectionEvent] = deque(maxlen=max_history)
        self._last_alert: dict[str, float] = {}  # defect_type → timestamp

        # Task 10: Temporal window for trend analysis
        self._temporal_window: deque[dict] = deque(maxlen=TEMPORAL_WINDOW_SIZE)

        # Task 19: Defect occurrence counter
        self._defect_counts: dict[str, int] = {}

    # =========================================================================
    # Task 10: Temporal Context
    # =========================================================================

    def get_temporal_context(self) -> list[dict]:
        """Return recent detections for reasoning context."""
        return list(self._temporal_window)

    def _analyze_trend(self, current_defect: str, current_severity: str) -> str:
        """
        Task 10: Determine if the situation is stable, worsening, or intermittent.

        Logic:
        - stable: same defect, same severity, consistent
        - worsening: severity escalating or defect recurring faster
        - intermittent: defect type alternating (appears/disappears)
        """
        if len(self._temporal_window) < 2:
            return "stable"

        recent = list(self._temporal_window)
        severity_rank = {"info": 0, "low": 0, "warning": 1, "medium": 1, "critical": 2, "high": 2}

        # Check for escalation
        recent_severities = [severity_rank.get(e.get("severity", "info"), 0) for e in recent]
        current_rank = severity_rank.get(current_severity, 0)

        if len(recent_severities) >= 2:
            avg_recent = sum(recent_severities[-3:]) / min(len(recent_severities), 3)
            if current_rank > avg_recent + 0.3:
                return "worsening"

        # Check for intermittent (defect type changes)
        recent_defects = [e.get("defect", "none") for e in recent]
        unique_defects = set(d for d in recent_defects if d != "none")
        if current_defect != "none" and "none" in recent_defects and current_defect in recent_defects:
            return "intermittent"

        # Check if same defect repeating with increasing frequency
        same_defect_times = [
            e["timestamp"] for e in recent
            if e.get("defect") == current_defect
        ]
        if len(same_defect_times) >= 3:
            intervals = [same_defect_times[i] - same_defect_times[i-1]
                         for i in range(1, len(same_defect_times))]
            if len(intervals) >= 2 and intervals[-1] < intervals[0] * 0.7:
                return "worsening"

        return "stable"

    # =========================================================================
    # Core Event Processing
    # =========================================================================

    def process(self, perception: PerceptionResult,
                decision: DecisionResult,
                frame_path: Optional[str] = None) -> Optional[InspectionEvent]:
        """
        Create an event from perception + decision results.
        Returns None if the event is suppressed by cooldown.
        """
        now = time.time()

        # Cooldown check for non-trivial defects
        if perception.defect.lower() != "none":
            last_time = self._last_alert.get(perception.defect, 0)
            if (now - last_time) < self.cooldown_seconds:
                logger.debug(f"Event suppressed (cooldown): {perception.defect}")
                return None
            self._last_alert[perception.defect] = now

        # Task 10: Compute trend
        trend = self._analyze_trend(perception.defect, decision.severity)

        # Task 19: Update defect counter
        if perception.defect.lower() != "none":
            self._defect_counts[perception.defect] = self._defect_counts.get(perception.defect, 0) + 1
        repeat_count = self._defect_counts.get(perception.defect, 0)

        # Task 19: Compute time delta from last event
        time_delta = None
        if self._history:
            time_delta = now - self._history[0].timestamp

        event = InspectionEvent(
            timestamp=now,
            defect=perception.defect,
            severity=decision.severity,
            priority=decision.priority,
            action=decision.action,
            rationale=decision.rationale,
            visual_evidence=perception.visual_evidence,
            confidence=perception.confidence,
            trend=trend,
            repeat_count=repeat_count,
            time_delta=time_delta,
            frame_path=frame_path,
        )

        self._history.appendleft(event)

        # Task 10: Update temporal window
        self._temporal_window.append({
            "defect": perception.defect,
            "severity": decision.severity,
            "timestamp": now,
        })

        return event

    @property
    def recent_events(self) -> list[InspectionEvent]:
        return list(self._history)

    @staticmethod
    def severity_color(severity: str) -> str:
        return {"info": "#22c55e", "warning": "#eab308", "critical": "#ef4444"}.get(severity, "#6b7280")

    @staticmethod
    def severity_emoji(severity: str) -> str:
        return {"info": "✅", "warning": "⚠️", "critical": "🚨"}.get(severity, "ℹ️")

    @staticmethod
    def trend_emoji(trend: str) -> str:
        return {"stable": "→", "worsening": "↗", "intermittent": "↔"}.get(trend, "→")
