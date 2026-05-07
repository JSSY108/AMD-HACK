"""
Frame Selection — Efficiency Layer (Task 5 enhanced).

Protects GPU inference from overload via:
  - Time gating (min interval between emissions)
  - Motion detection (skip static scenes)
  - Event-aware cooldown (suppress duplicate defect detections)
"""

import time
import logging
from typing import Optional

import cv2
import numpy as np

from config import FRAME_INTERVAL, MOTION_THRESHOLD, EVENT_COOLDOWN

logger = logging.getLogger(__name__)


class FrameSelector:
    """
    Filters frames to reduce unnecessary inference calls.

    Gates:
      1. Time gate: enforces minimum interval between emitted frames.
      2. Motion gate: skips frames too similar to the last emitted frame.
      3. Event-aware cooldown (Task 5): suppresses re-analysis when the
         same defect type was recently detected.
    """

    def __init__(
        self,
        min_interval: float = FRAME_INTERVAL,
        motion_threshold: float = MOTION_THRESHOLD,
        event_cooldown: float = EVENT_COOLDOWN,
    ):
        self.min_interval = min_interval
        self.motion_threshold = motion_threshold
        self.event_cooldown = event_cooldown

        # Internal state
        self._last_emit_time: float = 0.0
        self._last_emitted_frame: Optional[np.ndarray] = None

        # Task 5: event-aware filtering
        self._last_detected_defect: Optional[str] = None
        self._last_event_timestamp: float = 0.0

    def _compute_motion(self, frame: np.ndarray) -> float:
        """Compute mean absolute pixel difference vs last emitted frame."""
        if self._last_emitted_frame is None:
            return float("inf")  # First frame always passes

        # Convert to grayscale for comparison
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(self._last_emitted_frame, cv2.COLOR_BGR2GRAY)

        # Resize to match if shapes differ
        if curr_gray.shape != prev_gray.shape:
            prev_gray = cv2.resize(prev_gray, (curr_gray.shape[1], curr_gray.shape[0]))

        diff = cv2.absdiff(curr_gray, prev_gray).astype(np.float32)
        return float(np.mean(diff))

    def update_event_state(self, defect: str):
        """
        Task 5: Called after inference to record what was detected.
        Used to suppress duplicate detections within the cooldown window.
        """
        self._last_detected_defect = defect
        self._last_event_timestamp = time.monotonic()

    def select(self, timestamp: float, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Decide whether this frame should proceed to inference.

        Returns:
            The frame if it passes all gates, None otherwise.
        """
        now = time.monotonic()

        # Gate 1: Time interval
        if (now - self._last_emit_time) < self.min_interval:
            return None

        # Gate 2: Motion detection
        motion = self._compute_motion(frame)
        if motion < self.motion_threshold:
            logger.debug(f"Frame skipped: low motion ({motion:.2f} < {self.motion_threshold})")
            return None

        # Update state
        self._last_emit_time = now
        self._last_emitted_frame = frame.copy()

        return frame

    def should_skip_event(self, detected_defect: str) -> bool:
        """
        Task 5: Check if this defect detection should be suppressed
        because the same defect was detected within the cooldown window.

        Called AFTER inference, before triggering downstream logic.
        """
        if detected_defect == "none":
            return False  # Never suppress "no defect" results

        now = time.monotonic()
        if (
            self._last_detected_defect == detected_defect
            and (now - self._last_event_timestamp) < self.event_cooldown
        ):
            logger.info(
                f"Event suppressed: '{detected_defect}' within cooldown "
                f"({now - self._last_event_timestamp:.1f}s < {self.event_cooldown}s)"
            )
            return True

        return False

    def reset(self):
        """Reset all internal state."""
        self._last_emit_time = 0.0
        self._last_emitted_frame = None
        self._last_detected_defect = None
        self._last_event_timestamp = 0.0
