"""
Video Ingestion + Frame Abstraction Layer.

Wraps OpenCV VideoCapture to provide a generator-based frame source
from webcam, video file, or RTSP stream.
"""

import time
import logging
from typing import Generator, Optional

import cv2
import numpy as np

from config import MAX_FRAME_DIMENSION

logger = logging.getLogger(__name__)


def _resize_frame(frame: np.ndarray, max_dim: int = MAX_FRAME_DIMENSION) -> np.ndarray:
    """Resize frame so longest side ≤ max_dim, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


class FrameGrabber:
    """
    Provides frames from a video source.

    Args:
        source: int (webcam index), str (file path or RTSP URL)
    """

    def __init__(self, source: int | str = 0):
        self.source = source
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """Open the video source. Returns True if successful."""
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            return False
        logger.info(f"Opened video source: {self.source}")
        return True

    def release(self):
        """Release the video source."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        """Return the native FPS of the source (0 if unknown)."""
        if self._cap is None:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def frame_count(self) -> int:
        """Total frame count (0 for live streams)."""
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_snapshot(self) -> Optional[np.ndarray]:
        """Grab a single frame. Returns None on failure."""
        if self._cap is None or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        return _resize_frame(frame)

    def yield_frames(
        self, resize: bool = True
    ) -> Generator[tuple[float, np.ndarray], None, None]:
        """
        Generator yielding (timestamp_seconds, frame_ndarray) at capture rate.

        For video files: yields at native FPS pacing.
        For live streams: yields as fast as capture allows.
        """
        if self._cap is None:
            if not self.open():
                return

        native_fps = self.fps
        frame_delay = 1.0 / native_fps if native_fps > 0 else 0.033

        while True:
            t_start = time.monotonic()

            ret, frame = self._cap.read()
            if not ret:
                logger.info("End of video stream or read failure.")
                break

            timestamp = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            if resize:
                frame = _resize_frame(frame)

            yield (timestamp, frame)

            # Pace to native FPS for file playback
            elapsed = time.monotonic() - t_start
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.release()
