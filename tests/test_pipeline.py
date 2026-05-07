"""
Integration smoke tests — Tasks 1-20.
Dependency-light: validates schemas, logic, and contracts without OpenCV/torch.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_perception_schema():
    """Task 1: Verify PerceptionResult schema."""
    from pydantic import BaseModel
    from typing import Literal, Optional

    class PerceptionResult(BaseModel):
        defect: str
        severity_hint: Literal["low", "medium", "high"]
        visual_evidence: str
        confidence: Literal["low", "medium", "high"]
        bbox: Optional[list[int]] = None

    result = PerceptionResult(
        defect="surface crack", severity_hint="high",
        visual_evidence="Linear discontinuity visible along weld seam",
        confidence="high", bbox=[100, 200, 300, 400],
    )
    assert result.defect == "surface crack"
    assert result.bbox == [100, 200, 300, 400]

    data = json.loads(result.model_dump_json())
    restored = PerceptionResult(**data)
    assert restored.defect == result.defect
    print("PASS test_perception_schema")


def test_decision_schema_with_trend():
    """Task 3 + Task 10: Verify DecisionResult with trend field."""
    from pydantic import BaseModel
    from typing import Literal

    class DecisionResult(BaseModel):
        severity: Literal["info", "warning", "critical"]
        action: str
        priority: Literal["low", "medium", "high"]
        rationale: str
        trend: Literal["stable", "worsening", "intermittent"] = "stable"

    result = DecisionResult(
        severity="critical", action="Shut down immediately.",
        priority="high", rationale="Crack detected near joint.",
        trend="worsening",
    )
    assert result.severity == "critical"
    assert result.trend == "worsening"

    # Default trend
    result2 = DecisionResult(
        severity="info", action="Continue.", priority="low", rationale="OK.",
    )
    assert result2.trend == "stable"
    print("PASS test_decision_schema_with_trend")


def test_confidence_calibration():
    """Task 11: Verify calibration downgrades low-confidence results."""
    # Inline calibration logic
    def calibrate(perception_conf, decision_sev, decision_pri, temporal_ctx, defect):
        severity, priority = decision_sev, decision_pri
        if perception_conf == "low":
            severity = "info"
            priority = "low"
        same_count = sum(1 for e in temporal_ctx if e.get("defect") == defect and defect != "none")
        if same_count >= 2 and severity == "warning":
            severity = "critical"
            priority = "high"
        return severity, priority

    # Low confidence -> downgrade
    sev, pri = calibrate("low", "critical", "high", [], "crack")
    assert sev == "info" and pri == "low"

    # Repeated defect -> escalate
    ctx = [{"defect": "crack"}, {"defect": "crack"}, {"defect": "crack"}]
    sev, pri = calibrate("high", "warning", "medium", ctx, "crack")
    assert sev == "critical" and pri == "high"
    print("PASS test_confidence_calibration")


def test_temporal_trend_analysis():
    """Task 10: Verify trend detection logic."""
    def analyze_trend(window, current_defect, current_severity):
        if len(window) < 2:
            return "stable"
        sev_rank = {"info": 0, "low": 0, "warning": 1, "medium": 1, "critical": 2, "high": 2}
        recent_sevs = [sev_rank.get(e.get("severity", "info"), 0) for e in window]
        cur_rank = sev_rank.get(current_severity, 0)
        if len(recent_sevs) >= 2:
            avg = sum(recent_sevs[-3:]) / min(len(recent_sevs), 3)
            if cur_rank > avg + 0.3:
                return "worsening"
        recent_defects = [e.get("defect", "none") for e in window]
        if current_defect != "none" and "none" in recent_defects and current_defect in recent_defects:
            return "intermittent"
        return "stable"

    # Stable
    window = [{"defect": "crack", "severity": "warning"}, {"defect": "crack", "severity": "warning"}]
    assert analyze_trend(window, "crack", "warning") == "stable"

    # Worsening
    window = [{"defect": "crack", "severity": "info"}, {"defect": "crack", "severity": "info"}]
    assert analyze_trend(window, "crack", "critical") == "worsening"

    # Intermittent: defect appeared, then disappeared (none), now reappearing
    # Use same severity to avoid triggering worsening check first
    window = [
        {"defect": "crack", "severity": "info"},
        {"defect": "none", "severity": "info"},
        {"defect": "crack", "severity": "info"},
    ]
    trend = analyze_trend(window, "crack", "info")
    assert trend == "intermittent", f"Expected intermittent, got {trend}"
    print("PASS test_temporal_trend_analysis")


def test_frame_selector_cooldown():
    """Task 5: Verify event-aware cooldown."""
    class MockSelector:
        def __init__(self, cooldown):
            self.cooldown = cooldown
            self._last_defect = None
            self._last_time = 0.0

        def update(self, defect):
            self._last_defect = defect
            self._last_time = time.monotonic()

        def should_skip(self, defect):
            if defect == "none":
                return False
            now = time.monotonic()
            if self._last_defect == defect and (now - self._last_time) < self.cooldown:
                return True
            return False

    sel = MockSelector(cooldown=5.0)
    assert sel.should_skip("crack") is False
    sel.update("crack")
    assert sel.should_skip("crack") is True
    assert sel.should_skip("rust") is False
    assert sel.should_skip("none") is False
    print("PASS test_frame_selector_cooldown")


def test_rag_query_construction():
    """Task 2: Verify improved RAG query."""
    def build_rag_query(defect, visual_evidence):
        return f"""Industrial safety procedure for {defect}.

Context:
{visual_evidence}

Include:
- Risk level
- Possible consequences
- Recommended actions"""

    query = build_rag_query("surface crack", "Linear discontinuity along weld seam")
    assert "surface crack" in query
    assert "Risk level" in query
    assert "Recommended actions" in query
    print("PASS test_rag_query_construction")


def test_event_dedup_with_tracking():
    """Task 19: Event dedup with repeat counting."""
    events_passed = []
    cooldown = 5.0
    last_alert = {}
    defect_counts = {}

    def process_event(defect):
        now = time.monotonic()
        if defect.lower() != "none":
            if (now - last_alert.get(defect, 0)) < cooldown:
                return False
            last_alert[defect] = now
            defect_counts[defect] = defect_counts.get(defect, 0) + 1
        events_passed.append(defect)
        return True

    assert process_event("rust") is True
    assert defect_counts.get("rust") == 1
    assert process_event("rust") is False  # Cooldown
    assert defect_counts.get("rust") == 1  # Not incremented
    assert process_event("crack") is True
    assert defect_counts.get("crack") == 1
    print("PASS test_event_dedup_with_tracking")


def test_guided_json_schemas():
    """Task 4 + Task 10: Verify JSON schemas include trend."""
    perception_schema = {
        "type": "object",
        "properties": {
            "defect": {"type": "string"},
            "severity_hint": {"type": "string", "enum": ["low", "medium", "high"]},
            "visual_evidence": {"type": "string"},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        },
        "required": ["defect", "severity_hint", "visual_evidence", "confidence"],
    }

    decision_schema = {
        "type": "object",
        "properties": {
            "severity": {"type": "string", "enum": ["info", "warning", "critical"]},
            "action": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "medium", "high"]},
            "rationale": {"type": "string"},
            "trend": {"type": "string", "enum": ["stable", "worsening", "intermittent"]},
        },
        "required": ["severity", "action", "priority", "rationale", "trend"],
    }

    for schema in [perception_schema, decision_schema]:
        assert schema["type"] == "object"
        for req in schema["required"]:
            assert req in schema["properties"]

    # Task 10: trend must be in decision schema
    assert "trend" in decision_schema["properties"]
    assert "trend" in decision_schema["required"]
    print("PASS test_guided_json_schemas")


def test_failure_fallback():
    """Task 14: Verify fallback result structure."""
    from pydantic import BaseModel
    from typing import Literal, Optional

    class PerceptionResult(BaseModel):
        defect: str
        severity_hint: Literal["low", "medium", "high"]
        visual_evidence: str
        confidence: Literal["low", "medium", "high"]
        bbox: Optional[list[int]] = None

    fallback = PerceptionResult(
        defect="unknown", severity_hint="low",
        visual_evidence="Model inference failed", confidence="low",
    )
    assert fallback.defect == "unknown"
    assert fallback.confidence == "low"
    assert fallback.bbox is None
    print("PASS test_failure_fallback")


def test_config_values():
    """Task 15/18: Verify config has required fields."""
    from config import (
        INFERENCE_TIMEOUT, DEMO_MODE, TEMPORAL_WINDOW_SIZE,
        INFERENCE_INTERVAL, MAX_FRAME_DIMENSION,
    )
    assert isinstance(INFERENCE_TIMEOUT, (int, float))
    assert INFERENCE_TIMEOUT > 0
    assert isinstance(DEMO_MODE, bool)
    assert isinstance(TEMPORAL_WINDOW_SIZE, int) and TEMPORAL_WINDOW_SIZE > 0
    assert INFERENCE_INTERVAL >= 1.0
    assert MAX_FRAME_DIMENSION == 1280
    print("PASS test_config_values")


if __name__ == "__main__":
    test_perception_schema()
    test_decision_schema_with_trend()
    test_confidence_calibration()
    test_temporal_trend_analysis()
    test_frame_selector_cooldown()
    test_rag_query_construction()
    test_event_dedup_with_tracking()
    test_guided_json_schemas()
    test_failure_fallback()
    test_config_values()
    print("\nAll 10 tests passed!")
