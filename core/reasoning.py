"""
Reasoning Engine — Decision Layer.
Task 3: Schema + prompt. Task 10: Temporal context + trend. Task 11: Confidence calibration.

Transforms perception output + SOP context + temporal history into decisions.
"""

import json
import logging
from typing import Literal, Optional

from pydantic import BaseModel

from config import VLLM_BASE_URL, VLLM_API_KEY, MODEL_NAME
from core.perception import PerceptionResult

logger = logging.getLogger(__name__)


# Task 10: Updated Decision Schema with trend
class DecisionResult(BaseModel):
    severity: Literal["info", "warning", "critical"]
    action: str
    priority: Literal["low", "medium", "high"]
    rationale: str
    trend: Literal["stable", "worsening", "intermittent"] = "stable"


DECISION_JSON_SCHEMA = {
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

# Task 10: Updated reasoning prompt with temporal context
REASONING_PROMPT_TEMPLATE = """You are an industrial safety engineer.

Input:
Detection:
{perception_json}

SOP Context:
{sop_chunks}

Previous observations:
{temporal_context}

Task:
Determine:
- Severity (info, warning, critical)
- Priority (low, medium, high)
- Trend (stable, worsening, intermittent) based on previous observations
- Recommended action (max 150 chars, ONE clear concise instruction)
- Short rationale (max 2 sentences)

Rules:
- Base decisions on visible evidence and SOP
- If the same defect appears repeatedly or severity increases, mark trend as "worsening"
- If the defect appears and disappears, mark trend as "intermittent"
- Do NOT hallucinate
- Keep rationale concise

Return ONLY JSON:
{{
  "severity": "...",
  "action": "...",
  "priority": "...",
  "rationale": "...",
  "trend": "..."
}}"""

REASONING_SYSTEM_PROMPT = "You are an industrial safety decision engine. Output ONLY valid JSON."


def _parse_decision_json(raw: str) -> DecisionResult:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    data = json.loads(text)
    # Normalize enum fields to lowercase (vLLM sometimes returns "WARNING", "Intermittent", etc.)
    for key in ("severity", "priority", "trend"):
        if key in data and isinstance(data[key], str):
            data[key] = data[key].lower()
    return DecisionResult(**data)


# =============================================================================
# Task 11: Confidence Calibration
# =============================================================================

def calibrate_decision(decision: DecisionResult, perception: PerceptionResult,
                       temporal_context: list[dict]) -> DecisionResult:
    """
    Task 11: Heuristic confidence calibration.

    - If perception confidence is low → downgrade severity to info
    - If same defect appears repeatedly → boost severity
    """
    severity = decision.severity
    priority = decision.priority

    # Downgrade on low confidence
    if perception.confidence == "low":
        severity = "info"
        priority = "low"
        logger.debug("Calibration: downgraded to info due to low confidence")

    # Boost if same defect repeats in temporal window
    if temporal_context:
        same_defect_count = sum(
            1 for e in temporal_context
            if e.get("defect") == perception.defect and perception.defect.lower() != "none"
        )
        if same_defect_count >= 2 and severity == "warning":
            severity = "critical"
            priority = "high"
            logger.debug(f"Calibration: escalated to critical (repeat count: {same_defect_count})")
        elif same_defect_count >= 3 and severity == "info":
            severity = "warning"
            priority = "medium"
            logger.debug(f"Calibration: escalated to warning (repeat count: {same_defect_count})")

    return DecisionResult(
        severity=severity,
        priority=priority,
        action=decision.action,
        rationale=decision.rationale,
        trend=decision.trend,
    )


class ReasoningEngine:
    """
    Combines perception + SOP context + temporal history to produce decisions.
    Uses vLLM text-only endpoint (no image) with guided_json.
    """

    def __init__(self):
        self._client = None
        self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI
            self._client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
            logger.info("Reasoning engine initialized (vLLM).")
        except Exception as e:
            logger.warning(f"Reasoning vLLM client init failed: {e}")
            self._client = None

    def decide(self, perception: PerceptionResult, sop_chunks: list[str],
               temporal_context: Optional[list[dict]] = None) -> DecisionResult:
        """Run reasoning with temporal context and confidence calibration."""
        temporal_context = temporal_context or []

        # Short-circuit for no-defect cases
        if perception.defect.lower() == "none":
            return DecisionResult(
                severity="info",
                action="Continue monitoring. No intervention needed.",
                priority="low",
                rationale="No defect detected in visual inspection.",
                trend="stable",
            )

        perception_json = perception.model_dump_json(indent=2)
        sop_text = "\n---\n".join(sop_chunks) if sop_chunks else "No SOP context available."

        # Task 10: Format temporal context for prompt
        if temporal_context:
            tc_str = json.dumps(temporal_context[-5:], indent=2, default=str)
        else:
            tc_str = "No previous observations."

        user_prompt = REASONING_PROMPT_TEMPLATE.format(
            perception_json=perception_json,
            sop_chunks=sop_text,
            temporal_context=tc_str,
        )

        if self._client is not None:
            decision = self._decide_vllm(user_prompt)
        else:
            decision = self._decide_fallback(perception)

        # Task 11: Apply confidence calibration
        decision = calibrate_decision(decision, perception, temporal_context)

        # F10: Ensure action is strictly limited to 150 chars
        if len(decision.action) > 150:
            decision.action = decision.action[:147] + "..."

        return decision

    def _decide_vllm(self, user_prompt: str) -> DecisionResult:
        try:
            response = self._client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=256,
                temperature=0.1,
                extra_body={"guided_json": DECISION_JSON_SCHEMA},
            )
            return _parse_decision_json(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Reasoning vLLM call failed: {e}")
            return DecisionResult(
                severity="warning",
                action="Manual review recommended — automated reasoning unavailable.",
                priority="medium",
                rationale="Reasoning model call failed; defaulting to manual review.",
                trend="stable",
            )

    def _decide_fallback(self, perception: PerceptionResult) -> DecisionResult:
        """Rule-based fallback when vLLM is unavailable."""
        severity_map = {"high": "critical", "medium": "warning", "low": "info"}
        return DecisionResult(
            severity=severity_map.get(perception.severity_hint, "warning"),
            action=f"Investigate detected defect: {perception.defect}",
            priority=perception.severity_hint,
            rationale=f"Based on visual evidence: {perception.visual_evidence[:100]}",
            trend="stable",
        )
