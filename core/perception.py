"""
Multimodal Perception Layer (Task 1 schema + Task 4 vLLM optimization).

Two backends behind a unified PerceptionEngine interface:
  A. vLLM (preferred) — OpenAI-compatible API with guided_json
  B. HF Transformers (fallback) — direct model loading
"""

import io
import base64
import json
import logging
from typing import Literal, Optional

import cv2
import numpy as np
from pydantic import BaseModel, ValidationError

from config import VLLM_BASE_URL, VLLM_API_KEY, MODEL_NAME, MAX_FRAME_DIMENSION

logger = logging.getLogger(__name__)

# =============================================================================
# Task 1: Updated Perception Schema
# =============================================================================

class PerceptionResult(BaseModel):
    """Structured output from visual inspection."""
    defect: str
    severity_hint: Literal["low", "medium", "high"]
    visual_evidence: str
    confidence: Literal["low", "medium", "high"]
    bbox_2d: Optional[list[int]] = None  # Add this


# Task 1: Updated system prompt — grounded observation, no cause guessing
PERCEPTION_SYSTEM_PROMPT = """You are an expert industrial visual inspector.

Analyze the image and identify any visible defects or anomalies.
Locate the defect in the image and return the coordinates in the format [x1, y1, x2, y2]. If multiple hazards exist, prioritize the most critical one.

IMPORTANT:
- Only report what is visually observable.
- Do NOT guess internal causes.
- Focus on clear, physical evidence.

Return ONLY a JSON object:
{
  "defect": "...",
  "severity_hint": "low|medium|high",
  "visual_evidence": "...",
  "confidence": "low|medium|high",
  "bbox_2d": [x1, y1, x2, y2]
}

If no defect is visible:
- defect = "none"
- severity_hint = "low"
- visual_evidence = "No visible anomalies detected"
- confidence = "high"
- bbox_2d = null
"""

# JSON schema for vLLM guided decoding
PERCEPTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "defect": {"type": "string"},
        "severity_hint": {"type": "string", "enum": ["low", "medium", "high"]},
        "visual_evidence": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "bbox_2d": {
            "type": ["array", "null"],
            "items": {"type": "integer"},
            "minItems": 4,
            "maxItems": 4
        }
    },
    "required": ["defect", "severity_hint", "visual_evidence", "confidence", "bbox_2d"],
}


def _frame_to_base64(frame: np.ndarray, max_dim: int = MAX_FRAME_DIMENSION) -> str:
    """Convert BGR frame → resized RGB JPEG → base64 data URL."""
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    _, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _parse_perception_json(raw: str) -> PerceptionResult:
    """Parse raw model output into validated PerceptionResult."""
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    data = json.loads(text)
    return PerceptionResult(**data)


# =============================================================================
# Backend A: vLLM (Task 4 — persistent client, system prompt caching)
# =============================================================================

class VLLMPerceptionBackend:
    """
    Uses vLLM's OpenAI-compatible API with guided_json for structured output.

    Task 4: Client initialized once, system prompt is fixed,
    only user message changes per request.
    """

    def __init__(self, base_url: str = VLLM_BASE_URL, api_key: str = VLLM_API_KEY):
        from openai import OpenAI
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._system_message = {"role": "system", "content": PERCEPTION_SYSTEM_PROMPT}
        logger.info(f"vLLM perception backend initialized: {base_url}")

    def analyze(self, frame: np.ndarray) -> PerceptionResult:
        """Send frame to vLLM and return structured perception result."""
        image_url = _frame_to_base64(frame)

        user_message = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "Inspect this image for defects."},
            ],
        }

        try:
            response = self._client.chat.completions.create(
                model=MODEL_NAME,
                messages=[self._system_message, user_message],
                max_tokens=512,
                temperature=0.1,
                extra_body={"guided_json": PERCEPTION_JSON_SCHEMA},
            )
            raw = response.choices[0].message.content
            return _parse_perception_json(raw)

        except Exception as e:
            logger.error(f"vLLM perception call failed: {e}")
            # Task 14: Return safe fallback instead of crashing
            return PerceptionResult(
                defect="unknown",
                severity_hint="low",
                visual_evidence="Model inference failed — vLLM backend error",
                confidence="low",
            )


# =============================================================================
# Backend B: HF Transformers (fallback)
# =============================================================================

class HFPerceptionBackend:
    """
    Direct model loading via Hugging Face Transformers.
    Used when vLLM is unavailable.
    """

    def __init__(self):
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        """Lazy-load the model on first call."""
        if self._model is not None:
            return

        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        logger.info(f"Loading {MODEL_NAME} via HF Transformers (fallback)...")
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self._processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
        logger.info("HF model loaded successfully.")

    def analyze(self, frame: np.ndarray) -> PerceptionResult:
        """Run perception via HF Transformers pipeline."""
        self._ensure_loaded()

        import torch
        from qwen_vl_utils import process_vision_info
        from PIL import Image

        # Convert BGR → RGB PIL Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        messages = [
            {"role": "system", "content": PERCEPTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": "Inspect this image for defects."},
                ],
            },
        ]

        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=512, temperature=0.1)

        # Trim prompt tokens from output
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        raw = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        try:
            return _parse_perception_json(raw)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"HF output parse failed, returning safe default. Raw: {raw[:200]}")
            return PerceptionResult(
                defect="none",
                severity_hint="low",
                visual_evidence="Model output could not be parsed",
                confidence="low",
            )


# =============================================================================
# Unified Engine
# =============================================================================

class PerceptionEngine:
    """
    Unified perception interface. Tries vLLM first, falls back to HF.
    """

    def __init__(self, backend: str = "vllm"):
        if backend == "vllm":
            try:
                self._backend = VLLMPerceptionBackend()
                self._backend_name = "vllm"
            except Exception as e:
                logger.warning(f"vLLM backend init failed ({e}), falling back to HF Transformers.")
                self._backend = HFPerceptionBackend()
                self._backend_name = "hf"
        else:
            self._backend = HFPerceptionBackend()
            self._backend_name = "hf"

        logger.info(f"PerceptionEngine active backend: {self._backend_name}")

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def analyze(self, frame: np.ndarray) -> PerceptionResult:
        """Run perception on a single frame. Task 14: never raises."""
        try:
            return self._backend.analyze(frame)
        except Exception as e:
            logger.error(f"PerceptionEngine.analyze failed: {e}")
            return PerceptionResult(
                defect="unknown",
                severity_hint="low",
                visual_evidence="Model inference failed",
                confidence="low",
            )
