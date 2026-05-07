"""
Central configuration for the Multimodal Inspection System.
All tunable parameters in one place.
"""

import os
from pathlib import Path

# Provide Hugging Face Token via environment variable to avoid unauthenticated request warnings
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
FAISS_INDEX_PATH = PROJECT_ROOT / "knowledge" / "faiss_index"
DB_PATH = PROJECT_ROOT / "inspection.db"
FRAME_ARCHIVE_DIR = PROJECT_ROOT / "frame_archive"
DEMO_VIDEO_PATH = PROJECT_ROOT / "demo" / "sample_video.mp4"

# Ensure directories exist
FRAME_ARCHIVE_DIR.mkdir(exist_ok=True)
FAISS_INDEX_PATH.parent.mkdir(exist_ok=True)
(PROJECT_ROOT / "demo").mkdir(exist_ok=True)

# =============================================================================
# vLLM / Model Serving
# =============================================================================
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")  # vLLM default
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")

# =============================================================================
# Frame Selection (Task 5 / Task 8)
# =============================================================================
FRAME_INTERVAL = 1.0          # Minimum seconds between emitted frames
MOTION_THRESHOLD = 5.0        # Mean absolute pixel diff to detect motion
EVENT_COOLDOWN = 25.0          # Seconds to suppress duplicate defect alerts (Task 5)
MAX_FRAME_DIMENSION = 1280    # Resize longest side before inference (Task 8)

# =============================================================================
# RAG (Task 2)
# =============================================================================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RAG_TOP_K = 3
CHUNK_MAX_TOKENS = 512

# =============================================================================
# Inference (Task 8 / Task 18)
# =============================================================================
INFERENCE_INTERVAL = 2.0      # Seconds between inference calls (1-3 range)
INFERENCE_TIMEOUT = 5.0       # Task 18: max seconds per inference call
USE_BF16 = True               # BFloat16 precision for Qwen2.5-VL

# =============================================================================
# Temporal Awareness (Task 10)
# =============================================================================
TEMPORAL_WINDOW_SIZE = 5      # Number of recent detections for trend analysis

# =============================================================================
# Demo Mode (Task 15)
# =============================================================================
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

# =============================================================================
# Persistence (Task 19)
# =============================================================================
EVENT_LOG_MAX = 50            # Max events shown in UI log
