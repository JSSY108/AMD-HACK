#!/usr/bin/env python3
"""
Headless Video Evaluator — Tasks 21, 22, 23, 24.

Runs the full inspection pipeline (Frame Selection -> Perception -> RAG -> Reasoning)
on a video file without Gradio. Outputs annotated frames, JSONL log, latency metrics,
and an error log.

Usage:
    python tests/evaluate_video.py --video_path path/to/video.mp4
    python tests/evaluate_video.py --video_path path/to/video.mp4 --output_dir my_eval/
    python tests/evaluate_video.py --video_path path/to/video.mp4 --max_frames 20

Outputs (in --output_dir):
    run_log.jsonl       — One JSON object per processed frame
    errors.log          — Tracebacks from any failures
    frame_NNNN_sec.jpg  — Annotated frame images
    summary.txt         — Run summary with aggregate stats
"""

from __future__ import annotations

import sys
import json
import time
import argparse
import traceback
import logging
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Project root setup (stdlib only — no heavy deps at module level)
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("evaluator")


# =========================================================================
# Task 22 — Annotated frame rendering
# =========================================================================

def annotate_frame(cv2_mod, frame, perception, decision, timestamp_sec: float):
    """Draw bbox, severity label, and action text onto the frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    color_map = {
        "info":     (34, 197,  94),
        "warning":  (234, 179,  8),
        "critical": (239,  68, 68),
    }
    rgb = color_map.get(decision.severity, (107, 114, 128))
    bgr = (rgb[2], rgb[1], rgb[0])

    # Bounding box or fallback highlight
    if perception.defect.lower() not in ("none", "unknown"):
        if perception.bbox and len(perception.bbox) == 4:
            x1, y1, x2, y2 = [max(0, v) for v in perception.bbox]
            x2, y2 = min(w, x2), min(h, y2)
            cv2_mod.rectangle(out, (x1, y1), (x2, y2), bgr, 3)
        else:
            mx, my = w // 6, h // 6
            clen = min(w, h) // 8
            corners = [
                ((mx, my), (mx + clen, my)), ((mx, my), (mx, my + clen)),
                ((w - mx, my), (w - mx - clen, my)), ((w - mx, my), (w - mx, my + clen)),
                ((mx, h - my), (mx + clen, h - my)), ((mx, h - my), (mx, h - my - clen)),
                ((w - mx, h - my), (w - mx - clen, h - my)), ((w - mx, h - my), (w - mx, h - my - clen)),
            ]
            for p1, p2 in corners:
                cv2_mod.line(out, p1, p2, bgr, 3)

    # Top-left: severity label
    sev_tag = f"[{decision.severity.upper()}] {perception.defect}"
    (tw, th), _ = cv2_mod.getTextSize(sev_tag, cv2_mod.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2_mod.rectangle(out, (4, 4), (tw + 12, th + 14), bgr, -1)
    cv2_mod.putText(out, sev_tag, (8, th + 8), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Bottom bar: action text
    action_short = decision.action[:90]
    bar_h = 40
    overlay = out.copy()
    cv2_mod.rectangle(overlay, (0, h - bar_h), (w, h), bgr, -1)
    cv2_mod.addWeighted(overlay, 0.8, out, 0.2, 0, out)
    cv2_mod.putText(out, action_short, (8, h - 12), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Timestamp watermark
    ts_str = f"t={timestamp_sec:.1f}s"
    cv2_mod.putText(out, ts_str, (w - 110, 24), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    return out


# =========================================================================
# Task 23 — Latency formatter
# =========================================================================

def fmt_latency(label: str, seconds: float) -> str:
    return f"{label}: {seconds:.2f}s"


# =========================================================================
# Main evaluator (Tasks 21-24)
# =========================================================================

def evaluate(
    video_path: str,
    output_dir: str = "eval_output",
    max_frames: int | None = None,
):
    # ------------------------------------------------------------------
    # Deferred imports (heavy deps only loaded when actually running)
    # ------------------------------------------------------------------
    import cv2
    import numpy as np
    from config import INFERENCE_INTERVAL
    from core.frame_grabber import FrameGrabber
    from core.frame_selector import FrameSelector
    from core.perception import PerceptionEngine
    from core.rag import KnowledgeStore
    from core.reasoning import ReasoningEngine
    from core.event_logic import EventManager

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    jsonl_path   = out_path / "run_log.jsonl"
    error_path   = out_path / "errors.log"
    summary_path = out_path / "summary.txt"

    # Clear previous outputs
    for f in [jsonl_path, error_path, summary_path]:
        f.write_text("", encoding="utf-8")

    # ------------------------------------------------------------------
    # Initialise components
    # ------------------------------------------------------------------
    print("=" * 72)
    print("  MULTIMODAL INSPECTION — HEADLESS EVALUATOR")
    print(f"  Video  : {video_path}")
    print(f"  Output : {out_path.resolve()}")
    print("=" * 72)

    print("\n[1/4] Initialising perception engine...")
    t0 = time.perf_counter()
    perception_engine = PerceptionEngine(backend="vllm")
    print(f"       Backend: {perception_engine.backend_name}  ({time.perf_counter() - t0:.1f}s)")

    print("[2/4] Loading knowledge store...")
    t0 = time.perf_counter()
    knowledge_store = KnowledgeStore()
    rag_loaded = knowledge_store.load()
    print(f"       Index loaded: {rag_loaded}  ({time.perf_counter() - t0:.1f}s)")

    print("[3/4] Initialising reasoning engine...")
    t0 = time.perf_counter()
    reasoning_engine = ReasoningEngine()
    print(f"       Ready  ({time.perf_counter() - t0:.1f}s)")

    print("[4/4] Opening video...")
    grabber = FrameGrabber(source=video_path)
    if not grabber.open():
        print(f"FATAL: Cannot open video: {video_path}")
        sys.exit(1)
    total_frames = grabber.frame_count
    native_fps = grabber.fps
    duration_est = total_frames / native_fps if native_fps else 0
    print(f"       Frames: {total_frames}  FPS: {native_fps:.1f}  Duration: {duration_est:.1f}s")

    selector = FrameSelector()
    event_manager = EventManager(cooldown_seconds=5.0)  # shorter cooldown for eval

    # ------------------------------------------------------------------
    # Processing loop
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("  PROCESSING")
    print("-" * 72)

    processed_count = 0
    skipped_count = 0
    error_count = 0
    latencies: list[dict] = []
    last_inference_time = 0.0
    all_results: list[dict] = []

    for timestamp_sec, frame in grabber.yield_frames():
        now = time.monotonic()

        # --- Frame selection gate ---
        selected = selector.select(timestamp_sec, frame)
        if selected is None:
            skipped_count += 1
            continue
        if (now - last_inference_time) < INFERENCE_INTERVAL:
            skipped_count += 1
            continue

        last_inference_time = now

        if max_frames is not None and processed_count >= max_frames:
            print(f"\n  Reached --max_frames={max_frames}, stopping.")
            break

        # ============================================================
        # Task 24 — Failsafe wrapper
        # ============================================================
        try:
            ts_label = f"{int(timestamp_sec // 60):02d}:{int(timestamp_sec % 60):02d}"
            frame_tag = f"[Frame {timestamp_sec:6.1f}s | {ts_label}]"

            # --- Perception (Task 23: timed) ---
            t_vis_start = time.perf_counter()
            perception = perception_engine.analyze(selected)
            t_vis = time.perf_counter() - t_vis_start

            # --- RAG retrieval (Task 23: timed) ---
            t_rag_start = time.perf_counter()
            sop_chunks: list[str] = []
            if perception.defect.lower() not in ("none", "unknown") and rag_loaded:
                query = knowledge_store.build_rag_query(
                    perception.defect, perception.visual_evidence,
                )
                sop_chunks = knowledge_store.query(query)
            t_rag = time.perf_counter() - t_rag_start

            # --- Reasoning (Task 23: timed) ---
            temporal_ctx = event_manager.get_temporal_context()
            t_rsn_start = time.perf_counter()
            decision = reasoning_engine.decide(perception, sop_chunks, temporal_ctx)
            t_rsn = time.perf_counter() - t_rsn_start

            t_total = t_vis + t_rag + t_rsn

            # --- Event tracking ---
            event = event_manager.process(perception, decision)

            # --- Console output (Task 23) ---
            defect_str = perception.defect if perception.defect.lower() != "none" else "-"
            sev_icon = {"info": " ", "warning": "!", "critical": "X"}.get(decision.severity, "?")
            print(
                f"  {frame_tag}  [{sev_icon}] Defect: {defect_str:20s}  "
                f"| {fmt_latency('Vision', t_vis)} "
                f"| {fmt_latency('RAG', t_rag)} "
                f"| {fmt_latency('Reason', t_rsn)} "
                f"| Total: {t_total:.2f}s"
            )

            # --- Task 22: Save annotated frame ---
            annotated = annotate_frame(cv2, selected, perception, decision, timestamp_sec)
            frame_filename = f"frame_{int(timestamp_sec):04d}_sec.jpg"
            cv2.imwrite(str(out_path / frame_filename), annotated)

            # --- Task 22: Append to JSONL ---
            record = {
                "timestamp_sec": round(timestamp_sec, 2),
                "perception": perception.model_dump(),
                "rag_context_used": [c[:200] for c in sop_chunks],
                "reasoning": decision.model_dump(),
                "trend": decision.trend,
                "latency_ms": {
                    "vision": round(t_vis * 1000),
                    "rag": round(t_rag * 1000),
                    "reasoning": round(t_rsn * 1000),
                    "total": round(t_total * 1000),
                },
                "frame_file": frame_filename,
                "event_suppressed": event is None and perception.defect.lower() != "none",
            }
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            all_results.append(record)
            latencies.append({"vision": t_vis, "rag": t_rag, "reasoning": t_rsn, "total": t_total})
            processed_count += 1

        except Exception:
            # Task 24 — log error and continue
            error_count += 1
            tb = traceback.format_exc()
            logger.error(f"Frame {timestamp_sec:.1f}s failed:\n{tb}")
            with open(error_path, "a", encoding="utf-8") as ef:
                ef.write(f"\n{'='*60}\n")
                ef.write(f"Frame timestamp: {timestamp_sec:.2f}s\n")
                ef.write(f"Time: {datetime.now().isoformat()}\n")
                ef.write(tb)
            print(f"  {frame_tag}  ERROR — logged to errors.log")
            processed_count += 1
            continue

    grabber.release()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  EVALUATION COMPLETE")
    print("=" * 72)

    if latencies:
        avg_vis = sum(l["vision"] for l in latencies) / len(latencies)
        avg_rag = sum(l["rag"] for l in latencies) / len(latencies)
        avg_rsn = sum(l["reasoning"] for l in latencies) / len(latencies)
        avg_tot = sum(l["total"] for l in latencies) / len(latencies)
        max_tot = max(l["total"] for l in latencies)
        min_tot = min(l["total"] for l in latencies)
    else:
        avg_vis = avg_rag = avg_rsn = avg_tot = max_tot = min_tot = 0

    # Count defects
    defect_counts: dict[str, int] = {}
    severity_counts = {"info": 0, "warning": 0, "critical": 0}
    for r in all_results:
        d = r["perception"]["defect"]
        if d.lower() not in ("none", "unknown"):
            defect_counts[d] = defect_counts.get(d, 0) + 1
        sev = r["reasoning"]["severity"]
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    summary_lines = [
        f"Video:             {video_path}",
        f"Duration:          {duration_est:.1f}s",
        f"Total frames:      {total_frames}",
        f"Processed:         {processed_count}",
        f"Skipped:           {skipped_count}",
        f"Errors:            {error_count}",
        f"",
        f"--- Latency (seconds) ---",
        f"  Vision   avg={avg_vis:.2f}  ",
        f"  RAG      avg={avg_rag:.3f}  ",
        f"  Reason   avg={avg_rsn:.2f}  ",
        f"  Total    avg={avg_tot:.2f}  min={min_tot:.2f}  max={max_tot:.2f}",
        f"",
        f"--- Defects Detected ---",
    ]
    if defect_counts:
        for defect, count in sorted(defect_counts.items(), key=lambda x: -x[1]):
            summary_lines.append(f"  {defect}: {count}")
    else:
        summary_lines.append("  (none)")

    summary_lines.extend([
        f"",
        f"--- Severity Distribution ---",
        f"  info:     {severity_counts.get('info', 0)}",
        f"  warning:  {severity_counts.get('warning', 0)}",
        f"  critical: {severity_counts.get('critical', 0)}",
        f"",
        f"--- Output Files ---",
        f"  Log:       {jsonl_path.resolve()}",
        f"  Errors:    {error_path.resolve()}",
        f"  Frames:    {out_path.resolve()}/frame_*.jpg",
    ])

    summary_text = "\n".join(summary_lines)
    print(summary_text)
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"\n  Summary saved to: {summary_path.resolve()}")


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Headless video evaluator for the multimodal inspection pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/evaluate_video.py --video_path demo/sample.mp4
  python tests/evaluate_video.py --video_path demo/sample.mp4 --max_frames 10
  python tests/evaluate_video.py --video_path demo/sample.mp4 --output_dir results/run1/
        """,
    )
    parser.add_argument(
        "--video_path", type=str, required=True,
        help="Path to the input MP4 video file.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="eval_output",
        help="Directory to save results (default: eval_output/).",
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="Maximum number of frames to process (default: all).",
    )
    args = parser.parse_args()

    if not Path(args.video_path).exists():
        print(f"ERROR: Video file not found: {args.video_path}")
        sys.exit(1)

    evaluate(
        video_path=args.video_path,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
