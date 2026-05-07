import gradio as gr
import time
import json
import cv2
import sys
import os
from pathlib import Path

# Add project root to sys.path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from config import INFERENCE_INTERVAL
from core.frame_grabber import FrameGrabber
from core.frame_selector import FrameSelector
from core.perception import PerceptionEngine
from core.rag import KnowledgeStore
from core.reasoning import ReasoningEngine
from core.event_logic import EventManager

# CSS from Tasks F3 & F9 & F10 & F11 & F12 & F13
css = """
body, .gradio-container {
    background-color: #0b0d0f !important;
}
.monospace-text, textarea, input, span, div.markdown-text {
    font-family: monospace !important;
}
.priority-action {
    font-size: 1.1rem !important;
    font-weight: bold !important;
    min-height: 200px !important;
    max-height: 200px !important;
    overflow-y: auto !important;
}
.rationale-text textarea {
    font-size: 13px !important;
    line-height: 1.4 !important;
}
/* F11: Hide floating label on the image */
.gr-image .label-wrap {
    display: none !important;
}
/* F12: Add padding/margin to columns for breathing room */
.gap-column {
    padding: 0 10px;
}
/* F13: Banner styles */
.banner-normal {
    background-color: #1e293b;
    padding: 10px;
    border-radius: 5px;
    color: white;
}
.banner-critical {
    background-color: #b91c1c;
    padding: 10px;
    border-radius: 5px;
    color: white;
    font-weight: bold;
}
.banner-warning {
    background-color: #ca8a04;
    padding: 10px;
    border-radius: 5px;
    color: white;
    font-weight: bold;
}
"""

class AppState:
    paused = False
    event_history = []

state = AppState()

def annotate_frame(cv2_mod, frame, perception, decision, timestamp_sec: float):
    # From evaluate_video.py
    out = frame.copy()
    h, w = out.shape[:2]

    color_map = {
        "info":     (34, 197,  94),
        "warning":  (234, 179,  8),
        "critical": (239,  68, 68),
    }
    rgb = color_map.get(decision.severity, (107, 114, 128))
    bgr = (rgb[2], rgb[1], rgb[0])

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

    sev_tag = f"[{decision.severity.upper()}] {perception.defect}"
    (tw, th), _ = cv2_mod.getTextSize(sev_tag, cv2_mod.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2_mod.rectangle(out, (4, 4), (tw + 12, th + 14), bgr, -1)
    cv2_mod.putText(out, sev_tag, (8, th + 8), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    action_short = decision.action[:90]
    bar_h = 40
    overlay = out.copy()
    cv2_mod.rectangle(overlay, (0, h - bar_h), (w, h), bgr, -1)
    cv2_mod.addWeighted(overlay, 0.8, out, 0.2, 0, out)
    cv2_mod.putText(out, action_short, (8, h - 12), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    ts_str = f"t={timestamp_sec:.1f}s"
    cv2_mod.putText(out, ts_str, (w - 110, 24), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    return cv2_mod.cvtColor(out, cv2_mod.COLOR_BGR2RGB)


def run_inspection():
    video_path = "demo/foundry_clip.mp4"
    if not os.path.exists(video_path):
        video_path = "demo/FACTORY_VIDEO.mp4"

    print(f"Initializing with video {video_path}")
    
    perception_engine = PerceptionEngine(backend="vllm")
    knowledge_store = KnowledgeStore()
    rag_loaded = knowledge_store.load()
    reasoning_engine = ReasoningEngine()

    grabber = FrameGrabber(source=video_path)
    if not grabber.open():
        yield None, "🔴 ERROR: Video not found", [], {}, "", "", "{}", []
        return

    selector = FrameSelector()
    event_manager = EventManager(cooldown_seconds=5.0)

    last_inference_time = 0.0
    
    state.paused = False
    state.event_history = []
    
    yield None, "<div class='banner-normal monospace-text'>🟢 MONITORING: Initialization complete. Starting stream...</div>", [("Initializing...", "Priority Action")], {"INFO": 1.0}, "Waiting for frames...", "", "{}", []

    analysis_window = []
    latched_decision = None
    latched_time = 0.0
    latched_perception = None
    latched_sop = []

    for timestamp_sec, frame in grabber.yield_frames():
        while state.paused:
            time.sleep(0.2)
            
        now = time.monotonic()
        selected = selector.select(timestamp_sec, frame)
        if selected is None:
            continue
        if (now - last_inference_time) < INFERENCE_INTERVAL:
            continue

        last_inference_time = now
        loop_start = time.perf_counter()

        t_vis_start = time.perf_counter()
        perception = perception_engine.analyze(selected)
        t_vis = time.perf_counter() - t_vis_start

        t_rag_start = time.perf_counter()
        sop_chunks = []
        if perception.defect.lower() not in ("none", "unknown") and rag_loaded:
            query = knowledge_store.build_rag_query(perception.defect, perception.visual_evidence)
            sop_chunks = knowledge_store.query(query)
        t_rag = time.perf_counter() - t_rag_start

        t_rsn_start = time.perf_counter()
        temporal_ctx = event_manager.get_temporal_context()
        decision = reasoning_engine.decide(perception, sop_chunks, temporal_ctx)
        t_rsn = time.perf_counter() - t_rsn_start

        # F8: Analysis Smoothing
        analysis_window.append(decision)
        if len(analysis_window) > 3:
            analysis_window.pop(0)
            
        severity_order = {"critical": 3, "warning": 2, "info": 1, "none": 0, "unknown": 0}
        best_decision = max(analysis_window, key=lambda d: severity_order.get(d.severity.lower(), 0))

        # F6: Event Latching
        current_time = time.monotonic()
        current_sev_val = severity_order.get(best_decision.severity.lower(), 0)
        latched_sev_val = severity_order.get(latched_decision.severity.lower(), 0) if latched_decision else -1

        if (latched_decision is None or
            current_sev_val > latched_sev_val or
            (current_time - latched_time) > 7.0 or
            (perception.defect != latched_perception.defect and current_sev_val >= latched_sev_val)):
            
            latched_decision = best_decision
            latched_time = current_time
            latched_perception = perception
            latched_sop = sop_chunks

        time_elapsed = int(current_time - latched_time)
        action_data = [(latched_decision.action, f"Priority Action ({time_elapsed}s ago)")]
        severity_dict = {latched_decision.severity.upper(): 1.0}

        t_total = t_vis + t_rag + t_rsn
        event = event_manager.process(latched_perception, latched_decision)
        
        annotated_frame = annotate_frame(cv2, selected, latched_perception, latched_decision, timestamp_sec)

        if latched_perception.defect.lower() not in ("none", "unknown"):
            sev_class = "banner-critical" if latched_decision.severity.lower() == "critical" else "banner-warning"
            status_text = f"<div class='{sev_class} monospace-text'>🔴 ALERT: {latched_perception.defect.upper()}</div>"
        else:
            status_text = "<div class='banner-normal monospace-text'>🟢 MONITORING</div>"

        sop_markdown = "\\n\\n".join(latched_sop) if latched_sop else "No SOP relevant for current state."

        telemetry_dict = {
            "timestamp": timestamp_sec,
            "perception": latched_perception.model_dump(),
            "reasoning": latched_decision.model_dump(),
            "latency_sec": {
                "vision": round(t_vis, 3),
                "rag": round(t_rag, 3),
                "reasoning": round(t_rsn, 3),
                "total": round(t_total, 3)
            }
        }
        telemetry_json = json.dumps(telemetry_dict, indent=2)

        # F7: Add to Event Timeline
        thumb = cv2.resize(annotated_frame, (160, 120))
        caption = f"{timestamp_sec:.1f}s | {latched_decision.severity.upper()}"
        
        history_item = {
            "thumb": thumb,
            "caption": caption,
            "annotated_frame": annotated_frame,
            "status_text": status_text,
            "action_data": action_data,
            "severity_dict": severity_dict,
            "rationale": latched_decision.rationale,
            "sop_markdown": sop_markdown,
            "telemetry_json": telemetry_json
        }
        state.event_history.append(history_item)
        if len(state.event_history) > 100:
            state.event_history.pop(0)
            
        # F14: Auto-scroll timeline (reverse order)
        gallery_data = [(e["thumb"], e["caption"]) for e in reversed(state.event_history)]

        processing_time = time.perf_counter() - loop_start
        sleep_time = max(0, INFERENCE_INTERVAL - processing_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

        yield (
            annotated_frame,
            status_text,
            action_data,
            severity_dict,
            latched_decision.rationale,
            sop_markdown,
            telemetry_json,
            gallery_data
        )

    grabber.release()

with gr.Blocks() as demo:
    with gr.Row():
        toggle = gr.Radio(choices=["Simplified", "Technical"], value="Simplified", label="The Judge's Toggle")
    
    with gr.Row():
        gr.Markdown("# VULKAN INSIGHT")
        status_indicator = gr.HTML("<div class='banner-normal monospace-text'>🟢 MONITORING</div>")

    with gr.Row():
        with gr.Column(scale=1, elem_classes=["gap-column"]):
            resume_btn = gr.Button("Return to Live (Paused)", visible=False, variant="primary")
            timeline_gallery = gr.Gallery(label="Event Timeline", columns=1, height=600, object_fit="contain")

        with gr.Column(scale=3, elem_classes=["gap-column"]):
            gr.Markdown("### Live Industrial Feed", elem_classes=["monospace-text"])
            video_feed = gr.Image(interactive=False)
            
        with gr.Column(scale=2, elem_classes=["gap-column"]):
            priority_action = gr.HighlightedText(label="Priority Action", elem_classes=["priority-action", "monospace-text"])
            severity_level = gr.Label(label="Severity Level")
            rationale = gr.Textbox(label="Decision Rationale", lines=4, elem_classes=["monospace-text", "rationale-text"], interactive=False)

    with gr.Row(visible=False) as black_box:
        with gr.Tabs():
            with gr.Tab("Safety SOP"):
                sop_display = gr.Markdown(elem_classes=["monospace-text"])
            with gr.Tab("Telemetry"):
                telemetry_display = gr.Code(language="json", elem_classes=["monospace-text"])

    def update_visibility(mode):
        return gr.update(visible=(mode != "Simplified"))

    toggle.change(fn=update_visibility, inputs=toggle, outputs=black_box)

    def select_history_event(evt: gr.SelectData):
        state.paused = True
        idx = evt.index
        # Since gallery renders reversed history, we must match the index
        reversed_history = list(reversed(state.event_history))
        if idx < len(reversed_history):
            e = reversed_history[idx]
            return (
                e["annotated_frame"],
                e["status_text"],
                e["action_data"],
                e["severity_dict"],
                e["rationale"],
                e["sop_markdown"],
                e["telemetry_json"],
                gr.update(visible=True)
            )
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()

    def resume_live():
        state.paused = False
        return gr.update(visible=False)

    timeline_gallery.select(
        fn=select_history_event,
        outputs=[video_feed, status_indicator, priority_action, severity_level, rationale, sop_display, telemetry_display, resume_btn]
    )
    
    resume_btn.click(
        fn=resume_live,
        outputs=[resume_btn]
    )

    demo.load(
        fn=run_inspection,
        outputs=[video_feed, status_indicator, priority_action, severity_level, rationale, sop_display, telemetry_display, timeline_gallery]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, css=css, theme=gr.themes.Monochrome())
