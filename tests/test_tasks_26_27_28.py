#!/usr/bin/env python3
"""
Task 26 — RAG Stress Test + Task 27 — Comparative Reasoning Test
Task 28 — Latency Check with SOP context injected

Tests:
  1. Query "bright light flare" → must retrieve sop_charging_flares, NOT generic lighting
  2. Query "person near furnace" → must retrieve sop_personnel_zones
  3. Compare reasoning output with vs without SOP context
  4. Measure latency impact of RAG context on inference
"""

import sys
import json
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU for embeddings

from core.rag import KnowledgeStore
from core.perception import PerceptionResult
from core.reasoning import ReasoningEngine

def divider(title: str):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")

def main():
    # ── Load knowledge store ──
    divider("TASK 26 — RAG STRESS TEST")
    
    store = KnowledgeStore()
    t0 = time.perf_counter()
    loaded = store.load()
    t_load = time.perf_counter() - t0
    print(f"Index loaded: {loaded} ({t_load:.3f}s)")
    
    if not loaded:
        print("FATAL: FAISS index not found. Run index_knowledge.py first.")
        sys.exit(1)

    # ── Test 1: "bright light flare" must pull charging flares SOP ──
    print("\n--- Test 1: Query 'bright light flare during scrap charging' ---")
    query1 = store.build_rag_query(
        "thermal flare", 
        "Extreme white light saturating camera, visible sparks radiating from furnace opening"
    )
    t0 = time.perf_counter()
    results1 = store.query(query1, top_k=3)
    t_rag1 = time.perf_counter() - t0
    
    found_flares = any("SOP-Charging-Flares" in r or "charging flare" in r.lower() or "thermal reaction" in r.lower() for r in results1)
    found_generic_lighting = any("lighting guide" in r.lower() for r in results1)
    
    print(f"  Query time: {t_rag1:.4f}s")
    print(f"  ✅ Retrieved SOP-Charging-Flares: {found_flares}")
    print(f"  ✅ No generic lighting guide: {not found_generic_lighting}")
    for i, chunk in enumerate(results1):
        # Show first 150 chars of each chunk
        preview = chunk[:150].replace('\n', ' ')
        print(f"  Result {i+1}: {preview}...")
    
    # ── Test 2: "person near furnace" must pull personnel zones SOP ──
    print("\n--- Test 2: Query 'person detected near furnace' ---")
    query2 = store.build_rag_query(
        "personnel proximity violation",
        "Human figure visible within 10 meters of active furnace during charging phase"
    )
    t0 = time.perf_counter()
    results2 = store.query(query2, top_k=3)
    t_rag2 = time.perf_counter() - t0
    
    found_zones = any("SOP-Personnel-Zones" in r or "exclusion zone" in r.lower() or "zone a" in r.lower() for r in results2)
    
    print(f"  Query time: {t_rag2:.4f}s")
    print(f"  ✅ Retrieved SOP-Personnel-Zones: {found_zones}")
    for i, chunk in enumerate(results2):
        preview = chunk[:150].replace('\n', ' ')
        print(f"  Result {i+1}: {preview}...")

    # ── Test 3: "smoke density" must pull smoke SOP ──
    print("\n--- Test 3: Query 'dense smoke from furnace' ---")
    query3 = store.build_rag_query(
        "excessive smoke emission",
        "Dense gray smoke obscuring visibility in furnace bay, escaping from roof openings"
    )
    t0 = time.perf_counter()
    results3 = store.query(query3, top_k=3)
    t_rag3 = time.perf_counter() - t0
    
    found_smoke = any("SOP-Smoke-Density" in r or "smoke" in r.lower() and "fume" in r.lower() for r in results3)
    
    print(f"  Query time: {t_rag3:.4f}s")
    print(f"  ✅ Retrieved SOP-Smoke-Density: {found_smoke}")
    for i, chunk in enumerate(results3):
        preview = chunk[:150].replace('\n', ' ')
        print(f"  Result {i+1}: {preview}...")

    # ═══════════════════════════════════════════════════════════════
    divider("TASK 27 — COMPARATIVE REASONING TEST")
    # ═══════════════════════════════════════════════════════════════
    
    reasoning = ReasoningEngine()
    
    # Simulate a perception result for a thermal flare
    perception_flare = PerceptionResult(
        defect="extreme thermal flare",
        severity_hint="high",
        visual_evidence="Intense white light saturating camera sensor, visible spark shower from furnace opening, dense fume plume rising above extraction hood",
        confidence="high",
    )
    
    # ── WITHOUT SOP context ──
    print("\n--- Without SOP Context ---")
    t0 = time.perf_counter()
    decision_no_sop = reasoning.decide(perception_flare, sop_chunks=[], temporal_context=[])
    t_no_sop = time.perf_counter() - t0
    
    print(f"  Latency: {t_no_sop:.3f}s")
    print(f"  Severity: {decision_no_sop.severity}")
    print(f"  Action: {decision_no_sop.action}")
    print(f"  Rationale: {decision_no_sop.rationale}")
    print(f"  Trend: {decision_no_sop.trend}")
    
    # ── WITH SOP context (from RAG) ──
    print("\n--- With SOP Context (RAG-injected) ---")
    t0 = time.perf_counter()
    decision_with_sop = reasoning.decide(perception_flare, sop_chunks=results1, temporal_context=[])
    t_with_sop = time.perf_counter() - t0
    
    print(f"  Latency: {t_with_sop:.3f}s")
    print(f"  Severity: {decision_with_sop.severity}")
    print(f"  Action: {decision_with_sop.action}")
    print(f"  Rationale: {decision_with_sop.rationale}")
    print(f"  Trend: {decision_with_sop.trend}")
    
    # Check if SOP file is cited
    action_text = decision_with_sop.action + " " + decision_with_sop.rationale
    cites_sop = any(kw in action_text.lower() for kw in [
        "sop", "charging", "ventilation", "fume extraction", "extraction", "crane"
    ])
    print(f"\n  ✅ Cites SOP procedures: {cites_sop}")

    # ═══════════════════════════════════════════════════════════════
    divider("TASK 28 — LATENCY IMPACT CHECK")
    # ═══════════════════════════════════════════════════════════════
    
    print(f"\n  Reasoning WITHOUT SOP:  {t_no_sop:.3f}s")
    print(f"  Reasoning WITH SOP:    {t_with_sop:.3f}s")
    print(f"  Delta:                 {t_with_sop - t_no_sop:+.3f}s")
    print(f"  RAG retrieval time:    {max(t_rag1, t_rag2, t_rag3):.4f}s")
    
    total_with_rag = t_with_sop + t_rag1
    print(f"\n  Total (RAG + Reasoning with SOP): {total_with_rag:.3f}s")
    
    if total_with_rag < 2.0:
        print(f"  ✅ PASS: Total pipeline under 2.0s target ({total_with_rag:.3f}s)")
    else:
        print(f"  ⚠️  WARN: Total pipeline exceeds 2.0s target ({total_with_rag:.3f}s)")
    
    # ═══════════════════════════════════════════════════════════════
    divider("SUMMARY")
    # ═══════════════════════════════════════════════════════════════
    
    results = {
        "task_26_rag_stress_test": {
            "flare_query_correct": found_flares,
            "no_generic_lighting": not found_generic_lighting,
            "personnel_query_correct": found_zones,
            "smoke_query_correct": found_smoke,
            "avg_retrieval_ms": round((t_rag1 + t_rag2 + t_rag3) / 3 * 1000, 1),
        },
        "task_27_comparative_reasoning": {
            "without_sop_severity": decision_no_sop.severity,
            "without_sop_action": decision_no_sop.action,
            "with_sop_severity": decision_with_sop.severity,
            "with_sop_action": decision_with_sop.action,
            "cites_sop_procedures": cites_sop,
        },
        "task_28_latency": {
            "reasoning_no_sop_ms": round(t_no_sop * 1000),
            "reasoning_with_sop_ms": round(t_with_sop * 1000),
            "rag_retrieval_ms": round(t_rag1 * 1000, 1),
            "total_rag_plus_reasoning_ms": round(total_with_rag * 1000),
            "under_2s_target": total_with_rag < 2.0,
        },
    }
    
    output_path = Path("eval_output/tasks_26_27_28_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
