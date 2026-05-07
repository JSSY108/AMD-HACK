"""
Knowledge Indexing Script.

Reads .md files from knowledge/, chunks, embeds, and saves FAISS index.
Run once before starting the inspection system.

Usage:
    python knowledge/index_knowledge.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import KNOWLEDGE_DIR
from core.rag import KnowledgeStore


def main():
    print("📚 Knowledge Base Indexing")
    print("=" * 50)

    # Collect all .md files
    md_files = list(KNOWLEDGE_DIR.glob("*.md"))
    if not md_files:
        print("❌ No .md files found in knowledge/")
        return

    print(f"Found {len(md_files)} document(s):")
    for f in md_files:
        print(f"  • {f.name}")

    # Read documents
    documents = []
    for f in md_files:
        documents.append(f.read_text(encoding="utf-8"))

    # Ingest into vector store
    store = KnowledgeStore()
    store.ingest(documents)

    # Verify
    print("\n✅ Indexing complete. Testing retrieval...")
    test_query = "surface crack on structural component"
    results = store.query(test_query, top_k=2)
    print(f"\nQuery: '{test_query}'")
    for i, chunk in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)

    print("\n✅ Knowledge base ready.")


if __name__ == "__main__":
    main()
