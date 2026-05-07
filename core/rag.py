"""
Knowledge Integration Layer — RAG (Task 2).

FAISS vector store with sentence-transformers embeddings.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from config import EMBEDDING_MODEL, RAG_TOP_K, FAISS_INDEX_PATH, CHUNK_MAX_TOKENS

logger = logging.getLogger(__name__)


class KnowledgeStore:
    """
    Vector store for SOP documents.
    Embeds with sentence-transformers (GPU), stores in FAISS CPU index.
    """

    def __init__(self, embedding_model: str = EMBEDDING_MODEL,
                 index_path: Path = FAISS_INDEX_PATH, top_k: int = RAG_TOP_K):
        self.embedding_model_name = embedding_model
        self.index_path = Path(index_path)
        self.top_k = top_k
        self._model = None
        self._index = None
        self._chunks: list[str] = []

    def _ensure_model(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(self.embedding_model_name, device=device)
        logger.info(f"Embedding model loaded on {device}")

    def _chunk_text(self, text: str, max_chars: int = CHUNK_MAX_TOKENS * 4) -> list[str]:
        paragraphs = text.split("\n\n")
        chunks, current = [], ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) + 2 > max_chars and current:
                chunks.append(current.strip())
                current = para
            else:
                current = current + "\n\n" + para if current else para
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def ingest(self, documents: list[str]):
        import faiss
        self._ensure_model()
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self._chunk_text(doc))
        if not all_chunks:
            return
        logger.info(f"Embedding {len(all_chunks)} chunks...")
        embeddings = self._model.encode(all_chunks, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        self._chunks = all_chunks
        self.index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path / "index.faiss"))
        with open(self.index_path / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)
        logger.info(f"Indexed {len(all_chunks)} chunks")

    def load(self) -> bool:
        import faiss
        idx_f = self.index_path / "index.faiss"
        chk_f = self.index_path / "chunks.pkl"
        if not idx_f.exists() or not chk_f.exists():
            return False
        self._index = faiss.read_index(str(idx_f))
        with open(chk_f, "rb") as f:
            self._chunks = pickle.load(f)
        logger.info(f"Loaded index: {self._index.ntotal} vectors")
        return True

    def query(self, text: str, top_k: Optional[int] = None) -> list[str]:
        if self._index is None and not self.load():
            return []
        self._ensure_model()
        k = top_k or self.top_k
        qv = self._model.encode([text], normalize_embeddings=True).astype(np.float32)
        _, indices = self._index.search(qv, k)
        seen, results = set(), []
        for idx in indices[0]:
            if 0 <= idx < len(self._chunks) and self._chunks[idx] not in seen:
                seen.add(self._chunks[idx])
                results.append(self._chunks[idx])
        return results

    def build_rag_query(self, defect: str, visual_evidence: str) -> str:
        """Task 2: Improved query construction from perception output."""
        return f"""Industrial safety procedure for {defect}.

Context:
{visual_evidence}

Include:
- Risk level
- Possible consequences
- Recommended actions"""
