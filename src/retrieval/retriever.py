# retrieval/retriever.py
"""
Vector-based document retriever using FAISS and Sentence Transformers.
Model is loaded lazily on first use to reduce startup memory.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Module-level singleton — shared across all Retriever instances
_shared_model = None

def _get_model(model_name: str):
    """Load the embedding model once and reuse it (lazy singleton)."""
    global _shared_model
    if _shared_model is None:
        logger.info(f"Lazy-loading embedding model: {model_name}")
        try:
            import faiss  # noqa: F401 — verify faiss is available
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Required packages not installed. "
                "Run: pip install faiss-cpu sentence-transformers"
            ) from e
        _shared_model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded.")
    return _shared_model


class Retriever:
    """
    Retrieves relevant document chunks based on semantic similarity.
    Embedding model is loaded on the first retrieve() call, not at init.
    """

    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        score_threshold: float = 0.5,
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model_name = model_name
        self.top_k = top_k
        self.score_threshold = score_threshold
        self._model = None  # loaded lazily

        if not Path(index_path).exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        if not Path(metadata_path).exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        # Load FAISS index (lightweight — just reads the file)
        import faiss
        logger.info(f"Loading FAISS index from {index_path}")
        self.index: faiss.Index = faiss.read_index(index_path)

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata: List[Dict] = json.load(f)
        logger.info(f"Loaded {len(self.metadata)} document chunks (model not yet loaded)")

        if len(self.metadata) != self.index.ntotal:
            logger.warning(
                f"Metadata count ({len(self.metadata)}) != "
                f"Index size ({self.index.ntotal}). Data may be inconsistent."
            )

    @property
    def model(self):
        """Lazy property: load model on first access."""
        if self._model is None:
            self._model = _get_model(self.model_name)
        return self._model

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        query = query.strip()
        k = top_k or self.top_k

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = np.array(query_embedding, dtype="float32")

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            similarity_score = 1.0 / (1.0 + float(distance))
            if similarity_score >= self.score_threshold:
                chunk = dict(self.metadata[idx])
                chunk["similarity_score"] = similarity_score
                chunk["distance"] = float(distance)
                results.append(chunk)

        logger.info(f"Retrieved {len(results)} chunks (threshold: {self.score_threshold:.2f})")
        return results

    def retrieve_with_scores(self, query: str, top_k: Optional[int] = None) -> Tuple[List[Dict], List[float]]:
        chunks = self.retrieve(query, top_k)
        scores = [c.get("similarity_score", 0.0) for c in chunks]
        clean_chunks = [{k: v for k, v in c.items() if k != "similarity_score"} for c in chunks]
        return clean_chunks, scores

    def get_stats(self) -> Dict:
        return {
            "total_chunks": len(self.metadata),
            "index_size": self.index.ntotal,
            "index_type": type(self.index).__name__,
            "embedding_dim": self.index.d,
            "model_name": self.model_name,
            "model_loaded": self._model is not None,
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
        }