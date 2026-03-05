"""
knowledge/embedder.py — Embedding utilities for the knowledge layer.

Wraps sentence-transformers with a word-overlap fallback and exposes a
chromadb-compatible embedding function for use in KnowledgeManager.
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ── Optional sentence-transformers ────────────────────────────────────────────

try:
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction as _STEF
    _CHROMA_ST_AVAILABLE = True
except ImportError:
    _CHROMA_ST_AVAILABLE = False
    _STEF = None  # type: ignore[assignment, misc]


def make_chroma_embedding_fn(model_name: str = "all-MiniLM-L6-v2") -> Any:
    """
    Return a chromadb-compatible embedding function.

    Tries SentenceTransformerEmbeddingFunction first; falls back to
    DeterministicEmbeddingFn (no model required) when unavailable.
    """
    if _CHROMA_ST_AVAILABLE and _STEF is not None:
        try:
            return _STEF(model_name=model_name)
        except Exception as exc:
            logger.debug("Could not load SentenceTransformerEmbeddingFunction: %s", exc)
    return DeterministicEmbeddingFn()


class DeterministicEmbeddingFn:
    """
    Character-frequency embedding function — no model download required.

    Suitable for tests and environments without sentence-transformers.
    Produces 64-dimensional vectors based on character frequencies,
    giving some semantic signal for texts sharing common words.
    """

    DIMS = 64
    _CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?-_"

    def name(self) -> str:  # required by newer chromadb versions
        return "deterministic-char-freq"

    def is_legacy(self) -> bool:  # suppress chromadb DeprecationWarning
        return False

    def embed_documents(self, input: list[str]) -> list[list[float]]:  # newer chromadb upsert path
        return [self._embed(text) for text in input]

    def embed_query(self, input: list[str]) -> list[list[float]]:  # newer chromadb query path
        return [self._embed(text) for text in input]

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> list[float]:
        t = text.lower()
        n = max(1, len(t))
        # Character-frequency component (len(CHARS) dims)
        freq = [t.count(c) / n for c in self._CHARS]
        # Hash-based component to fill remaining dims
        h = hashlib.md5(text.encode()).digest()
        hash_dims = [b / 255.0 for b in h]
        combined = freq + hash_dims
        # Pad or truncate to DIMS
        while len(combined) < self.DIMS:
            combined.append(0.0)
        return combined[: self.DIMS]
