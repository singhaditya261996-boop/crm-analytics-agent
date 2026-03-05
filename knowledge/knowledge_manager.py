"""
knowledge/knowledge_manager.py — Central knowledge layer for Module 15.

Manages three knowledge sources:
  1. Base files  — knowledge/base/*.txt (FM context, glossary, competitors, etc.)
  2. Scraped     — knowledge/scraped/**/*.txt (weekly competitor / industry news)
  3. Meeting notes — knowledge/meeting_notes/notes_YYYYMMDD.txt (post-call notes)

Public API
----------
KnowledgeManager(knowledge_dir, chroma_path)
    .load_all_knowledge()
    .get_relevant_context(question, intent_type) -> dict
    .add_meeting_note(note_text, session_date) -> Path
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from knowledge.embedder import make_chroma_embedding_fn

logger = logging.getLogger(__name__)

# Benchmark keyword → YAML top-level key
_BENCHMARK_MAP: dict[str, list[str]] = {
    "win rate":             ["win_rates", "win_rate"],
    "win rates":            ["win_rates", "win_rate"],
    "sales cycle":          ["sales_cycle_days", "sales_cycle"],
    "pipeline":             ["pipeline_health", "pipeline_coverage"],
    "pipeline coverage":    ["pipeline_health", "pipeline_coverage"],
    "coverage ratio":       ["pipeline_health", "pipeline_coverage"],
    "revenue concentration":["revenue_concentration"],
    "deal size":            ["deal_size_brackets", "average_deal_size"],
    "dormant":              ["account_health", "dormant_account_days"],
    "at risk":              ["account_health"],
    "bid cost":             ["pipeline_health"],
}

# Approximate chars per token — used for context trimming
_CHARS_PER_TOKEN = 4
_MAX_CONTEXT_TOKENS = 1000
_MAX_CONTEXT_CHARS = _MAX_CONTEXT_TOKENS * _CHARS_PER_TOKEN


class KnowledgeManager:
    """
    Manages all industry knowledge and injects relevant context into LLM prompts.

    Parameters
    ----------
    knowledge_dir : path to the knowledge/ folder (default: "knowledge")
    chroma_path   : where to persist the chromadb database
    _client       : inject a chromadb client for testing (e.g. EphemeralClient)
    _embedding_fn : inject a custom embedding function for testing
    """

    COLLECTION_NAME = "industry_knowledge"

    def __init__(
        self,
        knowledge_dir: str | Path = "knowledge",
        chroma_path: str | Path = "data/.cache/chroma",
        _client: Any = None,
        _embedding_fn: Any = None,
        _collection_name: str | None = None,
    ) -> None:
        self._knowledge_dir = Path(knowledge_dir)
        self._benchmarks_path = self._knowledge_dir / "benchmarks.yaml"

        # ChromaDB client
        if _client is not None:
            self._chroma = _client
        else:
            import chromadb
            Path(chroma_path).mkdir(parents=True, exist_ok=True)
            self._chroma = chromadb.PersistentClient(path=str(chroma_path))

        ef = _embedding_fn if _embedding_fn is not None else make_chroma_embedding_fn()
        collection_name = _collection_name or self.COLLECTION_NAME
        self._collection = self._chroma.get_or_create_collection(
            name=collection_name,
            embedding_function=ef,
        )

        self._benchmarks: dict = self._load_benchmarks()

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_all_knowledge(self) -> int:
        """
        Embed all knowledge sources into chromadb.
        Returns the total number of chunks indexed.
        """
        total = 0
        total += self._embed_base_files()
        total += self._embed_scraped_content()
        total += self._embed_meeting_notes()
        logger.info("KnowledgeManager: indexed %d chunks total", total)
        return total

    def get_relevant_context(
        self,
        question: str,
        intent_type: str = "aggregation",
    ) -> dict:
        """
        Retrieve relevant knowledge chunks for this question.

        Returns
        -------
        dict with keys:
          document_context  — formatted text from matching chunks (str)
          benchmark_context — formatted benchmark text if a match found (str | None)
          has_benchmark     — bool
        """
        # Semantic search (returns empty string if collection is empty)
        document_context = self._semantic_search(question)

        # Benchmark keyword matching
        benchmark_context = self._get_benchmark_context(question)

        return {
            "document_context": document_context,
            "benchmark_context": benchmark_context,
            "has_benchmark": benchmark_context is not None,
        }

    def add_meeting_note(self, note_text: str, session_date: str | None = None) -> Path:
        """
        Save a meeting note to disk and immediately embed it into chromadb.

        Returns the path to the saved file.
        """
        if session_date is None:
            session_date = datetime.now(timezone.utc).strftime("%Y%m%d")

        notes_dir = self._knowledge_dir / "meeting_notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        note_path = notes_dir / f"notes_{session_date}.txt"

        # Append to file if it already exists for that day
        mode = "a" if note_path.exists() else "w"
        with open(note_path, mode) as fh:
            if mode == "a":
                fh.write("\n\n---\n\n")
            fh.write(note_text)

        # Embed chunks
        chunks = self._chunk_text(note_text)
        self._upsert_chunks(
            chunks=chunks,
            id_prefix=f"note_{session_date}",
            metadata={
                "source_type": "meeting_notes",
                "date": session_date,
                "source_file": note_path.name,
            },
        )
        logger.info("Meeting note saved and indexed: %s (%d chunks)", note_path.name, len(chunks))
        return note_path

    # ── Embedding helpers ──────────────────────────────────────────────────────

    def _embed_base_files(self) -> int:
        base_path = self._knowledge_dir / "base"
        if not base_path.exists():
            return 0
        total = 0
        for txt_file in sorted(base_path.glob("*.txt")):
            try:
                text = txt_file.read_text(encoding="utf-8", errors="replace")
                chunks = self._chunk_text(text)
                self._upsert_chunks(
                    chunks=chunks,
                    id_prefix=f"base_{txt_file.stem}",
                    metadata={
                        "source_type": "base",
                        "source_file": txt_file.name,
                        "category": txt_file.stem,
                    },
                )
                total += len(chunks)
            except Exception as exc:
                logger.warning("Could not embed base file %s: %s", txt_file.name, exc)
        return total

    def _embed_scraped_content(self) -> int:
        scraped_dir = self._knowledge_dir / "scraped"
        if not scraped_dir.exists():
            return 0
        total = 0
        for txt_file in sorted(scraped_dir.rglob("*.txt")):
            try:
                text = txt_file.read_text(encoding="utf-8", errors="replace")
                # Infer source type from parent folder name
                source_type = txt_file.parent.name  # equans_news, competitor_news, etc.
                chunks = self._chunk_text(text)
                self._upsert_chunks(
                    chunks=chunks,
                    id_prefix=f"scraped_{source_type}_{txt_file.stem}",
                    metadata={
                        "source_type": source_type,
                        "source_file": txt_file.name,
                        "scraped_date": "",
                    },
                )
                total += len(chunks)
            except Exception as exc:
                logger.warning("Could not embed scraped file %s: %s", txt_file.name, exc)
        return total

    def _embed_meeting_notes(self) -> int:
        notes_dir = self._knowledge_dir / "meeting_notes"
        if not notes_dir.exists():
            return 0
        total = 0
        for txt_file in sorted(notes_dir.glob("notes_*.txt")):
            try:
                date_part = txt_file.stem.replace("notes_", "")
                text = txt_file.read_text(encoding="utf-8", errors="replace")
                chunks = self._chunk_text(text)
                self._upsert_chunks(
                    chunks=chunks,
                    id_prefix=f"note_{date_part}",
                    metadata={
                        "source_type": "meeting_notes",
                        "date": date_part,
                        "source_file": txt_file.name,
                    },
                )
                total += len(chunks)
            except Exception as exc:
                logger.warning("Could not embed meeting note %s: %s", txt_file.name, exc)
        return total

    def _upsert_chunks(
        self,
        chunks: list[str],
        id_prefix: str,
        metadata: dict,
    ) -> None:
        if not chunks:
            return
        ids = [f"{id_prefix}_{i}" for i in range(len(chunks))]
        metadatas = [metadata] * len(chunks)
        try:
            self._collection.upsert(
                documents=chunks,
                ids=ids,
                metadatas=metadatas,
            )
        except Exception as exc:
            logger.warning("Chroma upsert failed for %s: %s", id_prefix, exc)

    # ── Semantic search ────────────────────────────────────────────────────────

    def _semantic_search(self, question: str, n_results: int = 5) -> str:
        try:
            count = self._collection.count()
            if count == 0:
                return ""
            actual_n = min(n_results, count)
            results = self._collection.query(
                query_texts=[question],
                n_results=actual_n,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.debug("Chroma query failed: %s", exc)
            return ""

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not docs:
            return ""

        chunks: list[str] = []
        for doc, meta, dist in zip(docs, metas, distances):
            # cosine distance: 0 = identical, 2 = opposite
            # We want similarity > 0.7 → distance < 0.6  (1 - sim ≈ dist/2)
            if dist is not None and dist > 1.2:
                continue
            source = meta.get("source_file", "unknown")
            date = meta.get("date", "")
            date_str = f" ({date})" if date else ""
            chunks.append(f"[{source}{date_str}]\n{doc}")

        context = "\n\n---\n\n".join(chunks)
        # Trim to max context length, preferring to keep earlier (higher-quality) chunks
        return context[: _MAX_CONTEXT_CHARS]

    # ── Benchmark matching ─────────────────────────────────────────────────────

    def _get_benchmark_context(self, question: str) -> str | None:
        q_lower = question.lower()
        matched_keys: list[str] = []
        for keyword, yaml_keys in _BENCHMARK_MAP.items():
            if keyword in q_lower:
                matched_keys.extend(yaml_keys)

        if not matched_keys:
            return None

        sections: list[str] = []
        seen: set[str] = set()
        for key in matched_keys:
            if key in seen or key not in self._benchmarks:
                continue
            seen.add(key)
            sections.append(self._format_benchmark(key, self._benchmarks[key]))

        if not sections:
            return None
        result = "\n\n".join(sections)
        return result[: _MAX_CONTEXT_CHARS // 2]  # benchmarks get half the budget

    def _format_benchmark(self, metric: str, data: Any) -> str:
        label = metric.replace("_", " ").title()
        if isinstance(data, dict):
            lines = [f"FM Industry Benchmark — {label}:"]
            for k, v in data.items():
                lines.append(f"  {k.replace('_', ' ')}: {v}")
            return "\n".join(lines)
        return f"FM Industry Benchmark — {label}: {data}"

    # ── Chunking ───────────────────────────────────────────────────────────────

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """Split text into overlapping word-based chunks."""
        words = text.split()
        if not words:
            return []
        chunks: list[str] = []
        step = max(1, chunk_size - overlap)
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    # ── Benchmarks loader ──────────────────────────────────────────────────────

    def _load_benchmarks(self) -> dict:
        if not self._benchmarks_path.exists():
            logger.warning("benchmarks.yaml not found at %s", self._benchmarks_path)
            return {}
        try:
            with open(self._benchmarks_path) as fh:
                return yaml.safe_load(fh) or {}
        except Exception as exc:
            logger.warning("Could not load benchmarks.yaml: %s", exc)
            return {}
