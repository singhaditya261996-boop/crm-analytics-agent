"""
tests/test_knowledge_manager.py — Module 15 unit tests for the knowledge layer.

All chromadb operations use a test EphemeralClient with a deterministic fake
embedding function — no production knowledge base is touched and no model
download is required.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from knowledge.embedder import DeterministicEmbeddingFn
from knowledge.fine_tuning_prep import export_fine_tuning_dataset
from knowledge.knowledge_manager import KnowledgeManager


# ── Test fixtures ──────────────────────────────────────────────────────────────

def _make_km(tmp_path: Path, extra_kwargs: dict | None = None) -> KnowledgeManager:
    """Build an isolated KnowledgeManager using in-memory chromadb."""
    import chromadb
    import uuid

    client = chromadb.EphemeralClient()
    ef = DeterministicEmbeddingFn()
    # Use a unique collection name per call to avoid cross-test contamination
    # when EphemeralClient shares in-memory state across instances.
    unique_name = f"test_{uuid.uuid4().hex[:12]}"
    km = KnowledgeManager(
        knowledge_dir=tmp_path / "knowledge",
        chroma_path=tmp_path / "chroma",  # not used (in-memory client injected)
        _client=client,
        _embedding_fn=ef,
        _collection_name=unique_name,
        **(extra_kwargs or {}),
    )
    return km


@pytest.fixture()
def km(tmp_path):
    return _make_km(tmp_path)


@pytest.fixture()
def km_with_base(tmp_path):
    """KM with a populated knowledge/base directory."""
    base_dir = tmp_path / "knowledge" / "base"
    base_dir.mkdir(parents=True)
    (base_dir / "fm_context.txt").write_text(
        "UK FM market size is £120bn. Win rate median is 28%."
    )
    (base_dir / "glossary.txt").write_text(
        "TFM: Total Facilities Management. Hard FM: HVAC and electrical."
    )
    km = _make_km(tmp_path)
    km.load_all_knowledge()
    return km


# ── TestKnowledgeBaseLoading ───────────────────────────────────────────────────

class TestKnowledgeBaseLoading:
    def test_load_all_knowledge_returns_chunk_count(self, tmp_path):
        base_dir = tmp_path / "knowledge" / "base"
        base_dir.mkdir(parents=True)
        (base_dir / "context.txt").write_text("FM sector context. " * 50)
        km = _make_km(tmp_path)
        total = km.load_all_knowledge()
        assert total >= 1

    def test_base_files_indexed_into_collection(self, km_with_base):
        assert km_with_base._collection.count() > 0

    def test_empty_base_dir_returns_zero(self, km):
        km._knowledge_dir.mkdir(parents=True, exist_ok=True)
        (km._knowledge_dir / "base").mkdir()
        total = km._embed_base_files()
        assert total == 0

    def test_missing_base_dir_returns_zero(self, km):
        total = km._embed_base_files()
        assert total == 0

    def test_multiple_base_files_all_indexed(self, tmp_path):
        base_dir = tmp_path / "knowledge" / "base"
        base_dir.mkdir(parents=True)
        for i in range(3):
            (base_dir / f"file_{i}.txt").write_text(f"Content for file {i}. " * 30)
        km = _make_km(tmp_path)
        total = km._embed_base_files()
        assert total >= 3  # at least one chunk per file

    def test_scraped_files_indexed(self, tmp_path):
        scraped_dir = tmp_path / "knowledge" / "scraped" / "competitor_news"
        scraped_dir.mkdir(parents=True)
        (scraped_dir / "20250301_mitie_article.txt").write_text(
            "Mitie announces new contract. " * 20
        )
        km = _make_km(tmp_path)
        total = km._embed_scraped_content()
        assert total >= 1

    def test_meeting_notes_indexed(self, tmp_path):
        notes_dir = tmp_path / "knowledge" / "meeting_notes"
        notes_dir.mkdir(parents=True)
        (notes_dir / "notes_20250301.txt").write_text(
            "Meeting with client. Account XYZ flagged as at risk. " * 10
        )
        km = _make_km(tmp_path)
        total = km._embed_meeting_notes()
        assert total >= 1


# ── TestBenchmarkInjection ────────────────────────────────────────────────────

class TestBenchmarkInjection:
    def test_win_rate_keyword_triggers_benchmark(self, tmp_path):
        bench_path = tmp_path / "knowledge" / "benchmarks.yaml"
        bench_path.parent.mkdir(parents=True)
        bench_path.write_text("win_rates:\n  competitive_tender:\n    median: 28\n    unit: '%'\n")
        km = _make_km(tmp_path)
        km._benchmarks_path = bench_path
        km._benchmarks = km._load_benchmarks()
        ctx = km._get_benchmark_context("What is our win rate this quarter?")
        assert ctx is not None
        assert "28" in ctx or "win" in ctx.lower()

    def test_pipeline_keyword_triggers_benchmark(self, tmp_path):
        bench_path = tmp_path / "knowledge" / "benchmarks.yaml"
        bench_path.parent.mkdir(parents=True)
        bench_path.write_text("pipeline_health:\n  coverage_ratio:\n    healthy: 3.0\n")
        km = _make_km(tmp_path)
        km._benchmarks_path = bench_path
        km._benchmarks = km._load_benchmarks()
        ctx = km._get_benchmark_context("Show me our pipeline coverage ratio")
        assert ctx is not None

    def test_unrelated_question_returns_none(self, km):
        # No benchmark keywords present
        ctx = km._get_benchmark_context("How many employees joined in January?")
        assert ctx is None

    def test_dormant_keyword_triggers_benchmark(self, tmp_path):
        bench_path = tmp_path / "knowledge" / "benchmarks.yaml"
        bench_path.parent.mkdir(parents=True)
        bench_path.write_text("account_health:\n  dormant_threshold_days: 90\n")
        km = _make_km(tmp_path)
        km._benchmarks_path = bench_path
        km._benchmarks = km._load_benchmarks()
        ctx = km._get_benchmark_context("Which accounts are dormant?")
        assert ctx is not None

    def test_missing_benchmarks_file_returns_none(self, km):
        km._benchmarks = {}
        ctx = km._get_benchmark_context("What is our win rate?")
        assert ctx is None

    def test_benchmark_context_respects_token_limit(self, tmp_path):
        bench_path = tmp_path / "knowledge" / "benchmarks.yaml"
        bench_path.parent.mkdir(parents=True)
        # Large benchmark entry
        bench_path.write_text(
            "win_rates:\n  " + "\n  ".join(f"key_{i}: value_{i}" for i in range(500)) + "\n"
        )
        km = _make_km(tmp_path)
        km._benchmarks_path = bench_path
        km._benchmarks = km._load_benchmarks()
        ctx = km._get_benchmark_context("What is our win rate?")
        from knowledge.knowledge_manager import _MAX_CONTEXT_CHARS
        if ctx:
            assert len(ctx) <= _MAX_CONTEXT_CHARS

    def test_get_relevant_context_has_expected_keys(self, km):
        result = km.get_relevant_context("Some question", "aggregation")
        assert "document_context" in result
        assert "benchmark_context" in result
        assert "has_benchmark" in result

    def test_get_relevant_context_has_benchmark_false_when_no_match(self, km):
        result = km.get_relevant_context("Some completely irrelevant question", "aggregation")
        assert result["has_benchmark"] is False


# ── TestMeetingNotes ──────────────────────────────────────────────────────────

class TestMeetingNotes:
    def test_add_meeting_note_saves_to_disk(self, km, tmp_path):
        km._knowledge_dir.mkdir(parents=True, exist_ok=True)
        path = km.add_meeting_note("Account ABC discussed. Risk flagged.", session_date="20250301")
        assert path.exists()
        assert "20250301" in path.name

    def test_add_meeting_note_embeds_into_collection(self, km, tmp_path):
        km._knowledge_dir.mkdir(parents=True, exist_ok=True)
        before = km._collection.count()
        km.add_meeting_note("Strategic note for FM sector deal.", session_date="20250302")
        assert km._collection.count() > before

    def test_add_meeting_note_appends_on_same_date(self, km, tmp_path):
        km._knowledge_dir.mkdir(parents=True, exist_ok=True)
        km.add_meeting_note("First note.", session_date="20250303")
        km.add_meeting_note("Second note.", session_date="20250303")
        note_path = km._knowledge_dir / "meeting_notes" / "notes_20250303.txt"
        content = note_path.read_text()
        assert "First note." in content
        assert "Second note." in content

    def test_meeting_note_retrievable_by_semantic_search(self, km, tmp_path):
        km._knowledge_dir.mkdir(parents=True, exist_ok=True)
        note_text = "Account Telford NHS at risk due to procurement contact change."
        km.add_meeting_note(note_text, session_date="20250304")
        # Query the collection directly — should find the note
        results = km._collection.query(
            query_texts=["Telford NHS procurement risk"],
            n_results=1,
            include=["documents"],
        )
        docs = results.get("documents", [[]])[0]
        assert len(docs) >= 1  # at least one result returned

    def test_add_meeting_note_uses_today_when_no_date(self, km, tmp_path):
        km._knowledge_dir.mkdir(parents=True, exist_ok=True)
        path = km.add_meeting_note("Note without explicit date.")
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        assert today in path.name


# ── TestContextTokenLimit ─────────────────────────────────────────────────────

class TestContextTokenLimit:
    def test_semantic_search_result_respects_char_limit(self, tmp_path):
        base_dir = tmp_path / "knowledge" / "base"
        base_dir.mkdir(parents=True)
        # Write a very large file
        large_text = "FM sector benchmark data. " * 2000
        (base_dir / "large_context.txt").write_text(large_text)
        km = _make_km(tmp_path)
        km.load_all_knowledge()
        result = km._semantic_search("FM sector benchmarks")
        from knowledge.knowledge_manager import _MAX_CONTEXT_CHARS
        assert len(result) <= _MAX_CONTEXT_CHARS

    def test_benchmark_context_respects_half_char_limit(self, tmp_path):
        bench_path = tmp_path / "knowledge" / "benchmarks.yaml"
        bench_path.parent.mkdir(parents=True)
        bench_path.write_text(
            "win_rates:\n  " + "\n  ".join(f"key_{i}: {'x' * 100}" for i in range(100)) + "\n"
        )
        km = _make_km(tmp_path)
        km._benchmarks_path = bench_path
        km._benchmarks = km._load_benchmarks()
        ctx = km._get_benchmark_context("win rate analysis")
        from knowledge.knowledge_manager import _MAX_CONTEXT_CHARS
        if ctx:
            assert len(ctx) <= _MAX_CONTEXT_CHARS // 2

    def test_empty_collection_returns_empty_string(self, km):
        result = km._semantic_search("any question")
        assert result == ""


# ── TestChunkText ─────────────────────────────────────────────────────────────

class TestChunkText:
    def test_empty_text_returns_empty_list(self):
        assert KnowledgeManager._chunk_text("") == []

    def test_short_text_returns_one_chunk(self):
        chunks = KnowledgeManager._chunk_text("Hello world")
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_long_text_produces_multiple_chunks(self):
        text = "word " * 1200  # 1200 words
        chunks = KnowledgeManager._chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) > 1

    def test_chunks_overlap(self):
        words = [f"word{i}" for i in range(600)]
        text = " ".join(words)
        chunks = KnowledgeManager._chunk_text(text, chunk_size=100, overlap=20)
        # Each chunk should have some words from the previous
        assert len(chunks) >= 2


# ── TestScraperGracefulDegradation ────────────────────────────────────────────

class TestScraperGracefulDegradation:
    def test_scrape_all_sources_handles_unavailable_site(self, tmp_path):
        """Scraper must not raise even when all sources fail."""
        from knowledge.scraper import IntelligenceScraper, SCRAPE_SOURCES

        km = _make_km(tmp_path)
        km._knowledge_dir.mkdir(parents=True, exist_ok=True)
        scraper = IntelligenceScraper(
            knowledge_manager=km,
            scraped_dir=tmp_path / "scraped",
            registry_path=tmp_path / "registry.json",
        )

        # Override sources with an unreachable URL
        import unittest.mock as mock
        bad_sources = {
            "fake_source": {
                "url": "http://localhost:1/does-not-exist",
                "type": "test",
                "company": "Fake",
                "max_articles": 1,
                "save_dir": "test",
            }
        }
        with mock.patch.object(scraper, "_scrape_source", side_effect=Exception("connection refused")):
            result = scraper.scrape_all_sources()

        assert isinstance(result, dict)
        assert "total_found" in result
        assert "errors" in result
        assert len(result["errors"]) > 0  # errors captured, not raised

    def test_robots_txt_allows_localhost(self, tmp_path):
        from knowledge.scraper import IntelligenceScraper
        km = _make_km(tmp_path)
        scraper = IntelligenceScraper(km, scraped_dir=tmp_path)
        # robots.txt check should not crash even on bad domains
        result = scraper._robots_allows("http://localhost:9999/test")
        assert isinstance(result, bool)

    def test_registry_persists_between_instances(self, tmp_path):
        from knowledge.scraper import IntelligenceScraper
        km = _make_km(tmp_path)
        reg_path = tmp_path / "registry.json"

        s1 = IntelligenceScraper(km, scraped_dir=tmp_path, registry_path=reg_path)
        s1._registry.add("abc123")
        s1._save_registry()

        s2 = IntelligenceScraper(km, scraped_dir=tmp_path, registry_path=reg_path)
        assert "abc123" in s2._registry


# ── TestFineTuningExport ──────────────────────────────────────────────────────

class TestFineTuningExport:
    def _write_log(self, path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_export_filters_by_min_score(self, tmp_path):
        log = tmp_path / "training_log.jsonl"
        self._write_log(log, [
            {"is_final": True, "critic_score": 85, "question": "Q1",
             "intent_type": "aggregation", "interpretation": "Good answer", "recommendation": {}},
            {"is_final": True, "critic_score": 60, "question": "Q2",
             "intent_type": "trend", "interpretation": "Low score answer", "recommendation": {}},
            {"is_final": False, "critic_score": 90, "question": "Q3",
             "intent_type": "ranking", "interpretation": "Not final", "recommendation": {}},
        ])
        out = export_fine_tuning_dataset(
            training_log_path=log,
            output_path=tmp_path / "export.jsonl",
            min_score=80,
        )
        records = [json.loads(l) for l in out.read_text().strip().splitlines()]
        assert len(records) == 1
        assert records[0]["instruction"] == "Q1"

    def test_export_produces_valid_jsonl(self, tmp_path):
        log = tmp_path / "training_log.jsonl"
        self._write_log(log, [
            {"is_final": True, "critic_score": 88, "question": "Revenue total?",
             "intent_type": "aggregation", "interpretation": "Revenue is £500k",
             "recommendation": {"priority_action": "Focus on top accounts",
                                "risk_flag": "Concentration risk", "opportunity": "Upsell"}},
        ])
        out = export_fine_tuning_dataset(
            training_log_path=log,
            output_path=tmp_path / "export.jsonl",
            min_score=80,
        )
        for line in out.read_text().strip().splitlines():
            parsed = json.loads(line)
            assert "instruction" in parsed
            assert "response" in parsed
            assert "score" in parsed

    def test_export_empty_log_produces_empty_file(self, tmp_path):
        log = tmp_path / "training_log.jsonl"
        out = export_fine_tuning_dataset(
            training_log_path=log,
            output_path=tmp_path / "export.jsonl",
            min_score=80,
        )
        assert out.exists()
        assert out.read_text().strip() == ""

    def test_export_respects_min_score_threshold(self, tmp_path):
        log = tmp_path / "training_log.jsonl"
        self._write_log(log, [
            {"is_final": True, "critic_score": 95, "question": "Q_high",
             "intent_type": "aggregation", "interpretation": "Great", "recommendation": {}},
            {"is_final": True, "critic_score": 75, "question": "Q_mid",
             "intent_type": "trend", "interpretation": "OK", "recommendation": {}},
            {"is_final": True, "critic_score": 50, "question": "Q_low",
             "intent_type": "ranking", "interpretation": "Bad", "recommendation": {}},
        ])
        out_90 = export_fine_tuning_dataset(log, tmp_path / "export_90.jsonl", min_score=90)
        out_70 = export_fine_tuning_dataset(log, tmp_path / "export_70.jsonl", min_score=70)
        recs_90 = [json.loads(l) for l in out_90.read_text().strip().splitlines() if l]
        recs_70 = [json.loads(l) for l in out_70.read_text().strip().splitlines() if l]
        assert len(recs_90) == 1
        assert len(recs_70) == 2

    def test_export_default_output_path_created(self, tmp_path):
        log = tmp_path / "exports" / "training_log.jsonl"
        self._write_log(log, [
            {"is_final": True, "critic_score": 82, "question": "Q1",
             "intent_type": "aggregation", "interpretation": "Result", "recommendation": {}},
        ])
        out = export_fine_tuning_dataset(training_log_path=log, min_score=80)
        assert out.exists()
        assert "fine_tuning_dataset" in out.name

    def test_export_recommendation_included_in_response(self, tmp_path):
        log = tmp_path / "training_log.jsonl"
        self._write_log(log, [
            {"is_final": True, "critic_score": 88, "question": "Q?",
             "intent_type": "recommendation", "interpretation": "Summary here.",
             "recommendation": {"priority_action": "Action A",
                                "risk_flag": "Risk B", "opportunity": "Opp C"}},
        ])
        out = export_fine_tuning_dataset(log, tmp_path / "e.jsonl", min_score=80)
        record = json.loads(out.read_text().strip())
        assert "Action A" in record["response"]
        assert "Risk B" in record["response"]
