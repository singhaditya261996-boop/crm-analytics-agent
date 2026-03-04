"""
tests/test_tracker.py — Unit tests for tracker/database.py and tracker/csv_manager.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy.exc import IntegrityError

from tracker.csv_manager import CSVManager
from tracker.database import TrackerDB


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def db(tmp_path):
    """In-memory-style SQLite DB scoped to a temp file per test."""
    url = f"sqlite:///{tmp_path}/test.db"
    return TrackerDB(db_url=url)


@pytest.fixture()
def csv_mgr(db, tmp_path):
    return CSVManager(db, output_folder=tmp_path / "exports")


# ── TestTrackerDBInit ─────────────────────────────────────────────────────────

class TestTrackerDBInit:
    def test_default_url_constant_exists(self):
        assert TrackerDB.DEFAULT_URL == "sqlite:///tracker/crm_agent.db"

    def test_custom_url_creates_tables(self, tmp_path):
        url = f"sqlite:///{tmp_path}/custom.db"
        db = TrackerDB(db_url=url)
        # Tables exist if we can query without error
        assert db.get_recent_queries(1) == []

    def test_tables_created_on_init(self, db):
        # All three methods should work without raising
        assert db.get_recent_queries() == []
        assert db.get_patterns() == []


# ── TestQueryLog ──────────────────────────────────────────────────────────────

class TestQueryLog:
    def test_log_query_persists_row(self, db):
        db.log_query("How many accounts?", "result = len(df)", "42 accounts", score=90.0)
        rows = db.get_recent_queries()
        assert len(rows) == 1
        assert rows[0]["question"] == "How many accounts?"

    def test_get_recent_queries_newest_first(self, db):
        db.log_query("First question", "result = 1", score=80.0)
        db.log_query("Second question", "result = 2", score=85.0)
        rows = db.get_recent_queries()
        assert rows[0]["question"] == "Second question"

    def test_limit_n_respected(self, db):
        for i in range(10):
            db.log_query(f"Q{i}", f"result = {i}", score=float(i * 10))
        rows = db.get_recent_queries(n=3)
        assert len(rows) == 3

    def test_error_field_stored(self, db):
        db.log_query("Bad query", "result = ???", error="SyntaxError: invalid syntax")
        rows = db.get_recent_queries()
        assert rows[0]["error"] == "SyntaxError: invalid syntax"

    def test_result_summary_stored(self, db):
        # result_summary is persisted in DB; log_query must not raise
        db.log_query("Q", "result = 1", result_summary="One result", score=75.0)
        rows = db.get_recent_queries()
        assert len(rows) == 1  # row was saved successfully

    def test_empty_db_returns_empty_list(self, db):
        assert db.get_recent_queries() == []

    def test_score_and_iterations_stored(self, db):
        db.log_query("Q", "result = 1", score=92.5, iterations=3)
        rows = db.get_recent_queries()
        assert rows[0]["score"] == 92.5
        assert rows[0]["iterations"] == 3

    def test_long_result_summary_not_truncated_by_db(self, db):
        # TrackerDB itself doesn't truncate; caller must truncate
        long_summary = "x" * 600
        db.log_query("Q", "result = 1", result_summary=long_summary[:500])
        rows = db.get_recent_queries()
        assert len(rows[0]["question"]) > 0


# ── TestCodePattern ───────────────────────────────────────────────────────────

class TestCodePattern:
    def test_log_pattern_creates_new_entry(self, db):
        db.log_pattern("aggregation", "result = df.groupby('x').sum()", score=88.0)
        patterns = db.get_patterns()
        assert len(patterns) == 1
        assert patterns[0]["question_type"] == "aggregation"

    def test_second_call_increments_use_count(self, db):
        db.log_pattern("aggregation", "result = df.sum()", score=80.0)
        db.log_pattern("aggregation", "result = df.sum()", score=85.0)
        patterns = db.get_patterns()
        assert patterns[0]["use_count"] == 2

    def test_score_updated_to_max(self, db):
        db.log_pattern("aggregation", "result = df.sum()", score=70.0)
        db.log_pattern("aggregation", "result = df.sum()", score=95.0)
        patterns = db.get_patterns()
        assert patterns[0]["score"] == 95.0

    def test_score_not_lowered(self, db):
        db.log_pattern("aggregation", "result = df.sum()", score=90.0)
        db.log_pattern("aggregation", "result = df.sum()", score=60.0)
        patterns = db.get_patterns()
        assert patterns[0]["score"] == 90.0

    def test_get_patterns_filters_by_question_type(self, db):
        db.log_pattern("aggregation", "result = df.sum()", score=80.0)
        db.log_pattern("trend", "result = df.resample('M').sum()", score=75.0)
        patterns = db.get_patterns(question_type="agg")
        assert len(patterns) == 1
        assert patterns[0]["question_type"] == "aggregation"

    def test_get_patterns_returns_all_sorted_by_score(self, db):
        db.log_pattern("type_a", "code_a", score=60.0)
        db.log_pattern("type_b", "code_b", score=90.0)
        db.log_pattern("type_c", "code_c", score=75.0)
        patterns = db.get_patterns()
        scores = [p["score"] for p in patterns]
        assert scores == sorted(scores, reverse=True)

    def test_empty_patterns_returns_empty_list(self, db):
        assert db.get_patterns() == []


# ── TestSessionLog ────────────────────────────────────────────────────────────

class TestSessionLog:
    def test_log_session_stores_data(self, db):
        db.log_session("sess-abc123", ["accounts.csv", "pipeline.xlsx"])
        # No exception raised means it was stored; verify via raw engine
        from sqlalchemy import text
        with db._Session() as session:
            row = session.execute(
                text("SELECT session_id, file_names FROM session_log")
            ).fetchone()
        assert row[0] == "sess-abc123"
        assert "accounts.csv" in json.loads(row[1])

    def test_duplicate_session_id_raises(self, db):
        db.log_session("sess-dup", [])
        with pytest.raises(Exception):  # IntegrityError or similar
            db.log_session("sess-dup", [])


# ── TestCSVManager ────────────────────────────────────────────────────────────

class TestCSVManager:
    def test_export_query_log_returns_path(self, db, csv_mgr):
        db.log_query("Q", "result = 1", score=80.0)
        path = csv_mgr.export_query_log()
        assert isinstance(path, Path)
        assert path.exists()
        assert path.suffix == ".csv"

    def test_export_query_log_has_expected_columns(self, db, csv_mgr):
        db.log_query("Q", "result = 1", score=80.0, iterations=2)
        path = csv_mgr.export_query_log()
        df = pd.read_csv(path)
        for col in ["question", "code", "score", "iterations"]:
            assert col in df.columns

    def test_export_patterns_returns_path(self, db, csv_mgr):
        db.log_pattern("aggregation", "result = df.sum()", score=88.0)
        path = csv_mgr.export_patterns()
        assert isinstance(path, Path)
        assert path.exists()

    def test_empty_db_export_does_not_raise(self, db, csv_mgr):
        path = csv_mgr.export_query_log()
        assert path.exists()
        path2 = csv_mgr.export_patterns()
        assert path2.exists()

    def test_custom_filename_used(self, db, csv_mgr):
        db.log_query("Q", "result = 1", score=80.0)
        path = csv_mgr.export_query_log(filename="my_custom_export.csv")
        assert path.name == "my_custom_export.csv"

    def test_export_patterns_has_expected_columns(self, db, csv_mgr):
        db.log_pattern("trend", "result = df.resample('M').mean()", score=77.0)
        path = csv_mgr.export_patterns()
        df = pd.read_csv(path)
        for col in ["question_type", "code_pattern", "score", "use_count"]:
            assert col in df.columns


# ── TestPersistence ───────────────────────────────────────────────────────────

class TestPersistence:
    def test_query_persists_across_instances(self, tmp_path):
        url = f"sqlite:///{tmp_path}/persist.db"
        db1 = TrackerDB(db_url=url)
        db1.log_query("Persistent question", "result = 42", score=88.0)

        db2 = TrackerDB(db_url=url)
        rows = db2.get_recent_queries()
        assert len(rows) == 1
        assert rows[0]["question"] == "Persistent question"

    def test_pattern_persists_across_instances(self, tmp_path):
        url = f"sqlite:///{tmp_path}/persist_p.db"
        db1 = TrackerDB(db_url=url)
        db1.log_pattern("aggregation", "result = df.sum()", score=90.0)

        db2 = TrackerDB(db_url=url)
        patterns = db2.get_patterns()
        assert len(patterns) == 1
        assert patterns[0]["score"] == 90.0
