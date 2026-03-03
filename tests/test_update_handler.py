"""
tests/test_update_handler.py — Unit tests for Module 11: UpdateHandler.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.update_handler import (
    ColumnDiff,
    FileDiff,
    FileClassification,
    FileFingerprint,
    UpdateHandler,
    UpdateSummary,
    _col_hash,
    _hash_rows,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_handler(tmp_path):
    return UpdateHandler(cache_dir=tmp_path / ".cache")


@pytest.fixture()
def sample_df():
    return pd.DataFrame({
        "id":   [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "val":  [10.0, 20.0, 30.0],
    })


@pytest.fixture()
def sample_df2():
    """Same columns, different rows (one extra)."""
    return pd.DataFrame({
        "id":   [1, 2, 3, 4],
        "name": ["Alice", "Bob", "Charlie", "Dave"],
        "val":  [10.0, 20.0, 30.0, 40.0],
    })


@pytest.fixture()
def corrected_df():
    """Same shape as sample_df, one cell value changed."""
    return pd.DataFrame({
        "id":   [1, 2, 3],
        "name": ["Alice", "Bob", "Charlotte"],   # Charlie → Charlotte
        "val":  [10.0, 20.0, 30.0],
    })


# ── TestFileFingerprint ────────────────────────────────────────────────────────

class TestFileFingerprint:
    def test_deterministic_hash(self, tmp_handler, sample_df):
        fp1 = tmp_handler.fingerprint(sample_df, "test.csv")
        fp2 = tmp_handler.fingerprint(sample_df, "test.csv")
        assert fp1.row_sample_hash == fp2.row_sample_hash
        assert fp1.col_hash == fp2.col_hash

    def test_identical_df_same_hash(self, tmp_handler, sample_df):
        df_copy = sample_df.copy()
        fp1 = tmp_handler.fingerprint(sample_df, "a.csv")
        fp2 = tmp_handler.fingerprint(df_copy, "a.csv")
        assert fp1.row_sample_hash == fp2.row_sample_hash

    def test_mutated_df_different_hash(self, tmp_handler, sample_df):
        mutated = sample_df.copy()
        mutated.iloc[0, 1] = "Xxxxxx"
        fp1 = tmp_handler.fingerprint(sample_df, "a.csv")
        fp2 = tmp_handler.fingerprint(mutated, "a.csv")
        assert fp1.row_sample_hash != fp2.row_sample_hash

    def test_col_hash_uses_sorted_columns(self, tmp_handler):
        df_a = pd.DataFrame({"b": [1], "a": [2]})
        df_b = pd.DataFrame({"a": [2], "b": [1]})
        fp_a = tmp_handler.fingerprint(df_a, "a.csv")
        fp_b = tmp_handler.fingerprint(df_b, "b.csv")
        assert fp_a.col_hash == fp_b.col_hash

    def test_row_count_correct(self, tmp_handler, sample_df):
        fp = tmp_handler.fingerprint(sample_df, "x.csv")
        assert fp.row_count == len(sample_df)

    def test_file_hash_populated(self, tmp_handler, sample_df):
        fp = tmp_handler.fingerprint(sample_df, "x.csv", file_bytes=b"hello")
        assert fp.file_hash != ""

    def test_file_hash_empty_when_no_bytes(self, tmp_handler, sample_df):
        fp = tmp_handler.fingerprint(sample_df, "x.csv")
        assert fp.file_hash == ""

    def test_serialisation_roundtrip(self, tmp_handler, sample_df):
        fp = tmp_handler.fingerprint(sample_df, "x.csv")
        restored = FileFingerprint.from_dict(fp.to_dict())
        assert restored.row_sample_hash == fp.row_sample_hash
        assert restored.col_hash == fp.col_hash
        assert restored.row_count == fp.row_count


# ── TestClassification ─────────────────────────────────────────────────────────

class TestClassification:
    def test_new_table(self, tmp_handler, sample_df):
        cls, new_fp, old_fp = tmp_handler.classify("mytable", sample_df)
        assert cls == FileClassification.NEW_TABLE
        assert old_fp is None

    def test_duplicate(self, tmp_handler, sample_df):
        tmp_handler.register("mytable", tmp_handler.fingerprint(sample_df, "mytable"))
        cls, _, _ = tmp_handler.classify("mytable", sample_df)
        assert cls == FileClassification.DUPLICATE

    def test_updated_version_extra_rows(self, tmp_handler, sample_df, sample_df2):
        tmp_handler.register("mytable", tmp_handler.fingerprint(sample_df, "mytable"))
        cls, _, _ = tmp_handler.classify("mytable", sample_df2)
        assert cls == FileClassification.UPDATED_VERSION

    def test_updated_version_new_column(self, tmp_handler, sample_df):
        tmp_handler.register("mytable", tmp_handler.fingerprint(sample_df, "mytable"))
        df_extra_col = sample_df.copy()
        df_extra_col["new_col"] = "x"
        cls, _, _ = tmp_handler.classify("mytable", df_extra_col)
        assert cls == FileClassification.UPDATED_VERSION

    def test_corrected_data(self, tmp_handler, sample_df, corrected_df):
        tmp_handler.register("mytable", tmp_handler.fingerprint(sample_df, "mytable"))
        cls, _, old_fp = tmp_handler.classify("mytable", corrected_df)
        assert cls == FileClassification.CORRECTED_DATA
        assert old_fp is not None

    def test_old_fp_returned_when_registered(self, tmp_handler, sample_df, corrected_df):
        fp = tmp_handler.fingerprint(sample_df, "mytable")
        tmp_handler.register("mytable", fp)
        _, new_fp, old_fp = tmp_handler.classify("mytable", corrected_df)
        assert old_fp.row_sample_hash == fp.row_sample_hash


# ── TestFileDiff ───────────────────────────────────────────────────────────────

class TestFileDiff:
    def test_rows_added(self, tmp_handler, sample_df, sample_df2):
        diff = tmp_handler.compute_diff(sample_df, sample_df2)
        assert diff.rows_added == 1

    def test_rows_removed(self, tmp_handler, sample_df, sample_df2):
        diff = tmp_handler.compute_diff(sample_df2, sample_df)
        assert diff.rows_removed == 1

    def test_no_changes_identical(self, tmp_handler, sample_df):
        diff = tmp_handler.compute_diff(sample_df, sample_df.copy())
        assert diff.rows_added == 0
        assert diff.rows_removed == 0
        assert diff.column_diffs == []

    def test_column_diffs_detected(self, tmp_handler, sample_df, corrected_df):
        diff = tmp_handler.compute_diff(sample_df, corrected_df)
        assert any(cd.col_name == "name" for cd in diff.column_diffs)

    def test_changed_cells_count(self, tmp_handler, sample_df, corrected_df):
        diff = tmp_handler.compute_diff(sample_df, corrected_df)
        name_diff = next(cd for cd in diff.column_diffs if cd.col_name == "name")
        assert name_diff.changed_cells == 1

    def test_sample_changes_content(self, tmp_handler, sample_df, corrected_df):
        diff = tmp_handler.compute_diff(sample_df, corrected_df)
        name_diff = next(cd for cd in diff.column_diffs if cd.col_name == "name")
        assert len(name_diff.sample_changes) >= 1
        old_val, new_val = name_diff.sample_changes[0]
        assert str(old_val) == "Charlie"
        assert str(new_val) == "Charlotte"

    def test_new_columns_detected(self, tmp_handler, sample_df):
        df_extra = sample_df.copy()
        df_extra["extra"] = 99
        diff = tmp_handler.compute_diff(sample_df, df_extra)
        assert "extra" in diff.new_columns

    def test_removed_columns_detected(self, tmp_handler, sample_df):
        df_less = sample_df.drop(columns=["val"])
        diff = tmp_handler.compute_diff(sample_df, df_less)
        assert "val" in diff.removed_columns

    def test_summary_text_rows_added(self, tmp_handler, sample_df, sample_df2):
        diff = tmp_handler.compute_diff(sample_df, sample_df2)
        text = diff.summary_text()
        assert "new row" in text

    def test_summary_text_corrected(self, tmp_handler, sample_df, corrected_df):
        diff = tmp_handler.compute_diff(sample_df, corrected_df)
        text = diff.summary_text()
        assert "corrected" in text

    def test_summary_text_no_changes(self, tmp_handler, sample_df):
        diff = tmp_handler.compute_diff(sample_df, sample_df.copy())
        assert diff.summary_text() == "No changes detected"


# ── TestVersionHistory ─────────────────────────────────────────────────────────

class TestVersionHistory:
    def test_save_creates_parquet(self, tmp_handler, sample_df):
        p = tmp_handler.save_version("mytable", sample_df)
        assert p.exists()
        assert p.suffix == ".parquet"

    def test_list_versions_newest_first(self, tmp_handler, sample_df):
        from datetime import timezone
        import datetime as dt

        ts1 = dt.datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        ts2 = dt.datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc)
        tmp_handler.save_version("mytable", sample_df, ts=ts1)
        tmp_handler.save_version("mytable", sample_df, ts=ts2)

        versions = tmp_handler.list_versions("mytable")
        assert len(versions) == 2
        # Newest first: ts2 > ts1
        assert versions[0]["ts_str"] > versions[1]["ts_str"]

    def test_load_version_restores_df(self, tmp_handler, sample_df):
        from datetime import timezone
        import datetime as dt
        ts = dt.datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        tmp_handler.save_version("mytable", sample_df, ts=ts)

        versions = tmp_handler.list_versions("mytable")
        loaded = tmp_handler.load_version("mytable", versions[0]["ts_str"])
        pd.testing.assert_frame_equal(loaded, sample_df)

    def test_multiple_versions_same_table(self, tmp_handler, sample_df, sample_df2):
        # microsecond precision in ts_str ensures two rapid saves get distinct filenames
        tmp_handler.save_version("mytable", sample_df)
        tmp_handler.save_version("mytable", sample_df2)
        versions = tmp_handler.list_versions("mytable")
        assert len(versions) == 2

    def test_row_count_in_list(self, tmp_handler, sample_df):
        tmp_handler.save_version("mytable", sample_df)
        versions = tmp_handler.list_versions("mytable")
        assert versions[0]["row_count"] == len(sample_df)


# ── TestMerge ─────────────────────────────────────────────────────────────────

class TestMerge:
    def test_dedup_keeps_latest(self, tmp_handler):
        old = pd.DataFrame({"id": [1, 2], "val": ["a", "b"]})
        new = pd.DataFrame({"id": [2, 3], "val": ["B", "c"]})
        merged = tmp_handler.merge_dataframes(old, new, key_col="id")
        # id=2 should have "B" (from new)
        row2 = merged[merged["id"] == 2]["val"].values[0]
        assert row2 == "B"

    def test_no_duplicates_in_result(self, tmp_handler):
        old = pd.DataFrame({"id": [1, 2], "val": ["a", "b"]})
        new = pd.DataFrame({"id": [2, 3], "val": ["B", "c"]})
        merged = tmp_handler.merge_dataframes(old, new, key_col="id")
        assert merged["id"].duplicated().sum() == 0

    def test_all_ids_present(self, tmp_handler):
        old = pd.DataFrame({"id": [1, 2], "val": ["a", "b"]})
        new = pd.DataFrame({"id": [2, 3], "val": ["B", "c"]})
        merged = tmp_handler.merge_dataframes(old, new, key_col="id")
        assert set(merged["id"].tolist()) == {1, 2, 3}

    def test_raises_for_missing_key_in_old(self, tmp_handler):
        old = pd.DataFrame({"x": [1]})
        new = pd.DataFrame({"id": [1]})
        with pytest.raises(ValueError, match="key_col 'id' not in old_df"):
            tmp_handler.merge_dataframes(old, new, key_col="id")

    def test_raises_for_missing_key_in_new(self, tmp_handler):
        old = pd.DataFrame({"id": [1]})
        new = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError, match="key_col 'id' not in new_df"):
            tmp_handler.merge_dataframes(old, new, key_col="id")


# ── TestAgendaImpact ──────────────────────────────────────────────────────────

class TestAgendaImpact:
    def test_flags_questions_when_update_detected(self, tmp_handler, sample_df, sample_df2):
        tmp_handler.register("mytable", tmp_handler.fingerprint(sample_df, "mytable"))
        agenda = {"Q1": "answer1", "Q2": "answer2"}
        stale = tmp_handler.check_agenda_impact(["mytable"], agenda)
        assert "Q1" in stale
        assert "Q2" in stale

    def test_empty_updated_names_returns_empty(self, tmp_handler):
        agenda = {"Q1": "a"}
        stale = tmp_handler.check_agenda_impact([], agenda)
        assert stale == set()

    def test_empty_agenda_returns_empty(self, tmp_handler, sample_df):
        tmp_handler.register("mytable", tmp_handler.fingerprint(sample_df, "mytable"))
        stale = tmp_handler.check_agenda_impact(["mytable"], {})
        assert stale == set()

    def test_no_registry_entry_flags_all(self, tmp_handler):
        agenda = {"Q1": "a", "Q2": "b"}
        stale = tmp_handler.check_agenda_impact(["unknown_table"], agenda)
        assert stale == {"Q1", "Q2"}


# ── TestSentinelFile ──────────────────────────────────────────────────────────

class TestSentinelFile:
    def test_write_read_lifecycle(self, tmp_handler):
        tmp_handler.write_sentinel([Path("/tmp/a.csv"), Path("/tmp/b.xlsx")])
        result = tmp_handler.read_sentinel()
        assert result is not None
        assert "/tmp/a.csv" in result
        assert "/tmp/b.xlsx" in result

    def test_read_returns_none_when_no_sentinel(self, tmp_handler):
        assert tmp_handler.read_sentinel() is None

    def test_clear_removes_sentinel(self, tmp_handler):
        tmp_handler.write_sentinel([Path("/tmp/x.csv")])
        tmp_handler.clear_sentinel()
        assert tmp_handler.read_sentinel() is None

    def test_write_string_paths(self, tmp_handler):
        tmp_handler.write_sentinel(["/some/path/file.csv"])
        result = tmp_handler.read_sentinel()
        assert "/some/path/file.csv" in result


# ── TestFolderWatcher ─────────────────────────────────────────────────────────

class TestFolderWatcher:
    def test_start_watcher_creates_observer(self, tmp_handler, tmp_path):
        """start_watcher should set _observer attribute."""
        watch_dir = tmp_path / "uploads"
        watch_dir.mkdir()
        try:
            tmp_handler.start_watcher(watch_dir)
            assert tmp_handler._observer is not None
        finally:
            tmp_handler.stop_watcher()

    def test_start_watcher_idempotent(self, tmp_handler, tmp_path):
        """Calling start_watcher twice should not error."""
        watch_dir = tmp_path / "uploads"
        watch_dir.mkdir()
        try:
            tmp_handler.start_watcher(watch_dir)
            obs1 = tmp_handler._observer
            tmp_handler.start_watcher(watch_dir)   # second call
            assert tmp_handler._observer is obs1    # same object
        finally:
            tmp_handler.stop_watcher()

    def test_stop_watcher_clears_observer(self, tmp_handler, tmp_path):
        watch_dir = tmp_path / "uploads"
        watch_dir.mkdir()
        tmp_handler.start_watcher(watch_dir)
        tmp_handler.stop_watcher()
        assert tmp_handler._observer is None

    def test_on_created_csv_writes_sentinel(self, tmp_handler, tmp_path):
        """Simulate a .csv file creation event and check sentinel is written."""
        from data.update_handler import _NewFileHandler, _WATCHDOG_OK
        if not _WATCHDOG_OK:
            pytest.skip("watchdog not installed")

        handler = _NewFileHandler(tmp_handler._sentinel_path)
        mock_event = MagicMock()
        mock_event.is_directory = False
        mock_event.src_path = str(tmp_path / "new_data.csv")

        handler.on_created(mock_event)

        result = tmp_handler.read_sentinel()
        assert result is not None
        assert str(tmp_path / "new_data.csv") in result

    def test_on_created_pdf_ignored(self, tmp_handler, tmp_path):
        """PDF files should not trigger a sentinel write."""
        from data.update_handler import _NewFileHandler, _WATCHDOG_OK
        if not _WATCHDOG_OK:
            pytest.skip("watchdog not installed")

        handler = _NewFileHandler(tmp_handler._sentinel_path)
        mock_event = MagicMock()
        mock_event.is_directory = False
        mock_event.src_path = str(tmp_path / "report.pdf")

        handler.on_created(mock_event)
        assert tmp_handler.read_sentinel() is None

    def test_on_created_directory_ignored(self, tmp_handler, tmp_path):
        """Directory events should be ignored."""
        from data.update_handler import _NewFileHandler, _WATCHDOG_OK
        if not _WATCHDOG_OK:
            pytest.skip("watchdog not installed")

        handler = _NewFileHandler(tmp_handler._sentinel_path)
        mock_event = MagicMock()
        mock_event.is_directory = True
        mock_event.src_path = str(tmp_path / "somefolder")

        handler.on_created(mock_event)
        assert tmp_handler.read_sentinel() is None

    def test_no_watchdog_raises_import_error(self, tmp_handler, tmp_path):
        """If watchdog is not available start_watcher raises ImportError."""
        watch_dir = tmp_path / "uploads"
        watch_dir.mkdir()
        with patch("data.update_handler._WATCHDOG_OK", False):
            with pytest.raises(ImportError):
                tmp_handler.start_watcher(watch_dir)


# ── TestRegistry ──────────────────────────────────────────────────────────────

class TestRegistry:
    def test_register_persists_to_json(self, tmp_handler, sample_df):
        fp = tmp_handler.fingerprint(sample_df, "mytable")
        tmp_handler.register("mytable", fp)
        raw = json.loads(tmp_handler._registry_path.read_text())
        assert "mytable" in raw

    def test_load_registry_restores_on_new_init(self, tmp_path, sample_df):
        handler1 = UpdateHandler(cache_dir=tmp_path / ".cache")
        fp = handler1.fingerprint(sample_df, "mytable")
        handler1.register("mytable", fp)

        # New handler instance pointing to same cache
        handler2 = UpdateHandler(cache_dir=tmp_path / ".cache")
        cls, _, old_fp = handler2.classify("mytable", sample_df)
        assert cls == FileClassification.DUPLICATE
        assert old_fp is not None

    def test_register_overwrites_existing(self, tmp_handler, sample_df, sample_df2):
        fp1 = tmp_handler.fingerprint(sample_df, "mytable")
        tmp_handler.register("mytable", fp1)

        fp2 = tmp_handler.fingerprint(sample_df2, "mytable")
        tmp_handler.register("mytable", fp2)

        raw = json.loads(tmp_handler._registry_path.read_text())
        stored = FileFingerprint.from_dict(raw["mytable"])
        assert stored.row_count == len(sample_df2)


# ── TestProcessUpdate ─────────────────────────────────────────────────────────

class TestProcessUpdate:
    def test_returns_update_summary(self, tmp_handler, sample_df):
        summary = tmp_handler.process_update("mytable", sample_df)
        assert isinstance(summary, UpdateSummary)
        assert summary.classification == FileClassification.NEW_TABLE

    def test_diff_computed_when_old_df_provided(
        self, tmp_handler, sample_df, corrected_df
    ):
        tmp_handler.register("mytable", tmp_handler.fingerprint(sample_df, "mytable"))
        summary = tmp_handler.process_update(
            "mytable", corrected_df, old_df=sample_df
        )
        assert summary.diff is not None
        assert summary.classification == FileClassification.CORRECTED_DATA

    def test_no_diff_for_duplicate(self, tmp_handler, sample_df):
        tmp_handler.register("mytable", tmp_handler.fingerprint(sample_df, "mytable"))
        summary = tmp_handler.process_update("mytable", sample_df, old_df=sample_df)
        assert summary.classification == FileClassification.DUPLICATE
        assert summary.diff is None
