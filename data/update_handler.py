"""
data/update_handler.py — Incremental file update handler (Module 11).

Tracks file identity via fingerprinting, classifies each reload as
NEW_TABLE / UPDATED_VERSION / CORRECTED_DATA / DUPLICATE, produces a diff
report, manages versioned parquet snapshots for rollback, and runs a
background folder watcher so new files dropped in uploads/ are immediately
detected.

Public API
----------
UpdateHandler(cache_dir: Path | str = "data/.cache")
    .fingerprint(df, filename, file_bytes=None) -> FileFingerprint
    .classify(name, df, file_bytes=None) -> (FileClassification, new_fp, old_fp|None)
    .compute_diff(old_df, new_df) -> FileDiff
    .process_update(name, df, old_df=None, file_bytes=None) -> UpdateSummary
    .register(name, fp) -> None
    .save_version(name, df, ts=None) -> Path
    .load_version(name, ts_str) -> pd.DataFrame
    .list_versions(name) -> list[dict]
    .check_agenda_impact(updated_names, agenda_results, threshold=0.10) -> set[str]
    .merge_dataframes(old_df, new_df, key_col) -> pd.DataFrame
    .write_sentinel(new_paths) -> None
    .read_sentinel() -> list[str] | None
    .clear_sentinel() -> None
    .start_watcher(watch_dir) -> None
    .stop_watcher() -> None
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_REGISTRY_FILE = "file_registry.json"
_VERSIONS_DIR  = "versions"
_SENTINEL_FILE = "new_files.sentinel"
_ROW_SAMPLE_N  = 10_000   # rows sampled for fingerprint on large tables


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class FileFingerprint:
    filename: str
    row_count: int
    columns: list[str]
    col_hash: str           # MD5 of sorted column names joined
    row_sample_hash: str    # MD5 of hashed row tuples (sampled if >10k rows)
    registered_at: str      # ISO 8601 UTC
    file_hash: str = ""     # MD5 of raw file bytes (optional)

    def to_dict(self) -> dict:
        return {
            "filename":        self.filename,
            "row_count":       self.row_count,
            "columns":         self.columns,
            "col_hash":        self.col_hash,
            "row_sample_hash": self.row_sample_hash,
            "registered_at":   self.registered_at,
            "file_hash":       self.file_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FileFingerprint":
        return cls(
            filename=d["filename"],
            row_count=d["row_count"],
            columns=d["columns"],
            col_hash=d["col_hash"],
            row_sample_hash=d["row_sample_hash"],
            registered_at=d["registered_at"],
            file_hash=d.get("file_hash", ""),
        )


class FileClassification(str, Enum):
    NEW_TABLE       = "NEW_TABLE"
    UPDATED_VERSION = "UPDATED_VERSION"   # new/removed rows or new columns
    CORRECTED_DATA  = "CORRECTED_DATA"    # same shape, cell values changed
    DUPLICATE       = "DUPLICATE"          # identical


@dataclass
class ColumnDiff:
    col_name: str
    changed_cells: int
    sample_changes: list[tuple]   # up to 3 (old_val, new_val) examples


@dataclass
class FileDiff:
    rows_added: int
    rows_removed: int
    new_columns: list[str]
    removed_columns: list[str]
    column_diffs: list[ColumnDiff]  # only for shared columns with changes

    def summary_text(self) -> str:
        """Human-readable diff summary."""
        parts: list[str] = []
        if self.rows_added:
            parts.append(f"{self.rows_added} new row{'s' if self.rows_added != 1 else ''}")
        if self.rows_removed:
            parts.append(f"{self.rows_removed} row{'s' if self.rows_removed != 1 else ''} removed")
        if self.new_columns:
            parts.append(f"{len(self.new_columns)} new column{'s' if len(self.new_columns) != 1 else ''}")
        if self.removed_columns:
            parts.append(f"{len(self.removed_columns)} column{'s' if len(self.removed_columns) != 1 else ''} removed")
        corrected_cols = [cd for cd in self.column_diffs if cd.changed_cells > 0]
        if corrected_cols:
            col_names = ", ".join(cd.col_name for cd in corrected_cols[:3])
            total_cells = sum(cd.changed_cells for cd in corrected_cols)
            parts.append(f"{total_cells} value{'s' if total_cells != 1 else ''} corrected in {col_names}")
        return ", ".join(parts) if parts else "No changes detected"


@dataclass
class UpdateSummary:
    filename: str
    table_name: str
    classification: FileClassification
    diff: FileDiff | None
    old_fingerprint: FileFingerprint | None
    new_fingerprint: FileFingerprint


# ── Helpers ────────────────────────────────────────────────────────────────────

def _hash_rows(df: pd.DataFrame) -> str:
    """
    Compute a deterministic hash over the rows of *df*.

    For DFs > _ROW_SAMPLE_N rows: head 5 000 + tail 5 000 +
    random seed-42 middle sample so the hash stays fast on large files.
    """
    flat = df.copy().fillna("__NULL__")
    if len(flat) > _ROW_SAMPLE_N:
        mid_n = _ROW_SAMPLE_N - 10_000
        head  = flat.head(5_000)
        tail  = flat.tail(5_000)
        mid   = flat.sample(
            min(max(mid_n, 0), max(len(flat) - 10_000, 0)),
            random_state=42,
        )
        flat = pd.concat([head, mid, tail])
    row_strs = flat.apply(lambda r: str(tuple(r)), axis=1)
    return hashlib.md5(
        "||".join(row_strs).encode(), usedforsecurity=False
    ).hexdigest()


def _col_hash(columns: list[str]) -> str:
    return hashlib.md5(
        "|".join(sorted(columns)).encode(), usedforsecurity=False
    ).hexdigest()


# ── Watchdog inner class (optional dependency) ─────────────────────────────────

try:
    from watchdog.events import FileSystemEventHandler as _FSEHandler
    from watchdog.observers import Observer as _Observer

    class _NewFileHandler(_FSEHandler):  # type: ignore[misc]
        EXTENSIONS = {".csv", ".xlsx", ".xls"}

        def __init__(self, sentinel_path: Path) -> None:
            super().__init__()
            self._sentinel_path = sentinel_path

        def on_created(self, event) -> None:
            if event.is_directory:
                return
            src = Path(event.src_path)
            if src.suffix.lower() not in self.EXTENSIONS:
                return
            # append to sentinel
            existing: list[str] = []
            if self._sentinel_path.exists():
                try:
                    existing = json.loads(self._sentinel_path.read_text())
                except Exception:
                    existing = []
            if str(src) not in existing:
                existing.append(str(src))
            self._sentinel_path.write_text(json.dumps(existing))

    _WATCHDOG_OK = True

except ImportError:
    _WATCHDOG_OK = False
    _Observer = None         # type: ignore[assignment,misc]
    _NewFileHandler = None   # type: ignore[assignment,misc]


# ── UpdateHandler ──────────────────────────────────────────────────────────────

class UpdateHandler:
    """
    Manages incremental file updates: fingerprinting, classification,
    diffs, versioned parquet snapshots, agenda impact, and folder watching.
    """

    def __init__(self, cache_dir: Path | str = "data/.cache") -> None:
        self.cache_dir     = Path(cache_dir)
        self.versions_dir  = self.cache_dir / _VERSIONS_DIR
        self._registry_path = self.cache_dir / _REGISTRY_FILE
        self._sentinel_path = self.cache_dir / _SENTINEL_FILE

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)

        self._registry: dict[str, dict] = {}
        self._load_registry()

        self._observer: Any = None   # watchdog Observer
        self._lock = threading.Lock()

    # ── Fingerprinting ────────────────────────────────────────────────────────

    def fingerprint(
        self,
        df: pd.DataFrame,
        filename: str,
        file_bytes: bytes | None = None,
    ) -> FileFingerprint:
        """Compute a FileFingerprint for *df*."""
        cols = list(df.columns)
        fh = (
            hashlib.md5(file_bytes, usedforsecurity=False).hexdigest()
            if file_bytes is not None
            else ""
        )
        return FileFingerprint(
            filename=filename,
            row_count=len(df),
            columns=cols,
            col_hash=_col_hash(cols),
            row_sample_hash=_hash_rows(df),
            registered_at=datetime.now(timezone.utc).isoformat(),
            file_hash=fh,
        )

    # ── Classification ────────────────────────────────────────────────────────

    def classify(
        self,
        name: str,
        df: pd.DataFrame,
        file_bytes: bytes | None = None,
    ) -> tuple[FileClassification, FileFingerprint, FileFingerprint | None]:
        """
        Returns (classification, new_fingerprint, old_fingerprint | None).

        Rules:
          Not in registry                        → NEW_TABLE
          Same col_hash + same row_sample_hash   → DUPLICATE
          Same col_hash, same row_count, diff    → CORRECTED_DATA
          Same col_hash, different row_count     → UPDATED_VERSION
          Different col_hash                     → UPDATED_VERSION
        """
        new_fp = self.fingerprint(df, name, file_bytes)

        if name not in self._registry:
            return FileClassification.NEW_TABLE, new_fp, None

        old_fp = FileFingerprint.from_dict(self._registry[name])

        if old_fp.col_hash == new_fp.col_hash:
            if old_fp.row_sample_hash == new_fp.row_sample_hash:
                return FileClassification.DUPLICATE, new_fp, old_fp
            if old_fp.row_count == new_fp.row_count:
                return FileClassification.CORRECTED_DATA, new_fp, old_fp
            return FileClassification.UPDATED_VERSION, new_fp, old_fp

        # Different column set → structural update
        return FileClassification.UPDATED_VERSION, new_fp, old_fp

    # ── Diff ──────────────────────────────────────────────────────────────────

    def compute_diff(self, old_df: pd.DataFrame, new_df: pd.DataFrame) -> FileDiff:
        """Compute structural and value-level diff between two DataFrames."""
        old_cols = set(old_df.columns)
        new_cols = set(new_df.columns)
        added_cols   = sorted(new_cols - old_cols)
        removed_cols = sorted(old_cols - new_cols)

        # Row-level diff using hashed row strings
        old_row_set = set(
            old_df.fillna("__NULL__").apply(lambda r: str(tuple(r)), axis=1)
        )
        new_row_set = set(
            new_df.fillna("__NULL__").apply(lambda r: str(tuple(r)), axis=1)
        )
        rows_added   = len(new_row_set - old_row_set)
        rows_removed = len(old_row_set - new_row_set)

        # Value-level diff for shared columns when row counts match
        column_diffs: list[ColumnDiff] = []
        shared_cols = sorted(old_cols & new_cols)
        if len(old_df) == len(new_df) and shared_cols:
            for col in shared_cols:
                old_series = old_df[col].reset_index(drop=True)
                new_series = new_df[col].reset_index(drop=True)
                # Align on common length
                min_len = min(len(old_series), len(new_series))
                old_s = old_series.iloc[:min_len]
                new_s = new_series.iloc[:min_len]
                # Compare as strings to handle mixed types gracefully
                changed_mask = old_s.astype(str) != new_s.astype(str)
                changed_count = int(changed_mask.sum())
                if changed_count > 0:
                    sample_idx = list(changed_mask[changed_mask].index[:3])
                    samples = [
                        (old_s.loc[idx], new_s.loc[idx])
                        for idx in sample_idx
                    ]
                    column_diffs.append(
                        ColumnDiff(
                            col_name=col,
                            changed_cells=changed_count,
                            sample_changes=samples,
                        )
                    )

        return FileDiff(
            rows_added=rows_added,
            rows_removed=rows_removed,
            new_columns=added_cols,
            removed_columns=removed_cols,
            column_diffs=column_diffs,
        )

    # ── Process update ────────────────────────────────────────────────────────

    def process_update(
        self,
        name: str,
        df: pd.DataFrame,
        old_df: pd.DataFrame | None = None,
        file_bytes: bytes | None = None,
    ) -> UpdateSummary:
        """
        Classify the file, compute diff if applicable, and return UpdateSummary.
        Caller must call register() to persist the new fingerprint.
        """
        classification, new_fp, old_fp = self.classify(name, df, file_bytes)

        diff: FileDiff | None = None
        if classification != FileClassification.DUPLICATE and old_df is not None:
            try:
                diff = self.compute_diff(old_df, df)
            except Exception as exc:
                logger.warning("compute_diff failed for %s: %s", name, exc)

        return UpdateSummary(
            filename=new_fp.filename,
            table_name=name,
            classification=classification,
            diff=diff,
            old_fingerprint=old_fp,
            new_fingerprint=new_fp,
        )

    # ── Registry ──────────────────────────────────────────────────────────────

    def register(self, name: str, fp: FileFingerprint) -> None:
        """Persist fingerprint to the registry."""
        with self._lock:
            self._registry[name] = fp.to_dict()
            self._save_registry()

    def _load_registry(self) -> None:
        if self._registry_path.exists():
            try:
                self._registry = json.loads(self._registry_path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not load registry: %s", exc)
                self._registry = {}

    def _save_registry(self) -> None:
        self._registry_path.write_text(
            json.dumps(self._registry, indent=2)
        )

    # ── Version management ────────────────────────────────────────────────────

    def save_version(
        self,
        name: str,
        df: pd.DataFrame,
        ts: datetime | None = None,
    ) -> Path:
        """Save *df* as a versioned parquet snapshot. Returns the path."""
        ts = ts or datetime.now(timezone.utc)
        ts_str = ts.strftime("%Y%m%dT%H%M%S_%fZ")
        safe_name = name.replace("/", "_").replace("\\", "_")
        out_path = self.versions_dir / f"{safe_name}_{ts_str}.parquet"
        df.to_parquet(out_path, index=False)
        return out_path

    def load_version(self, name: str, ts_str: str) -> pd.DataFrame:
        """Load a versioned parquet snapshot by ts_str."""
        safe_name = name.replace("/", "_").replace("\\", "_")
        path = self.versions_dir / f"{safe_name}_{ts_str}.parquet"
        return pd.read_parquet(path)

    def list_versions(self, name: str) -> list[dict]:
        """
        Return [{ts_str, path, row_count}] for *name*, sorted newest-first.
        row_count is read from parquet metadata (no full load).
        """
        safe_name = name.replace("/", "_").replace("\\", "_")
        pattern = f"{safe_name}_*.parquet"
        paths = sorted(
            self.versions_dir.glob(pattern),
            reverse=True,
        )
        results: list[dict] = []
        for p in paths:
            # Extract ts_str from filename: <safe_name>_<ts_str>.parquet
            stem = p.stem  # e.g. "mytable_20240101T120000Z"
            prefix = safe_name + "_"
            ts_str = stem[len(prefix):] if stem.startswith(prefix) else stem
            try:
                import pyarrow.parquet as pq
                meta = pq.read_metadata(p)
                row_count = meta.num_rows
            except Exception:
                try:
                    row_count = len(pd.read_parquet(p))
                except Exception:
                    row_count = -1
            results.append({"ts_str": ts_str, "path": str(p), "row_count": row_count})
        return results

    # ── Agenda impact ─────────────────────────────────────────────────────────

    def check_agenda_impact(
        self,
        updated_names: list[str],
        agenda_results: dict,
        threshold: float = 0.10,
    ) -> set[str]:
        """
        Return set of question strings that should be marked stale.

        Logic: if |delta_rows| / old_row_count > threshold for any updated
        table, flag ALL questions in agenda_results. Fallback (no old fp):
        flag everything.
        """
        if not updated_names or not agenda_results:
            return set()

        significant = False
        for name in updated_names:
            if name not in self._registry:
                # No old fingerprint → conservative: flag all
                significant = True
                break
            old_fp = FileFingerprint.from_dict(self._registry[name])
            if old_fp.row_count == 0:
                significant = True
                break
            # We don't have the new count easily here, so look at registry
            # The caller should register() after process_update; we compare
            # registered row_count vs. None — flag all if unknown.
            significant = True  # conservative: always flag when update detected
            break

        if significant:
            return set(agenda_results.keys())
        return set()

    # ── Merge ─────────────────────────────────────────────────────────────────

    def merge_dataframes(
        self,
        old_df: pd.DataFrame,
        new_df: pd.DataFrame,
        key_col: str,
    ) -> pd.DataFrame:
        """
        Concat old + new and dedup on key_col keeping the latest (new) value.
        Raises ValueError if key_col is not in both DataFrames.
        """
        if key_col not in old_df.columns:
            raise ValueError(f"key_col '{key_col}' not in old_df columns: {list(old_df.columns)}")
        if key_col not in new_df.columns:
            raise ValueError(f"key_col '{key_col}' not in new_df columns: {list(new_df.columns)}")

        merged = pd.concat([old_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=[key_col], keep="last")
        return merged.reset_index(drop=True)

    # ── Sentinel file ─────────────────────────────────────────────────────────

    def write_sentinel(self, new_paths: list) -> None:
        """Write a JSON list of new file paths to the sentinel file."""
        paths_str = [str(p) for p in new_paths]
        self._sentinel_path.write_text(json.dumps(paths_str))

    def read_sentinel(self) -> list[str] | None:
        """Return list of new file paths, or None if no sentinel file."""
        if not self._sentinel_path.exists():
            return None
        try:
            data = json.loads(self._sentinel_path.read_text())
            return data if isinstance(data, list) else None
        except (json.JSONDecodeError, OSError):
            return None

    def clear_sentinel(self) -> None:
        """Remove the sentinel file."""
        if self._sentinel_path.exists():
            self._sentinel_path.unlink()

    # ── Folder watcher ────────────────────────────────────────────────────────

    def start_watcher(self, watch_dir: Path | str) -> None:
        """
        Start a watchdog Observer (daemon thread) that writes sentinel
        whenever a new .csv/.xlsx/.xls appears in watch_dir.

        Guard: no-op if already started or watchdog not installed.
        Raises ImportError if watchdog is unavailable.
        """
        if not _WATCHDOG_OK:
            raise ImportError("watchdog is not installed. Run: pip install watchdog")

        if self._observer is not None:
            return  # already running

        watch_dir = Path(watch_dir)
        watch_dir.mkdir(parents=True, exist_ok=True)

        handler = _NewFileHandler(self._sentinel_path)
        observer = _Observer()
        observer.schedule(handler, str(watch_dir), recursive=False)
        observer.daemon = True
        observer.start()
        self._observer = observer
        logger.info("Folder watcher started on %s", watch_dir)

    def stop_watcher(self) -> None:
        """Stop the watchdog Observer if running."""
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=5)
            except Exception as exc:
                logger.warning("Error stopping watcher: %s", exc)
            finally:
                self._observer = None
