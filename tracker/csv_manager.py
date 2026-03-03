"""
tracker/csv_manager.py — Auto-export query history to CSV.

Reads from TrackerDB and writes timestamped CSVs to exports/.

Public API
----------
CSVManager(db: TrackerDB, output_folder: str | Path)
    .export_query_log(filename: str | None) -> Path
    .export_patterns(filename: str | None) -> Path
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class CSVManager:
    """Exports tracker data to CSV files for external review."""

    def __init__(self, db, output_folder: str | Path = "exports") -> None:
        self.db = db
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def export_query_log(self, filename: str | None = None) -> Path:
        """Write all query history rows to a CSV. Returns the file path."""
        rows = self.db.get_recent_queries(n=10_000)
        df = pd.DataFrame(rows)
        fname = filename or f"query_log_{_ts()}.csv"
        out = self.output_folder / fname
        df.to_csv(out, index=False)
        logger.info("Query log exported: %s (%d rows)", out, len(df))
        return out

    def export_patterns(self, filename: str | None = None) -> Path:
        """Write all code patterns to a CSV. Returns the file path."""
        rows = self.db.get_patterns()
        df = pd.DataFrame(rows)
        fname = filename or f"patterns_{_ts()}.csv"
        out = self.output_folder / fname
        df.to_csv(out, index=False)
        logger.info("Patterns exported: %s (%d rows)", out, len(df))
        return out


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
