"""
data/cache.py — Parquet-based DataFrame cache management.

Caches loaded DataFrames as Parquet files for fast re-loading.
Keyed by a hash of the source file path + mtime.

Public API
----------
CacheManager(cache_folder: str | Path)
    .get(key: str) -> pd.DataFrame | None
    .put(key: str, df: pd.DataFrame) -> None
    .invalidate(key: str) -> None
    .clear_all() -> None
    .list_cached() -> list[str]
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Stores DataFrames as Parquet files under cache_folder/.
    Keys are arbitrary strings (typically a hash of source path + mtime).
    """

    def __init__(self, cache_folder: str | Path) -> None:
        self.cache_folder = Path(cache_folder)
        self.cache_folder.mkdir(parents=True, exist_ok=True)

    # ── Public ────────────────────────────────────────────────────────────────

    def get(self, key: str) -> pd.DataFrame | None:
        """Return cached DataFrame for *key*, or None if not cached."""
        path = self._key_to_path(key)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            logger.debug("Cache hit: %s", key)
            return df
        except Exception as exc:
            logger.warning("Cache read failed for %s: %s", key, exc)
            return None

    def put(self, key: str, df: pd.DataFrame) -> None:
        """Write *df* to cache under *key*."""
        path = self._key_to_path(key)
        try:
            df.to_parquet(path, index=True, engine="pyarrow")
            logger.debug("Cache written: %s → %s", key, path)
        except Exception as exc:
            logger.warning("Cache write failed for %s: %s", key, exc)

    def invalidate(self, key: str) -> None:
        """Remove a single cache entry."""
        path = self._key_to_path(key)
        if path.exists():
            path.unlink()
            logger.info("Cache invalidated: %s", key)

    def clear_all(self) -> None:
        """Delete all cached Parquet files."""
        for p in self.cache_folder.glob("*.parquet"):
            p.unlink()
        logger.info("Cache cleared: %s", self.cache_folder)

    def list_cached(self) -> list[str]:
        """Return all cached keys (filename stems)."""
        return [p.stem for p in self.cache_folder.glob("*.parquet")]

    @staticmethod
    def make_key(file_path: Path | str, mtime: float) -> str:
        """Create a deterministic cache key from file path + modification time."""
        raw = f"{Path(file_path).resolve()}::{mtime}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    # ── Private ───────────────────────────────────────────────────────────────

    def _key_to_path(self, key: str) -> Path:
        return self.cache_folder / f"{key}.parquet"
