"""
data/loader.py — Multi-format file loader with intelligent schema detection.

Features
--------
- Folder scan: discovers .csv, .xlsx, .xls files automatically
- Smart type inference (TypeInferrer):
    • currency   — strips £/$€¥₹ and comma separators → float
    • percentage — strips % suffix → float
    • date       — parses mixed date strings → datetime64
    • identifier — high-cardinality int/str with ID-like name
    • categorical — low-cardinality strings
    • numeric, text, email, phone
- TableSchema: per-file/per-column schema with null counts + sample values
- Parquet cache: instant reload on repeat runs (data/.cache/)
- LoadingSummary: human-readable load report

Public API
----------
DataLoader(config: dict)
    .scan_folder(folder)         -> list[Path]
    .load_folder(folder)         -> FolderLoadResult
    .load(file_path)             -> dict[str, pd.DataFrame]   # backward compat
    .load_many(paths)            -> dict[str, pd.DataFrame]   # backward compat
    .detect_schema(df, name='')  -> TableSchema
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from data.cache import CacheManager

logger = logging.getLogger(__name__)

# ── Column / semantic type constants ─────────────────────────────────────────

SUPPORTED_TABULAR = frozenset({".csv", ".xlsx", ".xls"})
SUPPORTED_DOCUMENT = frozenset({".pptx", ".pdf", ".png", ".jpg", ".jpeg"})

_CURRENCY_NAMES: frozenset[str] = frozenset({
    "revenue", "amount", "price", "value", "cost", "salary", "fee",
    "budget", "spend", "income", "profit", "margin", "arpu", "arr",
    "mrr", "ltv", "acv", "tcv", "deal_value", "contract_value",
    "invoice", "payment", "charge", "gross", "net",
})
_PERCENTAGE_NAMES: frozenset[str] = frozenset({
    "pct", "percent", "rate", "ratio", "win_rate", "churn", "conversion",
    "growth", "margin_pct", "discount", "tax", "utilisation", "utilization",
    "completion", "coverage",
})
_ID_SUFFIXES: tuple[str, ...] = ("_id", "_key", "_ref", "_code", "_no", "_number", "_uuid", "_guid")
_ID_PREFIXES: tuple[str, ...] = ("id_", "ref_", "uuid_")
_EMAIL_HINTS: tuple[str, ...] = ("email", "mail", "e_mail")
_PHONE_HINTS: tuple[str, ...] = ("phone", "mobile", "tel", "fax", "cell")

_CURRENCY_SYMBOL_RE = re.compile(r"[£$€¥₹]")
_CURRENCY_VALUE_RE = re.compile(r"^[£$€¥₹\s]?-?[\d,]+\.?\d*$")
_PERCENTAGE_VALUE_RE = re.compile(r"^-?[\d,]+\.?\d*\s*%$")

_CATEGORICAL_MAX_UNIQUE = 50       # absolute cap
_CATEGORICAL_MAX_UNIQUE_RATIO = 0.10  # 10% of rows
_DATE_MIN_PARSE_RATE = 0.70        # 70% of non-nulls must parse as date
_CURRENCY_MIN_MATCH_RATE = 0.40    # 40% of non-nulls match currency pattern
_PERCENTAGE_MIN_MATCH_RATE = 0.40


# ── Schema dataclasses ────────────────────────────────────────────────────────

@dataclass
class ColumnSchema:
    name: str
    raw_dtype: str          # dtype as read from file (before casting)
    inferred_type: str      # 'currency' | 'percentage' | 'date' | 'identifier'
                            # | 'categorical' | 'numeric' | 'text' | 'email' | 'phone'
    null_count: int
    null_pct: float
    unique_count: int
    sample_values: list[Any]   # up to 3 non-null values (post-cast)

    def __str__(self) -> str:
        return (
            f"  {self.name!s:<30} {self.inferred_type:<12} "
            f"nulls={self.null_pct:.1f}%  unique={self.unique_count}  "
            f"samples={self.sample_values}"
        )


@dataclass
class TableSchema:
    filename: str
    sheet: str | None
    row_count: int
    col_count: int
    columns: list[ColumnSchema]
    cache_hit: bool = False
    load_time_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    # convenience lookups
    def column(self, name: str) -> ColumnSchema | None:
        return next((c for c in self.columns if c.name == name), None)

    def columns_of_type(self, inferred_type: str) -> list[ColumnSchema]:
        return [c for c in self.columns if c.inferred_type == inferred_type]

    def __str__(self) -> str:
        header = (
            f"{'[CACHED] ' if self.cache_hit else ''}"
            f"{self.filename}  ({self.row_count:,} rows × {self.col_count} cols)"
            f"  loaded in {self.load_time_ms:.0f}ms"
        )
        col_lines = "\n".join(str(c) for c in self.columns)
        return f"{header}\n{col_lines}"


@dataclass
class LoadingSummary:
    folder: str
    files_found: int
    files_loaded: int
    files_failed: int
    total_rows: int
    total_columns: int
    cache_hits: int
    cache_misses: int
    elapsed_ms: float
    errors: list[tuple[str, str]] = field(default_factory=list)   # (filename, msg)

    def print(self) -> None:
        lines = [
            "",
            "═" * 60,
            f"  CRM DATA LOADER — {self.folder}",
            "═" * 60,
            f"  Files found   : {self.files_found}",
            f"  Files loaded  : {self.files_loaded}",
            f"  Files failed  : {self.files_failed}",
            f"  Total rows    : {self.total_rows:,}",
            f"  Total columns : {self.total_columns}",
            f"  Cache hits    : {self.cache_hits}",
            f"  Cache misses  : {self.cache_misses}",
            f"  Elapsed       : {self.elapsed_ms:.0f} ms",
        ]
        if self.errors:
            lines.append("  Errors:")
            for fname, msg in self.errors:
                lines.append(f"    ✗ {fname}: {msg}")
        lines.append("═" * 60)
        print("\n".join(lines))


@dataclass
class FolderLoadResult:
    dataframes: dict[str, pd.DataFrame]
    schemas: dict[str, TableSchema]
    summary: LoadingSummary

    def print_summary(self) -> None:
        self.summary.print()
        for schema in self.schemas.values():
            print(f"\n{schema}")


# ── TypeInferrer ──────────────────────────────────────────────────────────────

class TypeInferrer:
    """
    Intelligently infers and casts column types in a raw DataFrame.

    Detection order (object/string columns):
      1. currency   — name keyword or >40% values match £$€¥₹ / comma-number pattern
      2. percentage — name keyword or >40% values end with %
      3. date       — pd.to_datetime succeeds for ≥70% of non-null values
      4. email      — name hint
      5. phone      — name hint
      6. identifier — name ends with _id/_key/etc. AND unique_pct > 80%
      7. categorical — unique_count ≤ 50 (or ≤10% of rows)
      8. text       — everything else

    For already-numeric columns:
      1. currency   — name keyword
      2. percentage — name keyword
      3. identifier — name hint + high uniqueness
      4. numeric    — default
    """

    def infer_and_cast(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[ColumnSchema]]:
        """
        Return (cast_df, column_schemas).
        cast_df has currencies as float, percentages as float, dates as datetime64.
        """
        df = df.copy()
        schemas: list[ColumnSchema] = []

        for col in df.columns:
            raw_dtype = str(df[col].dtype)
            series = df[col]
            inferred_type, cast_series = self._infer_column(series)
            df[col] = cast_series
            schemas.append(self._build_column_schema(col, raw_dtype, inferred_type, cast_series))

        return df, schemas

    # ── Per-column dispatch ───────────────────────────────────────────────────

    def _infer_column(self, series: pd.Series) -> tuple[str, pd.Series]:
        name = str(series.name).lower()

        # Already datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return "date", series

        # Bool — must be checked before numeric (bool is a numeric subtype in numpy)
        if pd.api.types.is_bool_dtype(series):
            return "categorical", series

        # Numeric (int / float) — type already set by pandas
        if pd.api.types.is_numeric_dtype(series):
            return self._infer_numeric(series, name), series

        # Object/string — the interesting path
        return self._infer_object(series, name)

    def _infer_numeric(self, series: pd.Series, name: str) -> str:
        if self._name_is_currency(name):
            return "currency"
        if self._name_is_percentage(name):
            return "percentage"
        if self._name_is_id(name):
            unique_pct = series.nunique() / max(len(series.dropna()), 1)
            if unique_pct > 0.80:
                return "identifier"
        return "numeric"

    def _infer_object(self, series: pd.Series, name: str) -> tuple[str, pd.Series]:
        non_null = series.dropna()

        # 1 — Currency
        if self._name_is_currency(name) or self._values_match_currency(non_null):
            cast = self._cast_currency(series)
            if cast is not None:
                return "currency", cast

        # 2 — Percentage
        if self._name_is_percentage(name) or self._values_match_percentage(non_null):
            cast = self._cast_percentage(series)
            if cast is not None:
                return "percentage", cast

        # 3 — Date
        cast_date = self._try_date(series)
        if cast_date is not None:
            return "date", cast_date

        # 4 — Email / Phone (name hints only — no expensive value scan)
        if any(h in name for h in _EMAIL_HINTS):
            return "email", series
        if any(h in name for h in _PHONE_HINTS):
            return "phone", series

        # 5 — Numeric string
        cast_num = pd.to_numeric(series, errors="coerce")
        if cast_num.notna().sum() / max(len(non_null), 1) > 0.90:
            return "numeric", cast_num

        # 6 — Identifier
        if self._name_is_id(name):
            unique_pct = series.nunique() / max(len(non_null), 1)
            if unique_pct > 0.80:
                return "identifier", series

        # 7 — Categorical vs text
        n_unique = series.nunique()
        n_rows = max(len(series), 1)
        if n_unique <= _CATEGORICAL_MAX_UNIQUE or n_unique / n_rows <= _CATEGORICAL_MAX_UNIQUE_RATIO:
            return "categorical", series

        return "text", series

    # ── Currency ──────────────────────────────────────────────────────────────

    def _name_is_currency(self, name: str) -> bool:
        return any(kw in name for kw in _CURRENCY_NAMES)

    def _values_match_currency(self, non_null: pd.Series) -> bool:
        if len(non_null) == 0:
            return False
        str_vals = non_null.astype(str).str.strip()
        match_count = str_vals.str.match(_CURRENCY_VALUE_RE).sum()
        has_symbol = str_vals.str.contains(_CURRENCY_SYMBOL_RE).any()
        return has_symbol and (match_count / len(non_null) >= _CURRENCY_MIN_MATCH_RATE)

    def _cast_currency(self, series: pd.Series) -> pd.Series | None:
        cleaned = (
            series.astype(str)
            .str.strip()
            .str.replace(_CURRENCY_SYMBOL_RE, "", regex=True)
            .str.replace(",", "", regex=False)
            .str.replace(r"\s", "", regex=True)
        )
        result = pd.to_numeric(cleaned, errors="coerce").astype(float)
        if result.notna().sum() == 0:
            return None
        return result

    # ── Percentage ────────────────────────────────────────────────────────────

    def _name_is_percentage(self, name: str) -> bool:
        return any(kw in name for kw in _PERCENTAGE_NAMES)

    def _values_match_percentage(self, non_null: pd.Series) -> bool:
        if len(non_null) == 0:
            return False
        str_vals = non_null.astype(str).str.strip()
        match_count = str_vals.str.match(_PERCENTAGE_VALUE_RE).sum()
        return match_count / len(non_null) >= _PERCENTAGE_MIN_MATCH_RATE

    def _cast_percentage(self, series: pd.Series) -> pd.Series | None:
        cleaned = (
            series.astype(str)
            .str.strip()
            .str.rstrip("%")
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        result = pd.to_numeric(cleaned, errors="coerce").astype(float)
        if result.notna().sum() == 0:
            return None
        return result

    # ── Date ─────────────────────────────────────────────────────────────────

    def _try_date(self, series: pd.Series) -> pd.Series | None:
        non_null = series.dropna()
        if len(non_null) == 0:
            return None
        # Quick sanity: skip columns that look nothing like dates
        sample = str(non_null.iloc[0])
        if not any(c.isdigit() for c in sample):
            return None
        # Try explicit formats first to avoid pandas 2.x UserWarnings
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y", "%Y-%m-%dT%H:%M:%S"):
            try:
                parsed = pd.to_datetime(series, errors="coerce", format=fmt)
                if parsed.notna().sum() / len(non_null) >= _DATE_MIN_PARSE_RATE:
                    return parsed
            except Exception:
                continue
        # Fallback: suppress pandas inference warnings
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
        if parsed.notna().sum() / len(non_null) >= _DATE_MIN_PARSE_RATE:
            return parsed
        return None

    # ── Identifier ───────────────────────────────────────────────────────────

    def _name_is_id(self, name: str) -> bool:
        return (
            any(name.endswith(s) for s in _ID_SUFFIXES)
            or any(name.startswith(p) for p in _ID_PREFIXES)
            or name in {"id", "key", "ref", "code", "uuid", "guid"}
        )

    # ── ColumnSchema builder ──────────────────────────────────────────────────

    @staticmethod
    def _build_column_schema(
        col: str,
        raw_dtype: str,
        inferred_type: str,
        cast_series: pd.Series,
    ) -> ColumnSchema:
        non_null = cast_series.dropna()
        sample = [
            v.isoformat() if hasattr(v, "isoformat") else v
            for v in non_null.head(3).tolist()
        ]
        return ColumnSchema(
            name=col,
            raw_dtype=raw_dtype,
            inferred_type=inferred_type,
            null_count=int(cast_series.isna().sum()),
            null_pct=round(cast_series.isna().mean() * 100, 2),
            unique_count=int(cast_series.nunique()),
            sample_values=sample,
        )


# ── DataLoader ────────────────────────────────────────────────────────────────

class DataLoader:
    """
    Orchestrates folder scanning, file loading, type inference, and caching.

    Usage
    -----
    loader = DataLoader(config)

    # New primary API (Module 2)
    result = loader.load_folder("data/uploads/")
    result.print_summary()
    df_accounts = result.dataframes["accounts"]

    # Legacy API (backward-compatible with Module 1 tests)
    dfs = loader.load("data/uploads/accounts.csv")
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        data_cfg = config.get("data", {})
        self.upload_folder = Path(data_cfg.get("upload_folder", "data/uploads"))
        self.upload_folder.mkdir(parents=True, exist_ok=True)
        cache_folder = Path(data_cfg.get("cache_folder", "data/.cache"))
        self.cache = CacheManager(cache_folder)
        self.inferrer = TypeInferrer()

    # ── Primary API ───────────────────────────────────────────────────────────

    def scan_folder(self, folder: Path | str) -> list[Path]:
        """Return all supported tabular files in *folder* (non-recursive)."""
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        found = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_TABULAR
        )
        logger.info("scan_folder(%s): found %d file(s)", folder, len(found))
        return found

    def load_folder(self, folder: Path | str) -> FolderLoadResult:
        """
        Scan *folder*, load every supported file, and return a FolderLoadResult.
        Results are cached as Parquet; subsequent calls return instantly.
        """
        t0 = time.perf_counter()
        paths = self.scan_folder(folder)

        all_dfs: dict[str, pd.DataFrame] = {}
        all_schemas: dict[str, TableSchema] = {}
        errors: list[tuple[str, str]] = []
        cache_hits = 0
        cache_misses = 0
        total_rows = 0
        total_cols = 0

        for path in paths:
            try:
                file_dfs, file_schemas = self._load_file_with_schema(path)
                for name, df in file_dfs.items():
                    all_dfs[name] = df
                    all_schemas[name] = file_schemas[name]
                    total_rows += len(df)
                    total_cols += len(df.columns)
                    if file_schemas[name].cache_hit:
                        cache_hits += 1
                    else:
                        cache_misses += 1
            except Exception as exc:
                logger.warning("Failed to load %s: %s", path.name, exc)
                errors.append((path.name, str(exc)))

        elapsed = (time.perf_counter() - t0) * 1000
        summary = LoadingSummary(
            folder=str(folder),
            files_found=len(paths),
            files_loaded=len(paths) - len(errors),
            files_failed=len(errors),
            total_rows=total_rows,
            total_columns=total_cols,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            elapsed_ms=round(elapsed, 1),
            errors=errors,
        )
        return FolderLoadResult(dataframes=all_dfs, schemas=all_schemas, summary=summary)

    def detect_schema(self, df: pd.DataFrame, name: str = "") -> TableSchema:
        """Build a TableSchema for an already-loaded DataFrame."""
        _, col_schemas = self.inferrer.infer_and_cast(df)
        return TableSchema(
            filename=name,
            sheet=None,
            row_count=len(df),
            col_count=len(df.columns),
            columns=col_schemas,
        )

    # ── Legacy API (backward-compatible) ─────────────────────────────────────

    def load(self, file_path: str | Path) -> dict[str, pd.DataFrame]:
        """Load a single file and return {name: DataFrame}."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_TABULAR | SUPPORTED_DOCUMENT:
            raise ValueError(f"Unsupported file type: {suffix!r}")
        if suffix in SUPPORTED_DOCUMENT:
            df = self._load_document(path)
            return {path.stem: df}
        file_dfs, _ = self._load_file_with_schema(path)
        return file_dfs

    def load_many(self, paths: list[str | Path]) -> dict[str, pd.DataFrame]:
        """Load multiple files, merging results into one dict."""
        all_dfs: dict[str, pd.DataFrame] = {}
        for p in paths:
            try:
                all_dfs.update(self.load(p))
            except Exception as exc:
                logger.warning("Skipping %s — %s", p, exc)
        return all_dfs

    # ── File loading + caching ────────────────────────────────────────────────

    def _load_file_with_schema(
        self, path: Path
    ) -> tuple[dict[str, pd.DataFrame], dict[str, TableSchema]]:
        """Load one file (possibly multi-sheet). Returns ({name: df}, {name: schema})."""
        suffix = path.suffix.lower()
        t0 = time.perf_counter()

        if suffix == ".csv":
            raw_map = {path.stem: pd.read_csv(path, low_memory=False)}
        elif suffix in (".xlsx", ".xls"):
            raw_map = self._read_excel_raw(path)
        else:
            raise ValueError(f"Unsupported tabular format: {suffix!r}")

        result_dfs: dict[str, pd.DataFrame] = {}
        result_schemas: dict[str, TableSchema] = {}

        for name, raw_df in raw_map.items():
            sheet = name.split("__", 1)[1] if "__" in name else None
            cache_key = CacheManager.make_key(path, path.stat().st_mtime)
            if len(raw_map) > 1:
                # Make sheet-specific cache key
                import hashlib
                cache_key = hashlib.sha1(f"{cache_key}::{sheet}".encode()).hexdigest()[:16]

            cached_df = self.cache.get(cache_key)
            if cached_df is not None:
                # Rebuild schema from cached (already cast) df
                _, col_schemas = self.inferrer.infer_and_cast(cached_df)
                schema = TableSchema(
                    filename=path.name,
                    sheet=sheet,
                    row_count=len(cached_df),
                    col_count=len(cached_df.columns),
                    columns=col_schemas,
                    cache_hit=True,
                    load_time_ms=round((time.perf_counter() - t0) * 1000, 1),
                )
                result_dfs[name] = cached_df
                result_schemas[name] = schema
                logger.info("Cache hit: %s", name)
            else:
                cast_df, col_schemas = self.inferrer.infer_and_cast(raw_df)
                self.cache.put(cache_key, cast_df)
                schema = TableSchema(
                    filename=path.name,
                    sheet=sheet,
                    row_count=len(cast_df),
                    col_count=len(cast_df.columns),
                    columns=col_schemas,
                    cache_hit=False,
                    load_time_ms=round((time.perf_counter() - t0) * 1000, 1),
                )
                result_dfs[name] = cast_df
                result_schemas[name] = schema
                logger.info("Loaded+cached: %s (%d rows)", name, len(cast_df))

        return result_dfs, result_schemas

    def _read_excel_raw(self, path: Path) -> dict[str, pd.DataFrame]:
        sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
        result = {}
        for sheet_name, df in sheets.items():
            key = f"{path.stem}__{sheet_name}" if len(sheets) > 1 else path.stem
            result[key] = df
        return result

    def _load_document(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".pptx":
            from formats.ppt_handler import PPTHandler
            return PPTHandler().extract(path)
        if suffix == ".pdf":
            from formats.pdf_handler import PDFHandler
            return PDFHandler().extract(path)
        if suffix in (".png", ".jpg", ".jpeg"):
            from formats.image_handler import ImageHandler
            return ImageHandler().extract(path)
        raise ValueError(f"No document handler for {suffix!r}")
