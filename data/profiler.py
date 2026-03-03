"""
data/profiler.py — Automatic data quality profiler for CRM DataFrames.

Runs on every load. Results feed the Streamlit sidebar panel, the LLM
context (so quality-related agenda questions are answered from real data),
and a plain-text export written to exports/data_quality_report.txt.

Quality rules
-------------
  HIGH_NULLS          — column null % > 30
  FULLY_EMPTY         — column null % = 100
  FUTURE_DATES        — datetime column contains dates beyond today
  ZERO_REVENUE        — currency column contains zero values
  NEGATIVE_REVENUE    — currency column contains negative values
  DUPLICATE_ROWS      — table has duplicate rows
  HIGH_OVERALL_NULL   — table-wide average null % > 30

Public API
----------
DataProfiler(config=None)
    .profile(df, name, schema=None)         -> DataProfile
    .profile_all(dataframes, schemas=None)  -> dict[str, DataProfile]
    .build_quality_report(profiles)         -> dict        # LLM-injectable
    .export_report(profiles, output_folder) -> Path        # writes .txt file

render_profile_ui(profiles, key_prefix="profiler")          # Streamlit sidebar
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────

NULL_WARN_THRESHOLD: float = 30.0      # % per column
NULL_ERROR_THRESHOLD: float = 100.0    # % — fully empty column
DUPE_WARN_THRESHOLD: float = 0.5      # % duplicate rows to warn
DUPE_ERROR_THRESHOLD: float = 5.0     # % duplicate rows to error
OVERALL_NULL_WARN: float = 30.0       # % average across all columns

# Issue severity strings
SEV_ERROR = "error"
SEV_WARNING = "warning"
SEV_INFO = "info"

# Rule codes
RULE_HIGH_NULLS = "HIGH_NULLS"
RULE_FULLY_EMPTY = "FULLY_EMPTY"
RULE_FUTURE_DATES = "FUTURE_DATES"
RULE_ZERO_REVENUE = "ZERO_REVENUE"
RULE_NEGATIVE_REVENUE = "NEGATIVE_REVENUE"
RULE_DUPLICATE_ROWS = "DUPLICATE_ROWS"
RULE_HIGH_OVERALL_NULL = "HIGH_OVERALL_NULL"

# Currency column name keywords (mirrors TypeInferrer)
_CURRENCY_NAMES: frozenset[str] = frozenset({
    "revenue", "amount", "price", "value", "cost", "salary", "fee",
    "budget", "spend", "income", "profit", "margin", "arpu", "arr",
    "mrr", "ltv", "acv", "tcv", "deal_value", "contract_value",
    "invoice", "payment", "charge", "gross", "net",
})

_SEVERITY_ICON = {SEV_ERROR: "🔴", SEV_WARNING: "🟡", SEV_INFO: "🔵"}
_SEVERITY_RANK = {SEV_ERROR: 0, SEV_WARNING: 1, SEV_INFO: 2}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DataQualityIssue:
    """A single data quality finding."""
    severity: str           # SEV_ERROR | SEV_WARNING | SEV_INFO
    table: str
    column: str | None      # None for table-level issues
    rule: str               # RULE_* constant
    description: str
    affected_rows: int      # count of bad rows (0 for structural issues)
    affected_pct: float     # fraction of table rows (0–100)

    @property
    def icon(self) -> str:
        return _SEVERITY_ICON.get(self.severity, "❓")

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "column": self.column,
            "rule": self.rule,
            "description": self.description,
            "affected_rows": self.affected_rows,
            "affected_pct": round(self.affected_pct, 2),
        }


@dataclass
class ColumnProfile:
    """Complete per-column quality profile."""
    name: str
    dtype: str
    inferred_type: str       # from schema or name heuristic
    null_count: int
    null_pct: float
    unique_count: int
    # Numeric stats
    min: Any = None
    max: Any = None
    mean: float | None = None
    std: float | None = None
    median: float | None = None
    zero_count: int = 0
    negative_count: int = 0
    # Categorical / text stats
    top_values: list[tuple[Any, int]] = field(default_factory=list)
    # Date stats
    future_date_count: int = 0
    # Issues found in this column
    issues: list[DataQualityIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self.inferred_type,
            "null_pct": self.null_pct,
            "unique_count": self.unique_count,
        }
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        if self.mean is not None:
            d["mean"] = round(self.mean, 2)
        if self.median is not None:
            d["median"] = round(self.median, 2)
        if self.zero_count:
            d["zero_count"] = self.zero_count
        if self.negative_count:
            d["negative_count"] = self.negative_count
        if self.top_values:
            d["top_values"] = self.top_values[:5]
        if self.future_date_count:
            d["future_date_count"] = self.future_date_count
        return d


@dataclass
class DataProfile:
    """Complete quality profile for one DataFrame."""
    name: str
    row_count: int
    col_count: int
    columns: list[ColumnProfile]
    duplicate_rows: int
    duplicate_pct: float
    overall_null_pct: float
    fully_empty_columns: list[str]
    issues: list[DataQualityIssue]         # table-level issues only
    profiled_at: str = field(default_factory=lambda: _now_iso())

    # ── Convenience accessors ─────────────────────────────────────────────────

    def all_issues(self) -> list[DataQualityIssue]:
        """Table-level + all column-level issues, sorted by severity."""
        col_issues = [i for cp in self.columns for i in cp.issues]
        combined = self.issues + col_issues
        return sorted(combined, key=lambda i: _SEVERITY_RANK.get(i.severity, 9))

    def issue_count(self, severity: str | None = None) -> int:
        issues = self.all_issues()
        if severity:
            issues = [i for i in issues if i.severity == severity]
        return len(issues)

    def column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)

    def columns_with_issues(self) -> list[ColumnProfile]:
        return [c for c in self.columns if c.issues]

    # ── LLM-injectable report dict ────────────────────────────────────────────

    def to_quality_report(self) -> dict[str, Any]:
        return {
            "rows": self.row_count,
            "columns": self.col_count,
            "duplicate_rows": self.duplicate_rows,
            "duplicate_pct": round(self.duplicate_pct, 2),
            "overall_null_pct": round(self.overall_null_pct, 2),
            "fully_empty_columns": self.fully_empty_columns,
            "issues": [i.to_dict() for i in self.all_issues()],
            "column_stats": {c.name: c.to_dict() for c in self.columns},
        }

    # ── Text export ───────────────────────────────────────────────────────────

    def to_text(self) -> str:
        lines: list[str] = [
            f"TABLE: {self.name}",
            f"  Rows          : {self.row_count:,}",
            f"  Columns       : {self.col_count}",
            f"  Duplicate rows: {self.duplicate_rows} ({self.duplicate_pct:.1f}%)",
            f"  Overall nulls : {self.overall_null_pct:.1f}%",
            f"  Fully empty   : {', '.join(self.fully_empty_columns) or 'none'}",
        ]
        all_iss = self.all_issues()
        if all_iss:
            lines.append(f"\n  Issues ({len(all_iss)}):")
            for iss in all_iss:
                col_label = f"[{iss.column}]" if iss.column else "[table]"
                lines.append(
                    f"    {iss.icon} [{iss.severity.upper():<7}] "
                    f"{col_label:<25} {iss.rule} — {iss.description}"
                )
        lines.append("\n  Column details:")
        for cp in self.columns:
            lines.append(self._column_text_line(cp))
        return "\n".join(lines)

    @staticmethod
    def _column_text_line(cp: ColumnProfile) -> str:
        parts = [f"    {cp.name:<30} | {cp.inferred_type:<12} | nulls={cp.null_pct:.1f}%"]
        if cp.mean is not None:
            parts.append(f"min={cp.min:.0f} max={cp.max:.0f} mean={cp.mean:.0f}")
        elif cp.min is not None:
            parts.append(f"min={cp.min} max={cp.max}")
        if cp.top_values:
            tv = ", ".join(f"{v}({c})" for v, c in cp.top_values[:3])
            parts.append(f"top=[{tv}]")
        if cp.future_date_count:
            parts.append(f"future={cp.future_date_count}")
        if cp.issues:
            parts.append(f"⚠×{len(cp.issues)}")
        return "  ".join(parts)


# ── DataProfiler ──────────────────────────────────────────────────────────────

class DataProfiler:
    """
    Profiles DataFrames for data quality and generates LLM-ready reports.

    Usage
    -----
    profiler = DataProfiler(config)
    profiles = profiler.profile_all(result.dataframes, result.schemas)
    report   = profiler.build_quality_report(profiles)
    path     = profiler.export_report(profiles)
    """

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        out_cfg = self.config.get("exports", {})
        self.output_folder = Path(out_cfg.get("output_folder", "exports"))

    # ── Public ────────────────────────────────────────────────────────────────

    def profile(
        self,
        df: pd.DataFrame,
        name: str = "dataframe",
        schema: Any | None = None,    # TableSchema from loader (optional)
    ) -> DataProfile:
        """
        Profile a single DataFrame.  Pass *schema* (TableSchema) to use
        inferred column types from the loader; otherwise types are detected
        by column-name heuristics.
        """
        schema_map: dict[str, Any] = {}
        if schema is not None and hasattr(schema, "columns"):
            schema_map = {c.name: c for c in schema.columns}

        col_profiles = [
            self._profile_column(df[col], name, schema_map.get(col))
            for col in df.columns
        ]

        dup_count = int(df.duplicated().sum())
        dup_pct = round(dup_count / max(len(df), 1) * 100, 2)
        overall_null = round(float(df.isna().mean().mean() * 100), 2)
        empty_cols = [c.name for c in col_profiles if c.null_pct == 100.0]

        tbl_issues: list[DataQualityIssue] = []

        # DUPLICATE_ROWS
        if dup_count > 0:
            sev = SEV_ERROR if dup_pct >= DUPE_ERROR_THRESHOLD else SEV_WARNING
            tbl_issues.append(DataQualityIssue(
                severity=sev, table=name, column=None,
                rule=RULE_DUPLICATE_ROWS,
                description=f"{dup_count} duplicate rows found ({dup_pct:.1f}%)",
                affected_rows=dup_count, affected_pct=dup_pct,
            ))

        # HIGH_OVERALL_NULL
        if overall_null > OVERALL_NULL_WARN:
            tbl_issues.append(DataQualityIssue(
                severity=SEV_WARNING, table=name, column=None,
                rule=RULE_HIGH_OVERALL_NULL,
                description=f"Table-wide null rate is {overall_null:.1f}%",
                affected_rows=0, affected_pct=overall_null,
            ))

        return DataProfile(
            name=name,
            row_count=len(df),
            col_count=len(df.columns),
            columns=col_profiles,
            duplicate_rows=dup_count,
            duplicate_pct=dup_pct,
            overall_null_pct=overall_null,
            fully_empty_columns=empty_cols,
            issues=tbl_issues,
        )

    def profile_all(
        self,
        dataframes: dict[str, pd.DataFrame],
        schemas: dict[str, Any] | None = None,
    ) -> dict[str, DataProfile]:
        """Profile every DataFrame. Optionally pass schemas dict from FolderLoadResult."""
        schemas = schemas or {}
        return {
            name: self.profile(df, name, schemas.get(name))
            for name, df in dataframes.items()
        }

    def build_quality_report(
        self, profiles: dict[str, DataProfile]
    ) -> dict[str, Any]:
        """
        Return a structured dict suitable for injection into the LLM system prompt.
        Also used by context_builder.py.
        """
        total_issues = sum(p.issue_count() for p in profiles.values())
        total_errors = sum(p.issue_count(SEV_ERROR) for p in profiles.values())
        total_warnings = sum(p.issue_count(SEV_WARNING) for p in profiles.values())
        total_infos = sum(p.issue_count(SEV_INFO) for p in profiles.values())

        return {
            "generated_at": _now_iso(),
            "summary": {
                "total_tables": len(profiles),
                "total_rows": sum(p.row_count for p in profiles.values()),
                "total_issues": total_issues,
                "errors": total_errors,
                "warnings": total_warnings,
                "infos": total_infos,
            },
            "tables": {name: p.to_quality_report() for name, p in profiles.items()},
        }

    def export_report(
        self,
        profiles: dict[str, DataProfile],
        output_folder: str | Path | None = None,
    ) -> Path:
        """
        Write a plain-text quality report to exports/data_quality_report.txt.
        Returns the path of the written file.
        """
        folder = Path(output_folder) if output_folder else self.output_folder
        folder.mkdir(parents=True, exist_ok=True)
        out_path = folder / "data_quality_report.txt"

        report = self.build_quality_report(profiles)
        summary = report["summary"]

        lines: list[str] = [
            "=" * 68,
            "  DATA QUALITY REPORT",
            f"  Generated : {report['generated_at']}",
            "=" * 68,
            "",
        ]
        for p in profiles.values():
            lines.append(p.to_text())
            lines.append("")

        lines += [
            "=" * 68,
            "  SUMMARY",
            f"  Tables        : {summary['total_tables']}",
            f"  Total rows    : {summary['total_rows']:,}",
            f"  Total issues  : {summary['total_issues']}",
            f"  Errors        : {summary['errors']}",
            f"  Warnings      : {summary['warnings']}",
            f"  Infos         : {summary['infos']}",
            "=" * 68,
        ]

        out_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Quality report written: %s", out_path)
        return out_path

    def summary_markdown(self, profiles: dict[str, DataProfile]) -> str:
        """Backward-compatible markdown table summary."""
        lines = [
            "| Dataset | Rows | Cols | Nulls% | Duplicates | Issues |",
            "|---------|------|------|--------|------------|--------|",
        ]
        for p in profiles.values():
            lines.append(
                f"| {p.name} | {p.row_count:,} | {p.col_count} "
                f"| {p.overall_null_pct:.1f}% | {p.duplicate_rows} "
                f"| {p.issue_count()} |"
            )
        return "\n".join(lines)

    # ── Per-column profiling ──────────────────────────────────────────────────

    def _profile_column(
        self,
        series: pd.Series,
        table_name: str,
        schema_col: Any | None = None,
    ) -> ColumnProfile:
        col_name = str(series.name)
        null_count = int(series.isna().sum())
        null_pct = round(series.isna().mean() * 100, 2)
        unique_count = int(series.nunique(dropna=True))
        inferred_type = (
            schema_col.inferred_type
            if schema_col is not None
            else self._heuristic_type(series)
        )

        cp = ColumnProfile(
            name=col_name,
            dtype=str(series.dtype),
            inferred_type=inferred_type,
            null_count=null_count,
            null_pct=null_pct,
            unique_count=unique_count,
        )

        non_null = series.dropna()

        # ── Numeric stats ────────────────────────────────────────────────
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            if len(non_null):
                cp.min = round(float(non_null.min()), 4)
                cp.max = round(float(non_null.max()), 4)
                cp.mean = round(float(non_null.mean()), 4)
                cp.std = round(float(non_null.std()), 4) if len(non_null) > 1 else 0.0
                cp.median = round(float(non_null.median()), 4)
                cp.zero_count = int((non_null == 0).sum())
                cp.negative_count = int((non_null < 0).sum())

        # ── Datetime stats ───────────────────────────────────────────────
        elif pd.api.types.is_datetime64_any_dtype(series):
            if len(non_null):
                cp.min = str(non_null.min().date())
                cp.max = str(non_null.max().date())
                today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
                if series.dt.tz is not None:
                    future_mask = series.dropna() > pd.Timestamp.now(tz=series.dt.tz)
                else:
                    future_mask = series.dropna() > today
                cp.future_date_count = int(future_mask.sum())

        # ── Categorical / text stats ─────────────────────────────────────
        else:
            if len(non_null):
                top = non_null.value_counts().head(5)
                cp.top_values = list(zip(top.index.tolist(), top.values.tolist()))

        # ── Quality rule checks ──────────────────────────────────────────
        cp.issues = (
            self._check_nulls(null_pct, col_name, table_name, len(series))
            + self._check_revenue(cp, col_name, inferred_type, table_name, len(series))
            + self._check_future_dates(cp, col_name, table_name, len(series))
        )

        return cp

    # ── Quality rules ─────────────────────────────────────────────────────────

    @staticmethod
    def _check_nulls(
        null_pct: float,
        col_name: str,
        table_name: str,
        row_count: int,
    ) -> list[DataQualityIssue]:
        if null_pct == 100.0:
            return [DataQualityIssue(
                severity=SEV_ERROR, table=table_name, column=col_name,
                rule=RULE_FULLY_EMPTY,
                description=f"Column '{col_name}' is completely empty (100% null)",
                affected_rows=row_count, affected_pct=100.0,
            )]
        if null_pct > NULL_WARN_THRESHOLD:
            return [DataQualityIssue(
                severity=SEV_WARNING, table=table_name, column=col_name,
                rule=RULE_HIGH_NULLS,
                description=f"Column '{col_name}' is {null_pct:.1f}% null (threshold: {NULL_WARN_THRESHOLD:.0f}%)",
                affected_rows=int(null_pct / 100 * row_count),
                affected_pct=null_pct,
            )]
        return []

    @staticmethod
    def _check_revenue(
        cp: ColumnProfile,
        col_name: str,
        inferred_type: str,
        table_name: str,
        row_count: int,
    ) -> list[DataQualityIssue]:
        if inferred_type != "currency":
            return []
        issues: list[DataQualityIssue] = []
        if cp.zero_count:
            zero_pct = round(cp.zero_count / max(row_count, 1) * 100, 2)
            issues.append(DataQualityIssue(
                severity=SEV_WARNING, table=table_name, column=col_name,
                rule=RULE_ZERO_REVENUE,
                description=f"Column '{col_name}' has {cp.zero_count} zero values ({zero_pct:.1f}%)",
                affected_rows=cp.zero_count, affected_pct=zero_pct,
            ))
        if cp.negative_count:
            neg_pct = round(cp.negative_count / max(row_count, 1) * 100, 2)
            issues.append(DataQualityIssue(
                severity=SEV_WARNING, table=table_name, column=col_name,
                rule=RULE_NEGATIVE_REVENUE,
                description=f"Column '{col_name}' has {cp.negative_count} negative values ({neg_pct:.1f}%)",
                affected_rows=cp.negative_count, affected_pct=neg_pct,
            ))
        return issues

    @staticmethod
    def _check_future_dates(
        cp: ColumnProfile,
        col_name: str,
        table_name: str,
        row_count: int,
    ) -> list[DataQualityIssue]:
        if not cp.future_date_count:
            return []
        fut_pct = round(cp.future_date_count / max(row_count, 1) * 100, 2)
        return [DataQualityIssue(
            severity=SEV_WARNING, table=table_name, column=col_name,
            rule=RULE_FUTURE_DATES,
            description=(
                f"Column '{col_name}' has {cp.future_date_count} dates "
                f"beyond today ({fut_pct:.1f}%)"
            ),
            affected_rows=cp.future_date_count, affected_pct=fut_pct,
        )]

    @staticmethod
    def _heuristic_type(series: pd.Series) -> str:
        """Lightweight type detection when no schema is available."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return "date"
        if pd.api.types.is_bool_dtype(series):
            return "categorical"
        if pd.api.types.is_numeric_dtype(series):
            name = str(series.name).lower()
            if any(kw in name for kw in _CURRENCY_NAMES):
                return "currency"
            if any(s in name for s in ("_id", "_key", "_ref", "_code", "_no", "_uuid")):
                return "identifier"
            return "numeric"
        name = str(series.name).lower()
        if any(h in name for h in ("email", "mail", "e_mail")):
            return "email"
        if any(h in name for h in ("phone", "mobile", "tel", "fax", "cell")):
            return "phone"
        if any(s in name for s in ("_id", "_key", "_ref", "_code", "_no", "_uuid")):
            return "identifier"
        if series.nunique() <= 50:
            return "categorical"
        return "text"


# ── Streamlit sidebar panel ───────────────────────────────────────────────────

def render_profile_ui(
    profiles: dict[str, DataProfile],
    key_prefix: str = "profiler",
) -> None:
    """
    Render a collapsible data quality panel in the Streamlit sidebar.

    Shows per-table summary rows, colour-coded issue badges, and
    expandable column-level detail.

    Raises ImportError if streamlit is not installed.
    """
    try:
        import streamlit as st
    except ImportError as exc:
        raise ImportError(
            "streamlit is required for render_profile_ui. "
            "Install it with: pip install streamlit"
        ) from exc

    if not profiles:
        st.caption("No data loaded — nothing to profile.")
        return

    total_issues = sum(p.issue_count() for p in profiles.values())
    total_errors = sum(p.issue_count(SEV_ERROR) for p in profiles.values())

    badge = (
        f"🔴 {total_errors} error(s)" if total_errors
        else f"🟡 {total_issues} warning(s)" if total_issues
        else "✅ No issues"
    )

    with st.expander(f"📋 Data Quality — {badge}", expanded=(total_errors > 0)):
        for p in profiles.values():
            _render_table_profile(st, p, key_prefix)


def _render_table_profile(st: Any, p: DataProfile, key_prefix: str) -> None:
    """Render one table's profile block."""
    all_iss = p.all_issues()
    err_count = p.issue_count(SEV_ERROR)
    warn_count = p.issue_count(SEV_WARNING)

    header_badge = (
        f"🔴 {err_count}E {warn_count}W" if err_count
        else f"🟡 {warn_count}W" if warn_count
        else "✅"
    )
    st.markdown(f"**{p.name}** — {p.row_count:,} rows · {p.col_count} cols · {header_badge}")

    if p.duplicate_rows:
        st.warning(
            f"Duplicate rows: {p.duplicate_rows} ({p.duplicate_pct:.1f}%)",
            icon="⚠️",
        )

    for iss in all_iss:
        if iss.severity == SEV_ERROR:
            st.error(f"{iss.rule}: {iss.description}", icon="🔴")
        elif iss.severity == SEV_WARNING:
            st.warning(f"{iss.rule}: {iss.description}", icon="🟡")

    with st.expander(f"Column details — {p.name}", expanded=False):
        for cp in p.columns:
            _render_column_row(st, cp)

    st.divider()


def _render_column_row(st: Any, cp: ColumnProfile) -> None:
    """Render one row per column inside the expander."""
    issue_badge = f" {'⚠' * len(cp.issues)}" if cp.issues else ""
    st.markdown(
        f"`{cp.name}` **{cp.inferred_type}** — "
        f"nulls: {cp.null_pct:.1f}% · unique: {cp.unique_count}{issue_badge}"
    )
    if cp.mean is not None:
        st.caption(
            f"min={cp.min:,.0f}  max={cp.max:,.0f}  "
            f"mean={cp.mean:,.0f}  median={cp.median:,.0f}"
        )
    elif cp.top_values:
        tv_str = " · ".join(f"{v} ({c})" for v, c in cp.top_values[:5])
        st.caption(f"Top values: {tv_str}")
    if cp.future_date_count:
        st.caption(f"⚠ {cp.future_date_count} future dates")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
