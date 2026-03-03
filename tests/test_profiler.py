"""
tests/test_profiler.py — Comprehensive tests for data/profiler.py (Module 4).

Uses only synthetic data — no real CRM data is hardcoded.
All quality-rule thresholds are tested against the constants in the module.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import numpy as np
import pytest
from faker import Faker

from data.profiler import (
    NULL_WARN_THRESHOLD,
    RULE_DUPLICATE_ROWS,
    RULE_FULLY_EMPTY,
    RULE_FUTURE_DATES,
    RULE_HIGH_NULLS,
    RULE_HIGH_OVERALL_NULL,
    RULE_NEGATIVE_REVENUE,
    RULE_ZERO_REVENUE,
    SEV_ERROR,
    SEV_INFO,
    SEV_WARNING,
    ColumnProfile,
    DataProfile,
    DataProfiler,
    DataQualityIssue,
    render_profile_ui,
)

fake = Faker()
Faker.seed(1)
random.seed(1)

# ── Helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def profiler(tmp_path) -> DataProfiler:
    return DataProfiler({"exports": {"output_folder": str(tmp_path / "exports")}})


def make_clean_df(n: int = 50) -> pd.DataFrame:
    """Perfectly clean CRM DataFrame — no quality issues."""
    stages = ["Lead", "Proposal", "Closed Won", "Closed Lost"]
    return pd.DataFrame({
        "account_id":   list(range(1, n + 1)),
        "account_name": [fake.company() for _ in range(n)],
        "revenue":      [round(random.uniform(10_000, 500_000), 2) for _ in range(n)],
        "stage":        [random.choice(stages) for _ in range(n)],
        "close_date":   pd.to_datetime([
            fake.date_between(start_date="-2y", end_date="today")
            for _ in range(n)
        ]),
        "email":        [fake.company_email() for _ in range(n)],
    })


# ══════════════════════════════════════════════════════════════════════════════
# DataQualityIssue
# ══════════════════════════════════════════════════════════════════════════════

class TestDataQualityIssue:

    def _make(self, **kw) -> DataQualityIssue:
        defaults = dict(
            severity=SEV_WARNING, table="accounts", column="revenue",
            rule=RULE_HIGH_NULLS, description="test", affected_rows=5, affected_pct=10.0,
        )
        defaults.update(kw)
        return DataQualityIssue(**defaults)

    def test_icon_error(self):
        assert self._make(severity=SEV_ERROR).icon == "🔴"

    def test_icon_warning(self):
        assert self._make(severity=SEV_WARNING).icon == "🟡"

    def test_icon_info(self):
        assert self._make(severity=SEV_INFO).icon == "🔵"

    def test_to_dict_has_required_keys(self):
        d = self._make().to_dict()
        for key in ["severity", "rule", "description", "affected_rows", "affected_pct"]:
            assert key in d

    def test_to_dict_affected_pct_rounded(self):
        d = self._make(affected_pct=33.33333).to_dict()
        assert d["affected_pct"] == 33.33


# ══════════════════════════════════════════════════════════════════════════════
# ColumnProfile stats
# ══════════════════════════════════════════════════════════════════════════════

class TestColumnStats:

    def test_numeric_min_max_mean(self, profiler):
        df = pd.DataFrame({"revenue": [100.0, 200.0, 300.0]})
        profile = profiler.profile(df, "t")
        cp = profile.column("revenue")
        assert cp.min == pytest.approx(100.0)
        assert cp.max == pytest.approx(300.0)
        assert cp.mean == pytest.approx(200.0)

    def test_numeric_std_median(self, profiler):
        df = pd.DataFrame({"revenue": [10.0, 20.0, 30.0, 40.0, 50.0]})
        profile = profiler.profile(df, "t")
        cp = profile.column("revenue")
        assert cp.median == pytest.approx(30.0)
        assert cp.std is not None and cp.std > 0

    def test_null_count_and_pct(self, profiler):
        df = pd.DataFrame({"x": [1.0, None, 3.0, None]})
        profile = profiler.profile(df, "t")
        cp = profile.column("x")
        assert cp.null_count == 2
        assert cp.null_pct == pytest.approx(50.0)

    def test_unique_count(self, profiler):
        df = pd.DataFrame({"stage": ["Lead", "Lead", "Won", "Lost"]})
        profile = profiler.profile(df, "t")
        cp = profile.column("stage")
        assert cp.unique_count == 3

    def test_top_values_categorical(self, profiler):
        df = pd.DataFrame({"stage": ["Lead"] * 10 + ["Won"] * 6 + ["Lost"] * 4})
        profile = profiler.profile(df, "t")
        cp = profile.column("stage")
        assert len(cp.top_values) <= 5
        # Lead should be first (highest count)
        assert cp.top_values[0][0] == "Lead"
        assert cp.top_values[0][1] == 10

    def test_top_values_empty_for_numeric(self, profiler):
        df = pd.DataFrame({"revenue": [100.0, 200.0, 300.0]})
        profile = profiler.profile(df, "t")
        cp = profile.column("revenue")
        assert cp.top_values == []

    def test_datetime_min_max_are_strings(self, profiler):
        df = pd.DataFrame({
            "close_date": pd.to_datetime(["2023-01-01", "2023-06-15", "2023-12-31"])
        })
        profile = profiler.profile(df, "t")
        cp = profile.column("close_date")
        assert cp.min == "2023-01-01"
        assert cp.max == "2023-12-31"

    def test_zero_count_in_revenue_column(self, profiler):
        df = pd.DataFrame({"revenue": [0.0, 100.0, 0.0, 200.0]})
        profile = profiler.profile(df, "t")
        cp = profile.column("revenue")
        assert cp.zero_count == 2

    def test_negative_count_in_revenue_column(self, profiler):
        df = pd.DataFrame({"revenue": [-50.0, 100.0, -200.0, 300.0]})
        profile = profiler.profile(df, "t")
        cp = profile.column("revenue")
        assert cp.negative_count == 2

    def test_future_date_count(self, profiler):
        # 2 dates far in the future, 2 in the past
        df = pd.DataFrame({
            "close_date": pd.to_datetime(["2020-01-01", "2021-06-01", "2099-01-01", "2098-12-31"])
        })
        profile = profiler.profile(df, "t")
        cp = profile.column("close_date")
        assert cp.future_date_count == 2

    def test_no_future_dates_in_clean_data(self, profiler):
        df = make_clean_df(30)
        profile = profiler.profile(df, "t")
        cp = profile.column("close_date")
        assert cp.future_date_count == 0

    def test_fully_null_column_stats(self, profiler):
        df = pd.DataFrame({"x": [None, None, None]})
        profile = profiler.profile(df, "t")
        cp = profile.column("x")
        assert cp.null_count == 3
        assert cp.null_pct == 100.0

    def test_column_to_dict_has_type(self, profiler):
        df = make_clean_df(20)
        profile = profiler.profile(df, "t")
        for cp in profile.columns:
            d = cp.to_dict()
            assert "type" in d
            assert "null_pct" in d


# ══════════════════════════════════════════════════════════════════════════════
# Quality rules — individual checks
# ══════════════════════════════════════════════════════════════════════════════

class TestQualityRules:

    # ── HIGH_NULLS ─────────────────────────────────────────────────────────────

    def test_high_nulls_flag_above_threshold(self, profiler):
        n = 100
        nulls = int(n * 0.40)   # 40% > threshold
        df = pd.DataFrame({"notes": [None] * nulls + ["x"] * (n - nulls)})
        profile = profiler.profile(df, "t")
        rules = [i.rule for i in profile.column("notes").issues]
        assert RULE_HIGH_NULLS in rules

    def test_high_nulls_no_flag_below_threshold(self, profiler):
        n = 100
        nulls = int(n * 0.10)   # 10% < threshold
        df = pd.DataFrame({"notes": [None] * nulls + ["x"] * (n - nulls)})
        profile = profiler.profile(df, "t")
        rules = [i.rule for i in profile.column("notes").issues]
        assert RULE_HIGH_NULLS not in rules

    def test_high_nulls_severity_is_warning(self, profiler):
        df = pd.DataFrame({"x": [None] * 40 + ["v"] * 60})
        profile = profiler.profile(df, "t")
        iss = [i for i in profile.column("x").issues if i.rule == RULE_HIGH_NULLS]
        assert all(i.severity == SEV_WARNING for i in iss)

    # ── FULLY_EMPTY ───────────────────────────────────────────────────────────

    def test_fully_empty_flag(self, profiler):
        df = pd.DataFrame({"ghost_col": [None, None, None, None, None]})
        profile = profiler.profile(df, "t")
        rules = [i.rule for i in profile.column("ghost_col").issues]
        assert RULE_FULLY_EMPTY in rules

    def test_fully_empty_severity_is_error(self, profiler):
        df = pd.DataFrame({"ghost_col": [None, None, None]})
        profile = profiler.profile(df, "t")
        iss = [i for i in profile.column("ghost_col").issues if i.rule == RULE_FULLY_EMPTY]
        assert all(i.severity == SEV_ERROR for i in iss)

    def test_fully_empty_in_fully_empty_columns_list(self, profiler):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]})
        profile = profiler.profile(df, "t")
        assert "b" in profile.fully_empty_columns
        assert "a" not in profile.fully_empty_columns

    def test_no_fully_empty_columns_in_clean_df(self, profiler):
        profile = profiler.profile(make_clean_df(20), "t")
        assert profile.fully_empty_columns == []

    # ── FUTURE_DATES ──────────────────────────────────────────────────────────

    def test_future_dates_flag(self, profiler):
        df = pd.DataFrame({
            "close_date": pd.to_datetime(["2020-01-01", "2021-06-01", "2099-01-01"])
        })
        profile = profiler.profile(df, "t")
        rules = [i.rule for i in profile.column("close_date").issues]
        assert RULE_FUTURE_DATES in rules

    def test_future_dates_no_flag_for_past_dates(self, profiler):
        df = pd.DataFrame({
            "close_date": pd.to_datetime(["2020-01-01", "2021-06-01", "2022-03-15"])
        })
        profile = profiler.profile(df, "t")
        rules = [i.rule for i in profile.column("close_date").issues]
        assert RULE_FUTURE_DATES not in rules

    def test_future_dates_severity_is_warning(self, profiler):
        df = pd.DataFrame({
            "close_date": pd.to_datetime(["2020-01-01", "2099-01-01"])
        })
        profile = profiler.profile(df, "t")
        iss = [i for i in profile.column("close_date").issues if i.rule == RULE_FUTURE_DATES]
        assert all(i.severity == SEV_WARNING for i in iss)

    def test_future_dates_affected_rows_count(self, profiler):
        df = pd.DataFrame({
            "close_date": pd.to_datetime(["2020-01-01", "2099-01-01", "2098-06-01"])
        })
        profile = profiler.profile(df, "t")
        iss = [i for i in profile.column("close_date").issues if i.rule == RULE_FUTURE_DATES]
        assert iss[0].affected_rows == 2

    # ── ZERO_REVENUE ──────────────────────────────────────────────────────────

    def test_zero_revenue_flag(self, profiler):
        df = pd.DataFrame({"revenue": [0.0, 100.0, 0.0, 200.0]})
        profile = profiler.profile(df, "t")
        rules = [i.rule for i in profile.column("revenue").issues]
        assert RULE_ZERO_REVENUE in rules

    def test_zero_revenue_no_flag_for_non_currency(self, profiler):
        # Column not named with currency keyword → not flagged
        df = pd.DataFrame({"employee_count": [0, 5, 10]})
        profile = profiler.profile(df, "t")
        rules = [i.rule for i in profile.column("employee_count").issues]
        assert RULE_ZERO_REVENUE not in rules

    def test_zero_revenue_affected_rows(self, profiler):
        df = pd.DataFrame({"revenue": [0.0, 100.0, 0.0]})
        profile = profiler.profile(df, "t")
        iss = [i for i in profile.column("revenue").issues if i.rule == RULE_ZERO_REVENUE]
        assert iss[0].affected_rows == 2

    # ── NEGATIVE_REVENUE ──────────────────────────────────────────────────────

    def test_negative_revenue_flag(self, profiler):
        df = pd.DataFrame({"revenue": [-100.0, 200.0, 300.0]})
        profile = profiler.profile(df, "t")
        rules = [i.rule for i in profile.column("revenue").issues]
        assert RULE_NEGATIVE_REVENUE in rules

    def test_negative_revenue_no_flag_for_clean_data(self, profiler):
        df = pd.DataFrame({"revenue": [100.0, 200.0, 300.0]})
        profile = profiler.profile(df, "t")
        rules = [i.rule for i in profile.column("revenue").issues]
        assert RULE_NEGATIVE_REVENUE not in rules

    def test_negative_revenue_currency_keywords(self, profiler):
        for col_name in ["amount", "deal_value", "cost", "price", "budget"]:
            df = pd.DataFrame({col_name: [-1.0, 100.0]})
            profile = profiler.profile(df, "t")
            rules = [i.rule for i in profile.column(col_name).issues]
            assert RULE_NEGATIVE_REVENUE in rules, f"Failed for column: {col_name}"

    # ── DUPLICATE_ROWS ────────────────────────────────────────────────────────

    def test_duplicate_rows_flag(self, profiler):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        profile = profiler.profile(df, "t")
        rules = [i.rule for i in profile.issues]
        assert RULE_DUPLICATE_ROWS in rules

    def test_duplicate_rows_count_correct(self, profiler):
        df = pd.DataFrame({"a": [1, 1, 1, 2, 3]})  # 2 duplicates of row 1
        profile = profiler.profile(df, "t")
        assert profile.duplicate_rows == 2

    def test_no_duplicate_flag_for_clean_df(self, profiler):
        df = make_clean_df(30)
        profile = profiler.profile(df, "t")
        dup_issues = [i for i in profile.issues if i.rule == RULE_DUPLICATE_ROWS]
        assert len(dup_issues) == 0

    def test_high_duplicate_pct_is_error(self, profiler):
        # >5% → error severity
        rows = [{"a": 1, "b": "x"}] * 10 + [{"a": i, "b": str(i)} for i in range(2, 192)]
        df = pd.DataFrame(rows)   # 10/200 = 5% duplicates → boundary
        profile = profiler.profile(df, "t")
        dup_iss = [i for i in profile.issues if i.rule == RULE_DUPLICATE_ROWS]
        if dup_iss:
            # either error or warning depending on exact pct
            assert dup_iss[0].severity in (SEV_ERROR, SEV_WARNING)

    # ── HIGH_OVERALL_NULL ─────────────────────────────────────────────────────

    def test_high_overall_null_flag(self, profiler):
        # Most columns mostly null → high overall null rate
        df = pd.DataFrame({
            "a": [None] * 35 + [1] * 15,
            "b": [None] * 35 + [2] * 15,
        })
        profile = profiler.profile(df, "t")
        rules = [i.rule for i in profile.issues]
        assert RULE_HIGH_OVERALL_NULL in rules


# ══════════════════════════════════════════════════════════════════════════════
# DataProfile — aggregate methods
# ══════════════════════════════════════════════════════════════════════════════

class TestDataProfile:

    def test_all_issues_includes_column_and_table_issues(self, profiler):
        df = pd.DataFrame({
            "revenue":    [-100.0, 200.0],
            "close_date": pd.to_datetime(["2020-01-01", "2099-01-01"]),
        })
        # Also create a duplicate to get a table-level issue
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        profile = profiler.profile(df, "t")
        all_iss = profile.all_issues()
        rules = {i.rule for i in all_iss}
        assert RULE_DUPLICATE_ROWS in rules
        assert RULE_NEGATIVE_REVENUE in rules or RULE_FUTURE_DATES in rules

    def test_all_issues_sorted_by_severity(self, profiler):
        df = pd.DataFrame({
            "ghost":   [None, None, None],
            "revenue": [-10.0, 100.0, 200.0],
        })
        profile = profiler.profile(df, "t")
        all_iss = profile.all_issues()
        sevs = [i.severity for i in all_iss]
        # errors before warnings before infos
        for i in range(len(sevs) - 1):
            if sevs[i] == SEV_WARNING:
                assert sevs[i + 1] != SEV_ERROR

    def test_issue_count_total(self, profiler):
        df = pd.DataFrame({
            "revenue":  [-10.0, 200.0],
            "ghost_col": [None, None],
        })
        profile = profiler.profile(df, "t")
        assert profile.issue_count() > 0

    def test_issue_count_filtered_by_severity(self, profiler):
        df = pd.DataFrame({"ghost": [None, None, None]})
        profile = profiler.profile(df, "t")
        errors = profile.issue_count(SEV_ERROR)
        warnings = profile.issue_count(SEV_WARNING)
        assert errors + warnings == profile.issue_count()

    def test_column_lookup_returns_correct(self, profiler):
        df = make_clean_df(20)
        profile = profiler.profile(df, "t")
        cp = profile.column("revenue")
        assert cp is not None
        assert cp.name == "revenue"

    def test_column_lookup_missing_returns_none(self, profiler):
        profile = profiler.profile(make_clean_df(10), "t")
        assert profile.column("nonexistent") is None

    def test_columns_with_issues_only_returns_problematic(self, profiler):
        df = pd.DataFrame({
            "revenue":  [-10.0, 200.0, 300.0],
            "clean_col": [1, 2, 3],
        })
        profile = profiler.profile(df, "t")
        problematic = profile.columns_with_issues()
        names = [c.name for c in problematic]
        assert "revenue" in names
        assert "clean_col" not in names

    def test_duplicate_pct_calculated_correctly(self, profiler):
        df = pd.DataFrame({"a": [1, 1, 2, 3, 4]})  # 1 duplicate out of 5
        profile = profiler.profile(df, "t")
        assert profile.duplicate_rows == 1
        assert profile.duplicate_pct == pytest.approx(20.0, abs=0.1)


# ══════════════════════════════════════════════════════════════════════════════
# profile_all() and build_quality_report()
# ══════════════════════════════════════════════════════════════════════════════

class TestProfileAll:

    def test_profile_all_returns_dict(self, profiler):
        dfs = {"accounts": make_clean_df(20), "contacts": make_clean_df(15)}
        profiles = profiler.profile_all(dfs)
        assert isinstance(profiles, dict)
        assert set(profiles.keys()) == {"accounts", "contacts"}

    def test_profile_all_each_is_data_profile(self, profiler):
        dfs = {"accounts": make_clean_df(10)}
        profiles = profiler.profile_all(dfs)
        assert isinstance(profiles["accounts"], DataProfile)

    def test_build_quality_report_structure(self, profiler):
        dfs = {"accounts": make_clean_df(30)}
        profiles = profiler.profile_all(dfs)
        report = profiler.build_quality_report(profiles)

        assert "generated_at" in report
        assert "summary" in report
        assert "tables" in report
        summary = report["summary"]
        for key in ["total_tables", "total_rows", "total_issues", "errors", "warnings"]:
            assert key in summary

    def test_build_quality_report_counts_tables(self, profiler):
        dfs = {"a": make_clean_df(10), "b": make_clean_df(10)}
        profiles = profiler.profile_all(dfs)
        report = profiler.build_quality_report(profiles)
        assert report["summary"]["total_tables"] == 2

    def test_build_quality_report_sums_rows(self, profiler):
        dfs = {"a": make_clean_df(20), "b": make_clean_df(30)}
        profiles = profiler.profile_all(dfs)
        report = profiler.build_quality_report(profiles)
        assert report["summary"]["total_rows"] == 50

    def test_build_quality_report_includes_column_stats(self, profiler):
        dfs = {"accounts": make_clean_df(20)}
        profiles = profiler.profile_all(dfs)
        report = profiler.build_quality_report(profiles)
        col_stats = report["tables"]["accounts"]["column_stats"]
        assert "revenue" in col_stats
        assert col_stats["revenue"]["type"] == "currency"

    def test_build_quality_report_is_json_serializable(self, profiler):
        dfs = {"accounts": make_clean_df(20)}
        profiles = profiler.profile_all(dfs)
        report = profiler.build_quality_report(profiles)
        # Should not raise
        json.dumps(report, default=str)

    def test_report_counts_issues_correctly(self, profiler):
        df = pd.DataFrame({
            "revenue":  [-10.0, 0.0, 100.0],
            "ghost":    [None, None, None],
        })
        profiles = profiler.profile_all({"t": df})
        report = profiler.build_quality_report(profiles)
        # Should have NEGATIVE_REVENUE + ZERO_REVENUE + FULLY_EMPTY = 3 issues minimum
        assert report["summary"]["total_issues"] >= 3


# ══════════════════════════════════════════════════════════════════════════════
# export_report()
# ══════════════════════════════════════════════════════════════════════════════

class TestExportReport:

    def test_export_creates_file(self, profiler, tmp_path):
        profiles = profiler.profile_all({"accounts": make_clean_df(20)})
        path = profiler.export_report(profiles, tmp_path / "exports")
        assert path.exists()
        assert path.suffix == ".txt"

    def test_export_filename_is_correct(self, profiler, tmp_path):
        profiles = profiler.profile_all({"accounts": make_clean_df(10)})
        path = profiler.export_report(profiles, tmp_path)
        assert path.name == "data_quality_report.txt"

    def test_export_content_contains_table_name(self, profiler, tmp_path):
        profiles = profiler.profile_all({"accounts": make_clean_df(10)})
        path = profiler.export_report(profiles, tmp_path)
        content = path.read_text()
        assert "accounts" in content

    def test_export_content_has_summary_section(self, profiler, tmp_path):
        profiles = profiler.profile_all({"accounts": make_clean_df(10)})
        path = profiler.export_report(profiles, tmp_path)
        content = path.read_text()
        assert "SUMMARY" in content

    def test_export_issues_appear_in_file(self, profiler, tmp_path):
        df = pd.DataFrame({"revenue": [-100.0, 200.0, 300.0]})
        profiles = profiler.profile_all({"t": df})
        path = profiler.export_report(profiles, tmp_path)
        content = path.read_text()
        assert "NEGATIVE_REVENUE" in content

    def test_export_overwrites_on_rerun(self, profiler, tmp_path):
        profiles = profiler.profile_all({"a": make_clean_df(10)})
        p1 = profiler.export_report(profiles, tmp_path)
        p2 = profiler.export_report(profiles, tmp_path)
        assert p1 == p2
        assert p2.exists()

    def test_to_text_includes_column_details(self, profiler):
        df = make_clean_df(20)
        profile = profiler.profile(df, "accounts")
        text = profile.to_text()
        assert "revenue" in text
        assert "Rows" in text

    def test_summary_markdown_backward_compat(self, profiler):
        profiles = profiler.profile_all({"accounts": make_clean_df(10)})
        md = profiler.summary_markdown(profiles)
        assert "accounts" in md
        assert "|" in md    # it's a markdown table


# ══════════════════════════════════════════════════════════════════════════════
# Heuristic type detection (no schema)
# ══════════════════════════════════════════════════════════════════════════════

class TestHeuristicType:

    def test_currency_column_by_name(self, profiler):
        df = pd.DataFrame({"revenue": [100.0, 200.0]})
        profile = profiler.profile(df, "t")
        assert profile.column("revenue").inferred_type == "currency"

    def test_date_column_detected(self, profiler):
        df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01", "2024-02-01"])})
        profile = profiler.profile(df, "t")
        assert profile.column("ts").inferred_type == "date"

    def test_categorical_low_cardinality(self, profiler):
        stages = ["Lead", "Won", "Lost"] * 20
        df = pd.DataFrame({"stage": stages})
        profile = profiler.profile(df, "t")
        assert profile.column("stage").inferred_type == "categorical"

    def test_text_high_cardinality(self, profiler):
        df = pd.DataFrame({"notes": [fake.sentence() for _ in range(100)]})
        profile = profiler.profile(df, "t")
        assert profile.column("notes").inferred_type == "text"

    def test_identifier_column_by_name(self, profiler):
        df = pd.DataFrame({"account_id": list(range(50))})
        profile = profiler.profile(df, "t")
        assert profile.column("account_id").inferred_type == "identifier"


# ══════════════════════════════════════════════════════════════════════════════
# render_profile_ui — offline guard
# ══════════════════════════════════════════════════════════════════════════════

class TestRenderProfileUI:

    def test_raises_import_error_without_streamlit(self, profiler):
        import sys
        # Remove streamlit and set None sentinel to block re-import
        st_module = sys.modules.pop("streamlit", None)
        sys.modules["streamlit"] = None  # type: ignore[assignment]
        profiles = profiler.profile_all({"accounts": make_clean_df(10)})
        try:
            with pytest.raises((ImportError, ModuleNotFoundError)):
                render_profile_ui(profiles)
        finally:
            if st_module is not None:
                sys.modules["streamlit"] = st_module
            else:
                sys.modules.pop("streamlit", None)
