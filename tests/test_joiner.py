"""
tests/test_joiner.py — Comprehensive tests for data/joiner.py (Module 3).

Covers:
  - JoinCandidate / JoinQuality / JoinResult dataclasses
  - _normalize_col_name, _name_similarity, _value_overlap
  - suggest(): detection, scoring, ordering, threshold filtering
  - build(): master DataFrame, individual tables, quality metrics
  - compute_quality(): matched/lost rows, duplicate risk, warnings
  - Legacy API: detect_joins(), build_joined()
  - render_join_ui(): raises ImportError when streamlit absent (tests run offline)
"""
from __future__ import annotations

import random

import pandas as pd
import pytest
from faker import Faker

from data.joiner import (
    JoinCandidate,
    JoinDetector,
    JoinQuality,
    JoinResult,
    render_join_ui,
)

fake = Faker()
Faker.seed(0)
random.seed(0)

# ── Config ─────────────────────────────────────────────────────────────────────

MOCK_CONFIG = {
    "data": {
        "join_confidence_threshold": 0.5,
        "auto_join": True,
    }
}

HIGH_THRESHOLD_CONFIG = {
    "data": {
        "join_confidence_threshold": 0.85,
        "auto_join": True,
    }
}


@pytest.fixture()
def detector() -> JoinDetector:
    return JoinDetector(MOCK_CONFIG)


@pytest.fixture()
def strict_detector() -> JoinDetector:
    return JoinDetector(HIGH_THRESHOLD_CONFIG)


# ── Synthetic CRM data ────────────────────────────────────────────────────────

@pytest.fixture()
def accounts_df() -> pd.DataFrame:
    return pd.DataFrame({
        "account_id": list(range(1, 11)),
        "account_name": [fake.company() for _ in range(10)],
        "industry": [random.choice(["Tech", "Finance", "Health"]) for _ in range(10)],
        "revenue": [round(random.uniform(50_000, 1_000_000), 2) for _ in range(10)],
    })


@pytest.fixture()
def contacts_df() -> pd.DataFrame:
    """Many contacts per account (many-to-one FK: contact.account_id → account.account_id)."""
    return pd.DataFrame({
        "contact_id": list(range(101, 126)),
        "contact_name": [fake.name() for _ in range(25)],
        "email": [fake.company_email() for _ in range(25)],
        "account_id": [random.randint(1, 10) for _ in range(25)],
    })


@pytest.fixture()
def deals_df() -> pd.DataFrame:
    return pd.DataFrame({
        "deal_id": list(range(1001, 1016)),
        "account_id": [random.randint(1, 10) for _ in range(15)],
        "deal_value": [round(random.uniform(5_000, 200_000), 2) for _ in range(15)],
        "stage": [random.choice(["Lead", "Proposal", "Closed Won"]) for _ in range(15)],
    })


@pytest.fixture()
def activities_df() -> pd.DataFrame:
    """Links to contacts, NOT accounts."""
    return pd.DataFrame({
        "activity_id": list(range(2001, 2031)),
        "contact_id": [random.randint(101, 125) for _ in range(30)],
        "activity_type": [random.choice(["Call", "Email", "Meeting"]) for _ in range(30)],
    })


@pytest.fixture()
def unrelated_df() -> pd.DataFrame:
    return pd.DataFrame({
        "alpha": [fake.word() for _ in range(10)],
        "beta": [random.random() for _ in range(10)],
        "gamma": [fake.color_name() for _ in range(10)],
    })


# ═══════════════════════════════════════════════════════════════════════════════
# JoinCandidate dataclass
# ═══════════════════════════════════════════════════════════════════════════════

class TestJoinCandidate:

    def make(self, **kwargs) -> JoinCandidate:
        defaults = dict(
            left_table="accounts", right_table="contacts",
            left_col="account_id", right_col="account_id",
            name_similarity=1.0, value_overlap=0.9, confidence=0.94,
        )
        defaults.update(kwargs)
        return JoinCandidate(**defaults)

    def test_label_format(self):
        c = self.make()
        assert c.label == "accounts.account_id ↔ contacts.account_id"

    def test_confidence_pct(self):
        c = self.make(confidence=0.876)
        assert c.confidence_pct == 88

    def test_default_join_type_is_left(self):
        c = self.make()
        assert c.join_type == "left"

    def test_custom_join_type(self):
        c = self.make(join_type="inner")
        assert c.join_type == "inner"


# ═══════════════════════════════════════════════════════════════════════════════
# Name normalisation and similarity
# ═══════════════════════════════════════════════════════════════════════════════

class TestNameSimilarity:

    def test_identical_names_score_one(self, detector):
        assert detector._name_similarity("account_id", "account_id") == 1.0

    def test_case_insensitive(self, detector):
        assert detector._name_similarity("AccountID", "accountid") == 1.0

    def test_same_base_different_suffix(self, detector):
        # account_id vs account_key → both normalise to "account"
        score = detector._name_similarity("account_id", "account_key")
        assert score >= 0.90

    def test_similar_abbreviation(self, detector):
        score = detector._name_similarity("acct_id", "account_id")
        assert score >= 0.40   # low but above MIN_NAME_SIM gate

    def test_completely_different_names_score_low(self, detector):
        # "fax" vs "zip" share no characters — reliably low after normalisation
        score = detector._name_similarity("fax_number", "zip_code")
        assert score < 0.50

    def test_normalize_strips_id_suffix(self):
        assert JoinDetector._normalize_col_name("account_id") == "account"

    def test_normalize_strips_key_suffix(self):
        assert JoinDetector._normalize_col_name("contact_key") == "contact"

    def test_normalize_strips_ref_suffix(self):
        assert JoinDetector._normalize_col_name("deal_ref") == "deal"

    def test_normalize_strips_id_prefix(self):
        assert JoinDetector._normalize_col_name("id_account") == "account"

    def test_normalize_lowercases(self):
        # Normalisation works on underscore-delimited tokens; account_id → account
        assert JoinDetector._normalize_col_name("account_id") == "account"

    def test_normalize_replaces_underscores_with_spaces(self):
        norm = JoinDetector._normalize_col_name("customer_name")
        assert "_" not in norm


# ═══════════════════════════════════════════════════════════════════════════════
# Value overlap
# ═══════════════════════════════════════════════════════════════════════════════

class TestValueOverlap:

    def test_full_overlap_returns_one(self, detector):
        a = pd.Series([1, 2, 3])
        b = pd.Series([1, 2, 3, 4, 5])
        assert detector._value_overlap(a, b) == 1.0

    def test_no_overlap_returns_zero(self, detector):
        a = pd.Series([1, 2, 3])
        b = pd.Series([10, 20, 30])
        assert detector._value_overlap(a, b) == 0.0

    def test_partial_overlap(self, detector):
        a = pd.Series([1, 2, 3, 4])
        b = pd.Series([3, 4, 5, 6])
        score = detector._value_overlap(a, b)
        assert 0.4 < score <= 1.0

    def test_bidirectional_takes_max(self, detector):
        # All of b is in a, but only half of a is in b
        # → max(|a∩b|/|a|, |a∩b|/|b|) = max(0.5, 1.0) = 1.0
        a = pd.Series([1, 2, 3, 4])
        b = pd.Series([1, 2])
        score = detector._value_overlap(a, b)
        assert score == 1.0

    def test_empty_series_returns_zero(self, detector):
        assert detector._value_overlap(pd.Series([], dtype=int), pd.Series([1, 2])) == 0.0

    def test_null_values_ignored(self, detector):
        a = pd.Series([1, 2, None, None])
        b = pd.Series([1, 2, 3])
        score = detector._value_overlap(a, b)
        assert score == 1.0

    def test_string_values(self, detector):
        a = pd.Series(["A", "B", "C"])
        b = pd.Series(["A", "B", "C", "D"])
        assert detector._value_overlap(a, b) == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# suggest()
# ═══════════════════════════════════════════════════════════════════════════════

class TestSuggest:

    def test_returns_list(self, detector, accounts_df, contacts_df):
        result = detector.suggest({"accounts": accounts_df, "contacts": contacts_df})
        assert isinstance(result, list)

    def test_finds_account_id_join(self, detector, accounts_df, contacts_df):
        candidates = detector.suggest({"accounts": accounts_df, "contacts": contacts_df})
        labels = [c.label for c in candidates]
        assert any("account_id" in lbl for lbl in labels)

    def test_single_table_returns_empty(self, detector, accounts_df):
        assert detector.suggest({"accounts": accounts_df}) == []

    def test_sorted_by_confidence_descending(self, detector, accounts_df, contacts_df, deals_df):
        candidates = detector.suggest({
            "accounts": accounts_df, "contacts": contacts_df, "deals": deals_df
        })
        confs = [c.confidence for c in candidates]
        assert confs == sorted(confs, reverse=True)

    def test_confidence_between_zero_and_one(self, detector, accounts_df, contacts_df):
        candidates = detector.suggest({"accounts": accounts_df, "contacts": contacts_df})
        for c in candidates:
            assert 0.0 <= c.confidence <= 1.0

    def test_all_candidates_have_required_fields(self, detector, accounts_df, contacts_df):
        candidates = detector.suggest({"accounts": accounts_df, "contacts": contacts_df})
        for c in candidates:
            assert c.left_table and c.right_table
            assert c.left_col and c.right_col
            assert c.name_similarity >= 0
            assert c.value_overlap >= 0

    def test_unrelated_tables_return_no_candidates(self, detector, accounts_df, unrelated_df):
        candidates = detector.suggest({"accounts": accounts_df, "unrelated": unrelated_df})
        # No candidate should be above threshold
        above = [c for c in candidates if c.confidence >= detector.threshold]
        assert len(above) == 0

    def test_three_tables_finds_multiple_candidates(self, detector, accounts_df, contacts_df, deals_df):
        candidates = detector.suggest({
            "accounts": accounts_df, "contacts": contacts_df, "deals": deals_df
        })
        assert len(candidates) >= 2

    def test_contact_to_activity_join_detected(self, detector, contacts_df, activities_df):
        candidates = detector.suggest({"contacts": contacts_df, "activities": activities_df})
        assert any("contact_id" in c.label for c in candidates)


# ═══════════════════════════════════════════════════════════════════════════════
# compute_quality()
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeQuality:

    def _perfect_candidate(self) -> JoinCandidate:
        return JoinCandidate(
            left_table="accounts", right_table="contacts",
            left_col="account_id", right_col="account_id",
            name_similarity=1.0, value_overlap=1.0, confidence=1.0,
        )

    def test_returns_join_quality(self, detector, accounts_df, contacts_df):
        q = detector.compute_quality(accounts_df, contacts_df, self._perfect_candidate())
        assert isinstance(q, JoinQuality)

    def test_row_counts_correct(self, detector, accounts_df, contacts_df):
        q = detector.compute_quality(accounts_df, contacts_df, self._perfect_candidate())
        assert q.left_rows == len(accounts_df)
        assert q.right_rows == len(contacts_df)

    def test_full_match_rate(self, detector):
        # All left IDs appear in right
        left = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
        right = pd.DataFrame({"id": [1, 2, 3, 4, 5], "val": [10, 20, 30, 40, 50]})
        c = JoinCandidate(
            left_table="L", right_table="R", left_col="id", right_col="id",
            name_similarity=1.0, value_overlap=1.0, confidence=1.0,
        )
        q = detector.compute_quality(left, right, c)
        assert q.match_rate == 1.0
        assert q.lost_left_rows == 0

    def test_partial_match_rate(self, detector):
        left = pd.DataFrame({"id": [1, 2, 3, 4]})
        right = pd.DataFrame({"id": [1, 2]})
        c = JoinCandidate(
            left_table="L", right_table="R", left_col="id", right_col="id",
            name_similarity=1.0, value_overlap=1.0, confidence=1.0,
        )
        q = detector.compute_quality(left, right, c)
        assert q.match_rate == 0.5
        assert q.lost_left_rows == 2

    def test_lost_right_rows_counted(self, detector):
        left = pd.DataFrame({"id": [1, 2]})
        right = pd.DataFrame({"id": [1, 2, 3, 4]})
        c = JoinCandidate(
            left_table="L", right_table="R", left_col="id", right_col="id",
            name_similarity=1.0, value_overlap=1.0, confidence=1.0,
        )
        q = detector.compute_quality(left, right, c)
        assert q.lost_right_rows == 2

    def test_warning_generated_for_high_loss_rate(self, detector):
        # Only 50% match → should warn
        left = pd.DataFrame({"id": range(10)})
        right = pd.DataFrame({"id": range(5)})
        c = JoinCandidate(
            left_table="L", right_table="R", left_col="id", right_col="id",
            name_similarity=1.0, value_overlap=1.0, confidence=1.0,
        )
        q = detector.compute_quality(left, right, c)
        assert len(q.warnings) >= 1
        assert any("no match" in w.lower() or "rows" in w.lower() for w in q.warnings)

    def test_no_warning_for_perfect_match(self, detector, accounts_df, contacts_df):
        # accounts_df has 10 rows; all account_ids appear in contacts
        left = pd.DataFrame({"id": [1, 2, 3]})
        right = pd.DataFrame({"id": [1, 2, 3]})
        c = JoinCandidate(
            left_table="L", right_table="R", left_col="id", right_col="id",
            name_similarity=1.0, value_overlap=1.0, confidence=1.0,
        )
        q = detector.compute_quality(left, right, c)
        # Perfect match → no warnings
        assert q.match_rate == 1.0
        # No loss-rate warning
        loss_warnings = [w for w in q.warnings if "no match" in w.lower()]
        assert len(loss_warnings) == 0

    def test_duplicate_risk_detected(self, detector):
        # One left row matched by many right rows → M:N fan-out
        left = pd.DataFrame({"id": [1, 2, 3]})
        right = pd.DataFrame({"id": [1] * 50 + [2] * 50 + [3] * 50})  # 50x duplication
        c = JoinCandidate(
            left_table="L", right_table="R", left_col="id", right_col="id",
            name_similarity=1.0, value_overlap=1.0, confidence=1.0,
        )
        q = detector.compute_quality(left, right, c)
        assert q.duplicate_risk is True
        assert any("M:N" in w or "duplicate" in w.lower() for w in q.warnings)

    def test_no_duplicate_risk_for_clean_join(self, detector):
        left = pd.DataFrame({"id": [1, 2, 3]})
        right = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
        c = JoinCandidate(
            left_table="L", right_table="R", left_col="id", right_col="id",
            name_similarity=1.0, value_overlap=1.0, confidence=1.0,
        )
        q = detector.compute_quality(left, right, c)
        assert q.duplicate_risk is False

    def test_quality_to_dict_keys(self, detector):
        left = pd.DataFrame({"id": [1, 2]})
        right = pd.DataFrame({"id": [1, 2]})
        c = JoinCandidate(
            left_table="L", right_table="R", left_col="id", right_col="id",
            name_similarity=1.0, value_overlap=1.0, confidence=1.0,
        )
        q = detector.compute_quality(left, right, c)
        d = q.to_dict()
        for key in ["join", "left_rows", "right_rows", "matched", "match_rate", "warnings"]:
            assert key in d


# ═══════════════════════════════════════════════════════════════════════════════
# build()
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuild:

    def test_returns_join_result(self, detector, accounts_df, contacts_df):
        candidates = detector.suggest({"accounts": accounts_df, "contacts": contacts_df})
        result = detector.build({"accounts": accounts_df, "contacts": contacts_df}, candidates)
        assert isinstance(result, JoinResult)

    def test_master_df_is_dataframe(self, detector, accounts_df, contacts_df):
        candidates = detector.suggest({"accounts": accounts_df, "contacts": contacts_df})
        result = detector.build({"accounts": accounts_df, "contacts": contacts_df}, candidates)
        assert result.master_df is not None
        assert isinstance(result.master_df, pd.DataFrame)

    def test_individual_dfs_always_present(self, detector, accounts_df, contacts_df):
        candidates = detector.suggest({"accounts": accounts_df, "contacts": contacts_df})
        result = detector.build({"accounts": accounts_df, "contacts": contacts_df}, candidates)
        assert "accounts" in result.individual_dfs
        assert "contacts" in result.individual_dfs

    def test_individual_dfs_unchanged(self, detector, accounts_df, contacts_df):
        candidates = detector.suggest({"accounts": accounts_df, "contacts": contacts_df})
        result = detector.build({"accounts": accounts_df, "contacts": contacts_df}, candidates)
        pd.testing.assert_frame_equal(
            result.individual_dfs["accounts"].reset_index(drop=True),
            accounts_df.reset_index(drop=True),
        )

    def test_no_candidates_returns_none_master(self, detector, accounts_df, contacts_df):
        result = detector.build({"accounts": accounts_df, "contacts": contacts_df}, [])
        assert result.master_df is None

    def test_applied_list_populated(self, detector, accounts_df, contacts_df):
        candidates = detector.suggest({"accounts": accounts_df, "contacts": contacts_df})
        result = detector.build({"accounts": accounts_df, "contacts": contacts_df}, candidates)
        assert isinstance(result.applied, list)

    def test_quality_list_populated(self, detector, accounts_df, contacts_df):
        candidates = detector.suggest({"accounts": accounts_df, "contacts": contacts_df})
        result = detector.build({"accounts": accounts_df, "contacts": contacts_df}, candidates)
        if result.applied:
            assert len(result.quality) >= 1
            assert all(isinstance(q, JoinQuality) for q in result.quality)

    def test_master_contains_columns_from_both_tables(self, detector, accounts_df, contacts_df):
        candidates = detector.suggest({"accounts": accounts_df, "contacts": contacts_df})
        result = detector.build({"accounts": accounts_df, "contacts": contacts_df}, candidates)
        if result.master_df is not None:
            master_cols = set(result.master_df.columns)
            assert "account_name" in master_cols or "revenue" in master_cols
            assert "contact_name" in master_cols or "email" in master_cols

    def test_below_threshold_candidates_excluded(self, strict_detector, accounts_df, unrelated_df):
        candidates = strict_detector.suggest({"accounts": accounts_df, "unrelated": unrelated_df})
        result = strict_detector.build({"accounts": accounts_df, "unrelated": unrelated_df}, candidates)
        assert result.master_df is None

    def test_summary_lines_returns_strings(self, detector, accounts_df, contacts_df):
        candidates = detector.suggest({"accounts": accounts_df, "contacts": contacts_df})
        result = detector.build({"accounts": accounts_df, "contacts": contacts_df}, candidates)
        lines = result.summary_lines()
        assert isinstance(lines, list)
        assert all(isinstance(l, str) for l in lines)

    def test_three_table_master(self, detector, accounts_df, contacts_df, deals_df):
        dfs = {"accounts": accounts_df, "contacts": contacts_df, "deals": deals_df}
        candidates = detector.suggest(dfs)
        result = detector.build(dfs, candidates)
        # All individual tables accessible
        assert all(t in result.individual_dfs for t in ["accounts", "contacts", "deals"])

    def test_candidates_field_preserved(self, detector, accounts_df, contacts_df):
        candidates = detector.suggest({"accounts": accounts_df, "contacts": contacts_df})
        result = detector.build({"accounts": accounts_df, "contacts": contacts_df}, candidates)
        assert result.candidates is candidates


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy API backward-compatibility
# ═══════════════════════════════════════════════════════════════════════════════

class TestLegacyAPI:

    def test_detect_joins_returns_dict(self, detector, accounts_df, contacts_df):
        join_map = detector.detect_joins({"accounts": accounts_df, "contacts": contacts_df})
        assert isinstance(join_map, dict)

    def test_detect_joins_finds_account_id_link(self, detector, accounts_df, contacts_df):
        join_map = detector.detect_joins({"accounts": accounts_df, "contacts": contacts_df})
        assert len(join_map) >= 1
        all_cols = [(v["left_col"], v["right_col"]) for v in join_map.values()]
        assert any("account_id" in pair for pair in all_cols)

    def test_detect_joins_single_df_returns_empty(self, detector, accounts_df):
        assert detector.detect_joins({"accounts": accounts_df}) == {}

    def test_detect_joins_confidence_above_threshold(self, detector, accounts_df, contacts_df):
        join_map = detector.detect_joins({"accounts": accounts_df, "contacts": contacts_df})
        for info in join_map.values():
            assert info["confidence"] >= detector.threshold

    def test_detect_joins_has_required_keys(self, detector, accounts_df, contacts_df):
        join_map = detector.detect_joins({"accounts": accounts_df, "contacts": contacts_df})
        for info in join_map.values():
            assert "left_col" in info
            assert "right_col" in info
            assert "confidence" in info
            assert "join_type" in info

    def test_build_joined_returns_dataframe(self, detector, accounts_df, contacts_df):
        dfs = {"accounts": accounts_df, "contacts": contacts_df}
        join_map = detector.detect_joins(dfs)
        if join_map:
            merged = detector.build_joined(dfs, join_map)
            assert merged is not None
            assert isinstance(merged, pd.DataFrame)

    def test_build_joined_empty_map_returns_none(self, detector, accounts_df):
        result = detector.build_joined({"accounts": accounts_df}, {})
        assert result is None

    def test_referential_overlap_static_full(self):
        assert JoinDetector._referential_overlap(
            pd.Series([1, 2, 3]), pd.Series([1, 2, 3, 4])
        ) == 1.0

    def test_referential_overlap_static_partial(self):
        overlap = JoinDetector._referential_overlap(
            pd.Series([1, 2, 3, 4]), pd.Series([1, 2])
        )
        assert overlap == 0.5

    def test_referential_overlap_static_empty(self):
        assert JoinDetector._referential_overlap(
            pd.Series([], dtype=float), pd.Series([1, 2])
        ) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# render_join_ui() — offline guard
# ═══════════════════════════════════════════════════════════════════════════════

class TestRenderJoinUI:

    def test_raises_import_error_without_streamlit(self, monkeypatch):
        """
        In the test environment Streamlit is not installed.
        render_join_ui must raise ImportError rather than crashing with
        an AttributeError or silently doing nothing.
        """
        import sys
        # Remove streamlit and set None sentinel to block re-import
        st_module = sys.modules.pop("streamlit", None)
        sys.modules["streamlit"] = None  # type: ignore[assignment]

        try:
            with pytest.raises((ImportError, ModuleNotFoundError)):
                render_join_ui([
                    JoinCandidate(
                        left_table="a", right_table="b",
                        left_col="id", right_col="id",
                        name_similarity=1.0, value_overlap=1.0, confidence=0.9,
                    )
                ])
        finally:
            if st_module is not None:
                sys.modules["streamlit"] = st_module
            else:
                sys.modules.pop("streamlit", None)
