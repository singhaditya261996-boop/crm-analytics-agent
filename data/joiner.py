"""
data/joiner.py — Auto-join detection, execution, and quality tracking.

Features
--------
- Column name fuzzy matching with key-suffix normalisation (_id, _key, _ref…)
- Bidirectional value-overlap scoring; requires >60% overlap to qualify
- Ranks ALL candidate pairs per table combination; caller picks which to apply
- JoinResult: master DataFrame + individual tables + per-join quality metrics
- JoinQuality: matched rows, lost rows, duplicate-risk flag, user warnings
- Streamlit sidebar UI helper: confirm/override joins, change join type live

Public API
----------
JoinDetector(config: dict)
    .suggest(dataframes)                    -> list[JoinCandidate]
    .build(dataframes, candidates)          -> JoinResult
    .compute_quality(left_df, right_df, c)  -> JoinQuality

render_join_ui(candidates, quality=None, key_prefix="join")
                                            -> list[JoinCandidate]   # approved only

# Backward-compatible aliases
JoinDetector.detect_joins(dataframes)       -> dict
JoinDetector.build_joined(dataframes, join_map) -> pd.DataFrame | None
"""
from __future__ import annotations

import itertools
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Minimum value-overlap for a candidate to be considered at all
_MIN_OVERLAP: float = 0.30
# Minimum name-similarity for a candidate to be considered
_MIN_NAME_SIM: float = 0.35
# Scoring weights
_W_NAME: float = 0.40
_W_OVERLAP: float = 0.60
# Quality thresholds
_WARN_LOSS_RATE: float = 0.20     # warn if >20% left rows have no match
_WARN_DUPLICATE_RATIO: float = 1.10  # warn if merged rows > left_rows * 1.10

# Suffixes/prefixes stripped before fuzzy name comparison
_STRIP_SUFFIXES = re.compile(
    r"(_id|_key|_ref|_code|_no|_num|_number|_uuid|_guid|_fk|_pk)$",
    re.IGNORECASE,
)
_STRIP_PREFIXES = re.compile(r"^(id_|fk_|pk_|ref_)", re.IGNORECASE)

JOIN_TYPES = ["left", "inner", "outer", "right"]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class JoinCandidate:
    """A proposed join between two tables."""
    left_table: str
    right_table: str
    left_col: str
    right_col: str
    name_similarity: float      # 0–1 fuzzy name score (normalised names)
    value_overlap: float        # 0–1 bidirectional referential integrity
    confidence: float           # weighted combination
    join_type: str = "left"     # left | inner | outer | right

    @property
    def label(self) -> str:
        return (
            f"{self.left_table}.{self.left_col} ↔ "
            f"{self.right_table}.{self.right_col}"
        )

    @property
    def confidence_pct(self) -> int:
        return round(self.confidence * 100)


@dataclass
class JoinQuality:
    """Row-level statistics for one applied join."""
    left_table: str
    right_table: str
    left_col: str
    right_col: str
    left_rows: int
    right_rows: int
    matched_rows: int           # rows in left that found a match in right
    lost_left_rows: int         # left rows with NO match in right
    lost_right_rows: int        # right rows with NO match in left
    match_rate: float           # matched_rows / left_rows  (0–1)
    duplicate_risk: bool        # merged row count > left_rows (M:N)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "join": f"{self.left_table} ↔ {self.right_table}",
            "left_rows": self.left_rows,
            "right_rows": self.right_rows,
            "matched": self.matched_rows,
            "lost_left": self.lost_left_rows,
            "lost_right": self.lost_right_rows,
            "match_rate": f"{self.match_rate:.0%}",
            "duplicate_risk": self.duplicate_risk,
            "warnings": self.warnings,
        }


@dataclass
class JoinResult:
    """
    Output of JoinDetector.build().

    Attributes
    ----------
    master_df        : merged DataFrame (None if no joins applied)
    individual_dfs   : original tables, always accessible by name
    candidates       : all detected candidates (including below threshold)
    applied          : candidates that were actually merged
    quality          : one JoinQuality per applied join
    """
    master_df: pd.DataFrame | None
    individual_dfs: dict[str, pd.DataFrame]
    candidates: list[JoinCandidate]
    applied: list[JoinCandidate]
    quality: list[JoinQuality]

    def summary_lines(self) -> list[str]:
        lines: list[str] = [
            f"Applied joins : {len(self.applied)}",
            f"Master rows   : {len(self.master_df) if self.master_df is not None else 'N/A'}",
        ]
        for q in self.quality:
            lines.append(
                f"  {q.left_table} ↔ {q.right_table}: "
                f"match={q.match_rate:.0%}  "
                f"lost_left={q.lost_left_rows}  "
                f"{'⚠ DUPLICATE RISK' if q.duplicate_risk else ''}"
            )
        return lines

    def print_summary(self) -> None:
        print("\n".join(self.summary_lines()))


# ── JoinDetector ──────────────────────────────────────────────────────────────

class JoinDetector:
    """
    Detects, scores, and executes joins across a collection of DataFrames.
    """

    def __init__(self, config: dict) -> None:
        data_cfg = config.get("data", {})
        self.threshold: float = data_cfg.get("join_confidence_threshold", 0.6)
        self.auto_join: bool = data_cfg.get("auto_join", True)

    # ── Primary API ───────────────────────────────────────────────────────────

    def suggest(
        self, dataframes: dict[str, pd.DataFrame]
    ) -> list[JoinCandidate]:
        """
        Score every column pair across every table pair.
        Returns all candidates with confidence ≥ _MIN_OVERLAP,
        sorted by confidence descending.
        Only the highest-scoring pair per (left_table, right_table) is
        included by default; pass include_all=True to the private scorer
        for UI purposes.
        """
        if len(dataframes) < 2:
            return []

        candidates: list[JoinCandidate] = []
        names = list(dataframes.keys())

        for left_name, right_name in itertools.combinations(names, 2):
            pair_candidates = self._score_pair(
                left_name, dataframes[left_name],
                right_name, dataframes[right_name],
            )
            candidates.extend(pair_candidates)

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates

    def build(
        self,
        dataframes: dict[str, pd.DataFrame],
        candidates: list[JoinCandidate],
    ) -> JoinResult:
        """
        Execute the approved *candidates* and return a JoinResult.
        Uses a left-join chain anchored on the most-referenced table.
        Individual DataFrames are always preserved in JoinResult.individual_dfs.
        """
        approved = [c for c in candidates if c.confidence >= self.threshold]

        if not approved:
            return JoinResult(
                master_df=None,
                individual_dfs=dict(dataframes),
                candidates=candidates,
                applied=[],
                quality=[],
            )

        quality_list: list[JoinQuality] = []
        master = self._build_master(dataframes, approved, quality_list)

        return JoinResult(
            master_df=master,
            individual_dfs=dict(dataframes),
            candidates=candidates,
            applied=approved,
            quality=quality_list,
        )

    def compute_quality(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        candidate: JoinCandidate,
    ) -> JoinQuality:
        """
        Compute row-level quality metrics for a single candidate join.
        Does NOT mutate the DataFrames.
        """
        lcol, rcol = candidate.left_col, candidate.right_col
        left_vals = set(left_df[lcol].dropna().unique())
        right_vals = set(right_df[rcol].dropna().unique())

        matched = len(left_vals & right_vals)
        lost_left = len(left_df) - int(left_df[lcol].isin(right_vals).sum())
        lost_right = len(right_df) - int(right_df[rcol].isin(left_vals).sum())
        match_rate = (
            int(left_df[lcol].isin(right_vals).sum()) / max(len(left_df), 1)
        )

        # Check duplicate risk: does merging fan out left rows?
        try:
            probe = left_df.head(500).merge(
                right_df, left_on=lcol, right_on=rcol, how="inner"
            )
            dup_risk = len(probe) > len(left_df.head(500)) * _WARN_DUPLICATE_RATIO
        except Exception:
            dup_risk = False

        warnings: list[str] = []
        if (1 - match_rate) > _WARN_LOSS_RATE:
            warnings.append(
                f"{lost_left} left rows ({(1-match_rate):.0%}) have no match in "
                f"{candidate.right_table}.{rcol}"
            )
        if dup_risk:
            warnings.append(
                f"M:N relationship detected — merging may create duplicate rows. "
                f"Consider aggregating {candidate.right_table} first."
            )

        return JoinQuality(
            left_table=candidate.left_table,
            right_table=candidate.right_table,
            left_col=lcol,
            right_col=rcol,
            left_rows=len(left_df),
            right_rows=len(right_df),
            matched_rows=matched,
            lost_left_rows=lost_left,
            lost_right_rows=lost_right,
            match_rate=round(match_rate, 4),
            duplicate_risk=dup_risk,
            warnings=warnings,
        )

    # ── Master DataFrame builder ──────────────────────────────────────────────

    def _build_master(
        self,
        dataframes: dict[str, pd.DataFrame],
        approved: list[JoinCandidate],
        quality_out: list[JoinQuality],
    ) -> pd.DataFrame | None:
        """
        Build a master DataFrame by left-joining approved candidates onto an
        anchor table (the table that appears most as left_table, or the largest).
        """
        # Pick anchor table: most frequent left_table, tie-break by row count
        freq: dict[str, int] = {}
        for c in approved:
            freq[c.left_table] = freq.get(c.left_table, 0) + 1
        anchor = max(freq, key=lambda t: (freq[t], len(dataframes.get(t, pd.DataFrame()))))

        master = dataframes[anchor].copy()
        merged_tables = {anchor}

        # Process candidates anchored on the anchor first, then rest
        ordered = sorted(approved, key=lambda c: (c.left_table != anchor, -c.confidence))

        for c in ordered:
            # Determine which side is already in master
            if c.left_table in merged_tables and c.right_table not in merged_tables:
                l_df = master
                r_df = dataframes[c.right_table].copy()
                l_col, r_col = c.left_col, c.right_col
            elif c.right_table in merged_tables and c.left_table not in merged_tables:
                # Flip candidate
                l_df = master
                r_df = dataframes[c.left_table].copy()
                l_col, r_col = c.right_col, c.left_col
            else:
                # Both already merged or neither — skip to avoid fan-out
                continue

            quality_out.append(
                self.compute_quality(dataframes[c.left_table], dataframes[c.right_table], c)
            )

            # Rename conflicting columns in right table before merge
            conflict_cols = set(l_df.columns) - {r_col}
            r_rename = {
                col: f"{c.right_table}__{col}"
                for col in r_df.columns
                if col != r_col and col in conflict_cols
            }
            if r_rename:
                r_df = r_df.rename(columns=r_rename)

            try:
                master = l_df.merge(
                    r_df,
                    left_on=l_col,
                    right_on=r_col,
                    how=c.join_type,
                    suffixes=("", f"_{c.right_table}"),
                )
                merged_tables.add(c.right_table)
                logger.info(
                    "Merged %s ↔ %s on %s=%s → %d rows",
                    anchor, c.right_table, l_col, r_col, len(master),
                )
            except Exception as exc:
                logger.warning("Merge failed (%s ↔ %s): %s", c.left_table, c.right_table, exc)

        return master if len(merged_tables) > 1 else None

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_pair(
        self,
        left_name: str,
        left_df: pd.DataFrame,
        right_name: str,
        right_df: pd.DataFrame,
    ) -> list[JoinCandidate]:
        """Return ALL scored candidates for one table pair (above MIN gates)."""
        results: list[JoinCandidate] = []

        for lcol in left_df.columns:
            for rcol in right_df.columns:
                name_sim = self._name_similarity(lcol, rcol)
                if name_sim < _MIN_NAME_SIM:
                    continue

                overlap = self._value_overlap(left_df[lcol], right_df[rcol])
                if overlap < _MIN_OVERLAP:
                    continue

                confidence = _W_NAME * name_sim + _W_OVERLAP * overlap
                results.append(JoinCandidate(
                    left_table=left_name,
                    right_table=right_name,
                    left_col=lcol,
                    right_col=rcol,
                    name_similarity=round(name_sim, 4),
                    value_overlap=round(overlap, 4),
                    confidence=round(confidence, 4),
                ))

        # Keep only the best candidate per (left_table, right_table) pair
        # for auto-suggestion; include all above threshold for UI
        if results:
            results.sort(key=lambda c: c.confidence, reverse=True)

        return results

    # ── Similarity helpers ────────────────────────────────────────────────────

    @staticmethod
    def _normalize_col_name(name: str) -> str:
        """Strip key suffixes/prefixes and normalise for comparison."""
        n = name.lower().strip()
        n = _STRIP_PREFIXES.sub("", n)
        n = _STRIP_SUFFIXES.sub("", n)
        n = n.replace("_", " ").replace("-", " ").strip()
        return n

    def _name_similarity(self, col_a: str, col_b: str) -> float:
        """
        Fuzzy ratio on normalised names.
        Exact original match → 1.0 (short-circuit).
        """
        if col_a.lower() == col_b.lower():
            return 1.0
        na = self._normalize_col_name(col_a)
        nb = self._normalize_col_name(col_b)
        if na == nb:
            return 0.95   # normalised match (e.g. account_id vs account_key)
        return fuzz.ratio(na, nb) / 100.0

    @staticmethod
    def _value_overlap(left: pd.Series, right: pd.Series) -> float:
        """
        Bidirectional referential integrity:
        max( |L∩R|/|L| , |L∩R|/|R| )
        Uses the more generous direction so foreign keys score well
        regardless of which side is the dimension table.

        Values are normalised to Python floats when numeric to avoid
        int64 / float64 string-representation mismatches (e.g. 1 vs 1.0).
        """
        def _to_set(s: pd.Series) -> set:
            vals = s.dropna()
            if len(vals) == 0:
                return set()
            as_num = pd.to_numeric(vals, errors="coerce")
            if as_num.notna().all():
                return set(as_num.tolist())   # Python floats: 1 == 1.0 in sets
            return set(vals.astype(str).str.strip().tolist())

        l_set = _to_set(left)
        r_set = _to_set(right)
        if not l_set or not r_set:
            return 0.0
        intersection = len(l_set & r_set)
        return max(intersection / len(l_set), intersection / len(r_set))

    # ── Backward-compatible aliases ───────────────────────────────────────────

    def detect_joins(
        self, dataframes: dict[str, pd.DataFrame]
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Legacy API — wraps suggest() into the old dict format."""
        candidates = self.suggest(dataframes)
        # Keep only highest-confidence candidate per table pair
        seen: set[tuple[str, str]] = set()
        join_map: dict[tuple[str, str], dict[str, Any]] = {}
        for c in candidates:
            key = (c.left_table, c.right_table)
            if key not in seen and c.confidence >= self.threshold:
                seen.add(key)
                join_map[key] = {
                    "left_col": c.left_col,
                    "right_col": c.right_col,
                    "confidence": c.confidence,
                    "join_type": c.join_type,
                }
        return join_map

    def build_joined(
        self,
        dataframes: dict[str, pd.DataFrame],
        join_map: dict[tuple[str, str], dict[str, Any]],
    ) -> pd.DataFrame | None:
        """Legacy API — converts old join_map dict to candidates and calls build()."""
        if not join_map:
            return None
        candidates = [
            JoinCandidate(
                left_table=left,
                right_table=right,
                left_col=info["left_col"],
                right_col=info["right_col"],
                name_similarity=1.0,
                value_overlap=1.0,
                confidence=info.get("confidence", 1.0),
                join_type=info.get("join_type", "left"),
            )
            for (left, right), info in join_map.items()
        ]
        result = self.build(dataframes, candidates)
        return result.master_df

    # ── Static helpers (kept public for tests) ────────────────────────────────

    @staticmethod
    def _referential_overlap(left: pd.Series, right: pd.Series) -> float:
        """Unidirectional overlap — kept for backward-compat with Module 1 tests."""
        left_set = set(left.dropna().unique())
        right_set = set(right.dropna().unique())
        if not left_set:
            return 0.0
        return len(left_set & right_set) / len(left_set)


# ── Streamlit sidebar UI ──────────────────────────────────────────────────────

def render_join_ui(
    candidates: list[JoinCandidate],
    quality: list[JoinQuality] | None = None,
    key_prefix: str = "join",
) -> list[JoinCandidate]:
    """
    Render join suggestion widgets in the Streamlit sidebar.

    For each candidate a user sees:
      - Table/column labels + confidence bar
      - Checkbox: confirm this join
      - Selectbox: override join type (left / inner / outer / right)
      - Quality stats (if quality is provided)

    Returns the list of *confirmed* JoinCandidates (with possibly-overridden
    join_type set).

    Raises ImportError if streamlit is not installed.
    """
    try:
        import streamlit as st
    except ImportError as exc:
        raise ImportError(
            "streamlit is required for render_join_ui. "
            "Install it with: pip install streamlit"
        ) from exc

    if not candidates:
        st.caption("No join candidates detected.")
        return []

    quality_map: dict[str, JoinQuality] = {}
    if quality:
        for q in quality:
            key = f"{q.left_table}↔{q.right_table}"
            quality_map[key] = q

    confirmed: list[JoinCandidate] = []

    st.subheader("🔗 Detected Joins")
    st.caption(
        "Review and confirm the joins below. "
        "Only confirmed joins are used to build the master table."
    )

    for i, c in enumerate(candidates):
        ui_key = f"{key_prefix}_{i}"
        conf_color = "🟢" if c.confidence >= 0.80 else "🟡" if c.confidence >= 0.60 else "🔴"
        header_label = f"{conf_color} {c.label}  ({c.confidence_pct}%)"

        with st.expander(header_label, expanded=(c.confidence >= 0.60)):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.progress(c.confidence, text=f"Confidence: {c.confidence_pct}%")
                st.markdown(
                    f"- **Name similarity**: {c.name_similarity:.0%}  \n"
                    f"- **Value overlap**: {c.value_overlap:.0%}"
                )

            with col2:
                join_type = st.selectbox(
                    "Join type",
                    options=JOIN_TYPES,
                    index=JOIN_TYPES.index(c.join_type),
                    key=f"{ui_key}_type",
                )

            q_key = f"{c.left_table}↔{c.right_table}"
            if q_key in quality_map:
                q = quality_map[q_key]
                st.markdown(
                    f"**Quality** — matched: {q.matched_rows} rows | "
                    f"lost left: {q.lost_left_rows} | "
                    f"lost right: {q.lost_right_rows} | "
                    f"match rate: {q.match_rate:.0%}"
                )
                for w in q.warnings:
                    st.warning(w, icon="⚠️")

            confirmed_flag = st.checkbox(
                "✅ Use this join",
                value=(c.confidence >= 0.60),
                key=f"{ui_key}_confirm",
            )

            if confirmed_flag:
                confirmed.append(JoinCandidate(
                    left_table=c.left_table,
                    right_table=c.right_table,
                    left_col=c.left_col,
                    right_col=c.right_col,
                    name_similarity=c.name_similarity,
                    value_overlap=c.value_overlap,
                    confidence=c.confidence,
                    join_type=join_type,
                ))

    if confirmed:
        st.success(f"{len(confirmed)} join(s) confirmed.", icon="✅")
    else:
        st.info("No joins confirmed. Each table will be queried independently.")

    return confirmed
