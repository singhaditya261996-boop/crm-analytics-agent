"""
agent/context_builder.py — Build rich schema context strings for the LLM.

The context tells the LLM what DataFrames are loaded, their columns,
dtypes, sample values, and join relationships so it can write correct
pandas code without hallucinating column names.

Public API
----------
ContextBuilder(dataframes: dict[str, pd.DataFrame])
    .build_schema_context() -> str
    .build_join_context(join_map: dict) -> str
    .build_full_context(join_map=None) -> str
"""
from __future__ import annotations

import pandas as pd
from typing import Any


class ContextBuilder:
    """
    Converts loaded DataFrames into a structured, token-efficient context
    string suitable for injection into an LLM system prompt.
    """

    MAX_SAMPLE_ROWS: int = 3
    MAX_UNIQUE_VALUES: int = 10

    def __init__(self, dataframes: dict[str, pd.DataFrame]) -> None:
        self.dataframes = dataframes

    # ── Public methods ────────────────────────────────────────────────────────

    def build_schema_context(self) -> str:
        """Return a markdown-formatted schema summary of all loaded DataFrames."""
        if not self.dataframes:
            return "No data loaded."

        sections: list[str] = ["## Loaded DataFrames\n"]
        for name, df in self.dataframes.items():
            sections.append(self._describe_dataframe(name, df))
        return "\n".join(sections)

    def build_join_context(self, join_map: dict[str, Any]) -> str:
        """
        Return a human-readable description of detected join relationships.

        Parameters
        ----------
        join_map : output from data.joiner.JoinDetector.detect_joins()
        """
        if not join_map:
            return "No join relationships detected."

        lines = ["## Detected Join Relationships\n"]
        for (left, right), info in join_map.items():
            lines.append(
                f"- **{left}** ↔ **{right}** "
                f"on `{info['left_col']}` = `{info['right_col']}` "
                f"(confidence: {info['confidence']:.0%})"
            )
        return "\n".join(lines)

    def build_full_context(self, join_map: dict | None = None) -> str:
        """Combine schema context and (optionally) join context."""
        parts = [self.build_schema_context()]
        if join_map:
            parts.append(self.build_join_context(join_map))
        return "\n\n".join(parts)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _describe_dataframe(self, name: str, df: pd.DataFrame) -> str:
        lines = [f"### `{name}`  ({len(df):,} rows × {len(df.columns)} columns)\n"]
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_pct = df[col].isna().mean() * 100
            sample = self._sample_values(df[col])
            lines.append(
                f"  - `{col}` ({dtype}) — nulls: {null_pct:.1f}% — samples: {sample}"
            )
        return "\n".join(lines)

    def _sample_values(self, series: pd.Series) -> str:
        non_null = series.dropna()
        if non_null.empty:
            return "*(all null)*"
        unique = non_null.unique()
        shown = unique[: self.MAX_UNIQUE_VALUES]
        formatted = ", ".join(repr(v) for v in shown)
        if len(unique) > self.MAX_UNIQUE_VALUES:
            formatted += f", … ({len(unique):,} unique)"
        return formatted
