"""
agent/chart_generator.py — LLM-driven chart generation for CRM query results.

Pipeline
--------
QueryResult → LLM decides if a chart helps → chart type + columns selected
→ Plotly Figure built → auto-saved as PNG → returned on QueryResult.chart

Supported chart types
---------------------
bar_pareto    : sorted bars + cumulative Pareto % line (concentration/80-20)
line_ma       : time-series line with rolling moving-average overlay
scatter       : correlation between two numeric variables
pie           : proportions (2–6 categories, donut style)
heatmap       : matrix / pivot-table heatmap
funnel        : pipeline stages from largest to smallest
horizontal_bar: rankings or pipeline ageing comparisons

Public API
----------
gen    = ChartGenerator(llm_client, exports_dir=Path("exports"))
result = await gen.generate_for_result(query_result)   # mutates + returns
fig    = gen.generate(decision, result_df)              # sync
path   = gen.save_png(fig)                              # sync
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ── Lazy plotly import — module loads cleanly even if plotly absent ────────────
try:
    import plotly.graph_objects as go
    _PLOTLY_OK = True
except ImportError:  # pragma: no cover
    go = None  # type: ignore[assignment]
    _PLOTLY_OK = False

# ─── Registry ─────────────────────────────────────────────────────────────────

CHART_TYPES: dict[str, str] = {
    "bar_pareto":     "Ranked bars with cumulative % Pareto line (concentration/80-20)",
    "line_ma":        "Time-series line with rolling moving-average overlay",
    "scatter":        "Correlation between two numeric variables",
    "pie":            "Proportions of a whole (2–6 categories, donut style)",
    "heatmap":        "Matrix or pivot-table heatmap",
    "funnel":         "Pipeline stages from largest to smallest",
    "horizontal_bar": "Rankings or pipeline ageing comparisons",
}

# ─── Prompt templates ─────────────────────────────────────────────────────────

_DECIDE_SYSTEM = """\
You are a data visualisation expert for CRM analytics.

Decide if a chart meaningfully adds value to this analysis result.

Respond with ONLY valid JSON — no markdown fences, no extra text:
{"needs_chart": true, "chart_type": "bar_pareto", "x_col": "col_name", "y_col": "col_name", "title": "Chart Title"}

Chart types:
  bar_pareto:     ranked bars with cumulative % Pareto line (concentration/80-20)
  line_ma:        time-series line with rolling moving-average overlay
  scatter:        correlation between two numeric variables
  pie:            proportions of a whole (2-6 categories only, donut style)
  heatmap:        matrix or pivot-table heatmap
  funnel:         pipeline stages from largest to smallest
  horizontal_bar: rankings or pipeline ageing comparisons

Rules:
  - chart_type must be exactly one of the seven names above
  - x_col and y_col must be column names present in the result
  - Set needs_chart to false for: single scalar, error message, fewer than 2 rows
  - For pie: only when 2-6 distinct categories exist
  - For heatmap: set x_col and y_col to any two valid column names"""

_DECIDE_USER = """\
Question: {question}
Intent type: {intent_type}
Result preview:
{result_summary}

Does this result benefit from a chart?"""


# ─── ChartDecision dataclass ──────────────────────────────────────────────────

@dataclass
class ChartDecision:
    """Structured LLM response: whether and how to chart a result."""
    needs_chart: bool = False
    chart_type: str = "bar_pareto"
    x_col: str = ""
    y_col: str = ""
    title: str = "Analysis Result"


# ─── Module-level chart builders (individually importable for testing) ─────────

def build_bar_pareto(
    df: pd.DataFrame, x_col: str, y_col: str, title: str
) -> Any:
    """
    Bar chart sorted descending with a secondary cumulative Pareto % line.
    Shows what fraction of total the top-N items account for (80/20 analysis).
    """
    _require_plotly()
    df_s = df[[x_col, y_col]].dropna().sort_values(y_col, ascending=False).copy()
    total = df_s[y_col].sum()
    df_s["_cum_pct"] = (df_s[y_col].cumsum() / total * 100) if total != 0 else 0.0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_s[x_col].astype(str), y=df_s[y_col],
        name=y_col, marker_color="#00B4D8",
    ))
    fig.add_trace(go.Scatter(
        x=df_s[x_col].astype(str), y=df_s["_cum_pct"],
        name="Cumulative %", yaxis="y2",
        mode="lines+markers",
        line=dict(color="#FF6B6B", width=2), marker=dict(size=6),
    ))
    fig.update_layout(
        title=title, template="plotly_dark",
        yaxis=dict(title=y_col),
        yaxis2=dict(
            title="Cumulative %", overlaying="y", side="right",
            range=[0, 110], showgrid=False,
        ),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def build_line_ma(
    df: pd.DataFrame, x_col: str, y_col: str, title: str, window: int = 3
) -> Any:
    """
    Time-series line chart with a rolling moving-average overlay.
    Sorts by x_col before plotting.
    """
    _require_plotly()
    df_s = df[[x_col, y_col]].dropna().sort_values(x_col).copy()
    df_s["_ma"] = df_s[y_col].rolling(window=window, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_s[x_col].astype(str), y=df_s[y_col],
        name=y_col, mode="lines+markers",
        line=dict(color="#00B4D8", width=2), marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=df_s[x_col].astype(str), y=df_s["_ma"],
        name=f"{window}-period MA", mode="lines",
        line=dict(color="#FFB703", width=2, dash="dash"),
    ))
    fig.update_layout(
        title=title, template="plotly_dark",
        xaxis_title=x_col, yaxis_title=y_col,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def build_scatter(
    df: pd.DataFrame, x_col: str, y_col: str, title: str
) -> Any:
    """Scatter plot — correlation between two numeric columns."""
    _require_plotly()
    df_clean = df[[x_col, y_col]].dropna()
    fig = go.Figure(go.Scatter(
        x=df_clean[x_col], y=df_clean[y_col],
        mode="markers",
        marker=dict(color="#00B4D8", size=8, opacity=0.75),
    ))
    fig.update_layout(
        title=title, template="plotly_dark",
        xaxis_title=x_col, yaxis_title=y_col,
    )
    return fig


def build_pie(
    df: pd.DataFrame, x_col: str, y_col: str, title: str
) -> Any:
    """Donut-style pie chart. Best for 2–6 distinct categories."""
    _require_plotly()
    df_clean = df[[x_col, y_col]].dropna()
    fig = go.Figure(go.Pie(
        labels=df_clean[x_col].astype(str),
        values=df_clean[y_col],
        hole=0.4,
        textinfo="label+percent",
        hoverinfo="label+value+percent",
    ))
    fig.update_layout(title=title, template="plotly_dark")
    return fig


def build_heatmap(df: pd.DataFrame, title: str) -> Any:
    """
    Heatmap over numeric columns in df.
    Works naturally with pivot tables — index → y-axis, columns → x-axis.
    """
    _require_plotly()
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        numeric = df.copy()

    y_labels = (
        numeric.index.astype(str).tolist()
        if not isinstance(numeric.index, pd.RangeIndex)
        else [str(i) for i in numeric.index]
    )
    fig = go.Figure(go.Heatmap(
        z=numeric.values,
        x=numeric.columns.astype(str).tolist(),
        y=y_labels,
        colorscale="Blues",
        hoverongaps=False,
    ))
    fig.update_layout(title=title, template="plotly_dark")
    return fig


def build_funnel(
    df: pd.DataFrame, x_col: str, y_col: str, title: str
) -> Any:
    """
    Pipeline funnel. x_col = stage names, y_col = counts/values.
    Sorted from largest to smallest (top → bottom of funnel).
    """
    _require_plotly()
    df_s = df[[x_col, y_col]].dropna().sort_values(y_col, ascending=False)
    fig = go.Figure(go.Funnel(
        y=df_s[x_col].astype(str),
        x=df_s[y_col],
        textinfo="value+percent initial",
        marker=dict(color="#0077B6"),
    ))
    fig.update_layout(title=title, template="plotly_dark")
    return fig


def build_horizontal_bar(
    df: pd.DataFrame, x_col: str, y_col: str, title: str
) -> Any:
    """
    Horizontal bar chart sorted ascending so the largest appears at the top.
    Useful for rankings, pipeline ageing, or comparison.
    """
    _require_plotly()
    df_s = df[[x_col, y_col]].dropna().sort_values(y_col, ascending=True)
    fig = go.Figure(go.Bar(
        x=df_s[y_col], y=df_s[x_col].astype(str),
        orientation="h", marker_color="#00B4D8",
    ))
    fig.update_layout(
        title=title, template="plotly_dark",
        xaxis_title=y_col, yaxis_title=x_col,
    )
    return fig


# ─── Dispatch table ───────────────────────────────────────────────────────────

_CHART_BUILDERS: dict[str, Any] = {
    "bar_pareto":     build_bar_pareto,
    "bar":            build_bar_pareto,      # backward-compat alias
    "line_ma":        build_line_ma,
    "scatter":        build_scatter,
    "pie":            build_pie,
    "heatmap":        build_heatmap,
    "funnel":         build_funnel,
    "horizontal_bar": build_horizontal_bar,
}


def _require_plotly() -> None:
    if not _PLOTLY_OK:  # pragma: no cover
        raise ImportError("Install plotly: pip install plotly")


# ─── ChartGenerator ───────────────────────────────────────────────────────────

class ChartGenerator:
    """
    Orchestrates the full chart pipeline:
    1. Ask LLM if a chart helps and which type/columns to use.
    2. Build the Plotly figure.
    3. Optionally auto-save as PNG to exports/charts/.
    4. Attach figure to QueryResult.chart.

    Usage
    -----
    gen    = ChartGenerator(llm_client, exports_dir=Path("exports"))
    result = await gen.generate_for_result(query_result)
    """

    def __init__(
        self,
        llm_client: Any,
        exports_dir: Path | None = None,
        auto_save: bool = True,
    ) -> None:
        self._llm = llm_client
        self._exports_dir = exports_dir or Path("exports")
        self._auto_save = auto_save

    # ── Async LLM decision ────────────────────────────────────────────────────

    async def decide(
        self,
        question: str,
        intent_type: str,
        result_df: pd.DataFrame,
    ) -> ChartDecision:
        """Ask the LLM whether and how to chart the result."""
        if result_df is None or result_df.empty or len(result_df) < 2:
            return ChartDecision(needs_chart=False)

        user = _DECIDE_USER.format(
            question=question,
            intent_type=intent_type,
            result_summary=_summarise_df(result_df),
        )
        try:
            raw = await self._llm.complete(_DECIDE_SYSTEM, user)
            return _parse_decision(raw)
        except Exception as exc:
            logger.warning("Chart decision LLM call failed: %s", exc)
            return ChartDecision(needs_chart=False)

    # ── Sync chart generation ─────────────────────────────────────────────────

    def generate(
        self,
        decision: ChartDecision,
        result_df: pd.DataFrame,
    ) -> Any:
        """
        Build a Plotly figure from a ChartDecision.
        Returns None if chart not needed, df is empty, or columns can't be resolved.
        """
        if not decision.needs_chart:
            return None
        if result_df is None or result_df.empty:
            return None

        chart_type = decision.chart_type.lower()

        # Heatmap uses the whole DataFrame — no per-column dispatch needed
        if chart_type == "heatmap":
            return build_heatmap(result_df, decision.title)

        x, y = self._resolve_columns(decision, result_df)
        if not x or not y:
            logger.debug(
                "Cannot resolve chart columns (x=%r y=%r) from %s",
                x, y, list(result_df.columns),
            )
            return None

        builder = _CHART_BUILDERS.get(chart_type, build_bar_pareto)
        try:
            return builder(result_df, x, y, decision.title)
        except Exception as exc:
            logger.warning("Chart build failed (%s): %s", chart_type, exc)
            return None

    # ── PNG export ────────────────────────────────────────────────────────────

    def save_png(
        self,
        fig: Any,
        exports_dir: Path | None = None,
    ) -> Path:
        """
        Write the figure as PNG to <exports_dir>/charts/chart_YYYYMMDD_HHMMSS.png.
        Returns the path whether or not the write succeeded.
        """
        out_dir = (exports_dir or self._exports_dir) / "charts"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"chart_{ts}.png"
        try:
            fig.write_image(str(out_path), width=1200, height=600, scale=2)
            logger.info("Chart saved → %s", out_path)
        except Exception as exc:
            logger.warning("PNG save failed (kaleido installed?): %s", exc)
        return out_path

    # ── End-to-end helper ─────────────────────────────────────────────────────

    async def generate_for_result(self, query_result: Any) -> Any:
        """
        Full pipeline: decide → generate → (optionally save PNG).
        Mutates query_result.chart in place and returns the same object.
        """
        from agent.query_engine import QueryResult  # local import avoids circular

        if not isinstance(query_result, QueryResult):
            return query_result

        result_df = query_result.result_df
        if result_df is None or result_df.empty:
            return query_result

        decision = await self.decide(
            query_result.question,
            query_result.intent_type,
            result_df,
        )
        fig = self.generate(decision, result_df)

        if fig is not None:
            query_result.chart = fig
            if self._auto_save:
                self.save_png(fig)

        return query_result

    # ── Column resolution ─────────────────────────────────────────────────────

    def _resolve_columns(
        self, decision: ChartDecision, df: pd.DataFrame
    ) -> tuple[str, str]:
        """
        Return (x_col, y_col) validated against df.columns.
        Falls back to auto-detection when the LLM-specified columns are absent.
        """
        x = decision.x_col if decision.x_col in df.columns else ""
        y = decision.y_col if decision.y_col in df.columns else ""

        if not x or not y:
            numeric = df.select_dtypes(include="number").columns.tolist()
            non_num = df.select_dtypes(exclude="number").columns.tolist()
            if not x:
                x = non_num[0] if non_num else (
                    numeric[0] if len(numeric) > 1 else ""
                )
            if not y:
                remaining = [c for c in numeric if c != x]
                y = remaining[0] if remaining else (numeric[0] if numeric else "")

        return x, y


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _summarise_df(df: pd.DataFrame) -> str:
    """Compact text representation of a DataFrame for LLM prompt injection."""
    if df is None or df.empty:
        return "Empty result."
    preview = df.head(5).to_string(index=False, max_colwidth=40)
    return f"Columns: {list(df.columns)}\nRows: {len(df)}\n{preview}"


def _parse_decision(raw: str) -> ChartDecision:
    """
    Extract and parse JSON from an LLM response string.
    Returns ChartDecision(needs_chart=False) on any parse failure.
    """
    # Strip any markdown fences
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    m = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if not m:
        logger.debug("No JSON in chart decision response: %.80s", raw)
        return ChartDecision(needs_chart=False)
    try:
        data = json.loads(m.group())
        chart_type = str(data.get("chart_type", "bar_pareto"))
        if chart_type not in _CHART_BUILDERS:
            chart_type = "bar_pareto"
        return ChartDecision(
            needs_chart=bool(data.get("needs_chart", False)),
            chart_type=chart_type,
            x_col=str(data.get("x_col", "")),
            y_col=str(data.get("y_col", "")),
            title=str(data.get("title", "Analysis Result")),
        )
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        logger.debug("Chart decision JSON parse failed: %s — %.80s", exc, raw)
        return ChartDecision(needs_chart=False)


# ─── Streamlit render helper ──────────────────────────────────────────────────

def render_chart_ui(fig: Any) -> None:
    """Render a Plotly figure inline in Streamlit."""
    try:
        import streamlit as st
    except ImportError as exc:
        raise ImportError(
            "streamlit is required for render_chart_ui()"
        ) from exc
    if fig is None:
        return
    st.plotly_chart(fig, use_container_width=True)
