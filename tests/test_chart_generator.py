"""
tests/test_chart_generator.py — Unit tests for agent/chart_generator.py (Module 7).

All LLM calls are mocked. No Ollama/Groq or real kaleido writes required.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agent.chart_generator import (
    CHART_TYPES,
    ChartDecision,
    ChartGenerator,
    _parse_decision,
    _summarise_df,
    build_bar_pareto,
    build_funnel,
    build_heatmap,
    build_horizontal_bar,
    build_line_ma,
    build_pie,
    build_scatter,
)


# ─── Mock LLM ─────────────────────────────────────────────────────────────────

class MockLLM:
    """Sequential async mock — returns responses in order."""

    def __init__(self, responses: list[str] | str = "") -> None:
        self._responses = responses if isinstance(responses, list) else [responses]
        self._idx = 0
        self.provider = "mock"

    async def complete(self, system: str, user: str, temperature=None) -> str:
        r = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return r


def _yes_json(chart_type: str, x: str, y: str, title: str = "Test") -> str:
    return json.dumps({
        "needs_chart": True, "chart_type": chart_type,
        "x_col": x, "y_col": y, "title": title,
    })


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def bar_df() -> pd.DataFrame:
    return pd.DataFrame({
        "account": ["Acme", "Beta", "Gamma", "Delta", "Epsilon"],
        "revenue": [500_000, 320_000, 250_000, 100_000, 75_000],
    })


@pytest.fixture
def trend_df() -> pd.DataFrame:
    return pd.DataFrame({
        "month": ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"],
        "revenue": [120_000, 145_000, 132_000, 178_000, 165_000, 200_000],
    })


@pytest.fixture
def scatter_df() -> pd.DataFrame:
    return pd.DataFrame({
        "deal_size": [50, 120, 200, 350, 80, 150],
        "close_days": [30, 60, 90, 120, 45, 75],
    })


@pytest.fixture
def pie_df() -> pd.DataFrame:
    return pd.DataFrame({
        "service_line": ["Hard FM", "Soft FM", "Projects"],
        "revenue_pct": [55, 30, 15],
    })


@pytest.fixture
def pivot_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Hard FM": [100_000, 150_000],
        "Soft FM": [80_000, 60_000],
    }, index=["Q1", "Q2"])


@pytest.fixture
def funnel_df() -> pd.DataFrame:
    return pd.DataFrame({
        "stage": ["Prospecting", "Qualification", "Proposal", "Negotiation", "Closed Won"],
        "count": [200, 120, 60, 30, 15],
    })


@pytest.fixture
def hbar_df() -> pd.DataFrame:
    return pd.DataFrame({
        "account": ["Acme", "Beta", "Gamma", "Delta"],
        "days_open": [210, 180, 95, 45],
    })


# ─── TestChartDecision ────────────────────────────────────────────────────────

class TestChartDecision:
    def test_defaults_need_no_chart(self):
        d = ChartDecision()
        assert d.needs_chart is False

    def test_defaults_chart_type_is_bar_pareto(self):
        assert ChartDecision().chart_type == "bar_pareto"

    def test_fields_set_correctly(self):
        d = ChartDecision(
            needs_chart=True, chart_type="funnel",
            x_col="stage", y_col="count", title="Pipeline",
        )
        assert d.needs_chart is True
        assert d.chart_type == "funnel"
        assert d.x_col == "stage"
        assert d.y_col == "count"
        assert d.title == "Pipeline"


# ─── TestParseDecision ────────────────────────────────────────────────────────

class TestParseDecision:
    def test_valid_json_returns_decision(self):
        raw = _yes_json("bar_pareto", "account", "revenue", "Revenue")
        d = _parse_decision(raw)
        assert d.needs_chart is True
        assert d.chart_type == "bar_pareto"
        assert d.x_col == "account"
        assert d.y_col == "revenue"
        assert d.title == "Revenue"

    def test_needs_chart_false(self):
        raw = json.dumps({"needs_chart": False, "chart_type": "bar_pareto",
                          "x_col": "", "y_col": "", "title": ""})
        d = _parse_decision(raw)
        assert d.needs_chart is False

    def test_invalid_json_returns_no_chart(self):
        d = _parse_decision("This is not JSON at all")
        assert d.needs_chart is False

    def test_empty_string_returns_no_chart(self):
        d = _parse_decision("")
        assert d.needs_chart is False

    def test_markdown_fence_stripped(self):
        raw = '```json\n{"needs_chart": true, "chart_type": "scatter", "x_col": "x", "y_col": "y", "title": "T"}\n```'
        d = _parse_decision(raw)
        assert d.needs_chart is True
        assert d.chart_type == "scatter"

    def test_unknown_chart_type_falls_back_to_bar_pareto(self):
        raw = json.dumps({"needs_chart": True, "chart_type": "unknown_type",
                          "x_col": "a", "y_col": "b", "title": "T"})
        d = _parse_decision(raw)
        assert d.chart_type == "bar_pareto"

    def test_all_chart_types_accepted(self):
        for ct in CHART_TYPES:
            raw = json.dumps({"needs_chart": True, "chart_type": ct,
                              "x_col": "a", "y_col": "b", "title": "T"})
            d = _parse_decision(raw)
            assert d.chart_type == ct

    def test_bar_alias_accepted(self):
        raw = json.dumps({"needs_chart": True, "chart_type": "bar",
                          "x_col": "a", "y_col": "b", "title": "T"})
        d = _parse_decision(raw)
        assert d.needs_chart is True


# ─── TestSummariseDF ─────────────────────────────────────────────────────────

class TestSummariseDf:
    def test_includes_column_names(self, bar_df):
        s = _summarise_df(bar_df)
        assert "account" in s and "revenue" in s

    def test_includes_row_count(self, bar_df):
        s = _summarise_df(bar_df)
        assert "5" in s

    def test_empty_df_returns_empty_message(self):
        s = _summarise_df(pd.DataFrame())
        assert "Empty" in s

    def test_none_returns_empty_message(self):
        s = _summarise_df(None)
        assert "Empty" in s


# ─── TestBuildBarPareto ───────────────────────────────────────────────────────

class TestBuildBarPareto:
    def test_returns_figure(self, bar_df):
        import plotly.graph_objects as go
        fig = build_bar_pareto(bar_df, "account", "revenue", "Pareto Test")
        assert isinstance(fig, go.Figure)

    def test_has_two_traces(self, bar_df):
        fig = build_bar_pareto(bar_df, "account", "revenue", "T")
        assert len(fig.data) == 2

    def test_first_trace_is_bar(self, bar_df):
        import plotly.graph_objects as go
        fig = build_bar_pareto(bar_df, "account", "revenue", "T")
        assert isinstance(fig.data[0], go.Bar)

    def test_second_trace_is_scatter_pareto_line(self, bar_df):
        import plotly.graph_objects as go
        fig = build_bar_pareto(bar_df, "account", "revenue", "T")
        assert isinstance(fig.data[1], go.Scatter)

    def test_handles_zero_total(self):
        df = pd.DataFrame({"cat": ["A", "B"], "val": [0, 0]})
        fig = build_bar_pareto(df, "cat", "val", "Zero")
        assert fig is not None  # should not raise

    def test_sorted_descending(self, bar_df):
        fig = build_bar_pareto(bar_df, "account", "revenue", "T")
        # First bar should be the largest value
        y_vals = list(fig.data[0].y)
        assert y_vals == sorted(y_vals, reverse=True)


# ─── TestBuildLineMa ─────────────────────────────────────────────────────────

class TestBuildLineMa:
    def test_returns_figure(self, trend_df):
        import plotly.graph_objects as go
        fig = build_line_ma(trend_df, "month", "revenue", "Trend Test")
        assert isinstance(fig, go.Figure)

    def test_has_two_traces(self, trend_df):
        fig = build_line_ma(trend_df, "month", "revenue", "T")
        assert len(fig.data) == 2

    def test_second_trace_is_moving_average(self, trend_df):
        fig = build_line_ma(trend_df, "month", "revenue", "T")
        assert "MA" in fig.data[1].name

    def test_custom_window_in_trace_name(self, trend_df):
        fig = build_line_ma(trend_df, "month", "revenue", "T", window=5)
        assert "5" in fig.data[1].name

    def test_both_traces_are_scatter(self, trend_df):
        import plotly.graph_objects as go
        fig = build_line_ma(trend_df, "month", "revenue", "T")
        for trace in fig.data:
            assert isinstance(trace, go.Scatter)


# ─── TestBuildScatter ────────────────────────────────────────────────────────

class TestBuildScatter:
    def test_returns_figure(self, scatter_df):
        import plotly.graph_objects as go
        fig = build_scatter(scatter_df, "deal_size", "close_days", "Scatter Test")
        assert isinstance(fig, go.Figure)

    def test_trace_mode_is_markers(self, scatter_df):
        fig = build_scatter(scatter_df, "deal_size", "close_days", "T")
        assert fig.data[0].mode == "markers"

    def test_one_trace(self, scatter_df):
        fig = build_scatter(scatter_df, "deal_size", "close_days", "T")
        assert len(fig.data) == 1


# ─── TestBuildPie ────────────────────────────────────────────────────────────

class TestBuildPie:
    def test_returns_figure(self, pie_df):
        import plotly.graph_objects as go
        fig = build_pie(pie_df, "service_line", "revenue_pct", "Pie Test")
        assert isinstance(fig, go.Figure)

    def test_is_donut_style(self, pie_df):
        import plotly.graph_objects as go
        fig = build_pie(pie_df, "service_line", "revenue_pct", "T")
        assert isinstance(fig.data[0], go.Pie)
        assert fig.data[0].hole == pytest.approx(0.4)

    def test_labels_match_categories(self, pie_df):
        fig = build_pie(pie_df, "service_line", "revenue_pct", "T")
        labels = list(fig.data[0].labels)
        assert "Hard FM" in labels


# ─── TestBuildHeatmap ────────────────────────────────────────────────────────

class TestBuildHeatmap:
    def test_returns_figure(self, pivot_df):
        import plotly.graph_objects as go
        fig = build_heatmap(pivot_df, "Heatmap Test")
        assert isinstance(fig, go.Figure)

    def test_trace_is_heatmap_type(self, pivot_df):
        import plotly.graph_objects as go
        fig = build_heatmap(pivot_df, "T")
        assert isinstance(fig.data[0], go.Heatmap)

    def test_y_labels_from_index(self, pivot_df):
        fig = build_heatmap(pivot_df, "T")
        assert "Q1" in fig.data[0].y
        assert "Q2" in fig.data[0].y


# ─── TestBuildFunnel ─────────────────────────────────────────────────────────

class TestBuildFunnel:
    def test_returns_figure(self, funnel_df):
        import plotly.graph_objects as go
        fig = build_funnel(funnel_df, "stage", "count", "Pipeline Funnel")
        assert isinstance(fig, go.Figure)

    def test_trace_is_funnel_type(self, funnel_df):
        import plotly.graph_objects as go
        fig = build_funnel(funnel_df, "stage", "count", "T")
        assert isinstance(fig.data[0], go.Funnel)

    def test_sorted_largest_first(self, funnel_df):
        fig = build_funnel(funnel_df, "stage", "count", "T")
        x_vals = list(fig.data[0].x)
        assert x_vals == sorted(x_vals, reverse=True)


# ─── TestBuildHorizontalBar ──────────────────────────────────────────────────

class TestBuildHorizontalBar:
    def test_returns_figure(self, hbar_df):
        import plotly.graph_objects as go
        fig = build_horizontal_bar(hbar_df, "account", "days_open", "Ageing")
        assert isinstance(fig, go.Figure)

    def test_orientation_is_horizontal(self, hbar_df):
        fig = build_horizontal_bar(hbar_df, "account", "days_open", "T")
        assert fig.data[0].orientation == "h"

    def test_trace_is_bar(self, hbar_df):
        import plotly.graph_objects as go
        fig = build_horizontal_bar(hbar_df, "account", "days_open", "T")
        assert isinstance(fig.data[0], go.Bar)


# ─── TestChartGeneratorDecide ────────────────────────────────────────────────

class TestChartGeneratorDecide:
    def test_valid_llm_response_returns_decision(self, bar_df):
        llm = MockLLM(_yes_json("bar_pareto", "account", "revenue", "Rev"))
        gen = ChartGenerator(llm)
        d = asyncio.run(gen.decide("Revenue?", "aggregation", bar_df))
        assert d.needs_chart is True
        assert d.chart_type == "bar_pareto"

    def test_needs_chart_false_from_llm(self, bar_df):
        llm = MockLLM(json.dumps({"needs_chart": False, "chart_type": "bar_pareto",
                                  "x_col": "", "y_col": "", "title": ""}))
        gen = ChartGenerator(llm)
        d = asyncio.run(gen.decide("Q", "aggregation", bar_df))
        assert d.needs_chart is False

    def test_llm_failure_returns_no_chart(self, bar_df):
        class FailLLM:
            provider = "mock"
            async def complete(self, s, u, temperature=None):
                raise ConnectionError("down")
        gen = ChartGenerator(FailLLM())
        d = asyncio.run(gen.decide("Q", "ranking", bar_df))
        assert d.needs_chart is False

    def test_empty_df_skips_llm_returns_no_chart(self):
        llm = MockLLM("should not be called")
        gen = ChartGenerator(llm)
        d = asyncio.run(gen.decide("Q", "aggregation", pd.DataFrame()))
        assert d.needs_chart is False
        assert llm._idx == 0  # LLM was never called

    def test_single_row_df_returns_no_chart(self, bar_df):
        llm = MockLLM("should not be called")
        gen = ChartGenerator(llm)
        d = asyncio.run(gen.decide("Q", "aggregation", bar_df.head(1)))
        assert d.needs_chart is False


# ─── TestChartGeneratorGenerate ──────────────────────────────────────────────

class TestChartGeneratorGenerate:
    def test_generate_bar_pareto(self, bar_df):
        import plotly.graph_objects as go
        gen = ChartGenerator(MockLLM())
        decision = ChartDecision(
            needs_chart=True, chart_type="bar_pareto",
            x_col="account", y_col="revenue", title="T",
        )
        fig = gen.generate(decision, bar_df)
        assert isinstance(fig, go.Figure)

    def test_generate_line_ma(self, trend_df):
        import plotly.graph_objects as go
        gen = ChartGenerator(MockLLM())
        decision = ChartDecision(
            needs_chart=True, chart_type="line_ma",
            x_col="month", y_col="revenue", title="Trend",
        )
        fig = gen.generate(decision, trend_df)
        assert isinstance(fig, go.Figure)

    def test_generate_scatter(self, scatter_df):
        import plotly.graph_objects as go
        gen = ChartGenerator(MockLLM())
        decision = ChartDecision(
            needs_chart=True, chart_type="scatter",
            x_col="deal_size", y_col="close_days", title="T",
        )
        fig = gen.generate(decision, scatter_df)
        assert isinstance(fig, go.Figure)

    def test_generate_pie(self, pie_df):
        import plotly.graph_objects as go
        gen = ChartGenerator(MockLLM())
        decision = ChartDecision(
            needs_chart=True, chart_type="pie",
            x_col="service_line", y_col="revenue_pct", title="T",
        )
        fig = gen.generate(decision, pie_df)
        assert isinstance(fig, go.Figure)

    def test_generate_heatmap(self, pivot_df):
        import plotly.graph_objects as go
        gen = ChartGenerator(MockLLM())
        decision = ChartDecision(
            needs_chart=True, chart_type="heatmap",
            x_col="Hard FM", y_col="Soft FM", title="Heat",
        )
        fig = gen.generate(decision, pivot_df)
        assert isinstance(fig, go.Figure)

    def test_generate_funnel(self, funnel_df):
        import plotly.graph_objects as go
        gen = ChartGenerator(MockLLM())
        decision = ChartDecision(
            needs_chart=True, chart_type="funnel",
            x_col="stage", y_col="count", title="Funnel",
        )
        fig = gen.generate(decision, funnel_df)
        assert isinstance(fig, go.Figure)

    def test_generate_horizontal_bar(self, hbar_df):
        import plotly.graph_objects as go
        gen = ChartGenerator(MockLLM())
        decision = ChartDecision(
            needs_chart=True, chart_type="horizontal_bar",
            x_col="account", y_col="days_open", title="T",
        )
        fig = gen.generate(decision, hbar_df)
        assert isinstance(fig, go.Figure)

    def test_needs_chart_false_returns_none(self, bar_df):
        gen = ChartGenerator(MockLLM())
        fig = gen.generate(ChartDecision(needs_chart=False), bar_df)
        assert fig is None

    def test_empty_df_returns_none(self):
        gen = ChartGenerator(MockLLM())
        decision = ChartDecision(needs_chart=True, chart_type="bar_pareto",
                                 x_col="a", y_col="b", title="T")
        fig = gen.generate(decision, pd.DataFrame())
        assert fig is None

    def test_invalid_columns_returns_none(self, bar_df):
        gen = ChartGenerator(MockLLM())
        # Only one numeric column — can't auto-detect both x and y
        df_only_text = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        decision = ChartDecision(needs_chart=True, chart_type="scatter",
                                 x_col="nonexistent", y_col="also_missing", title="T")
        fig = gen.generate(decision, df_only_text)
        assert fig is None


# ─── TestResolveColumns ──────────────────────────────────────────────────────

class TestResolveColumns:
    def test_valid_columns_used_as_is(self, bar_df):
        gen = ChartGenerator(MockLLM())
        d = ChartDecision(needs_chart=True, x_col="account", y_col="revenue")
        x, y = gen._resolve_columns(d, bar_df)
        assert x == "account"
        assert y == "revenue"

    def test_auto_detects_string_col_as_x(self, bar_df):
        gen = ChartGenerator(MockLLM())
        d = ChartDecision(needs_chart=True, x_col="", y_col="revenue")
        x, _ = gen._resolve_columns(d, bar_df)
        assert x == "account"  # first non-numeric col

    def test_auto_detects_numeric_col_as_y(self, bar_df):
        gen = ChartGenerator(MockLLM())
        d = ChartDecision(needs_chart=True, x_col="account", y_col="missing")
        _, y = gen._resolve_columns(d, bar_df)
        assert y == "revenue"  # first numeric col

    def test_both_missing_auto_detected(self):
        df = pd.DataFrame({"cat": ["A", "B"], "val1": [1, 2], "val2": [3, 4]})
        gen = ChartGenerator(MockLLM())
        d = ChartDecision(needs_chart=True, x_col="", y_col="")
        x, y = gen._resolve_columns(d, df)
        assert x == "cat"
        assert y in ("val1", "val2")


# ─── TestSavePng ─────────────────────────────────────────────────────────────

class TestSavePng:
    def test_returns_path_object(self, bar_df, tmp_path):
        gen = ChartGenerator(MockLLM(), exports_dir=tmp_path)
        fig = build_bar_pareto(bar_df, "account", "revenue", "T")
        # Mock write_image to avoid kaleido dependency in CI
        with patch.object(fig, "write_image"):
            path = gen.save_png(fig, tmp_path)
        assert isinstance(path, Path)

    def test_filename_has_chart_prefix(self, bar_df, tmp_path):
        gen = ChartGenerator(MockLLM(), exports_dir=tmp_path)
        fig = build_bar_pareto(bar_df, "account", "revenue", "T")
        with patch.object(fig, "write_image"):
            path = gen.save_png(fig, tmp_path)
        assert path.name.startswith("chart_")

    def test_filename_ends_with_png(self, bar_df, tmp_path):
        gen = ChartGenerator(MockLLM(), exports_dir=tmp_path)
        fig = build_bar_pareto(bar_df, "account", "revenue", "T")
        with patch.object(fig, "write_image"):
            path = gen.save_png(fig, tmp_path)
        assert path.suffix == ".png"

    def test_charts_subdir_created(self, bar_df, tmp_path):
        out_dir = tmp_path / "exports"
        gen = ChartGenerator(MockLLM(), exports_dir=out_dir)
        fig = build_bar_pareto(bar_df, "account", "revenue", "T")
        with patch.object(fig, "write_image"):
            path = gen.save_png(fig, out_dir)
        assert (out_dir / "charts").is_dir()

    def test_graceful_on_kaleido_failure(self, bar_df, tmp_path):
        gen = ChartGenerator(MockLLM(), exports_dir=tmp_path)
        fig = build_bar_pareto(bar_df, "account", "revenue", "T")
        # Simulate kaleido not available
        with patch.object(fig, "write_image", side_effect=Exception("kaleido missing")):
            path = gen.save_png(fig, tmp_path)
        # Should return path without raising
        assert isinstance(path, Path)


# ─── TestGenerateForResult ───────────────────────────────────────────────────

class TestGenerateForResult:
    def _make_qr(self, df: pd.DataFrame):
        from agent.query_engine import QueryResult
        return QueryResult(
            question="Top accounts?",
            code="result = df",
            result=df,
            intent_type="aggregation",
            result_df=df,
        )

    def test_chart_set_on_result(self, bar_df):
        llm = MockLLM(_yes_json("bar_pareto", "account", "revenue", "Rev"))
        gen = ChartGenerator(llm, auto_save=False)
        qr = self._make_qr(bar_df)
        result = asyncio.run(gen.generate_for_result(qr))
        assert result.chart is not None

    def test_no_chart_for_empty_df(self):
        llm = MockLLM("should not be called")
        gen = ChartGenerator(llm, auto_save=False)
        qr = self._make_qr(pd.DataFrame())
        result = asyncio.run(gen.generate_for_result(qr))
        assert result.chart is None

    def test_auto_save_calls_write_image(self, bar_df, tmp_path):
        llm = MockLLM(_yes_json("bar_pareto", "account", "revenue", "Rev"))
        gen = ChartGenerator(llm, exports_dir=tmp_path, auto_save=True)
        qr = self._make_qr(bar_df)
        # Patch write_image to avoid actual kaleido call
        import plotly.graph_objects as go
        with patch.object(go.Figure, "write_image") as mock_wi:
            asyncio.run(gen.generate_for_result(qr))
        mock_wi.assert_called_once()

    def test_no_auto_save_when_disabled(self, bar_df, tmp_path):
        llm = MockLLM(_yes_json("bar_pareto", "account", "revenue", "Rev"))
        gen = ChartGenerator(llm, exports_dir=tmp_path, auto_save=False)
        qr = self._make_qr(bar_df)
        import plotly.graph_objects as go
        with patch.object(go.Figure, "write_image") as mock_wi:
            asyncio.run(gen.generate_for_result(qr))
        mock_wi.assert_not_called()

    def test_non_query_result_returned_unchanged(self):
        llm = MockLLM()
        gen = ChartGenerator(llm, auto_save=False)
        result = asyncio.run(gen.generate_for_result("not a QueryResult"))
        assert result == "not a QueryResult"

    def test_returns_same_object(self, bar_df):
        llm = MockLLM(_yes_json("bar_pareto", "account", "revenue", "Rev"))
        gen = ChartGenerator(llm, auto_save=False)
        qr = self._make_qr(bar_df)
        result = asyncio.run(gen.generate_for_result(qr))
        assert result is qr  # mutated in-place
