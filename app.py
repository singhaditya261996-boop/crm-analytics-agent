"""
app.py — Streamlit entry point for the Equans CRM Analytics Agent (Module 9).

Four tabs: Chat · Weekly Agenda · Data Explorer · Insights Dashboard
All query results rendered through a single render_answer() component.
100% local — no data leaves this machine.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

from agent.chart_generator import ChartGenerator, render_chart_ui
from agent.llm_client import LLMClient
from agent.query_engine import INTENT_TYPES, PivotHandler, QueryEngine, QueryResult
from agent.self_improver import SelfImprover
from agent.session_exporter import SessionExporter
from tracker.csv_manager import CSVManager
from tracker.database import TrackerDB
from agenda.prompts import AGENDA_QUESTIONS, SECTION_TITLES, get_section
from data.joiner import JoinCandidate, JoinDetector, render_join_ui
from data.loader import DataLoader
from data.profiler import DataProfiler, render_profile_ui
from data.update_handler import FileClassification, UpdateHandler

try:
    import plotly.graph_objects as go
    import plotly.io as _pio
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False

# ── Constants ─────────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent / "config" / "settings.yaml"
APP_VERSION = "0.9.0"

_SECTION_COLORS = {
    1: "#1D4ED8",
    2: "#0F766E",
    3: "#7C3AED",
    4: "#C2410C",
    5: "#DC2626",
    6: "#16A34A",
}

_GRACEFUL_MARKERS = ("CANNOT_ANSWER", "I wasn't able to compute")

_DARK_CSS = """
<style>
[data-testid="stSidebar"] { background-color: #1a1a2e; }
.conf-high   { background:#16a34a; color:white; padding:2px 8px; border-radius:12px; font-size:.75rem; }
.conf-review { background:#d97706; color:white; padding:2px 8px; border-radius:12px; font-size:.75rem; }
.conf-low    { background:#dc2626; color:white; padding:2px 8px; border-radius:12px; font-size:.75rem; }
.intent-pill { background:#374151; color:#9ca3af; padding:2px 8px; border-radius:12px; font-size:.7rem; }
.upd-badge   { background:#dc2626; color:white; padding:3px 10px; border-radius:12px; font-weight:bold; font-size:.8rem; }
</style>
"""

_LIGHT_CSS = """
<style>
.conf-high   { background:#16a34a; color:white; padding:2px 8px; border-radius:12px; font-size:.75rem; }
.conf-review { background:#d97706; color:white; padding:2px 8px; border-radius:12px; font-size:.75rem; }
.conf-low    { background:#dc2626; color:white; padding:2px 8px; border-radius:12px; font-size:.75rem; }
.intent-pill { background:#e5e7eb; color:#374151; padding:2px 8px; border-radius:12px; font-size:.7rem; }
.upd-badge   { background:#dc2626; color:white; padding:3px 10px; border-radius:12px; font-weight:bold; font-size:.8rem; }
</style>
"""

# ── Config ────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Session state ─────────────────────────────────────────────────────────────

_SS_DEFAULTS: dict[str, Any] = {
    "chat_history":           [],
    "agenda_results":         {},
    "dataframes":             {},
    "profiles":               {},
    "join_candidates":        [],
    "confirmed_joins":        [],
    "table_versions":         {},
    "last_loaded":            None,
    "data_updates":           0,
    "file_hashes":            {},
    "dark_mode":              True,
    "llm_provider":           "ollama",
    "engine":                 None,
    "chart_gen":              None,
    "settings":               {},
    "auto_refresh_dashboard": False,
    "last_dashboard_refresh": None,
    "s_model":                None,
    "s_temperature":          0.1,
    "s_dormant_days":         90,
    "s_rec_mode":             True,
    "s_conf_threshold":       60,
    # Module 11 — update handler state
    "pending_updates":        {},
    "stale_questions":        set(),
    "watcher_started":        False,
    # Module 12 — tracker
    "tracker_db":             None,
}


def _init_session_state(cfg: dict) -> None:
    for k, v in _SS_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if not st.session_state.settings:
        st.session_state.settings = cfg


def _apply_theme() -> None:
    css = _DARK_CSS if st.session_state.dark_mode else _LIGHT_CSS
    st.markdown(css, unsafe_allow_html=True)


# ── File loading ──────────────────────────────────────────────────────────────

def _get_update_handler(cfg: dict) -> UpdateHandler:
    cache_dir = Path(cfg.get("data", {}).get("upload_folder", "data/uploads")).parent / ".cache"
    return UpdateHandler(cache_dir=cache_dir)


def _get_tracker(cfg: dict) -> TrackerDB:
    if st.session_state.tracker_db is None:
        db_url = cfg.get("tracker", {}).get("db_url", TrackerDB.DEFAULT_URL)
        st.session_state.tracker_db = TrackerDB(db_url=db_url)
    return st.session_state.tracker_db


def _load_files(uploaded_files: list, cfg: dict) -> None:
    upload_dir = Path(cfg.get("data", {}).get("upload_folder", "data/uploads"))
    upload_dir.mkdir(parents=True, exist_ok=True)

    handler = _get_update_handler(cfg)
    new_count = 0
    updated_names: list[str] = []

    for uf in uploaded_files:
        content = uf.read()
        uf.seek(0)

        file_path = upload_dir / uf.name
        with open(file_path, "wb") as fh:
            fh.write(content)

        try:
            loader = DataLoader(cfg)
            new_dfs = loader.load(file_path)
        except Exception as exc:
            st.error(f"Failed to load {uf.name}: {exc}")
            continue

        for name, df in new_dfs.items():
            old_df = st.session_state.dataframes.get(name)
            summary = handler.process_update(
                name, df, old_df=old_df, file_bytes=content
            )

            if summary.classification == FileClassification.DUPLICATE:
                continue

            if summary.classification == FileClassification.NEW_TABLE:
                st.session_state.dataframes[name] = df
                handler.register(name, summary.new_fingerprint)
            else:
                # UPDATED_VERSION or CORRECTED_DATA
                if old_df is not None:
                    handler.save_version(name, old_df)
                st.session_state.dataframes[name] = df
                handler.register(name, summary.new_fingerprint)
                st.session_state.pending_updates[name] = summary

            updated_names.append(name)
            new_count += 1

    if new_count > 0:
        st.session_state.data_updates = new_count
        st.session_state.last_loaded = datetime.now(timezone.utc)

        try:
            det = JoinDetector(cfg)
            st.session_state.join_candidates = det.suggest(st.session_state.dataframes)
        except Exception:
            st.session_state.join_candidates = []

        try:
            prof = DataProfiler(cfg)
            st.session_state.profiles = prof.profile_all(st.session_state.dataframes)
        except Exception:
            st.session_state.profiles = {}

        # Flag stale agenda questions
        stale = handler.check_agenda_impact(
            updated_names, st.session_state.agenda_results
        )
        st.session_state.stale_questions.update(stale)

        st.session_state.engine = None
        st.session_state.chart_gen = None
        st.success(f"Loaded {new_count} new/changed file(s).")


def _load_files_from_paths(paths: list[Path], cfg: dict) -> None:
    """Load files from disk paths (used by folder-watcher sentinel)."""
    import io as _io

    class _FakeSt:
        """Minimal stand-in for a Streamlit UploadedFile."""
        def __init__(self, path: Path) -> None:
            self.name = path.name
            self._data = path.read_bytes()

        def read(self) -> bytes:
            return self._data

        def seek(self, _pos: int) -> None:
            pass

    fake_files = []
    for p in paths:
        if p.exists():
            fake_files.append(_FakeSt(p))

    if fake_files:
        _load_files(fake_files, cfg)


# ── Engine lazy init ──────────────────────────────────────────────────────────

def _get_engine(cfg: dict) -> QueryEngine:
    if st.session_state.engine is None:
        llm = LLMClient(cfg)
        dfs = dict(st.session_state.dataframes)

        if st.session_state.confirmed_joins and dfs:
            try:
                det = JoinDetector(cfg)
                jr = det.build(dfs, st.session_state.confirmed_joins)
                if jr.master_df is not None and len(jr.master_df) > 0:
                    dfs["_joined"] = jr.master_df
            except Exception:
                pass

        quality_report: dict = {}
        if st.session_state.profiles:
            try:
                quality_report = DataProfiler(cfg).build_quality_report(
                    st.session_state.profiles
                )
            except Exception:
                pass

        tracker = _get_tracker(cfg)
        si = SelfImprover(llm_client=llm, config=cfg, tracker_db=tracker)

        st.session_state.engine = QueryEngine(
            llm_client=llm,
            dataframes=dfs,
            quality_report=quality_report,
            exports_dir=Path("exports"),
            self_improver=si,
        )
    return st.session_state.engine


def _get_chart_gen(cfg: dict) -> ChartGenerator:
    if st.session_state.chart_gen is None:
        llm = LLMClient(cfg)
        st.session_state.chart_gen = ChartGenerator(
            llm_client=llm,
            exports_dir=Path("exports"),
            auto_save=cfg.get("exports", {}).get("auto_save_charts", True),
        )
    return st.session_state.chart_gen


# ── Async bridge for ChartGenerator ──────────────────────────────────────────

def _run_async(coro: Any) -> Any:
    """Run an async coroutine safely from Streamlit's sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# ── Query runner ──────────────────────────────────────────────────────────────

def _run_query(question: str, cfg: dict) -> QueryResult:
    if not st.session_state.dataframes:
        return QueryResult(
            question=question,
            code="",
            result="CANNOT_ANSWER: No data loaded.",
            error=None,
            answer_text="No data loaded — upload CRM files in the sidebar.",
        )

    engine = _get_engine(cfg)
    with st.spinner("Analysing data..."):
        result = engine.query(question)

    cg = _get_chart_gen(cfg)
    with st.spinner("Generating chart..."):
        try:
            _run_async(cg.generate_for_result(result))
        except Exception:
            pass

    # Log query to tracker (non-fatal)
    try:
        _get_tracker(cfg).log_query(
            question=result.question,
            code=result.code,
            result_summary=str(result.answer_text or result.result or "")[:500],
            score=float(result.confidence_score),
            iterations=result.iterations,
            error=result.error,
        )
    except Exception:
        pass

    return result


# ── Data quality traffic light ────────────────────────────────────────────────

def _dq_traffic_light(profile: Any) -> str:
    try:
        if profile.issue_count("error") > 0 or profile.overall_null_pct >= 30:
            return "🔴"
        if profile.overall_null_pct >= 10:
            return "🟡"
        return "🟢"
    except Exception:
        return "⬜"


# ── Dashboard HTML export ─────────────────────────────────────────────────────

def _build_kpi_html(dfs: dict[str, pd.DataFrame]) -> str:
    all_cols = {
        col.lower(): (tbl, df)
        for tbl, df in dfs.items()
        for col in df.columns
    }

    def _try_sum(keywords: list[str]) -> str:
        for kw in keywords:
            for col_lower, (_, df) in all_cols.items():
                if kw in col_lower:
                    try:
                        v = pd.to_numeric(df[col_lower], errors="coerce").sum()
                        return f"£{v:,.0f}"
                    except Exception:
                        pass
        return "—"

    def _try_count(keywords: list[str]) -> str:
        for kw in keywords:
            for col_lower, (_, df) in all_cols.items():
                if kw in col_lower:
                    return f"{df[col_lower].nunique():,}"
        return f"{sum(len(d) for d in dfs.values()):,}"

    dark = st.session_state.get("dark_mode", True)
    bg = "#0e1117" if dark else "#f0f2f6"
    card_bg = "#1a1a2e" if dark else "#ffffff"
    text = "#e2e8f0" if dark else "#1e293b"
    sub = "#94a3b8" if dark else "#64748b"

    kpis = [
        ("Total Accounts",  _try_count(["account_id","account_name","account"]), None),
        ("Pipeline Value",  _try_sum(["pipeline_value","deal_value","value","amount"]), "vs £125k/opp"),
        ("Avg Deal Size",   _try_sum(["deal_size","deal_value","contract_value"]), "vs £210k FM"),
        ("Total Revenue",   _try_sum(["revenue","total_revenue"]), None),
    ]

    cards = ""
    for label, value, note in kpis:
        note_html = (
            f'<p style="color:{sub};font-size:.68rem;margin:2px 0 0;">{note}</p>'
            if note else ""
        )
        cards += (
            f'<div style="background:{card_bg};border-radius:10px;padding:14px 18px;'
            f'flex:1;min-width:130px;box-shadow:0 2px 6px rgba(0,0,0,.2);margin:4px;">'
            f'<p style="color:{sub};font-size:.72rem;margin:0 0 4px;">{label}</p>'
            f'<p style="color:{text};font-size:1.5rem;font-weight:700;margin:0;">{value}</p>'
            f'{note_html}</div>'
        )

    return (
        f'<html><body style="background:{bg};font-family:sans-serif;margin:0;padding:6px;">'
        f'<div style="display:flex;flex-wrap:wrap;gap:8px;">{cards}</div>'
        f'</body></html>'
    )


def _export_dashboard_html(dfs: dict[str, pd.DataFrame]) -> Path:
    charts_html = ""
    if _PLOTLY_OK:
        for name, df in list(dfs.items())[:4]:
            try:
                num_cols = df.select_dtypes(include="number").columns.tolist()
                if num_cols:
                    fig = go.Figure(go.Bar(
                        x=df.index.astype(str)[:20],
                        y=df[num_cols[0]].head(20),
                    ))
                    fig.update_layout(title=f"{name} — {num_cols[0]}", height=300)
                    charts_html += f"<h3>{name}</h3>" + _pio.to_html(
                        fig, full_html=False, include_plotlyjs="cdn"
                    )
            except Exception:
                pass

    kpi_section = _build_kpi_html(dfs)
    ts = datetime.now().strftime("%d %B %Y %H:%M")
    html = (
        f'<!DOCTYPE html><html><head><meta charset="utf-8">'
        f'<title>Equans CRM Dashboard — {ts}</title>'
        f'<style>body{{font-family:sans-serif;padding:20px;background:#f8fafc;}}'
        f'h1{{color:#1e293b;}}h3{{color:#475569;}}</style></head><body>'
        f'<h1>Equans CRM Analytics Dashboard</h1>'
        f'<p>Exported: {ts} &nbsp;|&nbsp; <em>Print this page (Ctrl+P) to save as PDF</em></p>'
        f'<hr><h2>KPI Summary</h2>{kpi_section}'
        f'<h2>Charts</h2>'
        f'{charts_html if charts_html else "<p>No numeric data available.</p>"}'
        f'</body></html>'
    )
    out_dir = Path("exports/sessions")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# render_answer — central reusable render component (used across all tabs)
# ═══════════════════════════════════════════════════════════════════════════════

def render_answer(result: QueryResult, *, expanded_rec: bool = True) -> None:
    """
    Render a QueryResult consistently across Chat, Agenda, and any other tab.

    Section order:
      1  Headline answer + confidence badge + intent pill
      2  Chart (if generated)
      3  Recommendation panel
      4  Data table (collapsible)
      5  Pivot export buttons (pivot intent only)
      6  What-if comparison table + bar chart (what_if intent only)
      7  Benchmark callout (if benchmark_used)
      8  Technical details (collapsible)
    """
    answer_text = result.answer_text or str(result.result or "*(no answer)*")
    is_failure = any(m in str(result.result or "") for m in _GRACEFUL_MARKERS)
    is_cannot = "CANNOT_ANSWER" in answer_text

    # ── S1: Headline + badges ───────────────────────────────────────────────
    if result.question in st.session_state.get("stale_questions", set()):
        st.warning("⚠️ Data updated since this answer was generated — results may have changed.")

    if result.confidence_score < 60 and not is_failure and not is_cannot:
        st.warning("⚠️ Low confidence result — consider verifying this manually")

    label = result.confidence_label
    badge_cls = (
        "conf-high" if "🟢" in label
        else "conf-review" if "🟡" in label
        else "conf-low"
    )
    intent_display = (
        INTENT_TYPES.get(result.intent_type, result.intent_type)
        .replace("_", " ").title()
    )

    col_ans, col_badge, col_intent = st.columns([6, 2, 2])
    with col_ans:
        st.markdown(f"### {answer_text}")
    with col_badge:
        st.markdown(
            f'<span class="{badge_cls}">{label}</span>', unsafe_allow_html=True
        )
    with col_intent:
        st.markdown(
            f'<span class="intent-pill">{intent_display}</span>',
            unsafe_allow_html=True,
        )

    if is_failure or is_cannot:
        st.error("❌ Unable to compute this answer.")
        st.info(
            "💡 **Suggestions:** Try rephrasing · Check the **Data Explorer** "
            "tab to see available columns · Ensure the relevant file is uploaded"
        )
        return

    # ── S2: Chart ───────────────────────────────────────────────────────────
    if result.chart is not None:
        render_chart_ui(result.chart)
        charts_dir = Path("exports/charts")
        if charts_dir.exists():
            pngs = sorted(charts_dir.glob("chart_*.png"), reverse=True)
            if pngs:
                with open(pngs[0], "rb") as f:
                    st.download_button(
                        "📥 Download chart as PNG",
                        data=f.read(),
                        file_name=pngs[0].name,
                        mime="image/png",
                        key=f"dl_chart_{result.timestamp.strftime('%H%M%S%f')}",
                    )

    # ── S3: Recommendation panel ────────────────────────────────────────────
    rec = result.recommendation or {}
    if any(rec.values()):
        with st.expander("💡 So What? — Strategic Recommendations",
                         expanded=expanded_rec):
            st.markdown(
                '<div style="background:#CCFBF1;padding:1rem 1.2rem;'
                'border-radius:8px;border-left:4px solid #0f766e;">'
                f'<p style="margin:0 0 8px;"><strong>\U0001f3af Priority Action:</strong> '
                f'{rec.get("priority_action","—")}</p>'
                f'<p style="margin:0 0 8px;"><strong>\u26a0\ufe0f&nbsp; Risk Flag:</strong> '
                f'{rec.get("risk_flag","—")}</p>'
                f'<p style="margin:0;"><strong>\U0001f4a1 Opportunity:</strong> '
                f'{rec.get("opportunity","—")}</p>'
                '</div>',
                unsafe_allow_html=True,
            )

    # ── S4: Data table ──────────────────────────────────────────────────────
    df_show = (
        result.pivot_df
        if result.intent_type == "pivot" and result.pivot_df is not None
        else result.result_df
    )
    if df_show is not None and not df_show.empty:
        with st.expander("Show data table", expanded=False):
            st.caption(f"Showing {len(df_show):,} rows")
            try:
                num_cols = df_show.select_dtypes(include="number").columns.tolist()
                if num_cols:
                    styled = df_show.style.background_gradient(
                        subset=num_cols, cmap="Blues", low=0.1
                    )
                    st.dataframe(styled, use_container_width=True)
                else:
                    st.dataframe(df_show, use_container_width=True)
            except Exception:
                st.dataframe(df_show, use_container_width=True)

    # ── S5: Pivot export ────────────────────────────────────────────────────
    if result.intent_type == "pivot" and result.pivot_df is not None:
        _k = result.timestamp.strftime("%H%M%S%f")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📥 Export as Excel Pivot", key=f"pivot_xlsx_{_k}"):
                try:
                    handler = PivotHandler()
                    metadata = {
                        "question":         result.question,
                        "code":             result.code,
                        "timestamp":        result.timestamp.isoformat(),
                        "confidence_score": result.confidence_score,
                        "provider_used":    result.provider_used,
                        "session_id":       result.session_id,
                    }
                    out = handler.export_to_excel(
                        result.pivot_df, result.result_df, Path("exports"),
                        metadata=metadata,
                    )
                    st.success(f"Saved: {out.name}")
                    with open(out, "rb") as fh:
                        st.download_button(
                            "⬇️ Download .xlsx",
                            data=fh.read(),
                            file_name=out.name,
                            mime=(
                                "application/vnd.openxmlformats-"
                                "officedocument.spreadsheetml.sheet"
                            ),
                            key=f"dl_pivot_{_k}",
                        )
                except Exception as exc:
                    st.error(f"Export failed: {exc}")
        with c2:
            st.download_button(
                "📥 Export as CSV",
                data=result.pivot_df.to_csv(index=False).encode(),
                file_name=f"pivot_{_k}.csv",
                mime="text/csv",
                key=f"pivot_csv_{_k}",
            )

    # ── S6: What-if comparison ──────────────────────────────────────────────
    if result.intent_type == "what_if" and result.result_df is not None:
        wdf = result.result_df
        st.subheader("📊 Scenario Comparison")

        def _colour_delta(val: Any) -> str:
            try:
                v = float(val)
                return "color:green;font-weight:bold" if v > 0 else (
                    "color:red;font-weight:bold" if v < 0 else ""
                )
            except (TypeError, ValueError):
                return ""

        try:
            delta_cols = [c for c in wdf.columns if "delta" in c.lower() or "change" in c.lower()]
            if delta_cols:
                st.dataframe(
                    wdf.style.map(_colour_delta, subset=delta_cols),
                    use_container_width=True,
                )
            else:
                st.dataframe(wdf, use_container_width=True)
        except Exception:
            st.dataframe(wdf, use_container_width=True)

        if _PLOTLY_OK:
            try:
                num_cols = wdf.select_dtypes(include="number").columns.tolist()
                if len(num_cols) >= 2:
                    label_col = wdf.columns[0]
                    fig_wif = go.Figure()
                    for nc in num_cols[:2]:
                        fig_wif.add_trace(
                            go.Bar(name=nc, x=wdf[label_col].astype(str), y=wdf[nc])
                        )
                    fig_wif.update_layout(barmode="group", title="Baseline vs Scenario")
                    st.plotly_chart(fig_wif, use_container_width=True)
            except Exception:
                pass

    # ── S7: Benchmark callout ───────────────────────────────────────────────
    if result.benchmark_used:
        st.info("📊 Benchmarked against FM sector data (BIFM 2024)")

    # ── S8: Quality badge + iteration history ────────────────────────────────
    if result.final_score > 0:
        n_iter = result.iterations
        model_tag = result.provider_used or "—"
        badge_label = (
            f"✨ Quality score: {result.final_score}/100 · "
            f"{n_iter} iteration{'s' if n_iter != 1 else ''} · {model_tag}"
        )
        with st.expander(badge_label, expanded=False):
            for rec in result.iteration_log:
                fb = (rec.get("critic_feedback") or "")[:120]
                st.caption(
                    f"Iteration {rec['iteration']}: score {rec['critic_score']} — \"{fb}\""
                )
            if not result.iteration_log:
                st.caption("No iteration details available.")

    # ── S9: Technical details ───────────────────────────────────────────────
    with st.expander("Show technical details", expanded=False):
        st.code(result.code or "# No code generated", language="python")
        st.caption(
            f"Iterations: {result.iterations} · "
            f"Score: {result.final_score} · "
            f"Provider: {result.provider_used or '—'} · "
            f"{result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ── Session exporter factory ──────────────────────────────────────────────────


def _get_exporter(cfg: dict) -> SessionExporter:
    """Return a SessionExporter wired to an LLMClient when data is loaded."""
    llm = LLMClient(cfg) if st.session_state.dataframes else None
    return SessionExporter(exports_dir=Path("exports"), llm_client=llm)


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

def _render_sidebar(cfg: dict) -> None:
    with st.sidebar:
        col_title, col_dm = st.columns([3, 1])
        with col_title:
            st.title("⚙️ Controls")
        with col_dm:
            dark = st.toggle("🌙", value=st.session_state.dark_mode, key="dm_toggle")
            if dark != st.session_state.dark_mode:
                st.session_state.dark_mode = dark
                st.rerun()

        # ── Start folder watcher (once) ───────────────────────────────────
        if not st.session_state.watcher_started and not st.session_state.get("_watcher_skip"):
            try:
                handler = _get_update_handler(cfg)
                upload_dir = Path(cfg.get("data", {}).get("upload_folder", "data/uploads"))
                handler.start_watcher(upload_dir)
                st.session_state.watcher_started = True
            except ImportError:
                st.session_state["_watcher_skip"] = True

        # ── Sentinel badge ────────────────────────────────────────────────
        _handler = _get_update_handler(cfg)
        new_files = _handler.read_sentinel()
        if new_files:
            st.sidebar.info(f"🔔 {len(new_files)} new file(s) detected in uploads/")
            if st.sidebar.button("📥 Load now", key="load_sentinel"):
                _handler.clear_sentinel()
                _load_files_from_paths([Path(p) for p in new_files], cfg)
                st.rerun()

        updates = st.session_state.data_updates
        if updates > 0:
            st.markdown(
                f'<span class="upd-badge">🔴 {updates} update(s)</span>',
                unsafe_allow_html=True,
            )
            st.caption("")

        # ── File uploader ────────────────────────────────────────────────
        st.subheader("📁 Data Upload")
        uploaded = st.file_uploader(
            "Upload CRM files",
            accept_multiple_files=True,
            type=["xlsx", "xls", "csv", "pptx", "pdf", "png", "jpg"],
            help="Supports Excel, CSV, PowerPoint, PDF, and images.",
        )
        if uploaded:
            _load_files(uploaded, cfg)
            st.session_state.data_updates = 0

        # ── Pending updates panel ─────────────────────────────────────────
        if st.session_state.pending_updates:
            st.sidebar.divider()
            n_pending = len(st.session_state.pending_updates)
            st.sidebar.markdown(f"### 🔔 {n_pending} Update(s) Pending")
            _upd_handler = _get_update_handler(cfg)
            for upd_name, summary in list(st.session_state.pending_updates.items()):
                with st.sidebar.expander(
                    f"📄 {upd_name} — {summary.classification.value}", expanded=False
                ):
                    if summary.diff:
                        st.caption(summary.diff.summary_text())
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("✅ Replace", key=f"upd_replace_{upd_name}"):
                            del st.session_state.pending_updates[upd_name]
                            st.rerun()
                    with c2:
                        if st.button("🔀 Merge…", key=f"upd_merge_{upd_name}"):
                            st.session_state[f"_merge_open_{upd_name}"] = True
                    if st.session_state.get(f"_merge_open_{upd_name}"):
                        key_col = st.selectbox(
                            "Key column for dedup",
                            list(st.session_state.dataframes[upd_name].columns),
                            key=f"merge_key_{upd_name}",
                        )
                        if st.button("Apply merge", key=f"merge_apply_{upd_name}"):
                            versions = _upd_handler.list_versions(upd_name)
                            if versions:
                                prev_df = _upd_handler.load_version(
                                    upd_name, versions[0]["ts_str"]
                                )
                                merged = _upd_handler.merge_dataframes(
                                    prev_df,
                                    st.session_state.dataframes[upd_name],
                                    key_col,
                                )
                                st.session_state.dataframes[upd_name] = merged
                                st.session_state.engine = None
                            st.session_state.pop(f"_merge_open_{upd_name}", None)
                            del st.session_state.pending_updates[upd_name]
                            st.rerun()

        # ── Loaded tables ────────────────────────────────────────────────
        dfs = st.session_state.dataframes
        if dfs:
            st.subheader("📋 Loaded Tables")
            confirmed_tables = {c.left_table for c in st.session_state.confirmed_joins} | {
                c.right_table for c in st.session_state.confirmed_joins
            }
            for name, df in dfs.items():
                joined = name in confirmed_tables
                versions = st.session_state.table_versions.get(name, [])
                last_ts = versions[-1][0].strftime("%H:%M") if versions else "—"
                traffic = _dq_traffic_light(st.session_state.profiles.get(name))
                st.markdown(
                    f"{traffic} **{name}** · {len(df):,} rows · "
                    f"{'🔗 joined' if joined else '⬜'} · {last_ts}"
                )

        # ── Join suggestions ─────────────────────────────────────────────
        if st.session_state.join_candidates:
            st.subheader("🔗 Join Suggestions")
            try:
                approved = render_join_ui(
                    st.session_state.join_candidates, key_prefix="sidebar_join"
                )
                if approved != st.session_state.confirmed_joins:
                    st.session_state.confirmed_joins = approved
                    st.session_state.engine = None
            except Exception as exc:
                st.caption(f"Join UI unavailable: {exc}")

        # ── Data quality summary ─────────────────────────────────────────
        with st.expander("📊 Data Quality Summary", expanded=False):
            profiles = st.session_state.profiles
            if profiles:
                try:
                    render_profile_ui(profiles, key_prefix="sidebar_dq")
                except Exception:
                    rows = [
                        {
                            "Table": name,
                            "Status": _dq_traffic_light(p),
                            "Null %": f"{p.overall_null_pct:.1f}%",
                            "Issues": p.issue_count(),
                        }
                        for name, p in profiles.items()
                    ]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.caption("No data loaded yet.")

        # ── LLM status + toggle ──────────────────────────────────────────
        st.subheader("🤖 LLM Status")
        provider = st.radio(
            "Active provider",
            ["ollama", "groq"],
            index=0 if st.session_state.llm_provider == "ollama" else 1,
            key="provider_radio",
            horizontal=True,
        )
        if provider != st.session_state.llm_provider:
            st.session_state.llm_provider = provider
            st.session_state.engine = None
            st.session_state.chart_gen = None
            cfg["llm_provider"] = provider
        st.caption("🟢 Ollama (local)" if provider == "ollama" else "☁️ Groq (cloud)")

        # ── Settings ─────────────────────────────────────────────────────
        with st.expander("⚙️ Settings", expanded=False):
            default_model = cfg.get("ollama", {}).get("model", "llama3.1:8b")
            st.session_state.s_model = st.text_input(
                "Model", value=st.session_state.s_model or default_model
            )
            st.session_state.s_temperature = st.slider(
                "Temperature", 0.0, 1.0,
                value=float(st.session_state.s_temperature), step=0.05,
            )
            st.session_state.s_dormant_days = st.number_input(
                "Dormant days threshold",
                value=int(st.session_state.s_dormant_days),
                step=1, min_value=1, max_value=365,
            )
            st.session_state.s_rec_mode = st.toggle(
                "Recommendation mode", value=st.session_state.s_rec_mode
            )
            st.session_state.s_conf_threshold = st.slider(
                "Confidence threshold", 0, 100,
                value=int(st.session_state.s_conf_threshold), step=5,
            )

        # ── Export session ───────────────────────────────────────────────
        st.divider()
        if st.button("📄 Export Session", use_container_width=True):
            try:
                exp = _get_exporter(cfg)
                engine = st.session_state.get("engine")
                path = exp.export_session_docx(
                    chat_history=st.session_state.chat_history,
                    agenda_results=st.session_state.agenda_results,
                    dataframes=st.session_state.dataframes,
                    profiles=st.session_state.profiles,
                    session_id=engine.session_id if engine else "—",
                    session_start=st.session_state.last_loaded or datetime.now(timezone.utc),
                    model_used=cfg.get("ollama", {}).get("model", "unknown"),
                )
                st.success(f"Saved: {path.name}")
                with open(path, "rb") as f:
                    st.download_button(
                        "⬇️ Download Word doc",
                        data=f.read(),
                        file_name=path.name,
                        mime=(
                            "application/vnd.openxmlformats-"
                            "officedocument.wordprocessingml.document"
                        ),
                        key="dl_session_doc",
                    )
            except Exception as exc:
                st.error(f"Export failed: {exc}")

        # ── End Session ──────────────────────────────────────────────────
        if st.button("🔴 End Session", type="secondary", use_container_width=True):
            st.session_state["session_ending"] = True

        if st.session_state.get("session_ending"):
            st.warning("End Session — auto-saving chat and actions...")
            try:
                exp = _get_exporter(cfg)
                engine = st.session_state.get("engine")
                sid = engine.session_id if engine else "session"
                txt_path = exp.export_chat_txt(
                    st.session_state.chat_history, session_id=sid
                )
                act_path = exp.export_actions_docx(st.session_state.agenda_results)
                st.info(f"Auto-saved: {txt_path.name}, {act_path.name}")
            except Exception as exc:
                st.error(f"Auto-save failed: {exc}")

            st.write("Export full session report?")
            c_yes, c_no = st.columns(2)
            with c_yes:
                if st.button("Yes, export", key="end_yes"):
                    try:
                        exp = _get_exporter(cfg)
                        engine = st.session_state.get("engine")
                        path = exp.export_session_docx(
                            chat_history=st.session_state.chat_history,
                            agenda_results=st.session_state.agenda_results,
                            dataframes=st.session_state.dataframes,
                            profiles=st.session_state.profiles,
                            session_id=engine.session_id if engine else "—",
                            session_start=st.session_state.last_loaded or datetime.now(timezone.utc),
                            model_used=cfg.get("ollama", {}).get("model", "unknown"),
                        )
                        st.success(f"Saved: {path.name}")
                        with open(path, "rb") as f:
                            st.download_button(
                                "⬇️ Download",
                                data=f.read(),
                                file_name=path.name,
                                mime=(
                                    "application/vnd.openxmlformats-"
                                    "officedocument.wordprocessingml.document"
                                ),
                                key="dl_end_session",
                            )
                        st.session_state.pop("session_ending", None)
                    except Exception as exc:
                        st.error(f"Export failed: {exc}")
            with c_no:
                if st.button("No, skip", key="end_no"):
                    st.session_state.pop("session_ending", None)
                    st.rerun()

        # ── Footer ───────────────────────────────────────────────────────
        last = st.session_state.last_loaded
        st.caption(
            f"v{APP_VERSION} · Last refresh: "
            f"{last.strftime('%H:%M:%S') if last else 'never'}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Chat
# ═══════════════════════════════════════════════════════════════════════════════

def _render_chat_tab(cfg: dict) -> None:
    col_h, col_export, col_clear = st.columns([4, 2, 1])
    with col_h:
        st.header("💬 Chat")
    with col_export:
        if st.button("💾 Export chat", key="chat_export_btn"):
            try:
                exp = _get_exporter(cfg)
                engine = st.session_state.get("engine")
                sid = engine.session_id if engine else "session"
                path = exp.export_chat_txt(st.session_state.chat_history, session_id=sid)
                with open(path, "rb") as f:
                    st.download_button(
                        "⬇️ Download .txt",
                        data=f.read(),
                        file_name=path.name,
                        mime="text/plain",
                        key="dl_chat_txt",
                    )
            except Exception as exc:
                st.error(f"Export failed: {exc}")
    with col_clear:
        if st.button("🗑 Clear", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    if not st.session_state.dataframes:
        st.info("👈 Upload CRM files in the sidebar to start asking questions.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.write(msg["content"])
            else:
                r = msg.get("result")
                if r is not None:
                    render_answer(r, expanded_rec=False)
                else:
                    st.write(msg.get("content", ""))

    question = st.chat_input("Ask anything about the CRM data...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            result = _run_query(question, cfg)
            render_answer(result, expanded_rec=False)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": result.answer_text, "result": result}
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Weekly Agenda
# ═══════════════════════════════════════════════════════════════════════════════

def _render_agenda_tab(cfg: dict) -> None:
    st.header("📋 Weekly Agenda")

    col_run, col_info = st.columns([2, 5])
    with col_run:
        run_all = st.button("▶️ Run Full Meeting", type="primary", key="run_full_meeting")
    with col_info:
        total_done = sum(len(v) for v in st.session_state.agenda_results.values())
        st.caption(f"{total_done}/{len(AGENDA_QUESTIONS)} questions run this session")

    if run_all:
        all_qs = list(AGENDA_QUESTIONS)
        prog = st.progress(0, text="Starting full meeting run...")
        start_t = datetime.now()
        for i, q in enumerate(all_qs):
            elapsed = max((datetime.now() - start_t).seconds, 1)
            avg = elapsed / (i + 1)
            remaining = int(avg * (len(all_qs) - i - 1))
            prog.progress(
                (i + 1) / len(all_qs),
                text=f"[{i+1}/{len(all_qs)}] {q.question[:55]}... · ~{remaining}s left",
            )
            r = _run_query(q.question, cfg)
            st.session_state.agenda_results.setdefault(q.section, {})[q.question] = r
        prog.empty()
        st.toast("✅ Full meeting analysis complete!")

    st.divider()

    for section_num in range(1, 7):
        title = SECTION_TITLES[section_num]
        color = _SECTION_COLORS[section_num]
        questions = get_section(section_num)
        sec_results = st.session_state.agenda_results.get(section_num, {})
        run_count = sum(1 for q in questions if q.question in sec_results)

        if run_count == 0:
            status = "⬜ Not started"
        elif run_count < len(questions):
            status = f"🔄 {run_count}/{len(questions)} run"
        else:
            status = "✅ All run"

        with st.expander(f"Section {section_num} — {title} · {status}", expanded=False):
            st.markdown(
                f'<div style="border-left:4px solid {color};padding-left:12px;'
                f'margin-bottom:10px;">'
                f'<strong style="color:{color}">{title}</strong></div>',
                unsafe_allow_html=True,
            )

            col_rs, col_exp, _ = st.columns([2, 2, 3])
            with col_rs:
                run_sec = st.button("▶ Run Full Section", key=f"run_sec_{section_num}")
            with col_exp:
                export_sec = st.button(
                    "📄 Export Section",
                    key=f"exp_sec_{section_num}",
                    disabled=(run_count == 0),
                )

            if run_sec:
                prog_s = st.progress(0, text="Running section...")
                for i, q in enumerate(questions):
                    prog_s.progress((i + 1) / len(questions), text=q.question[:60])
                    r = _run_query(q.question, cfg)
                    st.session_state.agenda_results.setdefault(section_num, {})[q.question] = r
                prog_s.empty()
                st.toast(f"✅ Section {section_num} complete!")
                sec_results = st.session_state.agenda_results.get(section_num, {})

            if export_sec and sec_results:
                try:
                    exp = _get_exporter(cfg)
                    path = exp.export_section_docx(section_num, sec_results)
                    st.success(f"Saved: {path.name}")
                    with open(path, "rb") as f:
                        st.download_button(
                            "⬇️ Download",
                            data=f.read(),
                            file_name=path.name,
                            mime=(
                                "application/vnd.openxmlformats-"
                                "officedocument.wordprocessingml.document"
                            ),
                            key=f"dl_sec_{section_num}",
                        )
                except Exception as exc:
                    st.error(f"Export failed: {exc}")

            st.markdown("---")

            for q in questions:
                col_btn, col_stat = st.columns([8, 1])
                with col_btn:
                    clicked = st.button(
                        q.question,
                        key=f"aq_{section_num}_{abs(hash(q.question)) % 999983}",
                        use_container_width=True,
                    )
                with col_stat:
                    st.caption("✅" if q.question in sec_results else "⬜")

                if clicked:
                    with st.spinner("Analysing data..."):
                        r = _run_query(q.question, cfg)
                    st.session_state.agenda_results.setdefault(section_num, {})[q.question] = r
                    sec_results = st.session_state.agenda_results.get(section_num, {})

                if q.question in sec_results:
                    render_answer(sec_results[q.question], expanded_rec=True)
                    st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Data Explorer
# ═══════════════════════════════════════════════════════════════════════════════

def _render_explorer_tab() -> None:
    st.header("🔍 Data Explorer")
    dfs = st.session_state.dataframes
    if not dfs:
        st.info("👈 Upload CRM data to start exploring.")
        return

    selected = st.selectbox("Select table", list(dfs.keys()), key="explorer_sel")
    df = dfs[selected]

    # ── Version history (parquet snapshots via UpdateHandler) ─────────────
    _exp_handler = _get_update_handler(st.session_state.settings)
    parquet_versions = _exp_handler.list_versions(selected)
    if parquet_versions:
        with st.expander(
            f"⏱ Version History ({len(parquet_versions)} snapshot(s))", expanded=False
        ):
            for v in parquet_versions:
                c_ts, c_rows, c_btn = st.columns([3, 1, 2])
                c_ts.caption(v["ts_str"])
                c_rows.caption(f"{v['row_count']:,} rows")
                confirm_key = f"_rb_confirm_{selected}_{v['ts_str']}"
                if c_btn.button("Rollback", key=f"rb_{selected}_{v['ts_str']}"):
                    st.session_state[confirm_key] = True
                if st.session_state.get(confirm_key):
                    st.warning("Roll back to this version?")
                    ca, cb = st.columns(2)
                    with ca:
                        if st.button("✅ Yes", key=f"rb_yes_{selected}_{v['ts_str']}"):
                            try:
                                st.session_state.dataframes[selected] = \
                                    _exp_handler.load_version(selected, v["ts_str"])
                                st.session_state.engine = None
                                st.session_state.pop(confirm_key, None)
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Rollback failed: {exc}")
                    with cb:
                        if st.button("❌ No", key=f"rb_no_{selected}_{v['ts_str']}"):
                            st.session_state.pop(confirm_key, None)

    # ── Search + table view ───────────────────────────────────────────────
    col_search, col_max = st.columns([4, 2])
    with col_search:
        search = st.text_input("Search all columns", placeholder="Filter rows...",
                               key="explorer_search")
    with col_max:
        max_rows = st.number_input("Max rows", value=100, step=50,
                                   min_value=10, max_value=5000, key="explorer_max")

    filtered = df
    if search:
        try:
            mask = (
                df.astype(str)
                .apply(lambda c: c.str.contains(search, case=False, na=False))
                .any(axis=1)
            )
            filtered = df[mask]
        except Exception:
            pass

    st.caption(
        f"Showing {min(int(max_rows), len(filtered)):,} of {len(filtered):,} rows "
        f"(filtered from {len(df):,} total)"
    )
    st.dataframe(filtered.head(int(max_rows)), use_container_width=True)

    # ── Column statistics ─────────────────────────────────────────────────
    with st.expander("📊 Column Statistics", expanded=False):
        col_sel = st.selectbox("Select column", df.columns.tolist(),
                               key="explorer_col_sel")
        series = df[col_sel]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Null %", f"{series.isnull().mean() * 100:.1f}%")
        c2.metric("Unique", f"{series.nunique():,}")
        if pd.api.types.is_numeric_dtype(series):
            c3.metric("Min", f"{series.min():,.2f}")
            c4.metric("Max", f"{series.max():,.2f}")
            st.caption(
                f"Mean: {series.mean():,.2f} · "
                f"Median: {series.median():,.2f} · "
                f"Std: {series.std():,.2f}"
            )
        else:
            top = series.value_counts().head(5)
            if not top.empty:
                st.caption(
                    "Top values: " + ", ".join(f"{v} ({c})" for v, c in top.items())
                )

    # ── Schema view ───────────────────────────────────────────────────────
    with st.expander("📐 Schema", expanded=False):
        schema_df = pd.DataFrame({
            "Column":   df.columns,
            "Dtype":    df.dtypes.astype(str).values,
            "Non-null": df.notnull().sum().values,
            "Null %":   (df.isnull().mean() * 100).round(1).values,
            "Unique":   df.nunique().values,
        })
        st.dataframe(schema_df, use_container_width=True)

    # ── Manual join builder ───────────────────────────────────────────────
    with st.expander("🔧 Manual Join Builder", expanded=False):
        table_names = list(dfs.keys())
        if len(table_names) < 2:
            st.caption("Load at least 2 tables to use the join builder.")
        else:
            jc1, jc2, jc3, jc4, jc5 = st.columns([2, 2, 2, 2, 1])
            with jc1:
                l_tbl = st.selectbox("Left table", table_names, key="mj_l_tbl")
            with jc2:
                l_col = st.selectbox("Left column",
                                     dfs[l_tbl].columns.tolist(), key="mj_l_col")
            with jc3:
                r_tbl = st.selectbox(
                    "Right table",
                    [t for t in table_names if t != l_tbl],
                    key="mj_r_tbl",
                )
            with jc4:
                r_col = st.selectbox("Right column",
                                     dfs[r_tbl].columns.tolist(), key="mj_r_col")
            with jc5:
                j_type = st.selectbox("Type", ["left", "inner", "outer"], key="mj_type")

            if st.button("➕ Add Join", key="mj_add"):
                new_c = JoinCandidate(
                    left_table=l_tbl, right_table=r_tbl,
                    left_col=l_col, right_col=r_col,
                    name_similarity=1.0, value_overlap=1.0,
                    confidence=1.0, join_type=j_type,
                )
                if new_c not in st.session_state.confirmed_joins:
                    st.session_state.confirmed_joins.append(new_c)
                    st.session_state.engine = None
                    st.success(f"Join added: {new_c.label}")
                else:
                    st.info("This join is already active.")


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Insights Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def _render_dashboard_tab() -> None:
    st.header("📊 Insights Dashboard")
    dfs = st.session_state.dataframes
    if not dfs:
        st.info("👈 Upload CRM data to generate the dashboard.")
        return

    col_ref, col_exp, col_pdf, col_auto = st.columns([2, 2, 2, 3])
    with col_ref:
        if st.button("🔄 Refresh Now", key="dash_refresh"):
            st.session_state.last_dashboard_refresh = datetime.now(timezone.utc)
            st.rerun()
    with col_exp:
        if st.button("📄 Export as HTML", key="dash_export"):
            try:
                path = _export_dashboard_html(dfs)
                st.success(f"Saved: {path.name}")
                st.caption("Open in browser → File → Print → Save as PDF")
                with open(path, "rb") as f:
                    st.download_button(
                        "⬇️ Download HTML",
                        data=f.read(),
                        file_name=path.name,
                        mime="text/html",
                        key="dl_dash_html",
                    )
            except Exception as exc:
                st.error(f"Export failed: {exc}")
    with col_pdf:
        if st.button("📑 Export as PDF", key="dash_pdf"):
            try:
                exp = SessionExporter(exports_dir=Path("exports"))
                path = exp.export_dashboard_pdf(dfs)
                st.success(f"Saved: {path.name}")
                with open(path, "rb") as f:
                    st.download_button(
                        "⬇️ Download PDF",
                        data=f.read(),
                        file_name=path.name,
                        mime="application/pdf",
                        key="dl_dash_pdf",
                    )
            except Exception as exc:
                st.error(f"PDF export failed: {exc}")
    with col_auto:
        auto = st.toggle(
            "Auto-refresh every 60s",
            value=st.session_state.auto_refresh_dashboard,
            key="auto_refresh_toggle",
        )
        st.session_state.auto_refresh_dashboard = auto
        if auto:
            last = st.session_state.last_dashboard_refresh
            if last is None:
                st.session_state.last_dashboard_refresh = datetime.now(timezone.utc)
            elif (datetime.now(timezone.utc) - last).seconds >= 60:
                st.session_state.last_dashboard_refresh = datetime.now(timezone.utc)
                st.rerun()

    # ── KPI cards ─────────────────────────────────────────────────────────
    st.subheader("Key Performance Indicators")
    st.components.v1.html(_build_kpi_html(dfs), height=220)

    if not _PLOTLY_OK:
        st.warning("Install plotly for charts: pip install plotly")
        return

    # ── Helpers ───────────────────────────────────────────────────────────
    def _find(df: pd.DataFrame, keywords: list[str]) -> str | None:
        for kw in keywords:
            for col in df.columns:
                if kw in col.lower():
                    return col
        return None

    def _pick(table_kws: list[str]) -> tuple[str, pd.DataFrame]:
        for kw in table_kws:
            for name, df in dfs.items():
                if kw in name.lower():
                    return name, df
        return next(iter(dfs.items()))

    # ── Row 1: Revenue Pareto | Pipeline Funnel ───────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Revenue Concentration (Pareto)")
        try:
            _, df_a = _pick(["account", "client", "customer"])
            rev = _find(df_a, ["revenue", "amount", "value", "deal_value"])
            nam = _find(df_a, ["account_name", "name", "client_name", "company"])
            if rev and nam:
                top = (
                    df_a.groupby(nam)[rev].sum()
                    .sort_values(ascending=False).head(20).reset_index()
                )
                total = top[rev].sum()
                top["_cum"] = top[rev].cumsum() / total * 100 if total > 0 else 0
                fig = go.Figure()
                fig.add_trace(go.Bar(x=top[nam], y=top[rev], name="Revenue"))
                fig.add_trace(go.Scatter(
                    x=top[nam], y=top["_cum"], name="Cum %",
                    yaxis="y2", mode="lines+markers",
                    line=dict(color="orange"),
                ))
                fig.update_layout(
                    yaxis2=dict(overlaying="y", side="right", range=[0, 110]),
                    height=380,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Columns 'account_name' and 'revenue' not found.")
        except Exception as exc:
            st.caption(f"Chart unavailable: {exc}")

    with col_b:
        st.subheader("Pipeline Stage Funnel")
        try:
            _, df_b = _pick(["opportunit", "pipeline", "deal"])
            stage = _find(df_b, ["stage", "status", "phase"])
            val = _find(df_b, ["value", "amount", "deal_value", "revenue"])
            if stage:
                if val:
                    fd = df_b.groupby(stage)[val].sum().reset_index()
                    fd = fd.sort_values(val, ascending=False)
                    y_vals, x_vals = fd[stage].astype(str).tolist(), fd[val].tolist()
                else:
                    vc = df_b[stage].value_counts()
                    y_vals, x_vals = vc.index.astype(str).tolist(), vc.values.tolist()
                fig = go.Figure(go.Funnel(
                    y=y_vals, x=x_vals, textinfo="value+percent initial"
                ))
                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Stage/status column not found.")
        except Exception as exc:
            st.caption(f"Chart unavailable: {exc}")

    # ── Row 2: Win rate by service line | Sales cycle trend ───────────────
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Win Rate by Service Line")
        try:
            _, df_c = _pick(["opportunit", "pipeline", "deal"])
            svc = _find(df_c, ["service_line", "service", "division", "category"])
            stg = _find(df_c, ["stage", "status", "outcome"])
            if svc and stg:
                won = df_c[df_c[stg].str.lower().isin(["won", "closed won", "win"])]
                total_s = df_c.groupby(svc).size()
                won_s = won.groupby(svc).size()
                wr = (won_s / total_s * 100).fillna(0).sort_values(
                    ascending=False).reset_index()
                wr.columns = [svc, "win_rate_pct"]
                fig = go.Figure(go.Bar(
                    x=wr[svc].astype(str), y=wr["win_rate_pct"],
                    text=(wr["win_rate_pct"].round(1).astype(str) + "%"),
                    textposition="outside",
                ))
                fig.add_hline(y=28, line_dash="dash", line_color="orange",
                              annotation_text="FM benchmark 28%")
                fig.update_layout(yaxis_title="Win Rate %", height=380,
                                  yaxis_range=[0, min(100, wr["win_rate_pct"].max() * 1.25 + 5)])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Service line or stage column not found.")
        except Exception as exc:
            st.caption(f"Chart unavailable: {exc}")

    with col_d:
        st.subheader("Sales Cycle Trend")
        try:
            _, df_d = _pick(["opportunit", "pipeline", "deal"])
            close_c = _find(df_d, ["close_date", "closed_date", "close", "end_date"])
            open_c = _find(df_d, ["created_date", "created_at", "start_date", "open_date"])
            if close_c and open_c:
                dc = df_d[[close_c, open_c]].dropna().copy()
                dc[close_c] = pd.to_datetime(dc[close_c], errors="coerce")
                dc[open_c] = pd.to_datetime(dc[open_c], errors="coerce")
                dc["days"] = (dc[close_c] - dc[open_c]).dt.days
                dc = dc.dropna(subset=["days"])
                dc["month"] = dc[close_c].dt.to_period("M").astype(str)
                monthly = dc.groupby("month")["days"].mean().reset_index()
                monthly.columns = ["month", "avg_days"]
                monthly["_ma"] = monthly["avg_days"].rolling(3, min_periods=1).mean()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=monthly["month"], y=monthly["avg_days"],
                    mode="lines+markers", name="Avg days", opacity=0.5,
                ))
                fig.add_trace(go.Scatter(
                    x=monthly["month"], y=monthly["_ma"],
                    mode="lines", name="3-month MA", line=dict(width=2),
                ))
                fig.add_hline(y=84, line_dash="dash", line_color="orange",
                              annotation_text="FM benchmark 84d")
                fig.update_layout(yaxis_title="Days", height=380)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Close date and/or open date columns not found.")
        except Exception as exc:
            st.caption(f"Chart unavailable: {exc}")

    # ── Full width: Top 10 accounts by revenue ────────────────────────────
    st.subheader("Top 10 Accounts by Revenue")
    try:
        _, df_t = _pick(["account", "client", "customer"])
        rev = _find(df_t, ["revenue", "amount", "value", "deal_value"])
        nam = _find(df_t, ["account_name", "name", "client_name", "company"])
        if rev and nam:
            top10 = (
                df_t.groupby(nam)[rev].sum()
                .sort_values(ascending=True).tail(10).reset_index()
            )
            fig = go.Figure(go.Bar(
                x=top10[rev], y=top10[nam].astype(str),
                orientation="h",
                text=top10[rev].apply(
                    lambda v: f"£{v:,.0f}" if pd.notna(v) else "—"
                ),
                textposition="outside",
            ))
            fig.update_layout(xaxis_title="Revenue", height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Account name and revenue columns not found.")
    except Exception as exc:
        st.caption(f"Chart unavailable: {exc}")

    # ── Full width: Account health heatmap ────────────────────────────────
    st.subheader("Account Health Heatmap")
    try:
        _, df_h = _pick(["account", "client", "customer"])
        num_cols = df_h.select_dtypes(include="number").columns.tolist()
        if len(num_cols) >= 2:
            heat = df_h[num_cols].head(30).fillna(0)
            nam = _find(df_h, ["account_name", "name", "client_name"])
            y_labels = (
                df_h[nam].head(30).astype(str).tolist()
                if nam else [str(i) for i in range(len(heat))]
            )
            fig = go.Figure(go.Heatmap(
                z=heat.values, x=heat.columns.tolist(),
                y=y_labels, colorscale="Blues",
            ))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Not enough numeric columns for heatmap.")
    except Exception as exc:
        st.caption(f"Chart unavailable: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 5 — History
# ═══════════════════════════════════════════════════════════════════════════════

def _render_history_tab(cfg: dict) -> None:
    st.header("🧠 Query History & Patterns")
    tracker = _get_tracker(cfg)
    csv_mgr = CSVManager(tracker, output_folder=Path("exports"))

    # ── Recent queries ────────────────────────────────────────────────────
    st.subheader("📜 Recent Queries")
    queries = tracker.get_recent_queries(n=50)
    if queries:
        df_q = pd.DataFrame(queries)[["timestamp", "question", "score", "iterations", "error"]]
        st.dataframe(df_q, use_container_width=True)
        if st.button("📥 Export Query Log CSV", key="hist_export_qlog"):
            path = csv_mgr.export_query_log()
            with open(path, "rb") as f:
                st.download_button(
                    "Save query_log.csv",
                    data=f.read(),
                    file_name=path.name,
                    mime="text/csv",
                    key="hist_dl_qlog",
                )
    else:
        st.info("No queries logged yet — run a question in the Chat or Agenda tab.")

    st.divider()

    # ── Pattern memory ────────────────────────────────────────────────────
    st.subheader("🔁 Pattern Memory")
    patterns = tracker.get_patterns()
    if patterns:
        df_p = pd.DataFrame(patterns)[["question_type", "score", "use_count", "code_pattern"]]
        st.dataframe(df_p, use_container_width=True)
        if st.button("📥 Export Patterns CSV", key="hist_export_patterns"):
            path = csv_mgr.export_patterns()
            with open(path, "rb") as f:
                st.download_button(
                    "Save patterns.csv",
                    data=f.read(),
                    file_name=path.name,
                    mime="text/csv",
                    key="hist_dl_patterns",
                )
    else:
        st.info(
            "No patterns stored yet — patterns are saved after successful "
            "self-improvement rewrites."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg = load_config()
    st.set_page_config(
        page_title=cfg.get("ui", {}).get("page_title", "CRM Analytics Agent"),
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_session_state(cfg)
    _apply_theme()
    _render_sidebar(cfg)

    st.title(f"📊 {cfg.get('ui', {}).get('page_title', 'Equans CRM Analytics Agent')}")
    st.caption("100% local · no data leaves this machine")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Chat",
        "📋 Weekly Agenda",
        "🔍 Data Explorer",
        "📊 Insights Dashboard",
        "🧠 History",
    ])
    with tab1:
        _render_chat_tab(cfg)
    with tab2:
        _render_agenda_tab(cfg)
    with tab3:
        _render_explorer_tab()
    with tab4:
        _render_dashboard_tab()
    with tab5:
        _render_history_tab(cfg)


def _cli_export_training_data() -> None:
    """Export final-iteration training records with score >= 75 to a clean JSONL."""
    import sys
    from agent.self_improver import SelfImprover

    cfg = load_config()
    si = SelfImprover(
        llm_client=None,   # not needed for export
        config=cfg,
        exports_dir=Path("exports"),
    )
    out = si.export_training_data()
    print(f"Exported to: {out}")
    sys.exit(0)


if __name__ == "__main__":
    import sys
    if "--export-training-data" in sys.argv:
        _cli_export_training_data()
    else:
        main()
