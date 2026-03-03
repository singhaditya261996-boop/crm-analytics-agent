# Equans CRM Analytics Agent

A fully local, offline AI-powered analytics assistant for CRM data — built with Streamlit, Ollama, and pandas.
**No data leaves the machine. No paid APIs required.**

Ask questions in plain English, get charts, pivot tables, recommendations, and weekly agenda answers — all powered by a local LLM running on your own hardware.

---

## Features

| Feature | Description |
|---------|-------------|
| **Natural language queries** | Ask "Which accounts haven't been contacted in 90 days?" and get a live answer with a chart |
| **Auto join detection** | Fuzzy-matches column names across tables and suggests joins |
| **Data quality profiling** | Null rates, type issues, outlier detection across all loaded tables |
| **Weekly agenda** | 23 pre-built CRM questions across 6 sections (Pipeline, Dormancy, Win/Loss, etc.) |
| **Chart generation** | Bar/Pareto, line + moving average, scatter, pie, heatmap, funnel, horizontal bar |
| **Incremental file updates** | Fingerprints each file upload, classifies as NEW / UPDATED / CORRECTED / DUPLICATE, diffs rows and cell values, saves versioned parquet snapshots for rollback |
| **Folder watcher** | Drop a file into `data/uploads/` and a sidebar badge appears instantly — no manual re-upload needed |
| **Pivot & What-If** | Auto-generates Excel pivot tables; what-if scenario comparison tables |
| **Session export** | Export full session as Word doc, section summaries, chat transcript, or PDF dashboard |
| **Stale data warnings** | If data is updated after an agenda answer is generated, the answer is flagged with a warning badge |
| **100% local** | Ollama as primary LLM, Groq as optional cloud fallback — switch with one env var |

---

## Architecture

```
crm-agent/
├── app.py                      # Streamlit entry point (4 tabs + sidebar)
├── agent/
│   ├── llm_client.py           # Ollama → Groq fallback, async-primary, JSONL logging
│   ├── query_engine.py         # English → pandas → QueryResult pipeline
│   ├── chart_generator.py      # 7 chart types, auto-saved PNGs
│   ├── context_builder.py      # Schema context injection for LLM prompts
│   └── self_improver.py        # Critic + rewriter + pattern memory
├── data/
│   ├── loader.py               # Multi-format loader (CSV, Excel, PDF, PPT, images)
│   ├── joiner.py               # Fuzzy join detection + referential overlap scoring
│   ├── profiler.py             # 7 data quality rules, column profiles, quality report
│   ├── update_handler.py       # Fingerprinting, diff, versioned parquet, folder watcher
│   └── cache.py                # SHA-1 mtime-based parquet cache
├── agenda/
│   └── prompts.py              # 23 pre-built weekly CRM questions across 6 sections
├── formats/
│   ├── ppt_handler.py          # PowerPoint text + table extraction
│   ├── pdf_handler.py          # PDF text + table extraction
│   └── image_handler.py        # Screenshot OCR via llava
├── tracker/
│   ├── database.py             # SQLite via SQLAlchemy
│   └── csv_manager.py          # CSV auto-export
├── config/
│   └── settings.yaml           # Models, thresholds, paths
├── knowledge/
│   └── benchmarks.yaml         # Industry benchmark values for callouts
└── tests/                      # 629 tests, 3 skipped
```

---

## Quick Start

### 1. Prerequisites

```bash
# Install Ollama — https://ollama.com
brew install ollama          # macOS
# or follow Linux/Windows instructions at ollama.com

# Pull the required models
ollama pull llama3.1:8b
ollama pull llava            # vision model for screenshot OCR
ollama serve                 # start the Ollama server
```

### 2. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** `pytesseract` requires the Tesseract binary.
> macOS: `brew install tesseract` | Ubuntu: `apt install tesseract-ocr`

### 3. Configure

```bash
cp .env.example .env
# Edit .env — set LLM_PROVIDER and optionally GROQ_API_KEY
```

Edit `config/settings.yaml` to customise models, dormancy thresholds, and upload paths.

### 4. Run

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Usage

### Chat tab
Type any CRM question in the chat box. The engine classifies the intent (aggregation, trend, recommendation, pivot, what-if, benchmark) and returns an answer with an optional chart and strategic recommendation.

### Weekly Agenda tab
Click any of the 23 pre-built questions. Results are cached per session. If the underlying data changes, answered questions are flagged with a ⚠️ stale badge.

### Data Explorer tab
Browse loaded tables, filter rows, inspect column statistics and schema, build manual joins, and roll back to a previous parquet snapshot of any table.

### Insights Dashboard tab
KPI cards, auto-refreshing charts, and a one-click HTML dashboard export.

---

## LLM Providers

| Provider | Privacy | Cost | Setup |
|----------|---------|------|-------|
| Ollama | 100% local | Free | `ollama pull llama3.1:8b` |
| Groq | Cloud | Free tier | Set `GROQ_API_KEY` in `.env` |

The agent tries Ollama first on every query. If Ollama is unavailable or times out, it automatically falls back to Groq. Switch the default with `LLM_PROVIDER=groq` in `.env`.

---

## Running Tests

```bash
pytest tests/ -v --tb=short
# 629 passed, 3 skipped
```

---

## Data Privacy

- All uploaded files stay in `data/uploads/` on your machine.
- Parquet caches and version snapshots live in `data/.cache/`.
- Query history and patterns are stored in a local SQLite DB (`tracker/`).
- LLM call logs (prompts + responses) are written to `exports/llm_log.jsonl` locally.
- Nothing is sent to external servers when using Ollama.
