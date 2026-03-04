# Equans CRM Analytics Agent

A fully local, offline AI-powered analytics assistant for CRM data — built with Streamlit, Ollama, and pandas.
**No data leaves the machine. No paid APIs required.**

Ask questions in plain English, get charts, pivot tables, recommendations, and weekly agenda answers — all powered by a local LLM running on your own hardware. Every answer is automatically reviewed by a critic agent and rewritten if it scores below 85/100.

---

## Features

| Feature | Description |
|---------|-------------|
| **Natural language queries** | Ask "Which accounts haven't been contacted in 90 days?" and get a live answer with a chart |
| **Intent-aware self-improvement** | Critic scores every answer 0–100 across 4 weighted dimensions; rewriter targets the weakest dimension up to 5 times |
| **Auto join detection** | Fuzzy-matches column names across tables and suggests joins with referential overlap scoring |
| **Data quality profiling** | Null rates, type issues, outlier detection, and a traffic-light quality report across all loaded tables |
| **Weekly agenda** | 23 pre-built CRM questions across 6 sections (Pipeline, Dormancy, Win/Loss, etc.) |
| **Chart generation** | Bar/Pareto, line + moving average, scatter, pie, heatmap, funnel, horizontal bar — auto-saved as PNGs |
| **Incremental file updates** | Fingerprints each upload, classifies as NEW / UPDATED / CORRECTED / DUPLICATE, diffs rows and cell values, saves versioned parquet snapshots for rollback |
| **Folder watcher** | Drop a file into `data/uploads/` and a sidebar badge appears instantly — no manual re-upload needed |
| **Pivot & What-If** | Auto-generates Excel pivot tables; what-if scenario comparison tables |
| **Session export** | Export full session as Word doc, section summaries, chat transcript, or PDF dashboard |
| **Stale data warnings** | If data is updated after an agenda answer is generated, the answer is flagged with a ⚠️ badge |
| **Pattern memory** | High-quality code patterns (score ≥ 90) stored in `data/.cache/pattern_memory.json` and injected as few-shot examples in future queries |
| **Query history** | Every query logged to SQLite (`tracker/crm_agent.db`); browsable in the History tab with CSV export |
| **100% local** | Ollama as primary LLM, Groq as optional cloud fallback — switch with one env var |

---

## Architecture

```
crm-agent/
├── app.py                      # Streamlit entry point (5 tabs + sidebar)
├── agent/
│   ├── llm_client.py           # Ollama → Groq fallback, async-primary, JSONL logging
│   ├── query_engine.py         # English → intent → pandas → QueryResult pipeline
│   ├── chart_generator.py      # 7 chart types, auto-saved PNGs
│   ├── self_improver.py        # Intent-aware critic + rewriter + pattern memory
│   └── session_exporter.py     # Word / PDF / TXT session exports
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
│   └── image_handler.py        # Screenshot OCR via llava vision model
├── tracker/
│   ├── database.py             # SQLite via SQLAlchemy (query log + pattern memory)
│   └── csv_manager.py          # CSV export of query history and patterns
├── config/
│   └── settings.yaml           # Models, thresholds, self-improvement config, paths
├── knowledge/
│   └── benchmarks.yaml         # Industry benchmark values injected into LLM prompts
└── tests/                      # 702 tests, 3 skipped
```

---

## Installation

### System requirements

- Python 3.10, 3.11, or 3.12
- macOS, Ubuntu 22.04+, or Windows 10+ (WSL2 recommended on Windows)
- 8 GB RAM minimum (16 GB recommended if running `llama3.1:8b` locally)
- 10 GB free disk space for model weights and parquet caches

### Step 1 — Clone the repository

```bash
git clone https://github.com/singhaditya261996-boop/crm-analytics-agent.git
cd crm-analytics-agent
```

### Step 2 — Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows PowerShell
```

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Install system binaries (for OCR and PDF)

```bash
# macOS
brew install tesseract poppler

# Ubuntu / Debian
sudo apt install tesseract-ocr poppler-utils

# Windows (WSL2) — same as Ubuntu
```

`tesseract` is only required if you upload screenshots or scanned PDFs. The app runs without it for CSV/Excel workflows.

### Step 5 — Configure

```bash
cp .env.example .env
# Open .env and set LLM_PROVIDER (ollama or groq) and optionally GROQ_API_KEY
```

The defaults in `config/settings.yaml` work out of the box for Ollama with `llama3.1:8b`.

---

## Ollama Setup (local LLM — recommended)

### Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows — download from https://ollama.com/download
```

### Start the server

```bash
ollama serve
# Ollama runs at http://localhost:11434 — keep this terminal open
```

### Pull the required models

```bash
ollama pull llama3.1:8b     # primary text model (~4.7 GB)
ollama pull llava            # vision model for screenshot OCR (~4.1 GB)
```

### Verify

```bash
ollama list
# NAME               SIZE
# llama3.1:8b        4.7 GB
# llava:latest       4.1 GB
```

### Use a different model (optional)

Edit `config/settings.yaml`:

```yaml
ollama:
  model: mistral:7b           # any model shown in `ollama list`
  vision_model: llava
```

### Troubleshooting Ollama

| Symptom | Fix |
|---------|-----|
| Sidebar shows "Ollama not reachable" | Run `ollama serve` in a terminal |
| Queries fall back to Groq immediately | Check `ollama serve` is running; verify `http://localhost:11434` returns JSON in your browser |
| Slow on first query | Model is loading from disk — subsequent queries are faster once the model is in memory |
| GPU not used on Apple Silicon | Upgrade Ollama to ≥ 0.2 — Metal GPU acceleration is automatic |

---

## Groq Setup (optional cloud fallback)

Groq provides free-tier API access to `llama-3.1-70b-versatile` — significantly more capable than the local 8b model for complex queries, but requires an internet connection.

### Get an API key

1. Visit <https://console.groq.com>
2. Sign in → **API Keys** → **Create new key**
3. Copy the key (starts with `gsk_...`)

### Configure

```bash
# In .env
GROQ_API_KEY=gsk_your_key_here
LLM_PROVIDER=groq              # to default to Groq every session
```

Or leave `LLM_PROVIDER=ollama` and Groq is used automatically only when Ollama fails or times out.

### Switch providers at runtime

The sidebar contains an **Active provider** radio button. Switching triggers a fresh engine on the next query — no restart needed.

### Privacy note

When Groq is active, your questions and short result summaries are sent to Groq's servers. Raw CRM data rows are never transmitted — only schema descriptions and answer snippets appear in prompts. See `agent/llm_client.py` for the exact prompt structure.

---

## First Run

1. Start Ollama: `ollama serve` (keep this terminal open)
2. Start the app: `streamlit run app.py`
3. Open **http://localhost:8501** in your browser
4. In the sidebar, click **Upload CRM files** and select one or more files:
   - Supported formats: `.xlsx`, `.xls`, `.csv`, `.pptx`, `.pdf`, `.png`, `.jpg`
   - Example: upload an accounts spreadsheet and an opportunities CSV
5. The app will automatically:
   - Type-infer all columns (currency, percentage, date, email, identifier)
   - Profile data quality (null rates, outliers, duplicates)
   - Suggest fuzzy-matched joins between tables
6. Review any join suggestions in the sidebar and approve or dismiss them. Or build a manual join in **Data Explorer → Manual Join Builder**.
7. Go to the **Chat** tab and ask your first question:
   - *"Which accounts haven't been contacted in 90 days?"*
   - *"Show me pipeline by stage as a funnel chart"*
   - *"What's our win rate by service line this quarter?"*

### What you will see in the answer

- **Quality badge** — `✨ Quality score: 91/100 · 2 iterations · llama3.1:8b`
  Expand it to see what the critic flagged and how the rewriter improved the answer.
- **Confidence pill** — `HIGH (91)` / `REVIEW (72)` / `LOW (45)`
- **Chart** — auto-generated Plotly figure (can be downloaded as PNG)
- **Recommendation** — strategic action, risk flag, and opportunity callout
- **Code** — the exact pandas code that produced the result (expandable)

### Expected terminal output on first run

```
Ollama is reachable at http://localhost:11434 (model: llama3.1:8b)
LLM log: exports/llm_log.jsonl
Tracker DB: tracker/crm_agent.db
```

---

## Adding New Field Mappings

### 1. TypeInferrer keyword lists (`data/loader.py`)

Add column-name synonyms to the keyword tuples near the top of the file:

```python
# Currency detection — extend _CURRENCY_KEYWORDS:
_CURRENCY_KEYWORDS = ("revenue", "value", "amount", "price", "cost",
                      "budget", "mrr", "arr", "tcv", "deal_value")  # ← add here

# Percentage detection — extend _PERCENTAGE_KEYWORDS:
_PERCENTAGE_KEYWORDS = ("rate", "pct", "percent", "ratio", "share",
                        "margin", "discount", "win_rate")           # ← add here

# Identifier detection — extend _ID_SUFFIXES or _ID_PREFIXES:
_ID_SUFFIXES = ("_id", "_key", "_ref", "_code", "_uuid", "_guid", "_num", "_ref")
```

Dates are inferred automatically by pattern matching — no list needed.

### 2. Dashboard column detection (`app.py`)

The Insights Dashboard locates revenue and account columns via keyword lists inside `_render_dashboard_tab()`. Add synonyms to the `_find()` calls:

```python
# Find the revenue column — add synonyms:
rev = _find(df_a, ["revenue", "amount", "value", "deal_value",
                   "contract_value", "tcv"])        # ← add here

# Find the account table — add table-name synonyms:
_, df_a = _pick(["account", "client", "customer",
                 "organisation", "company"])        # ← add here
```

### 3. KPI cards (`app.py`, `_build_kpi_html`)

KPI sidebar cards use the same `_try_sum()` keyword lists. Add synonyms in the `kpis` list:

```python
("Pipeline Value", _try_sum(["pipeline_value", "deal_value",
                              "opportunity_value", "tcv"]), "£{:,.0f}"),
```

### 4. Benchmark values (`knowledge/benchmarks.yaml`)

Add new industry benchmarks that the LLM will cite in qualifying answers:

```yaml
your_metric_name:
  your_source_median: "42% (YourSource 2024)"
  your_source_top_quartile: "58% (YourSource 2024)"
```

No restart required — `BenchmarkInjector` re-reads the YAML on every query.

### 5. Agenda questions (`agenda/prompts.py`)

Add new pre-built questions to the `AGENDA_QUESTIONS` list:

```python
AgendaQuestion(
    section=3,           # 1–6 (see SECTION_TITLES)
    question="Which service lines have the highest average deal size?",
    hint="Group by service_line, compute mean of deal_value, sort descending.",
),
```

---

## Configuration Reference

All configuration lives in `config/settings.yaml`:

```yaml
ollama:
  base_url: http://localhost:11434
  model: llama3.1:8b          # change to any model you have pulled
  vision_model: llava
  timeout_seconds: 60
  max_retries: 3
  temperature: 0.1

groq:
  api_key: ''                  # or set GROQ_API_KEY in .env
  model: llama-3.1-70b-versatile
  enabled: true

llm_provider: ollama           # ollama | groq

data:
  upload_folder: data/uploads/
  cache_folder: data/.cache/
  dormant_account_days: 90     # accounts with no activity beyond this are flagged
  join_confidence_threshold: 0.6
  auto_join: true

self_improvement:
  enabled: true
  max_iterations: 5            # max critic-rewriter cycles per query
  score_threshold: 85          # stop improving when critic score reaches this
  critic_temperature: 0.2
  log_all_iterations: true

exports:
  output_folder: exports/
  auto_save_charts: true
  log_llm_calls: true

tracker:
  db_url: "sqlite:///tracker/crm_agent.db"
```

---

## Running Tests

```bash
# Full suite
.venv/bin/python -m pytest tests/ -v --tb=short
# 702 passed, 3 skipped

# Individual modules
.venv/bin/python -m pytest tests/test_loader.py -v      # Module 2 — DataLoader
.venv/bin/python -m pytest tests/test_query_engine.py   # Module 6 — QueryEngine
.venv/bin/python -m pytest tests/test_self_improver.py  # Module 13 — SelfImprover
.venv/bin/python -m pytest tests/test_tracker.py        # Module 12 — TrackerDB

# All tests must pass without Ollama running — LLM calls are mocked in tests
```

---

## Troubleshooting

### "I wasn't able to compute this" in chat

**Cause:** The LLM generated pandas code that failed all retry attempts.

**Fix:**
1. Open **Data Explorer → Schema** and note the exact column names
2. Rephrase using exact column names: *"sum of `deal_value` grouped by `account_name`"*
3. If tables haven't been joined yet, approve a join suggestion in the sidebar first

### "No data loaded" even after uploading a file

**Cause:** File format not recognised, or file is password-protected.

**Fix:**
- Supported formats: `.xlsx`, `.xls`, `.csv`, `.pptx`, `.pdf`, `.png`, `.jpg`
- Remove password protection from Excel files before uploading
- Check the terminal for a specific load error message

### Queries are very slow (30+ seconds)

**Cause:** Model running on CPU only, or model cold-loading from disk.

**Fix:**
- Run `ollama ps` — confirm the model is loaded on GPU
- Apple Silicon: Metal GPU is used automatically with Ollama ≥ 0.2
- Reduce model size: `ollama pull llama3.2:3b` and update `settings.yaml`
- Set `OLLAMA_NUM_THREAD=8` in your shell to control CPU thread count

### Self-improvement loop makes every query slow

**Cause:** The critic + rewriter add 1–5 extra LLM round-trips per query.

**Fix:** Disable in `config/settings.yaml`:

```yaml
self_improvement:
  enabled: false
```

### "Export failed" when clicking Export Session

**Cause:** `python-docx` or `kaleido` is not installed.

**Fix:** `pip install python-docx kaleido`

### Plotly charts not appearing

**Cause:** `plotly` not installed, or `kaleido` missing for static export.

**Fix:** `pip install plotly kaleido`

### pytesseract errors on image upload

**Cause:** The Tesseract OCR binary is not installed.

**Fix:**
- macOS: `brew install tesseract`
- Ubuntu: `sudo apt install tesseract-ocr`
- Then restart the app

### SQLite tracker database locked

**Cause:** Two Streamlit instances running simultaneously sharing the same DB file.

**Fix:** Kill all Streamlit processes: `pkill -f streamlit`, then restart.

### Pattern memory not growing

**Cause:** No query has scored ≥ 90 yet in the critic loop — the critic is intentionally strict.

**Normal behaviour:** After 10–20 varied queries on a real dataset, high-quality patterns accumulate in `data/.cache/pattern_memory.json`. Patterns are reused as few-shot examples in future queries of the same intent type.

### Sidebar shows "Ollama status unknown"

**Cause:** No query has been run yet — the Ollama health check runs on the first query call.

**Fix:** Run any query in the Chat tab. The status updates to 🟢 or a warning automatically.

---

## Data Privacy

| Data | Location | Sent externally? |
|------|----------|-----------------|
| Uploaded CRM files | `data/uploads/` | Never |
| Parquet caches + snapshots | `data/.cache/` | Never |
| Query history + patterns | `tracker/crm_agent.db` | Never |
| LLM call logs (prompts + responses) | `exports/llm_log.jsonl` | Never |
| Self-improvement training log | `exports/training_log.jsonl` | Never |
| Questions + result summaries | Groq API (cloud) | **Only when Groq provider is active** |

Raw CRM data rows are never included in LLM prompts. Only schema descriptions, aggregated results, and answer summaries appear in prompt context.

---

## Licence

MIT — see [LICENSE](LICENSE).
