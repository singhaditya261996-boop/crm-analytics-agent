# Equans CRM Analytics Agent

A fully local, offline AI-powered CRM analytics agent built with Streamlit, Ollama, and pandas.
**No data leaves the machine. No paid APIs required.**

---

## Architecture

```
crm-agent/
├── app.py                      # Streamlit entry point
├── agent/
│   ├── llm_client.py           # Ollama + Groq abstraction
│   ├── query_engine.py         # English → pandas → result
│   ├── context_builder.py      # Schema context for LLM
│   ├── chart_generator.py      # Auto chart generation
│   └── self_improver.py        # Critic + rewriter + pattern memory
├── data/
│   ├── loader.py               # Multi-file loader + schema detector
│   ├── joiner.py               # Auto join detection
│   ├── profiler.py             # Data quality analysis
│   ├── update_handler.py       # Incremental file update detection
│   └── cache.py                # Parquet cache management
├── formats/
│   ├── ppt_handler.py          # PowerPoint text + table extraction
│   ├── pdf_handler.py          # PDF text + table extraction
│   └── image_handler.py        # Screenshot OCR via llava
├── agenda/
│   └── prompts.py              # Pre-built weekly agenda questions
├── tracker/
│   ├── database.py             # SQLite via SQLAlchemy
│   └── csv_manager.py          # CSV auto-export
├── exports/
│   ├── charts/                 # Auto-saved chart PNGs
│   └── sessions/               # Session Word doc exports
├── config/
│   └── settings.yaml
└── tests/
```

---

## Quick Start

### 1. Prerequisites

```bash
# Install Ollama (https://ollama.com)
brew install ollama          # macOS
# or follow Linux/Windows instructions at ollama.com

# Pull the required models
ollama pull llama3.1:8b
ollama pull llava            # vision model for screenshot OCR
```

### 2. Install Python Dependencies

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
# Edit .env — at minimum set LLM_PROVIDER=ollama
```

Edit `config/settings.yaml` to customise models, thresholds, and paths.

### 4. Run

```bash
streamlit run app.py
```

---

## LLM Providers

| Provider | Privacy | Cost  | Setup            |
|----------|---------|-------|------------------|
| Ollama   | 100% local | Free | Pull model once |
| Groq     | Cloud   | Free tier | Set `GROQ_API_KEY` |

Switch via `LLM_PROVIDER=ollama` or `LLM_PROVIDER=groq` in `.env`.

---

## Running Tests

```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## Data Privacy

- All uploaded files stay in `data/uploads/` on your machine.
- Parquet caches live in `data/.cache/`.
- Query history and patterns are stored in a local SQLite DB (`tracker/`).
- Nothing is sent to external servers when using Ollama.
