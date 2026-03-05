"""
knowledge/fine_tuning_prep.py — Fine-tuning dataset exporter (Module 15, optional).

Reads the self-improvement training log produced by Module 13 and exports
high-quality Q&A pairs in instruction-tuning format (JSONL).

Intended for post-engagement use — see README_FINE_TUNING.md for details.

CLI usage:
    python app.py --export-fine-tuning [--min-score 80]

Direct usage:
    from knowledge.fine_tuning_prep import export_fine_tuning_dataset
    path = export_fine_tuning_dataset(min_score=80)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_TRAINING_LOG = Path("exports/training_log.jsonl")
_DEFAULT_MIN_SCORE = 80


def export_fine_tuning_dataset(
    training_log_path: str | Path = _DEFAULT_TRAINING_LOG,
    output_path: str | Path | None = None,
    min_score: int = _DEFAULT_MIN_SCORE,
) -> Path:
    """
    Export high-quality Q&A pairs for fine-tuning.

    Filters training_log.jsonl for records where:
      - is_final == True
      - critic_score >= min_score

    Formats each as an instruction fine-tuning pair and writes to JSONL.

    Parameters
    ----------
    training_log_path : path to exports/training_log.jsonl
    output_path       : destination; defaults to exports/fine_tuning_dataset_YYYYMMDD.jsonl
    min_score         : minimum critic score to include (default 80)

    Returns
    -------
    Path to the exported JSONL file.
    """
    training_log_path = Path(training_log_path)
    if output_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d")
        output_path = training_log_path.parent / f"fine_tuning_dataset_{ts}.jsonl"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_records = _load_jsonl(training_log_path)
    high_quality = [
        r for r in all_records
        if r.get("is_final") and int(r.get("critic_score", 0)) >= min_score
    ]

    pairs: list[dict] = []
    for entry in high_quality:
        rec = entry.get("recommendation") or {}
        if isinstance(rec, str):
            try:
                rec = json.loads(rec)
            except Exception:
                rec = {}
        pair = {
            "instruction": entry.get("question", ""),
            "context": f"Intent: {entry.get('intent_type', 'unknown')}",
            "response": _format_response(entry, rec),
            "score": entry.get("critic_score", 0),
            "source_session": entry.get("session_id", ""),
            "intent_type": entry.get("intent_type", "unknown"),
        }
        if pair["instruction"] and pair["response"]:
            pairs.append(pair)

    _save_jsonl(pairs, output_path)

    intent_types = sorted({p["intent_type"] for p in pairs})
    logger.info(
        "Fine-tuning export: %d pairs from %d total logs "
        "(min_score=%d, %d intent types: %s) → %s",
        len(pairs), len(all_records), min_score,
        len(intent_types), ", ".join(intent_types), output_path,
    )
    print(f"Exported {len(pairs)} high-quality pairs to {output_path}")
    print(f"Filtered from {len(all_records)} total logged queries")
    if intent_types:
        print(f"Coverage: {len(intent_types)} intent types: {', '.join(intent_types)}")

    return output_path


# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_response(entry: dict, rec: dict) -> str:
    interp = entry.get("interpretation") or entry.get("answer_text") or ""
    if not rec:
        return interp
    priority = rec.get("priority_action", "")
    risk = rec.get("risk_flag", "")
    opportunity = rec.get("opportunity", "")
    rec_text = ""
    if any([priority, risk, opportunity]):
        rec_text = (
            "\n\nRECOMMENDATION:\n"
            f"Priority Action: {priority}\n"
            f"Risk Flag: {risk}\n"
            f"Opportunity: {opportunity}"
        )
    return f"{interp}{rec_text}".strip()


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _save_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
