"""
agent/self_improver.py — Intent-aware critic-rewriter self-improvement loop (Module 13).

Architecture
------------
QueryResult (from QueryEngine)
    ↓
Critic Agent — intent-aware scoring across 4 dimensions (total = 100)
    ↓
total_score >= threshold (85)?  YES → deliver  NO → Rewriter targets highest_priority_fix
    ↓
Re-execute if code was rewritten, otherwise only rewrite text
    ↓
Critic re-scores → repeat up to max_iterations
    ↓
Return best-scoring iteration
    ↓
Log every iteration to exports/training_log.jsonl
    ↓
If final_score >= 90: store pattern in data/.cache/pattern_memory.json

Public API
----------
SelfImprover(llm_client, config, tracker_db=None, cache_dir="data/.cache", exports_dir="exports")
    .improve(query_result, dataframes) -> QueryResult
    .export_training_data(output_path=None) -> Path
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent.query_engine import QueryResult, _execute_code

logger = logging.getLogger(__name__)

# ── Optional sentence-transformers ────────────────────────────────────────────

try:
    from sentence_transformers import SentenceTransformer as _ST
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    _ST = None  # type: ignore[assignment,misc]

_EMBEDDER: Any = None


def _get_embedder() -> Any:
    global _EMBEDDER
    if _EMBEDDER is None and _ST_AVAILABLE:
        try:
            _EMBEDDER = _ST("all-MiniLM-L6-v2")
        except Exception as exc:
            logger.debug("Could not load sentence-transformer model: %s", exc)
    return _EMBEDDER


def _cosine_sim(a: list, b: list) -> float:
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 1e-9 else 0.0


def _word_overlap_sim(q1: str, q2: str) -> float:
    """Fallback similarity when sentence-transformers unavailable."""
    s1 = set(re.findall(r"\w+", q1.lower()))
    s2 = set(re.findall(r"\w+", q2.lower()))
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


# ── Intent-aware rubric weights ───────────────────────────────────────────────

RUBRIC_WEIGHTS: dict[str, dict[str, int]] = {
    "aggregation":    {"code_correctness": 30, "result_completeness": 25, "interpretation_quality": 30, "chart_appropriateness": 15},
    "ranking":        {"code_correctness": 30, "result_completeness": 25, "interpretation_quality": 30, "chart_appropriateness": 15},
    "trend":          {"code_correctness": 30, "result_completeness": 25, "interpretation_quality": 30, "chart_appropriateness": 15},
    "pivot":          {"code_correctness": 25, "result_completeness": 25, "interpretation_quality": 25, "excel_export_quality": 25},
    "what_if":        {"scenario_logic": 35, "result_completeness": 25, "interpretation_quality": 25, "recommendation_quality": 15},
    "recommendation": {"data_grounding": 35, "specificity": 30, "interpretation_quality": 20, "benchmark_usage": 15},
    "data_quality":   {"code_correctness": 30, "result_completeness": 30, "interpretation_quality": 25, "chart_appropriateness": 15},
    "benchmark":      {"code_correctness": 30, "result_completeness": 30, "interpretation_quality": 25, "chart_appropriateness": 15},
}

# ── Prompt templates ──────────────────────────────────────────────────────────

_CRITIC_SYSTEM = """\
You are a rigorous senior data analyst reviewing an AI-generated CRM analysis \
for a consulting engagement.

Intent type: {intent_type}

Score using this rubric (total must equal 100):
{rubric}

Be strict — 85+ means ready to present to a client. 70-84 needs improvement. \
Below 70 has significant issues.

Return ONLY valid JSON:
{{
  "total_score": <0-100>,
  "dimension_scores": {{{dim_keys}}},
  "specific_feedback": "Exactly what is wrong and why",
  "rewrite_instructions": {{
    "code": "Specific instruction if code needs rewriting, else null",
    "interpretation": "Specific instruction if interpretation needs rewriting, else null",
    "recommendation": "Specific instruction if recommendation needs rewriting, else null",
    "chart": "Specific instruction if chart type needs changing, else null",
    "export": "Specific instruction if export needs fixing, else null"
  }},
  "highest_priority_fix": "code | interpretation | recommendation | chart | export"
}}"""

_REWRITER_SYSTEM = """\
You are improving a specific part of a CRM analytics response.
Apply the critic's instructions exactly.
Return ONLY the rewritten content — no explanation, no preamble."""

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class CriticResult:
    total_score: int
    dimension_scores: dict[str, int]
    specific_feedback: str
    rewrite_instructions: dict[str, str | None]
    highest_priority_fix: str  # "code" | "interpretation" | "recommendation" | "chart" | "export"


# ── SelfImprover ──────────────────────────────────────────────────────────────

class SelfImprover:
    """
    Intent-aware critic-rewriter loop.

    Every QueryResult passes through improve() before being returned to the user.
    The loop runs silently; users see only the best-scoring result plus a badge.
    """

    SIMILARITY_THRESHOLD = 0.85
    PATTERN_SCORE_THRESHOLD = 90
    TRAINING_EXPORT_MIN_SCORE = 75

    def __init__(
        self,
        llm_client: Any,
        config: dict,
        tracker_db: Any = None,
        cache_dir: str | Path = "data/.cache",
        exports_dir: str | Path = "exports",
    ) -> None:
        self.llm = llm_client
        cfg = config.get("self_improvement", {})
        self.enabled: bool = cfg.get("enabled", True)
        self.max_iterations: int = cfg.get("max_iterations", 5)
        self.score_threshold: int = cfg.get("score_threshold", 85)
        self.critic_temperature: float = cfg.get("critic_temperature", 0.2)
        self._tracker_db = tracker_db

        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._pattern_file = self._cache_dir / "pattern_memory.json"
        self._training_log = Path(exports_dir) / "training_log.jsonl"

    # ── Public API ────────────────────────────────────────────────────────────

    def improve(
        self,
        qr: QueryResult,
        dataframes: dict[str, pd.DataFrame],
    ) -> QueryResult:
        """
        Run the critic-rewriter loop on *qr*.
        Returns the best-scoring QueryResult within max_iterations.
        """
        if not self.enabled:
            return qr

        similar_codes = self._get_similar_patterns(qr.question)
        query_id = f"q_{qr.timestamp.strftime('%Y%m%d_%H%M%S%f')}"

        best_qr = qr
        best_score = 0
        iteration_log: list[dict] = []
        current = qr

        for iteration in range(1, self.max_iterations + 1):
            critic = self._critique(current)
            is_final = (
                critic.total_score >= self.score_threshold
                or iteration == self.max_iterations
            )

            record: dict = {
                "iteration": iteration,
                "critic_score": critic.total_score,
                "critic_feedback": critic.specific_feedback,
                "dimension_scores": critic.dimension_scores,
                "highest_priority_fix": critic.highest_priority_fix,
                "is_final": is_final,
            }
            iteration_log.append(record)
            self._log_training(query_id, current, critic, iteration, is_final)

            if critic.total_score > best_score:
                best_score = critic.total_score
                best_qr = current

            if critic.total_score >= self.score_threshold:
                logger.info(
                    "Self-improvement passed at iteration %d (score=%d)",
                    iteration, critic.total_score,
                )
                break

            if iteration == self.max_iterations:
                logger.warning("Max iterations reached. Best score: %d", best_score)
                break

            try:
                current = self._rewrite(current, critic, dataframes, similar_codes)
            except Exception as exc:
                logger.warning("Rewrite failed at iteration %d: %s", iteration, exc)
                break

        # Persist high-quality pattern
        if best_score >= self.PATTERN_SCORE_THRESHOLD:
            self._save_pattern(best_qr, best_score)

        # Module 12 tracker_db pattern logging (rewrite-improved only)
        if best_score >= self.score_threshold and len(iteration_log) > 1 and self._tracker_db is not None:
            try:
                self._tracker_db.log_pattern(
                    question_type=best_qr.intent_type or "unknown",
                    code_pattern=best_qr.code,
                    score=float(best_score),
                )
            except Exception as exc:
                logger.debug("TrackerDB pattern logging failed: %s", exc)

        # Return best with updated metadata
        best_qr.final_score = best_score
        best_qr.iterations = len(iteration_log)
        best_qr.iteration_log = iteration_log
        return best_qr

    def export_training_data(self, output_path: Path | str | None = None) -> Path:
        """
        Export final-iteration records with score >= TRAINING_EXPORT_MIN_SCORE.
        Returns the output path.
        """
        if output_path is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_path = self._training_log.parent / f"training_export_{ts}.jsonl"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        if self._training_log.exists():
            with open(self._training_log) as f_in, open(output_path, "w") as f_out:
                for line in f_in:
                    try:
                        rec = json.loads(line)
                        if rec.get("is_final") and rec.get("critic_score", 0) >= self.TRAINING_EXPORT_MIN_SCORE:
                            f_out.write(line)
                            count += 1
                    except json.JSONDecodeError:
                        continue
        logger.info("Exported %d training records to %s", count, output_path)
        return output_path

    # ── Critic ────────────────────────────────────────────────────────────────

    def _critique(self, qr: QueryResult) -> CriticResult:
        intent = qr.intent_type or "aggregation"
        rubric = RUBRIC_WEIGHTS.get(intent, RUBRIC_WEIGHTS["aggregation"])

        rubric_lines = "\n".join(
            f"- {k.replace('_', ' ').title()}: {v} points"
            for k, v in rubric.items()
        )
        dim_keys = ", ".join(f'"{k}": <score>' for k in rubric)

        system = _CRITIC_SYSTEM.format(
            intent_type=intent,
            rubric=rubric_lines,
            dim_keys=dim_keys,
        )

        result_repr = repr(qr.result)[:400] if qr.result is not None else f"ERROR: {qr.error}"
        rec_repr = json.dumps(qr.recommendation or {}, indent=2)[:300]

        user = (
            f"Question: {qr.question}\n\n"
            f"Code used:\n```python\n{qr.code or '# no code generated'}\n```\n\n"
            f"Result summary: {result_repr}\n\n"
            f"Interpretation: {qr.answer_text or '(none)'}\n\n"
            f"Recommendation: {rec_repr}"
        )

        raw = self.llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.critic_temperature,
        )

        try:
            data = json.loads(raw)
            return CriticResult(
                total_score=int(data.get("total_score", 50)),
                dimension_scores=data.get("dimension_scores", {}),
                specific_feedback=str(data.get("specific_feedback", "")),
                rewrite_instructions=data.get("rewrite_instructions", {}),
                highest_priority_fix=str(data.get("highest_priority_fix", "code")),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.warning("Critic returned non-JSON: %s", raw[:200])
            return CriticResult(
                total_score=50,
                dimension_scores={},
                specific_feedback="Could not parse critic response",
                rewrite_instructions={"code": "Re-examine the code logic"},
                highest_priority_fix="code",
            )

    # ── Rewriter ──────────────────────────────────────────────────────────────

    def _rewrite(
        self,
        qr: QueryResult,
        critic: CriticResult,
        dataframes: dict[str, pd.DataFrame],
        similar_codes: list[str],
    ) -> QueryResult:
        """Rewrite only the dimension identified as highest_priority_fix."""
        fix = critic.highest_priority_fix
        instructions = critic.rewrite_instructions.get(fix) or critic.specific_feedback

        if fix == "code":
            new_code = self._rewrite_code(qr, critic, similar_codes)
            new_result = _execute_code(new_code, dataframes)
            new_interp = self._call_rewriter(
                question=qr.question,
                intent=qr.intent_type,
                fix="interpretation",
                original=qr.answer_text or "",
                instructions="Rewrite the interpretation for the new result.",
                feedback=f"New result: {repr(new_result)[:300]}",
            )
            new_qr = dataclasses.replace(
                qr,
                code=new_code,
                result=new_result,
                answer_text=new_interp,
                result_df=new_result if isinstance(new_result, pd.DataFrame) else qr.result_df,
                error=None,
                iteration_log=[],
            )
            return new_qr

        elif fix == "recommendation":
            new_rec = self._rewrite_recommendation(qr, instructions, critic)
            return dataclasses.replace(qr, recommendation=new_rec, iteration_log=[])

        else:
            # "interpretation", "chart", "export" — rewrite answer_text
            new_text = self._call_rewriter(
                question=qr.question,
                intent=qr.intent_type,
                fix=fix,
                original=qr.answer_text or "",
                instructions=instructions or "",
                feedback=critic.specific_feedback,
            )
            return dataclasses.replace(qr, answer_text=new_text, iteration_log=[])

    def _rewrite_code(
        self,
        qr: QueryResult,
        critic: CriticResult,
        similar_codes: list[str],
    ) -> str:
        few_shot = ""
        if similar_codes:
            examples = "\n\n".join(
                f"Example {i+1}:\n```python\n{c}\n```"
                for i, c in enumerate(similar_codes[:3])
            )
            few_shot = f"\n\nSimilar successful patterns from memory:\n{examples}"

        instructions = critic.rewrite_instructions.get("code") or critic.specific_feedback
        user = (
            f"Question: {qr.question}\n"
            f"Intent type: {qr.intent_type}\n"
            f"What needs fixing: code\n"
            f"Critic feedback: {critic.specific_feedback}\n"
            f"Rewrite instructions: {instructions}\n\n"
            f"Original code:\n```python\n{qr.code}\n```"
            f"{few_shot}\n\n"
            f"Rewrite ONLY the code. Store the result in a variable called 'result'. "
            f"Return only the code block."
        )
        raw = self.llm.chat(
            [{"role": "system", "content": _REWRITER_SYSTEM}, {"role": "user", "content": user}]
        )
        match = _CODE_FENCE_RE.search(raw)
        return match.group(1).strip() if match else raw.strip()

    def _call_rewriter(
        self,
        question: str,
        intent: str,
        fix: str,
        original: str,
        instructions: str,
        feedback: str,
    ) -> str:
        user = (
            f"Question: {question}\n"
            f"Intent type: {intent}\n"
            f"What needs fixing: {fix}\n"
            f"Critic feedback: {feedback}\n"
            f"Rewrite instructions: {instructions}\n\n"
            f"Original {fix}:\n{original}\n\n"
            f"Rewrite ONLY the {fix}. Return only the rewritten text, no preamble."
        )
        return self.llm.chat(
            [{"role": "system", "content": _REWRITER_SYSTEM}, {"role": "user", "content": user}]
        ).strip()

    def _rewrite_recommendation(
        self,
        qr: QueryResult,
        instructions: str,
        critic: CriticResult,
    ) -> dict:
        user = (
            f"Question: {qr.question}\n"
            f"Intent type: {qr.intent_type}\n"
            f"Critic feedback: {critic.specific_feedback}\n"
            f"Instructions: {instructions}\n\n"
            f"Original recommendation:\n{json.dumps(qr.recommendation or {}, indent=2)}\n\n"
            f"Rewrite the recommendation. Return ONLY valid JSON:\n"
            f'{{"priority_action": "...", "risk_flag": "...", "opportunity": "..."}}'
        )
        raw = self.llm.chat(
            [{"role": "system", "content": _REWRITER_SYSTEM}, {"role": "user", "content": user}]
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return qr.recommendation or {}

    # ── Pattern memory ────────────────────────────────────────────────────────

    def _load_pattern_memory(self) -> list[dict]:
        if not self._pattern_file.exists():
            return []
        try:
            return json.loads(self._pattern_file.read_text())
        except (json.JSONDecodeError, OSError):
            return []

    def _save_pattern(self, qr: QueryResult, score: int) -> None:
        patterns = self._load_pattern_memory()
        embedder = _get_embedder()
        embedding: list[float] = []
        if embedder is not None:
            try:
                embedding = embedder.encode(qr.question).tolist()
            except Exception:
                pass

        entry = {
            "question": qr.question,
            "question_embedding": embedding,
            "intent_type": qr.intent_type or "unknown",
            "pandas_pattern": qr.code,
            "interpretation_style": (qr.answer_text or "")[:500],
            "score": score,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": qr.session_id,
        }
        # Avoid exact duplicates (same code)
        if not any(p.get("pandas_pattern") == qr.code for p in patterns):
            patterns.append(entry)
            try:
                self._pattern_file.write_text(json.dumps(patterns, indent=2))
            except OSError as exc:
                logger.debug("Could not write pattern memory: %s", exc)

    def _get_similar_patterns(self, question: str) -> list[str]:
        """Return pandas code patterns from past similar questions."""
        patterns = self._load_pattern_memory()
        if not patterns:
            return []

        embedder = _get_embedder()
        results: list[tuple[float, str]] = []

        for p in patterns:
            if embedder is not None and p.get("question_embedding"):
                try:
                    q_emb = embedder.encode(question).tolist()
                    sim = _cosine_sim(q_emb, p["question_embedding"])
                except Exception:
                    sim = _word_overlap_sim(question, p.get("question", ""))
            else:
                sim = _word_overlap_sim(question, p.get("question", ""))

            if sim >= self.SIMILARITY_THRESHOLD:
                results.append((sim, p["pandas_pattern"]))

        results.sort(reverse=True)
        return [code for _, code in results[:3]]

    # ── Training logger ───────────────────────────────────────────────────────

    def _log_training(
        self,
        query_id: str,
        qr: QueryResult,
        critic: CriticResult,
        iteration: int,
        is_final: bool,
    ) -> None:
        record = {
            "session_id": qr.session_id,
            "query_id": query_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": qr.question,
            "intent_type": qr.intent_type,
            "iteration": iteration,
            "code": qr.code,
            "result_summary": repr(qr.result)[:300] if qr.result is not None else "",
            "interpretation": qr.answer_text,
            "recommendation": qr.recommendation,
            "critic_score": critic.total_score,
            "dimension_scores": critic.dimension_scores,
            "critic_feedback": critic.specific_feedback,
            "provider": qr.provider_used,
            "model": "",
            "is_final": is_final,
        }
        try:
            self._training_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self._training_log, "a") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:
            logger.debug("Training log write failed: %s", exc)
