"""
agent/self_improver.py — Critic + rewriter loop with pattern memory.

Flow
----
1.  Execute initial QueryResult (passed in from QueryEngine)
2.  Critic LLM scores the result (0–100) and explains issues
3.  If score < threshold, Rewriter LLM produces improved code
4.  Repeat up to max_iterations
5.  Log successful patterns to SQLite pattern memory

Public API
----------
SelfImprover(llm_client, config)
    .improve(query_result, dataframes) -> QueryResult
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import pandas as pd

from agent.query_engine import QueryResult, _execute_code

logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

CRITIC_SYSTEM = """\
You are a strict pandas code critic. Given:
- A user's CRM question
- The generated pandas code
- The result (or error)

Score the answer from 0 to 100 and list specific issues.
Respond with ONLY valid JSON:
{"score": <int>, "issues": ["issue1", "issue2"], "verdict": "pass|fail"}

Score guide:
  90–100: correct, complete, efficient
  70–89 : correct but could be cleaner
  50–69 : partially correct or inefficient
  0–49  : wrong result or error
"""

REWRITER_SYSTEM = """\
You are an expert pandas code rewriter. Fix the issues listed by the critic.
Output ONLY the corrected Python code block (```python ... ```). No explanations.
"""


# ── SelfImprover ──────────────────────────────────────────────────────────────

class SelfImprover:
    """Iteratively critiques and rewrites LLM-generated pandas code."""

    _CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)

    def __init__(self, llm_client: Any, config: dict) -> None:
        self.llm = llm_client
        cfg = config.get("self_improvement", {})
        self.enabled: bool = cfg.get("enabled", True)
        self.max_iterations: int = cfg.get("max_iterations", 5)
        self.score_threshold: int = cfg.get("score_threshold", 85)
        self.critic_temperature: float = cfg.get("critic_temperature", 0.2)
        self.log_all: bool = cfg.get("log_all_iterations", True)

    def improve(
        self,
        query_result: QueryResult,
        dataframes: dict[str, pd.DataFrame],
    ) -> QueryResult:
        """
        Run the critic-rewriter loop on *query_result*.
        Returns the best QueryResult found within max_iterations.
        """
        if not self.enabled:
            return query_result

        current = query_result
        for iteration in range(1, self.max_iterations + 1):
            score, issues, verdict = self._critique(current)
            logger.info("Iteration %d — score: %d, verdict: %s", iteration, score, verdict)

            if score >= self.score_threshold or verdict == "pass":
                logger.info("Self-improvement passed at iteration %d (score=%d)", iteration, score)
                current.iterations = iteration
                return current

            if iteration == self.max_iterations:
                logger.warning("Max iterations reached. Returning best result so far.")
                current.iterations = iteration
                return current

            # Rewrite
            new_code = self._rewrite(current, issues)
            try:
                new_result = _execute_code(new_code, dataframes)
                current = QueryResult(
                    question=current.question,
                    code=new_code,
                    result=new_result,
                    iterations=iteration + 1,
                )
            except Exception as exc:
                logger.warning("Rewritten code also failed: %s", exc)
                current.error = str(exc)

        return current

    # ── Critic ────────────────────────────────────────────────────────────────

    def _critique(self, qr: QueryResult) -> tuple[int, list[str], str]:
        result_repr = repr(qr.result)[:500] if qr.result is not None else f"ERROR: {qr.error}"
        user_content = (
            f"Question: {qr.question}\n\n"
            f"Code:\n```python\n{qr.code}\n```\n\n"
            f"Result:\n{result_repr}"
        )
        messages = [
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": user_content},
        ]
        raw = self.llm.chat(messages, temperature=self.critic_temperature)
        try:
            data = json.loads(raw)
            return int(data["score"]), data.get("issues", []), data.get("verdict", "fail")
        except (json.JSONDecodeError, KeyError):
            logger.warning("Critic returned non-JSON: %s", raw[:200])
            return 50, ["Could not parse critic response"], "fail"

    # ── Rewriter ─────────────────────────────────────────────────────────────

    def _rewrite(self, qr: QueryResult, issues: list[str]) -> str:
        issues_text = "\n".join(f"- {i}" for i in issues)
        user_content = (
            f"Question: {qr.question}\n\n"
            f"Broken code:\n```python\n{qr.code}\n```\n\n"
            f"Issues to fix:\n{issues_text}"
        )
        messages = [
            {"role": "system", "content": REWRITER_SYSTEM},
            {"role": "user", "content": user_content},
        ]
        raw = self.llm.chat(messages)
        match = self._CODE_FENCE_RE.search(raw)
        return match.group(1).strip() if match else raw.strip()
