"""
tests/test_self_improver.py — Unit tests for agent/self_improver.py

No Ollama/Groq connection required — uses mock LLM clients.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from agent.query_engine import QueryResult
from agent.self_improver import SelfImprover

MOCK_CONFIG = {
    "self_improvement": {
        "enabled": True,
        "max_iterations": 3,
        "score_threshold": 85,
        "critic_temperature": 0.2,
        "log_all_iterations": False,
    }
}

ACCOUNTS_DF = pd.DataFrame({
    "account_id": [1, 2, 3],
    "revenue": [100_000, 200_000, 150_000],
    "stage": ["Closed Won", "Prospecting", "Negotiation"],
})


# ── Mock LLM helpers ──────────────────────────────────────────────────────────

def make_critic_llm(score: int, verdict: str = "pass") -> object:
    class _M:
        def chat(self, messages, temperature=None):
            return json.dumps({"score": score, "issues": [], "verdict": verdict})
    return _M()


def make_rewriter_llm(new_code: str) -> object:
    class _M:
        def chat(self, messages, temperature=None):
            return f"```python\n{new_code}\n```"
    return _M()


class CyclicLLM:
    """First call returns critic fail, second returns rewriter code, third returns critic pass."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, temperature=None):
        self.calls += 1
        if self.calls == 1:
            # Critic: fail
            return json.dumps({"score": 40, "issues": ["wrong aggregation"], "verdict": "fail"})
        elif self.calls == 2:
            # Rewriter: fixed code
            return "```python\nresult = accounts['revenue'].sum()\n```"
        else:
            # Critic: pass
            return json.dumps({"score": 90, "issues": [], "verdict": "pass"})


# ── Test: disabled ────────────────────────────────────────────────────────────

def test_improve_disabled_returns_original():
    config = {**MOCK_CONFIG, "self_improvement": {**MOCK_CONFIG["self_improvement"], "enabled": False}}
    improver = SelfImprover(make_critic_llm(40), config)
    qr = QueryResult(question="q", code="result = 1", result=1)
    result = improver.improve(qr, {"accounts": ACCOUNTS_DF})
    assert result is qr


# ── Test: already good ────────────────────────────────────────────────────────

def test_improve_passes_on_high_score():
    improver = SelfImprover(make_critic_llm(95, "pass"), MOCK_CONFIG)
    qr = QueryResult(question="q", code="result = accounts['revenue'].sum()", result=450_000)
    result = improver.improve(qr, {"accounts": ACCOUNTS_DF})
    assert result.error is None
    assert result.iterations == 1


# ── Test: rewrite cycle ───────────────────────────────────────────────────────

def test_improve_rewrites_on_low_score():
    improver = SelfImprover(CyclicLLM(), MOCK_CONFIG)
    qr = QueryResult(question="q", code="result = accounts.head()", result=ACCOUNTS_DF.head())
    result = improver.improve(qr, {"accounts": ACCOUNTS_DF})
    # Should have improved after rewrite
    assert result.result == 450_000 or result.iterations > 1


# ── Test: max iterations cap ──────────────────────────────────────────────────

def test_improve_stops_at_max_iterations():
    # Critic always returns fail
    improver = SelfImprover(make_critic_llm(10, "fail"), MOCK_CONFIG)
    qr = QueryResult(question="q", code="result = 0", result=0)

    class AlwaysFailLLM:
        def chat(self, messages, temperature=None):
            # Critic: always fail; Rewriter: same broken code
            content = messages[0].get("content", "")
            if "critic" in content.lower() or "score" in content.lower():
                return json.dumps({"score": 10, "issues": ["always wrong"], "verdict": "fail"})
            return "```python\nresult = 0\n```"

    improver.llm = AlwaysFailLLM()
    result = improver.improve(qr, {"accounts": ACCOUNTS_DF})
    assert result.iterations <= MOCK_CONFIG["self_improvement"]["max_iterations"]


# ── Test: critic JSON parse failure ──────────────────────────────────────────

def test_improve_handles_bad_critic_json():
    class BadCriticLLM:
        c = 0
        def chat(self, messages, temperature=None):
            self.c += 1
            if self.c <= 3:
                return "This is not JSON at all."
            return json.dumps({"score": 90, "issues": [], "verdict": "pass"})

    improver = SelfImprover(BadCriticLLM(), MOCK_CONFIG)
    qr = QueryResult(question="q", code="result = 1", result=1)
    # Should not raise, just handle gracefully
    result = improver.improve(qr, {"accounts": ACCOUNTS_DF})
    assert result is not None
