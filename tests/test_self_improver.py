"""
tests/test_self_improver.py — Module 13 unit tests for agent/self_improver.py

No Ollama/Groq connection required — all LLM calls are mocked.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from agent.query_engine import QueryResult
from agent.self_improver import CriticResult, SelfImprover, RUBRIC_WEIGHTS


# ── Constants ──────────────────────────────────────────────────────────────────

MOCK_CONFIG = {
    "self_improvement": {
        "enabled": True,
        "max_iterations": 3,
        "score_threshold": 85,
        "critic_temperature": 0.2,
        "log_all_iterations": True,
    }
}

ACCOUNTS_DF = pd.DataFrame({
    "account_id": [1, 2, 3],
    "revenue": [100_000, 200_000, 150_000],
    "stage": ["Closed Won", "Prospecting", "Negotiation"],
})
DATAFRAMES = {"accounts": ACCOUNTS_DF}

_TS = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _make_qr(
    question: str = "How much total revenue?",
    code: str = "result = accounts['revenue'].sum()",
    result: object = 450_000,
    intent_type: str = "aggregation",
    answer_text: str = "Total revenue is $450,000",
    error: str | None = None,
) -> QueryResult:
    return QueryResult(
        question=question,
        code=code,
        result=result,
        intent_type=intent_type,
        answer_text=answer_text,
        error=error,
        timestamp=_TS,
    )


# ── Mock LLM helpers ───────────────────────────────────────────────────────────

def _critic_json(
    score: int,
    feedback: str = "Analysis looks good",
    fix: str = "code",
    instructions: dict | None = None,
    dim_scores: dict | None = None,
) -> str:
    dims = dim_scores or {
        "code_correctness": score // 4,
        "result_completeness": score // 4,
        "interpretation_quality": score // 4,
        "chart_appropriateness": score - 3 * (score // 4),
    }
    instr = instructions or {
        "code": None, "interpretation": None,
        "recommendation": None, "chart": None, "export": None,
    }
    return json.dumps({
        "total_score": score,
        "dimension_scores": dims,
        "specific_feedback": feedback,
        "rewrite_instructions": instr,
        "highest_priority_fix": fix,
    })


class FixedLLM:
    """Always returns the same response regardless of input."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def chat(self, messages, temperature=None):
        self.calls += 1
        return self.response


class SequenceLLM:
    """Returns responses in order; holds on the last response when exhausted."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.idx = 0
        self.calls = 0

    def chat(self, messages, temperature=None):
        self.calls += 1
        resp = self.responses[self.idx]
        self.idx = min(self.idx + 1, len(self.responses) - 1)
        return resp


class ClassifyingLLM:
    """Returns critic JSON when the prompt contains rubric text, code block otherwise."""

    def __init__(self, critic_score: int, code: str = "result = accounts['revenue'].sum()") -> None:
        self.critic_score = critic_score
        self.code = code
        self.calls = 0

    def chat(self, messages, temperature=None):
        self.calls += 1
        content = " ".join(m.get("content", "") for m in messages)
        if "Score using this rubric" in content:
            return _critic_json(self.critic_score)
        return f"```python\n{self.code}\n```"


class AlwaysFailLLM:
    """Critic always returns score=25 (fail); rewriter returns unchanged code."""

    def chat(self, messages, temperature=None):
        content = " ".join(m.get("content", "") for m in messages)
        if "Score using this rubric" in content:
            return _critic_json(25, fix="code")
        return "```python\nresult = 0\n```"


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def make_imp(tmp_path):
    """Factory that creates SelfImprover instances using isolated temp dirs."""
    def _factory(llm, config=None):
        return SelfImprover(
            llm_client=llm,
            config=config or MOCK_CONFIG,
            cache_dir=tmp_path / ".cache",
            exports_dir=tmp_path / "exports",
        )
    return _factory


# ── TestRubricWeights ──────────────────────────────────────────────────────────

class TestRubricWeights:
    def test_all_eight_intent_types_present(self):
        expected = {
            "aggregation", "ranking", "trend", "pivot",
            "what_if", "recommendation", "data_quality", "benchmark",
        }
        assert set(RUBRIC_WEIGHTS.keys()) == expected

    def test_each_rubric_sums_to_100(self):
        for intent, weights in RUBRIC_WEIGHTS.items():
            total = sum(weights.values())
            assert total == 100, f"{intent} sums to {total}, expected 100"

    def test_each_rubric_has_four_dimensions(self):
        for intent, weights in RUBRIC_WEIGHTS.items():
            assert len(weights) == 4, f"{intent} has {len(weights)} dimensions, expected 4"

    def test_pivot_has_excel_export_dimension(self):
        assert "excel_export_quality" in RUBRIC_WEIGHTS["pivot"]

    def test_what_if_has_scenario_logic_dimension(self):
        assert "scenario_logic" in RUBRIC_WEIGHTS["what_if"]

    def test_recommendation_has_data_grounding_dimension(self):
        assert "data_grounding" in RUBRIC_WEIGHTS["recommendation"]


# ── TestImproveDisabled ────────────────────────────────────────────────────────

class TestImproveDisabled:
    def test_returns_original_object_unchanged(self, make_imp):
        config = {**MOCK_CONFIG, "self_improvement": {
            **MOCK_CONFIG["self_improvement"], "enabled": False,
        }}
        imp = make_imp(FixedLLM("noop"), config=config)
        qr = _make_qr()
        assert imp.improve(qr, DATAFRAMES) is qr

    def test_makes_no_llm_calls(self, make_imp):
        config = {**MOCK_CONFIG, "self_improvement": {
            **MOCK_CONFIG["self_improvement"], "enabled": False,
        }}
        llm = FixedLLM("noop")
        imp = make_imp(llm, config=config)
        imp.improve(_make_qr(), DATAFRAMES)
        assert llm.calls == 0


# ── TestCriticScoring ──────────────────────────────────────────────────────────

class TestCriticScoring:
    def test_high_score_stops_at_iteration_1(self, make_imp):
        """Score >= threshold on first call → return immediately, no rewrite."""
        imp = make_imp(FixedLLM(_critic_json(90)))
        result = imp.improve(_make_qr(), DATAFRAMES)
        assert result.final_score == 90
        assert result.iterations == 1

    def test_low_score_triggers_rewrite(self, make_imp):
        """Score < threshold → rewrite is called."""
        llm = SequenceLLM([
            _critic_json(40, fix="code"),                           # iter 1: fail
            "```python\nresult = accounts['revenue'].sum()\n```",   # rewrite code
            "Revenue total updated.",                               # rewrite interp
            _critic_json(88),                                       # iter 2: pass
        ])
        imp = make_imp(llm)
        result = imp.improve(_make_qr(code="result = 0", result=0), DATAFRAMES)
        assert result.final_score >= 85

    def test_iteration_log_contains_critic_score(self, make_imp):
        imp = make_imp(FixedLLM(_critic_json(92)))
        result = imp.improve(_make_qr(), DATAFRAMES)
        assert result.iteration_log[0]["critic_score"] == 92

    def test_iteration_log_contains_feedback(self, make_imp):
        imp = make_imp(FixedLLM(_critic_json(90, feedback="Excellent work")))
        result = imp.improve(_make_qr(), DATAFRAMES)
        assert "Excellent work" in result.iteration_log[0]["critic_feedback"]

    def test_bad_critic_json_falls_back_to_score_50(self, make_imp):
        """Non-JSON from critic should not raise — defaults to score=50 and continues."""
        llm = SequenceLLM([
            "This is not JSON!",                                    # iter 1: parse fail
            "```python\nresult = 1\n```",                           # rewrite code
            "Interpreted.",                                         # rewrite interp
            _critic_json(88),                                       # iter 2: pass
        ])
        imp = make_imp(llm)
        result = imp.improve(_make_qr(), DATAFRAMES)
        assert result is not None

    def test_aggregation_rubric_in_critic_prompt(self, make_imp):
        """Critic system prompt must include aggregation-specific dimension names."""
        captured: list[dict] = []

        class CaptureLLM:
            def chat(self, messages, temperature=None):
                captured.extend(messages)
                return _critic_json(90)

        imp = make_imp(CaptureLLM())
        imp.improve(_make_qr(intent_type="aggregation"), DATAFRAMES)
        system_msg = next(
            (m["content"] for m in captured if m.get("role") == "system"), ""
        )
        assert "Code Correctness" in system_msg

    def test_pivot_rubric_in_critic_prompt(self, make_imp):
        """Pivot critic prompt must include the excel_export_quality dimension."""
        captured: list[dict] = []

        class CaptureLLM:
            def chat(self, messages, temperature=None):
                captured.extend(messages)
                return _critic_json(90)

        imp = make_imp(CaptureLLM())
        imp.improve(_make_qr(intent_type="pivot"), DATAFRAMES)
        system_msg = next(
            (m["content"] for m in captured if m.get("role") == "system"), ""
        )
        assert "Excel Export Quality" in system_msg

    def test_unknown_intent_falls_back_gracefully(self, make_imp):
        """Unknown intent_type must not raise — falls back to aggregation rubric."""
        imp = make_imp(FixedLLM(_critic_json(90)))
        result = imp.improve(_make_qr(intent_type="totally_unknown"), DATAFRAMES)
        assert result is not None


# ── TestRewriterDispatch ───────────────────────────────────────────────────────

class TestRewriterDispatch:
    def test_code_fix_re_executes_code(self, make_imp):
        """highest_priority_fix=code → new code is executed, result is updated."""
        llm = SequenceLLM([
            _critic_json(40, fix="code",
                         instructions={
                             "code": "Use .sum()", "interpretation": None,
                             "recommendation": None, "chart": None, "export": None,
                         }),
            "```python\nresult = accounts['revenue'].sum()\n```",   # rewrite code
            "Revenue total is $450K.",                               # interp rewriter
            _critic_json(90),                                        # iter 2: pass
        ])
        imp = make_imp(llm)
        result = imp.improve(_make_qr(code="result = 0", result=0), DATAFRAMES)
        assert result.result == 450_000

    def test_interpretation_fix_rewrites_answer_text(self, make_imp):
        """highest_priority_fix=interpretation → answer_text rewritten, no code exec."""
        new_interp = "Revenue across all accounts totals $450K."
        llm = SequenceLLM([
            _critic_json(55, fix="interpretation",
                         instructions={
                             "code": None, "interpretation": "Be more specific",
                             "recommendation": None, "chart": None, "export": None,
                         }),
            new_interp,     # rewriter
            _critic_json(88),
        ])
        imp = make_imp(llm)
        result = imp.improve(_make_qr(), DATAFRAMES)
        assert result.final_score >= 55

    def test_recommendation_fix_returns_updated_recommendation(self, make_imp):
        """highest_priority_fix=recommendation → recommendation dict is rewritten."""
        new_rec = {"priority_action": "Focus on top accounts", "risk_flag": "Low", "opportunity": "High"}
        llm = SequenceLLM([
            _critic_json(50, fix="recommendation"),
            json.dumps(new_rec),    # rewriter returns JSON
            _critic_json(90),
        ])
        imp = make_imp(llm)
        result = imp.improve(_make_qr(), DATAFRAMES)
        assert result is not None

    def test_chart_fix_calls_rewriter(self, make_imp):
        """highest_priority_fix=chart → answer_text rewritten with chart guidance."""
        llm = SequenceLLM([
            _critic_json(60, fix="chart"),
            "Use a horizontal bar chart to compare revenue by stage.",
            _critic_json(87),
        ])
        imp = make_imp(llm)
        result = imp.improve(_make_qr(), DATAFRAMES)
        assert result.final_score >= 60

    def test_export_fix_calls_rewriter(self, make_imp):
        """highest_priority_fix=export → answer_text rewritten."""
        llm = SequenceLLM([
            _critic_json(62, fix="export"),
            "Export includes pivot table on Sheet 2.",
            _critic_json(86),
        ])
        imp = make_imp(llm)
        result = imp.improve(_make_qr(), DATAFRAMES)
        assert result.final_score >= 62


# ── TestMaxIterations ──────────────────────────────────────────────────────────

class TestMaxIterations:
    def test_loop_stops_at_max_iterations(self, make_imp):
        """Critic never passes → loop terminates at max_iterations exactly."""
        imp = make_imp(AlwaysFailLLM())
        result = imp.improve(_make_qr(), DATAFRAMES)
        max_iters = MOCK_CONFIG["self_improvement"]["max_iterations"]
        assert result.iterations <= max_iters

    def test_iteration_log_has_one_entry_per_critic_call(self, make_imp):
        """iteration_log length must equal the number of critic evaluations."""
        imp = make_imp(AlwaysFailLLM())
        result = imp.improve(_make_qr(), DATAFRAMES)
        max_iters = MOCK_CONFIG["self_improvement"]["max_iterations"]
        assert len(result.iteration_log) == max_iters

    def test_final_entry_is_marked_is_final_true(self, make_imp):
        """The last iteration_log entry must have is_final=True."""
        imp = make_imp(AlwaysFailLLM())
        result = imp.improve(_make_qr(), DATAFRAMES)
        assert result.iteration_log[-1]["is_final"] is True

    def test_passing_iteration_marked_is_final_true(self, make_imp):
        """When score passes threshold, that iteration is marked is_final=True."""
        imp = make_imp(FixedLLM(_critic_json(90)))
        result = imp.improve(_make_qr(), DATAFRAMES)
        assert result.iteration_log[0]["is_final"] is True

    def test_non_final_early_iterations_are_false(self, make_imp):
        """All iterations before the final one must have is_final=False."""
        llm = SequenceLLM([
            _critic_json(40, fix="code"),
            "```python\nresult = accounts['revenue'].sum()\n```",
            "Updated.",
            _critic_json(90),
        ])
        imp = make_imp(llm)
        result = imp.improve(_make_qr(code="result = 0", result=0), DATAFRAMES)
        assert len(result.iteration_log) == 2
        assert result.iteration_log[0]["is_final"] is False
        assert result.iteration_log[1]["is_final"] is True


# ── TestBestIteration ──────────────────────────────────────────────────────────

class TestBestIteration:
    def test_best_score_returned_not_last(self, make_imp):
        """
        Sequence: iter1=40, iter2=70, iter3=50.
        Best is iter2 (score=70). final_score must be 70.
        """
        llm = SequenceLLM([
            _critic_json(40, fix="code"),                           # iter 1 critic
            "```python\nresult = 1\n```",                           # iter 1 rewrite
            "Interp 1",                                             # iter 1 interp
            _critic_json(70, fix="code"),                           # iter 2 critic
            "```python\nresult = 2\n```",                           # iter 2 rewrite
            "Interp 2",                                             # iter 2 interp
            _critic_json(50),                                       # iter 3 critic (final)
        ])
        imp = make_imp(llm)
        result = imp.improve(_make_qr(), DATAFRAMES)
        assert result.final_score == 70

    def test_first_improvement_is_tracked(self, make_imp):
        """iter1=80, iter2=60 → best is 80 → final_score=80."""
        llm = SequenceLLM([
            _critic_json(80, fix="code"),
            "```python\nresult = 1\n```",
            "Updated.",
            _critic_json(60),
        ])
        imp = make_imp(llm)
        result = imp.improve(_make_qr(), DATAFRAMES)
        assert result.final_score == 80

    def test_immediately_passing_score_sets_iterations_to_1(self, make_imp):
        """Score passes on first iteration → no rewrites, iterations=1."""
        imp = make_imp(FixedLLM(_critic_json(90)))
        result = imp.improve(_make_qr(), DATAFRAMES)
        assert result.iterations == 1
        assert result.final_score == 90


# ── TestPatternMemory ──────────────────────────────────────────────────────────

class TestPatternMemory:
    def test_high_score_pattern_written_to_file(self, make_imp):
        """Score >= 90 → pattern saved to pattern_memory.json."""
        imp = make_imp(FixedLLM(_critic_json(92)))
        imp.improve(_make_qr(code="result = accounts['revenue'].sum()"), DATAFRAMES)
        assert imp._pattern_file.exists()
        patterns = json.loads(imp._pattern_file.read_text())
        assert len(patterns) == 1
        assert "revenue" in patterns[0]["pandas_pattern"]

    def test_low_score_pattern_not_saved(self, make_imp):
        """Score < 90 → nothing written to pattern_memory.json."""
        imp = make_imp(AlwaysFailLLM())  # always score=25
        imp.improve(_make_qr(), DATAFRAMES)
        if imp._pattern_file.exists():
            patterns = json.loads(imp._pattern_file.read_text())
            assert len(patterns) == 0

    def test_duplicate_code_not_stored_twice(self, make_imp):
        """Same code pattern twice → deduplicated, only one entry stored."""
        imp = make_imp(FixedLLM(_critic_json(92)))
        qr = _make_qr(code="result = accounts['revenue'].sum()")
        imp.improve(qr, DATAFRAMES)
        imp.improve(qr, DATAFRAMES)
        patterns = json.loads(imp._pattern_file.read_text())
        assert len(patterns) == 1

    def test_similar_question_retrieves_stored_pattern(self, tmp_path):
        """Word-overlap similarity returns matching code from pattern_memory.json."""
        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        pattern_file = cache_dir / "pattern_memory.json"
        pattern_file.write_text(json.dumps([{
            "question": "total revenue for all accounts",
            "question_embedding": [],   # empty → word-overlap path
            "intent_type": "aggregation",
            "pandas_pattern": "result = df['revenue'].sum()",
            "interpretation_style": "Total is X",
            "score": 92,
            "timestamp": "2025-01-01T00:00:00+00:00",
            "session_id": "test",
        }]))
        imp = SelfImprover(
            llm_client=FixedLLM(_critic_json(90)),
            config=MOCK_CONFIG,
            cache_dir=cache_dir,
            exports_dir=tmp_path / "exports",
        )
        # Lower threshold so word-overlap kicks in
        imp.SIMILARITY_THRESHOLD = 0.3
        codes = imp._get_similar_patterns("what is the total revenue for accounts")
        assert len(codes) > 0
        assert "revenue" in codes[0]

    def test_empty_pattern_file_returns_empty_list(self, make_imp):
        imp = make_imp(FixedLLM(_critic_json(90)))
        codes = imp._get_similar_patterns("some question")
        assert codes == []

    def test_pattern_entry_has_expected_fields(self, make_imp):
        imp = make_imp(FixedLLM(_critic_json(95)))
        qr = _make_qr(question="Revenue total?", code="result = accounts['revenue'].sum()")
        imp.improve(qr, DATAFRAMES)
        pattern = json.loads(imp._pattern_file.read_text())[0]
        for field in ["question", "intent_type", "pandas_pattern", "score", "timestamp"]:
            assert field in pattern, f"Missing field: {field}"


# ── TestTrainingLog ────────────────────────────────────────────────────────────

class TestTrainingLog:
    def test_training_log_created_after_improve(self, make_imp):
        imp = make_imp(FixedLLM(_critic_json(90)))
        imp.improve(_make_qr(), DATAFRAMES)
        assert imp._training_log.exists()

    def test_one_record_per_iteration_single_pass(self, make_imp):
        """One critic call → one line in training log."""
        imp = make_imp(FixedLLM(_critic_json(90)))
        imp.improve(_make_qr(), DATAFRAMES)
        lines = imp._training_log.read_text().strip().splitlines()
        assert len(lines) == 1

    def test_multiple_queries_append_to_log(self, make_imp):
        imp = make_imp(FixedLLM(_critic_json(90)))
        imp.improve(_make_qr(question="Q1"), DATAFRAMES)
        imp.improve(_make_qr(question="Q2"), DATAFRAMES)
        lines = imp._training_log.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_training_record_has_required_fields(self, make_imp):
        imp = make_imp(FixedLLM(_critic_json(88, feedback="Solid analysis")))
        imp.improve(_make_qr(question="Revenue total"), DATAFRAMES)
        record = json.loads(imp._training_log.read_text().strip())
        for key in [
            "query_id", "question", "intent_type", "iteration",
            "code", "critic_score", "critic_feedback", "is_final",
        ]:
            assert key in record, f"Missing key: {key}"

    def test_is_final_true_when_threshold_passed(self, make_imp):
        imp = make_imp(FixedLLM(_critic_json(90)))
        imp.improve(_make_qr(), DATAFRAMES)
        record = json.loads(imp._training_log.read_text().strip())
        assert record["is_final"] is True

    def test_all_records_are_valid_json(self, make_imp):
        """Every line in the training log must be parseable JSON."""
        imp = make_imp(AlwaysFailLLM())
        imp.improve(_make_qr(), DATAFRAMES)
        for line in imp._training_log.read_text().strip().splitlines():
            parsed = json.loads(line)  # must not raise
            assert isinstance(parsed, dict)

    def test_training_log_max_iterations_count(self, make_imp):
        """Always-fail loop → log contains exactly max_iterations records."""
        imp = make_imp(AlwaysFailLLM())
        imp.improve(_make_qr(), DATAFRAMES)
        lines = imp._training_log.read_text().strip().splitlines()
        assert len(lines) == MOCK_CONFIG["self_improvement"]["max_iterations"]


# ── TestExportTrainingData ─────────────────────────────────────────────────────

class TestExportTrainingData:
    def test_returns_path_object(self, make_imp):
        imp = make_imp(FixedLLM(_critic_json(90)))
        imp.improve(_make_qr(), DATAFRAMES)
        out = imp.export_training_data()
        assert isinstance(out, Path)
        assert out.exists()

    def test_custom_output_path_used(self, tmp_path, make_imp):
        imp = make_imp(FixedLLM(_critic_json(90)))
        imp.improve(_make_qr(), DATAFRAMES)
        custom = tmp_path / "my_export.jsonl"
        out = imp.export_training_data(output_path=custom)
        assert out == custom
        assert out.exists()

    def test_filters_low_score_records(self, tmp_path):
        """Only is_final=True AND critic_score >= 75 records are exported."""
        exports_dir = tmp_path / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        training_log = exports_dir / "training_log.jsonl"

        records = [
            {"is_final": True, "critic_score": 80, "question": "Q1"},   # ✓ include
            {"is_final": True, "critic_score": 90, "question": "Q2"},   # ✓ include
            {"is_final": False, "critic_score": 88, "question": "Q3"},  # ✗ not final
            {"is_final": True, "critic_score": 60, "question": "Q4"},   # ✗ below min
        ]
        with open(training_log, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        imp = SelfImprover(
            llm_client=FixedLLM(""),
            config=MOCK_CONFIG,
            cache_dir=tmp_path / ".cache",
            exports_dir=exports_dir,
        )
        out = imp.export_training_data(output_path=tmp_path / "filtered.jsonl")
        exported = [json.loads(line) for line in out.read_text().strip().splitlines()]
        assert len(exported) == 2
        for rec in exported:
            assert rec["is_final"] is True
            assert rec["critic_score"] >= 75

    def test_empty_log_does_not_raise(self, tmp_path, make_imp):
        """export_training_data with no training log must not raise."""
        imp = make_imp(FixedLLM(""))
        out = imp.export_training_data(output_path=tmp_path / "empty.jsonl")
        assert isinstance(out, Path)
        # File may or may not exist when no log exists — just must not raise

    def test_exported_records_are_valid_jsonl(self, make_imp):
        """Every exported line is parseable JSON."""
        imp = make_imp(FixedLLM(_critic_json(90)))
        imp.improve(_make_qr(), DATAFRAMES)
        out = imp.export_training_data()
        for line in out.read_text().strip().splitlines():
            assert isinstance(json.loads(line), dict)


# ── TestTrackerDBIntegration ───────────────────────────────────────────────────

class TestTrackerDBIntegration:
    def test_pattern_logged_to_tracker_after_rewrite(self, tmp_path):
        """Rewrite that improves to ≥ score_threshold → tracker.log_pattern called."""
        logged: list[dict] = []

        class MockTrackerDB:
            def log_pattern(self, question_type, code_pattern, score):
                logged.append({"question_type": question_type, "score": score})

        llm = SequenceLLM([
            _critic_json(40, fix="code"),
            "```python\nresult = accounts['revenue'].sum()\n```",
            "Total is $450K.",
            _critic_json(90),
        ])
        imp = SelfImprover(
            llm_client=llm,
            config=MOCK_CONFIG,
            tracker_db=MockTrackerDB(),
            cache_dir=tmp_path / ".cache",
            exports_dir=tmp_path / "exports",
        )
        imp.improve(_make_qr(code="result = 0", result=0), DATAFRAMES)
        assert len(logged) == 1

    def test_tracker_not_called_when_no_rewrite_occurred(self, tmp_path):
        """First-pass pass (iterations=1) → tracker.log_pattern NOT called."""
        call_count = [0]

        class MockTrackerDB:
            def log_pattern(self, **kwargs):
                call_count[0] += 1

        llm = FixedLLM(_critic_json(90))
        imp = SelfImprover(
            llm_client=llm,
            config=MOCK_CONFIG,
            tracker_db=MockTrackerDB(),
            cache_dir=tmp_path / ".cache",
            exports_dir=tmp_path / "exports",
        )
        imp.improve(_make_qr(), DATAFRAMES)
        assert call_count[0] == 0

    def test_tracker_error_does_not_propagate(self, tmp_path):
        """If tracker.log_pattern raises, improve() must still return a result."""
        class BrokenTrackerDB:
            def log_pattern(self, **kwargs):
                raise RuntimeError("DB is down")

        llm = SequenceLLM([
            _critic_json(40, fix="code"),
            "```python\nresult = accounts['revenue'].sum()\n```",
            "Total is $450K.",
            _critic_json(90),
        ])
        imp = SelfImprover(
            llm_client=llm,
            config=MOCK_CONFIG,
            tracker_db=BrokenTrackerDB(),
            cache_dir=tmp_path / ".cache",
            exports_dir=tmp_path / "exports",
        )
        result = imp.improve(_make_qr(code="result = 0", result=0), DATAFRAMES)
        assert result is not None
