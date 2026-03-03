"""tests/test_agenda_prompts.py — Tests for agenda/prompts.py (Module 8)."""
from __future__ import annotations

import pytest

from agenda.prompts import (
    AGENDA_CATEGORIES,
    AGENDA_QUESTIONS,
    SECTION_TITLES,
    AgendaQuestion,
    get_agenda_by_category,
    get_all_questions,
    get_section,
    get_section_titles,
)

# ── valid value sets (mirrors Module 6 / 7 constants) ───────────────────────
_VALID_CHART_HINTS = {
    "bar_pareto", "line_ma", "scatter", "pie",
    "heatmap", "funnel", "horizontal_bar",
}
_VALID_INTENT_HINTS = {
    "aggregation", "comparison", "ranking", "trend",
    "pivot", "recommendation", "what_if", "data_quality", "benchmark",
}


# ─────────────────────────────────────────────────────────────────────────────
class TestAgendaQuestion:
    def test_fields_set_correctly(self):
        q = AgendaQuestion(
            section=1,
            section_title="Test Section",
            question="How many?",
            chart_hint="pie",
            intent_hint="aggregation",
        )
        assert q.section == 1
        assert q.section_title == "Test Section"
        assert q.question == "How many?"
        assert q.chart_hint == "pie"
        assert q.intent_hint == "aggregation"

    def test_chart_hint_defaults_to_none(self):
        q = AgendaQuestion(section=1, section_title="S", question="Q")
        assert q.chart_hint is None

    def test_intent_hint_defaults_to_none(self):
        q = AgendaQuestion(section=1, section_title="S", question="Q")
        assert q.intent_hint is None

    def test_category_property_returns_section_title(self):
        q = AgendaQuestion(section=2, section_title="Opportunity Performance", question="Q")
        assert q.category == "Opportunity Performance"

    def test_category_property_matches_section_title_exactly(self):
        q = AgendaQuestion(section=5, section_title="Data Quality & Gaps", question="Q")
        assert q.category == q.section_title


# ─────────────────────────────────────────────────────────────────────────────
class TestSectionTitles:
    def test_has_exactly_6_sections(self):
        assert len(SECTION_TITLES) == 6

    def test_keys_are_1_to_6(self):
        assert set(SECTION_TITLES.keys()) == {1, 2, 3, 4, 5, 6}

    def test_section_1_title(self):
        assert SECTION_TITLES[1] == "Account-Level Insights"

    def test_section_2_title(self):
        assert SECTION_TITLES[2] == "Opportunity Performance"

    def test_section_3_title(self):
        assert SECTION_TITLES[3] == "Strategic Questions"

    def test_section_4_title(self):
        assert SECTION_TITLES[4] == "Workstream Progress"

    def test_section_5_title(self):
        assert SECTION_TITLES[5] == "Data Quality & Gaps"

    def test_section_6_title(self):
        assert SECTION_TITLES[6] == "Actions & Priorities"


# ─────────────────────────────────────────────────────────────────────────────
class TestAgendaQuestions:
    def test_total_count_is_23(self):
        assert len(AGENDA_QUESTIONS) == 23

    def test_all_are_agenda_question_instances(self):
        for q in AGENDA_QUESTIONS:
            assert isinstance(q, AgendaQuestion)

    def test_all_have_section_numbers(self):
        for q in AGENDA_QUESTIONS:
            assert q.section is not None

    def test_section_numbers_in_range_1_to_6(self):
        for q in AGENDA_QUESTIONS:
            assert 1 <= q.section <= 6

    def test_all_have_non_empty_question(self):
        for q in AGENDA_QUESTIONS:
            assert isinstance(q.question, str)
            assert len(q.question.strip()) > 0

    def test_no_duplicate_questions(self):
        questions = [q.question for q in AGENDA_QUESTIONS]
        assert len(questions) == len(set(questions))

    def test_section_title_matches_section_titles_dict(self):
        for q in AGENDA_QUESTIONS:
            assert q.section_title == SECTION_TITLES[q.section]

    def test_ordered_by_section(self):
        sections = [q.section for q in AGENDA_QUESTIONS]
        assert sections == sorted(sections)


# ─────────────────────────────────────────────────────────────────────────────
class TestSectionCounts:
    def test_section_1_has_4_questions(self):
        assert len(get_section(1)) == 4

    def test_section_2_has_5_questions(self):
        assert len(get_section(2)) == 5

    def test_section_3_has_3_questions(self):
        assert len(get_section(3)) == 3

    def test_section_4_has_5_questions(self):
        assert len(get_section(4)) == 5

    def test_section_5_has_4_questions(self):
        assert len(get_section(5)) == 4

    def test_section_6_has_2_questions(self):
        assert len(get_section(6)) == 2

    def test_section_counts_sum_to_total(self):
        total = sum(len(get_section(s)) for s in range(1, 7))
        assert total == len(AGENDA_QUESTIONS)

    def test_invalid_section_returns_empty(self):
        assert get_section(0) == []
        assert get_section(7) == []
        assert get_section(99) == []

    def test_all_returned_items_have_correct_section(self):
        for s in range(1, 7):
            for q in get_section(s):
                assert q.section == s


# ─────────────────────────────────────────────────────────────────────────────
class TestGetAgendaByCategory:
    def test_returns_questions_for_valid_category(self):
        result = get_agenda_by_category("Account-Level Insights")
        assert len(result) == 4

    def test_case_insensitive_match(self):
        upper = get_agenda_by_category("ACCOUNT-LEVEL INSIGHTS")
        lower = get_agenda_by_category("account-level insights")
        mixed = get_agenda_by_category("Account-Level Insights")
        assert len(upper) == len(lower) == len(mixed) == 4

    def test_returns_empty_for_unknown_category(self):
        assert get_agenda_by_category("Nonexistent") == []

    def test_returns_list(self):
        result = get_agenda_by_category("Strategic Questions")
        assert isinstance(result, list)

    def test_all_returned_items_match_category(self):
        for q in get_agenda_by_category("Opportunity Performance"):
            assert q.section_title == "Opportunity Performance"

    def test_backward_compat_category_property(self):
        """category property on returned items must equal the queried section_title."""
        for q in get_agenda_by_category("Workstream Progress"):
            assert q.category == "Workstream Progress"


# ─────────────────────────────────────────────────────────────────────────────
class TestGetAllQuestions:
    def test_returns_list_of_strings(self):
        result = get_all_questions()
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)

    def test_count_matches_agenda_questions(self):
        assert len(get_all_questions()) == len(AGENDA_QUESTIONS)

    def test_all_non_empty(self):
        for q in get_all_questions():
            assert len(q.strip()) > 0

    def test_order_matches_agenda_questions(self):
        assert get_all_questions() == [q.question for q in AGENDA_QUESTIONS]


# ─────────────────────────────────────────────────────────────────────────────
class TestGetSectionTitles:
    def test_returns_6_titles(self):
        assert len(get_section_titles()) == 6

    def test_ordered_section_1_first(self):
        assert get_section_titles()[0] == "Account-Level Insights"

    def test_ordered_section_6_last(self):
        assert get_section_titles()[-1] == "Actions & Priorities"

    def test_matches_section_titles_values_in_order(self):
        expected = [SECTION_TITLES[i] for i in range(1, 7)]
        assert get_section_titles() == expected

    def test_no_duplicates(self):
        titles = get_section_titles()
        assert len(titles) == len(set(titles))


# ─────────────────────────────────────────────────────────────────────────────
class TestAgendaCategories:
    def test_is_sorted(self):
        assert AGENDA_CATEGORIES == sorted(AGENDA_CATEGORIES)

    def test_no_duplicates(self):
        assert len(AGENDA_CATEGORIES) == len(set(AGENDA_CATEGORIES))

    def test_count_is_6(self):
        assert len(AGENDA_CATEGORIES) == 6

    def test_contains_all_section_titles(self):
        for title in SECTION_TITLES.values():
            assert title in AGENDA_CATEGORIES


# ─────────────────────────────────────────────────────────────────────────────
class TestChartHints:
    def test_all_chart_hints_are_valid_or_none(self):
        for q in AGENDA_QUESTIONS:
            if q.chart_hint is not None:
                assert q.chart_hint in _VALID_CHART_HINTS, (
                    f"Unknown chart_hint '{q.chart_hint}' on: {q.question!r}"
                )

    def test_pareto_questions_use_bar_pareto(self):
        # Revenue 80% and profit concentration are Pareto charts
        pareto_qs = [q for q in AGENDA_QUESTIONS if q.chart_hint == "bar_pareto"]
        assert len(pareto_qs) >= 2
        texts = [q.question for q in pareto_qs]
        assert any("revenue" in t.lower() or "profit" in t.lower() for t in texts)

    def test_funnel_question_uses_funnel_hint(self):
        funnel_qs = [q for q in AGENDA_QUESTIONS if q.chart_hint == "funnel"]
        assert len(funnel_qs) >= 1
        assert any("stage" in q.question.lower() or "drop" in q.question.lower()
                   for q in funnel_qs)

    def test_heatmap_questions_use_heatmap_hint(self):
        heatmap_qs = [q for q in AGENDA_QUESTIONS if q.chart_hint == "heatmap"]
        assert len(heatmap_qs) >= 2

    def test_scatter_questions_exist(self):
        scatter_qs = [q for q in AGENDA_QUESTIONS if q.chart_hint == "scatter"]
        assert len(scatter_qs) >= 3

    def test_section_6_questions_have_no_chart_hint(self):
        for q in get_section(6):
            assert q.chart_hint is None, (
                f"Section 6 question should have no chart hint: {q.question!r}"
            )


# ─────────────────────────────────────────────────────────────────────────────
class TestIntentHints:
    def test_all_intent_hints_are_valid_or_none(self):
        for q in AGENDA_QUESTIONS:
            if q.intent_hint is not None:
                assert q.intent_hint in _VALID_INTENT_HINTS, (
                    f"Unknown intent_hint '{q.intent_hint}' on: {q.question!r}"
                )

    def test_section_5_questions_all_have_data_quality_intent(self):
        for q in get_section(5):
            assert q.intent_hint == "data_quality", (
                f"Section 5 question should have data_quality intent: {q.question!r}"
            )

    def test_section_6_questions_have_recommendation_intent(self):
        for q in get_section(6):
            assert q.intent_hint == "recommendation"

    def test_strategic_section_has_recommendation_intents(self):
        for q in get_section(3):
            assert q.intent_hint == "recommendation"

    def test_ranking_intent_on_pareto_questions(self):
        ranking_qs = [q for q in AGENDA_QUESTIONS if q.intent_hint == "ranking"]
        assert len(ranking_qs) >= 1


# ─────────────────────────────────────────────────────────────────────────────
class TestContent:
    def test_dormant_account_question_exists(self):
        all_q = get_all_questions()
        assert any("dormant" in q.lower() for q in all_q)

    def test_90_days_dormant_question_in_section_1(self):
        qs = get_section(1)
        assert any("90" in q.question for q in qs)

    def test_win_rate_question_exists(self):
        all_q = get_all_questions()
        assert any("win rate" in q.lower() for q in all_q)

    def test_win_rate_by_service_line_in_section_2(self):
        qs = get_section(2)
        assert any("service line" in q.question.lower() and "win rate" in q.question.lower()
                   for q in qs)

    def test_pareto_revenue_question_in_section_1(self):
        qs = get_section(1)
        assert any("80%" in q.question or "revenue" in q.question.lower() for q in qs)

    def test_pipeline_funnel_question_in_section_2(self):
        qs = get_section(2)
        assert any("funnel" == q.chart_hint for q in qs)

    def test_workstream_ws1_present(self):
        qs = get_section(4)
        assert any("WS1" in q.question for q in qs)

    def test_workstream_ws2_present(self):
        qs = get_section(4)
        assert any("WS2" in q.question for q in qs)

    def test_workstream_ws3_present(self):
        qs = get_section(4)
        assert any("WS3" in q.question for q in qs)

    def test_workstream_ws4_present(self):
        qs = get_section(4)
        assert any("WS4" in q.question for q in qs)

    def test_workstream_ws5_present(self):
        qs = get_section(4)
        assert any("WS5" in q.question for q in qs)

    def test_section_6_action_items_present(self):
        qs = get_section(6)
        texts = [q.question.lower() for q in qs]
        assert any("action" in t or "flagged" in t for t in texts)

    def test_section_6_data_fixes_present(self):
        qs = get_section(6)
        texts = [q.question.lower() for q in qs]
        assert any("data fix" in t or "owner" in t for t in texts)

    def test_duplicate_detection_question_in_section_5(self):
        qs = get_section(5)
        assert any("duplicate" in q.question.lower() for q in qs)

    def test_null_rates_question_in_section_5(self):
        qs = get_section(5)
        assert any("null" in q.question.lower() for q in qs)

    def test_cross_sell_question_in_section_4(self):
        qs = get_section(4)
        assert any("cross-sell" in q.question.lower() for q in qs)

    def test_tenure_vs_opportunity_question_in_section_1(self):
        qs = get_section(1)
        assert any("tenure" in q.question.lower() for q in qs)

    def test_deal_size_bracket_question_in_section_2(self):
        qs = get_section(2)
        assert any("bracket" in q.question.lower() or "£100k" in q.question for q in qs)
