"""
agenda/prompts.py — Pre-built weekly CRM agenda questions.

Organised into 6 sections matching the standing weekly meeting format.
Each question is designed to work against standard CRM DataFrames loaded
by the agent and can be triggered as a one-click button in the UI.

Public API
----------
AGENDA_QUESTIONS   : list[AgendaQuestion]   – all questions, ordered by section
AGENDA_CATEGORIES  : list[str]              – sorted unique section titles
SECTION_TITLES     : dict[int, str]         – section number → title

get_agenda_by_category(category: str)  -> list[AgendaQuestion]
get_section(section_num: int)          -> list[AgendaQuestion]
get_all_questions()                    -> list[str]
get_section_titles()                   -> list[str]  (ordered 1–6)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgendaQuestion:
    """A single pre-built agenda question.

    Parameters
    ----------
    section:       Section number (1–6).
    section_title: Human-readable section heading.
    question:      The question text sent to the query engine.
    chart_hint:    Preferred chart type (bar_pareto | line_ma | scatter |
                   pie | heatmap | funnel | horizontal_bar | None).
    intent_hint:   Suggested intent for the query engine (aggregation |
                   comparison | ranking | trend | pivot | recommendation |
                   what_if | data_quality | benchmark | None).
    """

    section: int
    section_title: str
    question: str
    chart_hint: str | None = None
    intent_hint: str | None = None

    @property
    def category(self) -> str:
        """Backward-compat alias — returns section_title."""
        return self.section_title


SECTION_TITLES: dict[int, str] = {
    1: "Account-Level Insights",
    2: "Opportunity Performance",
    3: "Strategic Questions",
    4: "Workstream Progress",
    5: "Data Quality & Gaps",
    6: "Actions & Priorities",
}

AGENDA_QUESTIONS: list[AgendaQuestion] = [

    # ── Section 1 — Account-Level Insights ──────────────────────────────────
    AgendaQuestion(
        section=1,
        section_title=SECTION_TITLES[1],
        question="Which accounts make up 80% of revenue?",
        chart_hint="bar_pareto",
        intent_hint="ranking",
    ),
    AgendaQuestion(
        section=1,
        section_title=SECTION_TITLES[1],
        question="Show profit concentration by account.",
        chart_hint="bar_pareto",
        intent_hint="ranking",
    ),
    AgendaQuestion(
        section=1,
        section_title=SECTION_TITLES[1],
        question="Which accounts are dormant — no activity in the last 90 days?",
        chart_hint=None,
        intent_hint="data_quality",
    ),
    AgendaQuestion(
        section=1,
        section_title=SECTION_TITLES[1],
        question=(
            "Compare client tenure vs opportunity volume — "
            "which longest-standing clients have the lowest opportunity volume?"
        ),
        chart_hint="scatter",
        intent_hint="comparison",
    ),

    # ── Section 2 — Opportunity Performance ─────────────────────────────────
    AgendaQuestion(
        section=2,
        section_title=SECTION_TITLES[2],
        question="What is the overall win rate broken down by service line?",
        chart_hint="bar_pareto",
        intent_hint="aggregation",
    ),
    AgendaQuestion(
        section=2,
        section_title=SECTION_TITLES[2],
        question=(
            "What is the win rate by deal size bracket "
            "(small under £100k, mid £100k–£500k, large over £500k)?"
        ),
        chart_hint="horizontal_bar",
        intent_hint="comparison",
    ),
    AgendaQuestion(
        section=2,
        section_title=SECTION_TITLES[2],
        question="What is the average sales cycle length in days by service line?",
        chart_hint="horizontal_bar",
        intent_hint="aggregation",
    ),
    AgendaQuestion(
        section=2,
        section_title=SECTION_TITLES[2],
        question=(
            "Which opportunities have been in the pipeline for 90 or more days "
            "without a stage progression?"
        ),
        chart_hint=None,
        intent_hint="data_quality",
    ),
    AgendaQuestion(
        section=2,
        section_title=SECTION_TITLES[2],
        question=(
            "Show the pipeline stage drop-off — "
            "at which stage is leakage highest?"
        ),
        chart_hint="funnel",
        intent_hint="trend",
    ),

    # ── Section 3 — Strategic Questions ─────────────────────────────────────
    AgendaQuestion(
        section=3,
        section_title=SECTION_TITLES[3],
        question=(
            "Which accounts do we win deals with despite having low engagement scores?"
        ),
        chart_hint="scatter",
        intent_hint="recommendation",
    ),
    AgendaQuestion(
        section=3,
        section_title=SECTION_TITLES[3],
        question=(
            "Which strategically large accounts have low opportunity volume or revenue — "
            "potential under-exploited relationships?"
        ),
        chart_hint="scatter",
        intent_hint="recommendation",
    ),
    AgendaQuestion(
        section=3,
        section_title=SECTION_TITLES[3],
        question=(
            "Which service lines have a high win rate but low pipeline volume — "
            "are we underselling?"
        ),
        chart_hint="scatter",
        intent_hint="recommendation",
    ),

    # ── Section 4 — Workstream Progress ─────────────────────────────────────
    AgendaQuestion(
        section=4,
        section_title=SECTION_TITLES[4],
        question=(
            "WS1: What is the CRM data quality score — "
            "show null rates, completeness, and field consistency across all tables."
        ),
        chart_hint="bar_pareto",
        intent_hint="data_quality",
    ),
    AgendaQuestion(
        section=4,
        section_title=SECTION_TITLES[4],
        question=(
            "WS2: Segment accounts by revenue, tenure, and activity level — "
            "show the cluster breakdown."
        ),
        chart_hint="scatter",
        intent_hint="aggregation",
    ),
    AgendaQuestion(
        section=4,
        section_title=SECTION_TITLES[4],
        question=(
            "WS3: What is the pipeline health score, "
            "weighted by age, deal value, and stage?"
        ),
        chart_hint="horizontal_bar",
        intent_hint="aggregation",
    ),
    AgendaQuestion(
        section=4,
        section_title=SECTION_TITLES[4],
        question=(
            "WS4: Which factors correlate most strongly with won deals — "
            "show win/loss drivers."
        ),
        chart_hint="heatmap",
        intent_hint="recommendation",
    ),
    AgendaQuestion(
        section=4,
        section_title=SECTION_TITLES[4],
        question=(
            "WS5: Show the cross-sell map — "
            "which accounts have revenue in one service line but not others?"
        ),
        chart_hint="heatmap",
        intent_hint="recommendation",
    ),

    # ── Section 5 — Data Quality & Gaps ─────────────────────────────────────
    AgendaQuestion(
        section=5,
        section_title=SECTION_TITLES[5],
        question="Which columns have the highest null rates across all tables?",
        chart_hint="bar_pareto",
        intent_hint="data_quality",
    ),
    AgendaQuestion(
        section=5,
        section_title=SECTION_TITLES[5],
        question="Which opportunities are missing close dates or deal values?",
        chart_hint=None,
        intent_hint="data_quality",
    ),
    AgendaQuestion(
        section=5,
        section_title=SECTION_TITLES[5],
        question="Are there duplicate accounts — show any detected duplicates.",
        chart_hint=None,
        intent_hint="data_quality",
    ),
    AgendaQuestion(
        section=5,
        section_title=SECTION_TITLES[5],
        question=(
            "What percentage of the pipeline has missing service line classifications — "
            "show unclassified opportunities."
        ),
        chart_hint="pie",
        intent_hint="data_quality",
    ),

    # ── Section 6 — Actions & Priorities ────────────────────────────────────
    AgendaQuestion(
        section=6,
        section_title=SECTION_TITLES[6],
        question=(
            "Auto-generate this session's flagged issues as a prioritised action item list."
        ),
        chart_hint=None,
        intent_hint="recommendation",
    ),
    AgendaQuestion(
        section=6,
        section_title=SECTION_TITLES[6],
        question=(
            "List all data fixes agreed in this session, "
            "with a responsible owner field for each item."
        ),
        chart_hint=None,
        intent_hint="recommendation",
    ),
]


def get_agenda_by_category(category: str) -> list[AgendaQuestion]:
    """Return all agenda questions matching *category* / section_title (case-insensitive)."""
    cat = category.lower()
    return [q for q in AGENDA_QUESTIONS if q.section_title.lower() == cat]


def get_section(section_num: int) -> list[AgendaQuestion]:
    """Return all questions belonging to *section_num* (1–6)."""
    return [q for q in AGENDA_QUESTIONS if q.section == section_num]


def get_all_questions() -> list[str]:
    """Return just the question strings for all agenda items, in section order."""
    return [q.question for q in AGENDA_QUESTIONS]


def get_section_titles() -> list[str]:
    """Return section titles in section-number order (1–6)."""
    return [SECTION_TITLES[i] for i in sorted(SECTION_TITLES)]


AGENDA_CATEGORIES: list[str] = sorted({q.section_title for q in AGENDA_QUESTIONS})
