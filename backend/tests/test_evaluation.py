import json
from pathlib import Path

from evaluation.run_evaluation import (
    Result,
    handles_unsupported_safely,
    has_unsupported_warning,
    summarize,
)

ROOT = Path(__file__).resolve().parents[2]


def test_explicit_documentation_warning_passes_without_citations():
    answer = "This detail is not confirmed by the FlowDesk documentation."

    assert has_unsupported_warning(answer) is True
    assert handles_unsupported_safely(answer, []) is True


def test_warning_with_false_file_citation_fails():
    answer = "I don't see this feature in the FlowDesk documentation."

    assert handles_unsupported_safely(answer, ["01-getting-started.md"]) is False


def test_generic_guess_does_not_count_as_safe_unsupported_handling():
    answer = "Most SaaS products probably support this feature."

    assert has_unsupported_warning(answer) is False
    assert handles_unsupported_safely(answer, []) is False


def test_technical_fallback_does_not_count_as_product_warning():
    answer = "I couldn't generate an answer. Please try asking again."

    assert handles_unsupported_safely(answer, []) is False


def test_every_evaluation_case_has_a_reviewed_expected_answer():
    dataset = json.loads((ROOT / "evaluation" / "questions.json").read_text(encoding="utf-8"))

    assert len(dataset) == 40
    assert all(item["expected_answer"].strip() for item in dataset)


def _result(identifier: str, manual_correct: bool | None) -> Result:
    return Result(
        id=identifier,
        question="Question",
        answerable=True,
        expected_file="article.md",
        expected_answer="Expected answer",
        answer="Actual answer",
        cited_files=["article.md"],
        citation_correct=True,
        refused=False,
        latency_seconds=1.0,
        input_tokens=1,
        cached_input_tokens=0,
        output_tokens=1,
        file_search_calls=1,
        estimated_cost_usd=0.01,
        manual_correct=manual_correct,
    )


def test_resume_answer_accuracy_is_hidden_until_every_answer_is_reviewed():
    summary = summarize([_result("one", True), _result("two", None)])

    assert summary["manually_reviewed_answers"] == 1
    assert summary["manual_review_complete"] is False
    assert summary["manual_answer_accuracy"] is None
    assert summary["resume_answer_accuracy_ready"] is False


def test_completed_manual_review_produces_resume_ready_accuracy():
    summary = summarize([_result("one", True), _result("two", False)])

    assert summary["manual_review_complete"] is True
    assert summary["manual_answer_accuracy"] == 0.5
    assert summary["resume_answer_accuracy_ready"] is True
