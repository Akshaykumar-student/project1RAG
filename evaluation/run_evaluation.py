import argparse
import asyncio
import json
import os
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

from backend.config import Settings
from backend.rag import FALLBACK_ANSWER, RAGService

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Result:
    id: str
    question: str
    answerable: bool
    expected_file: str | None
    expected_answer: str
    answer: str
    cited_files: list[str]
    citation_correct: bool
    refused: bool
    latency_seconds: float
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    file_search_calls: int
    estimated_cost_usd: float
    warned_unsupported: bool = False
    unsupported_handled: bool = False
    manual_correct: bool | None = None
    manual_notes: str | None = None


INPUT_COST_PER_MILLION = 0.25
CACHED_INPUT_COST_PER_MILLION = 0.025
OUTPUT_COST_PER_MILLION = 2.00
FILE_SEARCH_COST_PER_CALL = 2.50 / 1000
OUTPUT_PATH = ROOT / "evaluation" / "latest_results.json"
UNSUPPORTED_WARNING_PATTERNS = (
    re.compile(
        r"not (?:confirmed|documented|covered|specified|stated) (?:by|in) "
        r"(?:the )?flowdesk (?:documentation|docs)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:flowdesk|the) (?:documentation|docs) (?:does not|doesn't|do not|don't) "
        r"(?:confirm|document|mention|cover|specify|state|indicate|say|list|show)",
        re.IGNORECASE,
    ),
    re.compile(
        r"i (?:do not|don't|couldn't|cannot|can't) (?:see|find).*"
        r"flowdesk (?:documentation|docs)",
        re.IGNORECASE | re.DOTALL,
    ),
)


def has_unsupported_warning(answer: str) -> bool:
    return any(pattern.search(answer) for pattern in UNSUPPORTED_WARNING_PATTERNS)


def handles_unsupported_safely(answer: str, cited_files: list[str]) -> bool:
    return not cited_files and has_unsupported_warning(answer)


def estimated_cost(usage: dict[str, int], file_search_calls: int) -> float:
    input_tokens = usage.get("input_tokens", 0)
    cached_tokens = min(usage.get("cached_input_tokens", 0), input_tokens)
    uncached_tokens = input_tokens - cached_tokens
    return round(
        (uncached_tokens * INPUT_COST_PER_MILLION / 1_000_000)
        + (cached_tokens * CACHED_INPUT_COST_PER_MILLION / 1_000_000)
        + (usage.get("output_tokens", 0) * OUTPUT_COST_PER_MILLION / 1_000_000)
        + (file_search_calls * FILE_SEARCH_COST_PER_CALL),
        6,
    )


def save_report(results: list[Result], complete: bool) -> None:
    report = {
        "complete": complete,
        "summary": summarize(results),
        "pricing": {
            "model": "gpt-5-mini",
            "input_per_million_tokens_usd": INPUT_COST_PER_MILLION,
            "cached_input_per_million_tokens_usd": CACHED_INPUT_COST_PER_MILLION,
            "output_per_million_tokens_usd": OUTPUT_COST_PER_MILLION,
            "file_search_per_1000_calls_usd": 2.50,
            "note": "Estimate excludes vector-store storage and taxes.",
        },
        "results": [asdict(result) for result in results],
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


async def evaluate(limit: int | None = None) -> list[Result]:
    load_dotenv(ROOT / ".env")
    settings = Settings()
    if not settings.configured:
        raise RuntimeError("Configure OPENAI_API_KEY and ingest the knowledge base first.")

    dataset = json.loads((ROOT / "evaluation" / "questions.json").read_text(encoding="utf-8"))
    if limit:
        dataset = dataset[:limit]
    service = RAGService(
        settings.openai_api_key,
        settings.vector_store_id,
        settings.openai_model,
    )
    results = []
    for index, item in enumerate(dataset, start=1):
        started = time.perf_counter()
        for attempt in range(1, 4):
            try:
                response = await service.ask(item["question"])
                break
            except Exception:
                if attempt == 3:
                    save_report(results, complete=False)
                    raise
                await asyncio.sleep(2**attempt)
        latency_seconds = round(time.perf_counter() - started, 3)
        cited_files = [citation.filename for citation in response.citations]
        refused = FALLBACK_ANSWER.lower() in response.answer.lower()
        warned_unsupported = has_unsupported_warning(response.answer)
        expected = item["expected_file"]
        usage = service.last_usage
        results.append(
            Result(
                id=item["id"],
                question=item["question"],
                answerable=item["answerable"],
                expected_file=expected,
                expected_answer=item["expected_answer"],
                answer=response.answer,
                cited_files=cited_files,
                citation_correct=expected in cited_files if expected else not cited_files,
                refused=refused,
                latency_seconds=latency_seconds,
                input_tokens=usage.get("input_tokens", 0),
                cached_input_tokens=usage.get("cached_input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                file_search_calls=service.last_file_search_calls,
                estimated_cost_usd=estimated_cost(usage, service.last_file_search_calls),
                warned_unsupported=warned_unsupported,
                unsupported_handled=(
                    not item["answerable"]
                    and handles_unsupported_safely(response.answer, cited_files)
                ),
            )
        )
        save_report(results, complete=False)
        print(f"[{index}/{len(dataset)}] {item['id']} ({latency_seconds:.2f}s)", flush=True)
    return results


def summarize(results: list[Result]) -> dict[str, float | int | None]:
    answerable = [result for result in results if result.answerable]
    unsupported = [result for result in results if not result.answerable]
    citation_accuracy = (
        sum(result.citation_correct for result in answerable) / len(answerable) if answerable else 0
    )
    unsupported_handling_accuracy = (
        sum(result.unsupported_handled for result in unsupported) / len(unsupported)
        if unsupported
        else 0
    )
    latencies = [result.latency_seconds for result in results]
    sorted_latencies = sorted(latencies)
    p95_index = max(0, min(len(sorted_latencies) - 1, int(len(sorted_latencies) * 0.95)))
    manually_reviewed = [result for result in results if result.manual_correct is not None]
    manual_review_complete = bool(results) and len(manually_reviewed) == len(results)
    manual_answer_accuracy = (
        round(sum(bool(result.manual_correct) for result in manually_reviewed) / len(results), 4)
        if manual_review_complete
        else None
    )
    return {
        "questions": len(results),
        "manually_reviewed_answers": len(manually_reviewed),
        "manual_review_complete": manual_review_complete,
        "manual_answer_accuracy": manual_answer_accuracy,
        "resume_answer_accuracy_ready": manual_review_complete,
        "citation_accuracy": round(citation_accuracy, 4),
        "unsupported_handling_accuracy": round(unsupported_handling_accuracy, 4),
        "unsupported_technical_fallbacks": sum(result.refused for result in unsupported),
        "average_latency_seconds": round(statistics.mean(latencies), 3) if latencies else None,
        "p95_latency_seconds": sorted_latencies[p95_index] if sorted_latencies else None,
        "input_tokens": sum(result.input_tokens for result in results),
        "cached_input_tokens": sum(result.cached_input_tokens for result in results),
        "output_tokens": sum(result.output_tokens for result in results),
        "file_search_calls": sum(result.file_search_calls for result in results),
        "estimated_cost_usd": round(sum(result.estimated_cost_usd for result in results), 6),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate FlowDesk grounding, citations, and unsupported-question handling."
    )
    parser.add_argument("--limit", type=int, help="Evaluate only the first N questions.")
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Apply current scoring rules to latest_results.json without API calls.",
    )
    args = parser.parse_args()
    if args.rescore:
        report = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        dataset = json.loads(
            (ROOT / "evaluation" / "questions.json").read_text(encoding="utf-8")
        )
        expected_answers = {item["id"]: item["expected_answer"] for item in dataset}
        results = []
        for item in report["results"]:
            item["expected_answer"] = expected_answers[item["id"]]
            item["warned_unsupported"] = has_unsupported_warning(item["answer"])
            item["unsupported_handled"] = bool(
                not item["answerable"]
                and handles_unsupported_safely(item["answer"], item["cited_files"])
            )
            item.setdefault("manual_correct", None)
            item.setdefault("manual_notes", None)
            results.append(
                Result(
                    **{
                        name: item[name]
                        for name in Result.__dataclass_fields__
                    }
                )
            )
        save_report(results, complete=bool(report.get("complete")))
        print(json.dumps(summarize(results), indent=2))
        print(f"Rescored existing results in {OUTPUT_PATH}")
        return
    results = await evaluate(args.limit)
    save_report(results, complete=True)
    print(json.dumps(summarize(results), indent=2))
    print(f"Saved detailed results to {OUTPUT_PATH}")


if __name__ == "__main__":
    if not os.getenv("PYTHONPATH"):
        os.environ["PYTHONPATH"] = str(ROOT)
    asyncio.run(main())
