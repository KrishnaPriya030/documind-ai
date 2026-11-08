"""Evaluate DocuMind AI answer quality on a labelled handbook question set.

Default mode is safe/offline: retrieval is measured for every question, while
answer generation is evaluated only for deterministic local answer/refusal paths.
It does not call Gemini unless --use-gemini is explicitly passed.

Run from the project root:
    python evaluate_answer_quality.py
    python evaluate_answer_quality.py --use-gemini  # may consume API quota

The results CSV is written to evaluation_results/answer_quality_results.csv.
This is a small hand-labelled smoke test, not a statistically reliable benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

from app.answer_generator import NOT_FOUND_MESSAGE, answer_question
from app.vector_store import VectorStore

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "answer_quality_cases.json"
DEFAULT_OUTPUT = ROOT / "evaluation_results" / "answer_quality_results.csv"


def normalize(text: str) -> str:
    """Normalize whitespace and lowercase text for simple expected-term checks."""
    return re.sub(r"\s+", " ", text).strip().lower()


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        cases = json.load(file)
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"Evaluation dataset must be a non-empty JSON list: {path}")

    required = {"id", "question", "answerable_from_handbook", "expected_pages"}
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"Case {index} must be a JSON object.")
        missing = required - case.keys()
        if missing:
            raise ValueError(f"Case {index} is missing fields: {sorted(missing)}")
    return cases


def check_expected_terms(answer: str, required_terms: list[str]) -> tuple[bool | None, list[str]]:
    """Return a simple lexical fact check; this is not full semantic grading."""
    if not required_terms:
        return None, []
    normalized_answer = normalize(answer)
    missing = [term for term in required_terms if normalize(term) not in normalized_answer]
    return not missing, missing


def evaluate_case(
    store: VectorStore,
    case: dict[str, Any],
    top_k: int,
    use_gemini: bool,
) -> dict[str, Any]:
    question = str(case["question"]).strip()
    answerable_expected = bool(case["answerable_from_handbook"])
    expected_pages = [int(page) for page in case.get("expected_pages", [])]
    required_citation_pages = [
        int(page) for page in case.get("required_citation_pages", [])
    ]

    retrieved = store.search(question, k=top_k)
    retrieved_pages = [int(item["page"]) for item in retrieved if item.get("page") is not None]
    retrieved_page_set = set(retrieved_pages)
    expected_pages_found = [p for p in expected_pages if p in retrieved_page_set]
    retrieval_recall = (
        len(expected_pages_found) / len(expected_pages) if expected_pages else None
    )
    all_expected_pages_found = (
        len(expected_pages_found) == len(expected_pages) if expected_pages else None
    )

    # Without --use-gemini, check only known local answer/refusal routes. Normal
    # answerable questions are skipped for generation so no API call happens.
    deterministic_local_case = (
        not answerable_expected
        or case.get("id") == "school_count_conflict"
        or not retrieved
    )
    should_generate = use_gemini or deterministic_local_case

    row: dict[str, Any] = {
        "id": case["id"],
        "question": question,
        "answerable_expected": answerable_expected,
        "retrieved_pages": ",".join(map(str, retrieved_pages)),
        "expected_pages": ",".join(map(str, expected_pages)),
        "expected_pages_found": ",".join(map(str, expected_pages_found)),
        "retrieval_recall_at_k": "" if retrieval_recall is None else round(retrieval_recall, 4),
        "all_expected_pages_found": "" if all_expected_pages_found is None else all_expected_pages_found,
        "answer_evaluated": should_generate,
        "answer_found": "",
        "answerability_correct": "",
        "expected_terms_match": "",
        "missing_expected_terms": "",
        "citations_valid": "",
        "cited_pages": "",
        "required_citation_pages_found": "",
        "answer": "",
        "error": "",
    }

    if not should_generate:
        row["answer"] = "SKIPPED (offline mode: no Gemini call made)"
        return row

    try:
        result = answer_question(question, retrieved)
    except Exception as exc:  # keep partial results if the API is unavailable
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["answer"] = "ERROR"
        return row

    answer = str(result.get("answer", ""))
    found = bool(result.get("found"))
    citations_valid = bool(result.get("citations_ok"))
    cited = result.get("cited", [])
    cited_pages = sorted({int(page) for _doc, page in cited})

    row["answer_found"] = found
    row["answerability_correct"] = found == answerable_expected
    row["citations_valid"] = citations_valid
    row["cited_pages"] = ",".join(map(str, cited_pages))
    row["answer"] = answer

    expected_terms = case.get("expected_answer_terms", [])
    terms_match, missing_terms = check_expected_terms(answer, expected_terms)
    row["expected_terms_match"] = "" if terms_match is None else terms_match
    row["missing_expected_terms"] = "; ".join(missing_terms)

    # Citation-page coverage is assessed only for an answer (refusals do not need
    # citations). For conflicting evidence, all required pages must be cited.
    if found and required_citation_pages:
        missing_citation_pages = [p for p in required_citation_pages if p not in cited_pages]
        row["required_citation_pages_found"] = not missing_citation_pages
        if missing_citation_pages:
            row["error"] = (
                f"Missing required citation page(s): {', '.join(map(str, missing_citation_pages))}"
            )
    elif not found:
        row["required_citation_pages_found"] = True
    else:
        row["required_citation_pages_found"] = ""

    return row


def safe_ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator}/{denominator} ({numerator / denominator:.1%})"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id", "question", "answerable_expected", "retrieved_pages", "expected_pages",
        "expected_pages_found", "retrieval_recall_at_k", "all_expected_pages_found",
        "answer_evaluated", "answer_found", "answerability_correct",
        "expected_terms_match", "missing_expected_terms", "citations_valid",
        "cited_pages", "required_citation_pages_found", "answer", "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]], output_path: Path, use_gemini: bool) -> None:
    retrieval_rows = [row for row in rows if row.get("expected_pages")]
    retrieval_rows_with_recall = [row for row in retrieval_rows if row.get("retrieval_recall_at_k") != ""]
    all_pages_pass = sum(row.get("all_expected_pages_found") is True or row.get("all_expected_pages_found") == "True" for row in retrieval_rows)

    evaluated = [row for row in rows if row.get("answer_evaluated") is True or row.get("answer_evaluated") == "True"]
    answerability_checked = [row for row in evaluated if row.get("answerability_correct") is not None and row.get("answerability_correct") != ""]
    answerability_pass = sum(row.get("answerability_correct") is True or row.get("answerability_correct") == "True" for row in answerability_checked)
    term_checked = [row for row in evaluated if row.get("expected_terms_match") is not None and row.get("expected_terms_match") != ""]
    term_pass = sum(row.get("expected_terms_match") is True or row.get("expected_terms_match") == "True" for row in term_checked)
    citation_checked = [row for row in evaluated if row.get("citations_valid") is not None and row.get("citations_valid") != ""]
    citation_pass = sum(row.get("citations_valid") is True or row.get("citations_valid") == "True" for row in citation_checked)
    required_cite_checked = [row for row in evaluated if row.get("required_citation_pages_found") is not None and row.get("required_citation_pages_found") != ""]
    required_cite_pass = sum(row.get("required_citation_pages_found") is True or row.get("required_citation_pages_found") == "True" for row in required_cite_checked)

    print("\n" + "=" * 72)
    print("ANSWER-QUALITY EVALUATION SUMMARY")
    print("Mode:", "Gemini enabled (may consume API quota)" if use_gemini else "Offline safe mode (no Gemini calls)")
    print(f"Cases processed: {len(rows)}")
    print(f"Retrieval: all expected pages found for {all_pages_pass}/{len(retrieval_rows)} cases with expected pages")
    if retrieval_rows_with_recall:
        recall_sum = sum(float(row["retrieval_recall_at_k"]) for row in retrieval_rows_with_recall)
        print(f"Mean expected-page Recall@k: {recall_sum / len(retrieval_rows_with_recall):.1%}")
    print(f"Answer generation evaluated: {len(evaluated)}/{len(rows)}")
    print("Answerability decision accuracy:", safe_ratio(answerability_pass, len(answerability_checked)))
    print("Expected-answer-term match:", safe_ratio(term_pass, len(term_checked)))
    print("Citation validity:", safe_ratio(citation_pass, len(citation_checked)))
    print("Required citation-page coverage:", safe_ratio(required_cite_pass, len(required_cite_checked)))
    errors = [row for row in rows if row.get("error")]
    print(f"Cases with errors: {len(errors)}")
    if errors:
        for row in errors:
            print(f"  ERROR {row['id']}: {row['error']}")
    skipped = [row for row in rows if not (row.get("answer_evaluated") is True or row.get("answer_evaluated") == "True")]
    if skipped:
        print("Answer generation skipped (no API calls):", ", ".join(row["id"] for row in skipped))
    print("CSV report:", output_path)
    print("Note: expected-term matching is a simple lexical check, not a full semantic or hallucination evaluation.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to answer_quality_cases.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CSV report path")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks to inspect (default: 5)")
    parser.add_argument(
        "--use-gemini",
        action="store_true",
        help="Generate answers for ordinary answerable questions. This can consume Gemini API quota.",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    try:
        cases = load_cases(args.dataset)
        store = VectorStore()
    except Exception as exc:
        print(f"Could not initialize evaluation: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    for case in cases:
        print(f"\n[{case['id']}] {case['question']}")
        try:
            row = evaluate_case(store, case, args.top_k, args.use_gemini)
        except Exception as exc:
            row = {
                "id": case.get("id", "unknown"),
                "question": case.get("question", ""),
                "answerable_expected": case.get("answerable_from_handbook", ""),
                "answer_evaluated": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        pages = row.get("retrieved_pages", "")
        expected_found = row.get("expected_pages_found", "")
        print(f"  Retrieved pages: {pages or 'none'}")
        if row.get("expected_pages"):
            print(f"  Expected pages found: {expected_found} / expected {row['expected_pages']}")
        if row.get("answer_evaluated") is True:
            print(f"  Answer: {row.get('answer', '')}")
            print(f"  Answerability correct: {row.get('answerability_correct', '')}")
            print(f"  Citations valid: {row.get('citations_valid', '')}")
        elif row.get("answer_evaluated") == "True":
            print(f"  Answer: {row.get('answer', '')}")
            print(f"  Answerability correct: {row.get('answerability_correct', '')}")
            print(f"  Citations valid: {row.get('citations_valid', '')}")
        else:
            print("  Answer generation skipped (safe offline mode).")

        # With --use-gemini, stop after the first generation failure to avoid
        # repeatedly hitting a quota/error condition for every remaining case.
        if args.use_gemini and row.get("error") and "Gemini answer generation failed" in row["error"]:
            print("Stopping after first Gemini generation error to avoid repeated failed requests.")
            break

    write_csv(args.output, rows)
    print_summary(rows, args.output, args.use_gemini)

    has_errors = any(row.get("error") for row in rows)
    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
