"""Evaluate DocuMind AI retrieval against a small labelled question set.

This script is local-only: it does not call Gemini or any external API.
Run it from the project root with the virtual environment activated:
    python evaluate_local_retrieval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from app.vector_store import VectorStore

DATASET_PATH = Path("evaluation_questions.json")
TOP_K = 5


def load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path.resolve()}\n"
            "Place evaluation_questions.json in your project root."
        )
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise ValueError("Evaluation dataset must be a non-empty JSON list.")
    for row in data:
        if not isinstance(row, dict) or not row.get("question"):
            raise ValueError("Every dataset item needs a question.")
        if "answerable_from_handbook" not in row or "expected_pages" not in row:
            raise ValueError(
                "Every item needs answerable_from_handbook and expected_pages."
            )
    return data


def main() -> int:
    try:
        dataset = load_dataset(DATASET_PATH)
        store = VectorStore()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    answerable_rows = [r for r in dataset if r["answerable_from_handbook"]]
    answerable_with_pages = [r for r in answerable_rows if r["expected_pages"]]
    single_page_hits = 0
    required_page_total = 0
    required_page_hits = 0
    all_required_found = 0

    print(f"Dataset: {DATASET_PATH.resolve()}")
    print(f"Top K: {TOP_K} | Questions: {len(dataset)}")
    print("Note: unanswerable questions are shown for inspection only; retrieval scores cannot prove refusal correctness.\n")

    for row in dataset:
        question = row["question"]
        expected_pages = {int(p) for p in row["expected_pages"]}
        results = store.search(question, k=TOP_K)
        returned_pages = [int(item["page"]) for item in results if item.get("page") is not None]
        page_set = set(returned_pages)

        print("=" * 78)
        print(f"[{row['id']}] {question}")
        print(
            "Label: "
            + ("ANSWERABLE" if row["answerable_from_handbook"] else "UNANSWERABLE")
            + f" | Expected pages: {sorted(expected_pages) if expected_pages else 'none'}"
        )
        print(f"Returned pages (top {TOP_K}): {returned_pages}")

        if row["answerable_from_handbook"] and expected_pages:
            found = expected_pages & page_set
            required_page_hits += len(found)
            required_page_total += len(expected_pages)
            all_found = expected_pages.issubset(page_set)
            if all_found:
                all_required_found += 1
            if len(expected_pages) == 1 and found:
                single_page_hits += 1
            print(f"Expected pages found: {sorted(found)}")
            print(f"All expected pages found: {'PASS' if all_found else 'MISS'}")
        else:
            print("Retrieval alone does not determine whether the final answer should refuse.")

        for rank, item in enumerate(results, start=1):
            snippet = " ".join(str(item.get("text", "")).split())[:190]
            print(f"  Rank {rank} | page {item.get('page')} | method {item.get('retrieval_method')} | {snippet}")

    print("\n" + "=" * 78)
    print("RETRIEVAL SUMMARY")
    if answerable_with_pages:
        print(
            "Single-page question Hit@5: "
            f"{single_page_hits}/{sum(len(r['expected_pages']) == 1 for r in answerable_with_pages)}"
        )
    if required_page_total:
        print(
            f"Expected-page recall@{TOP_K}: "
            f"{required_page_hits}/{required_page_total} "
            f"({required_page_hits / required_page_total:.1%})"
        )
        print(
            f"Questions with ALL expected pages in top {TOP_K}: "
            f"{all_required_found}/{len(answerable_with_pages)}"
        )
    print("\nThis is a small hand-labelled smoke test, not a statistically reliable benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
