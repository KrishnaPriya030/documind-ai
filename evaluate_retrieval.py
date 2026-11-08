"""Compare dense, BM25 and hybrid retrieval: Recall@5 and MRR."""
import csv
import json
import sys
from pathlib import Path

from app.retrieval import Retriever

# Usage: python evaluate_retrieval.py [path to questions json]
QUESTIONS_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evaluation/questions.json")
OUTPUT_CSV = Path("index") / f"retrieval_results_{QUESTIONS_PATH.stem}.csv"
K = 5


def hit_rank(pages, acceptable, require_all=False):
    """Rank (1-based) of the first acceptable page, or None if there is no hit.

    If require_all is True (conflict questions) every acceptable page must
    appear, and the rank is that of the last one to appear.
    """
    if require_all:
        ranks = [pages.index(p) + 1 for p in acceptable if p in pages]
        return max(ranks) if len(ranks) == len(acceptable) else None
    for rank, page in enumerate(pages, start=1):
        if page in acceptable:
            return rank
    return None


def main():
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    answerable = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]

    print("Loading retriever...")
    r = Retriever()

    # Every method returns a list of result dicts. Hybrid variants are an
    # experiment: run exactly these, report all of them.
    methods = {
        "Dense": lambda q: r.dense(q, K),
        "BM25": lambda q: r.keyword(q, K),
        "Hybrid depth=5": lambda q: r.hybrid(q, K, depth=5),
        "Hybrid depth=20": lambda q: r.hybrid(q, K, depth=20),
        "Hybrid depth=20, dense x2": lambda q: r.hybrid(q, K, depth=20, w_dense=2.0),
    }

    rows = []
    summary = []
    for name, fn in methods.items():
        hits, rr_sum = 0, 0.0
        for q in answerable:
            pages = [item["page"] for item in fn(q["question"])]
            rank = hit_rank(pages, set(q["acceptable_pages"]), q.get("require_all", False))
            if rank:
                hits += 1
                rr_sum += 1 / rank
            rows.append({
                "method": name,
                "question": q["question"],
                "category": q.get("category", ""),
                "acceptable_pages": q["acceptable_pages"],
                "top5_pages": pages,
                "hit_rank": rank if rank else "none",
            })
        n = len(answerable)
        summary.append((name, hits, n, hits / n, rr_sum / n))

    print("\n" + "=" * 62)
    print(f"{'Method':<28}{'Hits':>8}{'Recall@5':>12}{'MRR':>10}")
    print("=" * 62)
    for name, hits, n, recall, mrr in summary:
        print(f"{name:<28}{f'{hits}/{n}':>8}{recall:>12.2%}{mrr:>10.3f}")
    print("=" * 62)
    print(f"Answerable questions: {len(answerable)}. One question = {1 / len(answerable):.0%}.")

    # Top-1 dense score for answerable and unanswerable questions: the data
    # for the "not found" threshold analysis.
    print("\nAnswerable questions: top-1 dense score and page")
    for q in answerable:
        top = r.dense(q["question"], 1)[0]
        print(f"  {top['score']:.4f} | page {top['page']} | {q['question']}")

    print("\nUnanswerable questions: top-1 dense score and page")
    for q in unanswerable:
        top = r.dense(q["question"], 1)[0]
        print(f"  {top['score']:.4f} | page {top['page']} | {q['question']}")

    OUTPUT_CSV.parent.mkdir(exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-question results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()