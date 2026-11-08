"""Run the FULL pipeline (retrieval + Gemini) on a question file and measure it.

Usage:  python evaluate_answers.py [path to questions json]
Default: evaluation/test_questions.json

Automatic measures: correct refusals, false refusals, citation validity,
and whether a cited page is one of the acceptable pages.
By hand: open the CSV and fill the 'answer_correct' column (1 or 0).
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

from app.answer_generator import NOT_FOUND_MESSAGE
from app.retrieval import Retriever
from ask import SCORE_CUTOFF, TOP_K, DailyQuotaError, answer_with_retry

QUESTIONS_PATH = (
    Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evaluation/test_questions.json")
)
OUTPUT_CSV = Path("index") / f"answer_results_{QUESTIONS_PATH.stem}.csv"
PAUSE_SECONDS = int(os.getenv("PAUSE_SECONDS", "12"))  # raise it if you hit rate limits

FIELDS = [
    "question", "category", "answerable", "status", "layer", "top_score",
    "retrieved_pages", "answer", "found", "cited", "citations_ok",
    "cites_acceptable_page", "expected_answer", "answer_correct",
]


def run_one(retriever, question):
    """Run the same steps as ask.py and return a result dict."""
    top = retriever.dense(question, 1)
    top_score = top[0]["score"] if top else 0.0
    chunks = retriever.hybrid(question, k=TOP_K, depth=5)
    pages = [c["page"] for c in chunks]
    if top_score < SCORE_CUTOFF:
        return {"layer": "score_check", "top_score": top_score, "pages": pages,
                "answer": NOT_FOUND_MESSAGE, "found": False, "cited": [],
                "citations_ok": True}
    result = answer_with_retry(question, chunks)
    return {"layer": "llm", "top_score": top_score, "pages": pages, **result}


def summarize(rows):
    """Compute the automatic measures from finished rows (errors are excluded)."""
    ok = [r for r in rows if r["status"] == "ok"]
    answerable = [r for r in ok if r["answerable"]]
    unanswerable = [r for r in ok if not r["answerable"]]
    found = [r for r in answerable if r["found"]]
    return {
        "errors": len(rows) - len(ok),
        "answerable": len(answerable),
        "answered": len(found),
        "false_refusals": len(answerable) - len(found),
        "citations_ok": sum(1 for r in found if r["citations_ok"]),
        "cites_acceptable": sum(1 for r in found if r["cites_acceptable_page"]),
        "unanswerable": len(unanswerable),
        "correct_refusals": sum(1 for r in unanswerable if not r["found"]),
    }


def save(rows):
    OUTPUT_CSV.parent.mkdir(exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_finished_rows():
    """Reuse rows from an earlier run (keeps your hand-marked answer_correct values)."""
    finished = {}
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["status"] == "ok":
                    for key in ("answerable", "found", "citations_ok", "cites_acceptable_page"):
                        r[key] = r[key] == "True"
                    finished[r["question"]] = r
    return finished


def main():
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    finished = load_finished_rows()
    if finished:
        print(f"Resuming: {len(finished)} questions already done, they will be skipped.")

    print("Loading retrieval system...")
    retriever = Retriever()
    rows = []
    stopped = False

    for n, q in enumerate(questions, start=1):
        print(f"\n[{n}/{len(questions)}] {q['question']}", flush=True)
        if q["question"] in finished:
            rows.append(finished[q["question"]])
            print("  -> already done, skipped")
            continue
        row = {
            "question": q["question"], "category": q.get("category", ""),
            "answerable": q["answerable"], "expected_answer": q.get("expected_answer", ""),
            "answer_correct": "",
        }
        try:
            r = run_one(retriever, q["question"])
            acceptable = set(q.get("acceptable_pages", []))
            row.update({
                "status": "ok", "layer": r["layer"], "top_score": round(r["top_score"], 4),
                "retrieved_pages": r["pages"], "answer": r["answer"], "found": r["found"],
                "cited": r["cited"], "citations_ok": r["citations_ok"],
                "cites_acceptable_page": any(p in acceptable for _, p in r["cited"]),
            })
            print(f"  -> {'answered' if r['found'] else 'not found'} (via {r['layer']})")
            used_llm = r["layer"] == "llm"
        except DailyQuotaError:
            print("  -> DAILY QUOTA used up. Stopping. Run the script again later:")
            print("     finished questions are saved and will be skipped.")
            stopped = True
            break
        except RuntimeError as err:
            row.update({"status": f"error: {str(err)[:80]}", "found": "", "answer": ""})
            print(f"  -> ERROR, will retry on the next run: {str(err)[:80]}")
            used_llm = True
        rows.append(row)
        save(rows)  # save after every question so a crash loses nothing
        if used_llm and n < len(questions):
            time.sleep(PAUSE_SECONDS)

    save(rows)
    s = summarize(rows)
    print("\n" + "=" * 60)
    print("ANSWER EVALUATION (automatic measures)" + ("  [INCOMPLETE RUN]" if stopped else ""))
    print("=" * 60)
    print(f"Questions finished:             {s['answerable'] + s['unanswerable']}/{len(questions)}")
    print(f"Errors (excluded):              {s['errors']}")
    print(f"Answerable questions:           {s['answerable']}")
    print(f"  answered:                     {s['answered']}/{s['answerable']}")
    print(f"  false refusals:               {s['false_refusals']}/{s['answerable']}")
    print(f"  citations valid:              {s['citations_ok']}/{s['answered']}")
    print(f"  cites an acceptable page:     {s['cites_acceptable']}/{s['answered']}")
    print(f"Unanswerable questions:         {s['unanswerable']}")
    print(f"  correct refusals:             {s['correct_refusals']}/{s['unanswerable']}")
    print(f"\nNow open {OUTPUT_CSV} and fill 'answer_correct' (1 or 0) by hand.")


if __name__ == "__main__":
    main()