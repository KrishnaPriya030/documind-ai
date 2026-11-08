"""Ask DocuMind AI a question.

Usage:  python ask.py "How many supplementary exams can a student take?"
        python ask.py            (it will prompt for a question)
"""
import re
import sys
import time

from app.answer_generator import NOT_FOUND_MESSAGE, answer_question
from app.retrieval import Retriever

SCORE_CUTOFF = 0.40  # layer 1 of "not found": provisional, from the tuning set
TOP_K = 5


class DailyQuotaError(RuntimeError):
    """The daily quota is used up: retrying today will not help."""


def retry_delay_seconds(message):
    """Read the wait time Google suggests, e.g. 'retry in 34.5s' or retryDelay '34s'."""
    match = re.search(r"retry(?:Delay|\s+in)\D{0,6}(\d+(?:\.\d+)?)\s*s", message, re.I)
    return float(match.group(1)) if match else None


def answer_with_retry(question, chunks, max_attempts=4):
    """Retry temporary errors (503, 429, 504). Never retry real bugs or daily limits."""
    for attempt in range(1, max_attempts + 1):
        try:
            return answer_question(question, chunks)
        except RuntimeError as err:
            text = str(err)
            if "PerDay" in text:
                raise DailyQuotaError(text) from err
            temporary = any(code in text for code in ("503", "429", "504"))
            if not temporary or attempt == max_attempts:
                raise
            wait = min((retry_delay_seconds(text) or 5 * 3 ** (attempt - 1)) + 1, 120)
            print(f"Temporary error. Retrying in {wait:.0f}s...", flush=True)
            time.sleep(wait)


def show_passages(chunks):
    print("\nRETRIEVED EVIDENCE")
    print("-" * 60)
    for i, c in enumerate(chunks, start=1):
        print(f"[{i}] {c['doc']} | PDF page {c['page']}")
        print(f"    {c['text'][:200]}...")


def main():
    question = " ".join(sys.argv[1:]).strip() or input("Enter your question: ").strip()
    if not question:
        print("Please enter a question.")
        return

    print("Loading retrieval system...")
    retriever = Retriever()

    # Layer 1: if even the best semantic match is very weak, skip the LLM.
    top = retriever.dense(question, 1)
    top_score = top[0]["score"] if top else 0.0
    chunks = retriever.hybrid(question, k=TOP_K, depth=5)

    if top_score < SCORE_CUTOFF:
        print(f"\nTop similarity {top_score:.2f} is below {SCORE_CUTOFF}.")
        print(f"\nANSWER: {NOT_FOUND_MESSAGE}")
        return

    show_passages(chunks)

    # Layer 2: the prompt tells the model to refuse when evidence is missing.
    try:
        result = answer_with_retry(question, chunks)
    except DailyQuotaError:
        print("\nThe daily quota for this model is used up. Try again later or use another model.")
        print("The retrieved evidence above is still valid.")
        return
    except RuntimeError as err:
        print(f"\nThe answer service failed: {err}")
        print("The retrieved evidence above is still valid.")
        return

    print("\nANSWER")
    print("-" * 60)
    print(result["answer"])

    if not result["found"]:
        return
    if result["citations_ok"]:
        print("\nCitations checked: all point to supplied sources.")
        for doc, page in result["cited"]:
            for c in chunks:
                if c["doc"] == doc and c["page"] == page:
                    print(f"\nSource passage ({doc}, page {page}):\n{c['text']}")
                    break
    else:
        print("\nWARNING: citation check failed.")
        print(f"  Cited: {result['cited']}")
        print(f"  Not in supplied sources: {result['invalid_citations']}")


if __name__ == "__main__":
    main()