from app.vector_store import VectorStore


QUESTIONS = [
    "What is the vision and mission of the University of Kerala?",
    "How many schools and departments does the University of Kerala have?",
    "Which law governs the University of Kerala?",
    "What accreditation and ranking information is provided about the university?",
    "What information is provided about the Department of Computer Science?"
]

# This question is intentionally outside the handbook's subject matter.
UNKNOWN_QUESTION = (
    "What was the exact stock market closing price of Apple "
    "on 15 January 2025?"
)


def display_results(store, question, k=3):
    print("\n" + "=" * 90)
    print("QUESTION:", question)
    print("=" * 90)

    results = store.search(question, k=k)

    for rank, result in enumerate(results, start=1):
        print(f"\nResult {rank}")
        print(f"Document: {result['doc']}")
        print(f"PDF page: {result['page']}")
        print(f"Similarity score: {result['score']:.4f}")
        print(f"Text preview: {result['text'][:150]}")

    return results


def main():
    store = VectorStore()

    # Test the five answerable questions.
    for question in QUESTIONS:
        display_results(store, question, k=3)

    # Test a question the handbook is not expected to answer.
    display_results(store, UNKNOWN_QUESTION, k=3)

    print("\nAll retrieval tests completed.")


if __name__ == "__main__":
    main()