
from app.vector_store import VectorStore, tokenize

store = VectorStore()

queries = [
    "registrar email",
    "registrar email addresses",
    "registrar@keralauniversity.ac.in",
    "University Fact Sheet registrar email addresses",
]

for query in queries:
    print("\n" + "=" * 80)
    print("QUERY:", query)
    print("=" * 80)

    scores = store.bm25.get_scores(tokenize(query))

    # Show the 10 highest-ranked chunks.
    top_rows = scores.argsort()[::-1][:10]

    for rank, row in enumerate(top_rows, start=1):
        chunk = store.chunks[int(row)]

        print(
            f"\nRank: {rank} | "
            f"Chunk: {row} | "
            f"Page: {chunk['page']} | "
            f"BM25: {scores[row]:.4f}"
        )
        print(chunk["text"][:350])

    # Check the known Fact Sheet chunk directly.
    target_chunk = store.chunks[99]
    target_score = scores[99]

    print("\nFACT SHEET CHUNK")
    print("Chunk index: 99")
    print("Page:", target_chunk["page"])
    print("BM25 score:", round(float(target_score), 4))
