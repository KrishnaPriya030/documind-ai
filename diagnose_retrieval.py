from app.vector_store import VectorStore

store = VectorStore()

question = "How many departments are listed in the University Fact Sheet?"

for method_name in ("dense_search", "bm25_search", "hybrid_search"):
    method = getattr(store, method_name)
    results = method(question, k=10)

    print(f"\n{'=' * 65}")
    print(f"{method_name.upper()}")
    print("=" * 65)

    for rank, chunk in enumerate(results, start=1):
        print(
            f"Rank {rank} | Page {chunk['page']} | "
            f"Score {chunk['score']:.4f}"
        )
        print(chunk["text"][:250].replace("\n", " "))