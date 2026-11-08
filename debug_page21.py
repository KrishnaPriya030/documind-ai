from app.vector_store import VectorStore


store = VectorStore()

questions = [
    "How many schools does the University of Kerala have?",
    "What is the number of schools listed in the University Fact Sheet?",
    "What registrar email addresses are listed in the University fact sheet?",
    "registrar@keralauniversity.ac.in",
]

# Find all chunks from physical PDF page 21.
target_chunks = [
    (i, chunk)
    for i, chunk in enumerate(store.chunks)
    if chunk.get("page") == 21
]

print(f"\nTotal chunks on physical PDF page 21: {len(target_chunks)}")

for question in questions:
    print("\n" + "=" * 80)
    print("QUESTION:", question)
    print("=" * 80)

    dense_results = store.dense_search(
        question,
        k=len(store.chunks),
    )

    bm25_results = store.bm25_search(
        question,
        k=len(store.chunks),
    )

    hybrid_results = store.hybrid_search(
        question,
        k=10,
        candidate_k=100,
    )

    for method_name, results in [
        ("DENSE", dense_results),
        ("BM25", bm25_results),
    ]:
        ranks = {
            result["chunk_id"]: rank
            for rank, result in enumerate(results, start=1)
        }

        print(f"\n{method_name} — PAGE 21 CHUNKS")

        for chunk_id, chunk in target_chunks:
            rank = ranks.get(chunk_id)

            result = next(
                (
                    item
                    for item in results
                    if item["chunk_id"] == chunk_id
                ),
                None,
            )

            score = result["score"] if result else None

            print(
                f"Chunk {chunk_id} | Rank: {rank} | "
                f"Score: {score}"
            )
            print(chunk["text"][:350])
            print("-" * 40)

    print("\nHYBRID — TOP 10 RESULTS")

    for rank, result in enumerate(hybrid_results, start=1):
        print(
            f"Rank {rank} | Page {result['page']} | "
            f"Chunk {result['chunk_id']} | "
            f"RRF {result['rrf_score']:.6f} | "
            f"Dense rank {result['dense_rank']} | "
            f"BM25 rank {result['bm25_rank']}"
        )
        print(result["text"][:200])
        print("-" * 40)