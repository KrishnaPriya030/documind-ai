"""Smoke-test the dense, BM25, and hybrid retrievers on key handbook questions."""
from app.vector_store import VectorStore

QUESTIONS = [
    "How many schools does the University of Kerala have?",
    "How many schools are listed in the University Fact Sheet?",
    "What registrar email addresses are listed in the University fact sheet?",
    "What is the Registrar's email address?",
    "registrar@keralauniversity.ac.in",
]

def show_results(method_name, results):
    print(f"\n--- {method_name} ---")
    for rank, item in enumerate(results, start=1):
        print(
            f"Rank {rank:>2} | page {item['page']:>3} | "
            f"chunk {item.get('chunk_id')} | score={item['score']:.6f}"
        )
        if method_name == "HYBRID":
            print(
                f"          RRF={item['rrf_score']:.6f} | "
                f"dense_rank={item.get('dense_rank')} | "
                f"BM25_rank={item.get('bm25_rank')} | "
                f"cosine={item.get('semantic_score')} | "
                f"BM25_score={item.get('keyword_score')}"
            )
        print(f"          {item['text'][:280].replace(chr(10), ' ')}")

def main():
    store = VectorStore()
    for question in QUESTIONS:
        print("\n" + "=" * 100)
        print("QUESTION:", question)
        show_results("DENSE", store.dense_search(question, k=10))
        show_results("BM25", store.bm25_search(question, k=10))
        show_results("HYBRID", store.hybrid_search(question, k=10, candidate_k=100))

if __name__ == "__main__":
    main()
