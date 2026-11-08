
from app.vector_store import VectorStore

store = VectorStore()

questions = [
    "University Fact Sheet number of departments",
    "No. of Departments 43",
    "How many departments and schools are listed in the University Fact Sheet?",
]

for question in questions:
    print("\n" + "=" * 70)
    print("QUESTION:", question)
    print("=" * 70)

    results = store.search(question, k=10)

    for rank, chunk in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Document: {chunk.get('doc')}")
        print(f"Physical PDF page: {chunk.get('page')}")
        print(f"Text: {chunk.get('text', '')[:700]}")
