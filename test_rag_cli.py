
from app.answer_generator import generate_answer
from app.vector_store import VectorStore


def main():
    print("=" * 60)
    print("DOCUMIND AI - COMPLETE RAG PIPELINE")
    print("=" * 60)

    # Load the existing retrieval index.
    print("\nLoading the existing hybrid retrieval system...")
    store = VectorStore()

    print("Retrieval system loaded successfully.")

    # Accept a question from the user.
    question = input("\nEnter your question: ").strip()

    if not question:
        print("Please enter a valid question.")
        return

    # Retrieve relevant passages using hybrid search.
    print("\nRetrieving relevant passages...")

    retrieved_chunks = store.search(question, k=5)

    if not retrieved_chunks:
        print("\nNo relevant passages were retrieved.")
        return

    print(f"\nRetrieved {len(retrieved_chunks)} passages.")

    # Display the retrieved evidence for debugging.
    print("\n" + "-" * 60)
    print("RETRIEVED EVIDENCE")
    print("-" * 60)

    for index, chunk in enumerate(retrieved_chunks, start=1):
        print(f"\nPassage {index}")
        print(f"Document: {chunk.get('doc', 'Unknown')}")
        print(f"Physical PDF page: {chunk.get('page', 'Unknown')}")
        print(f"Text: {chunk.get('text', '')[:500]}")

    # Generate an answer using Gemini.
    print("\n" + "-" * 60)
    print("GENERATING ANSWER")
    print("-" * 60)

    answer = generate_answer(question, retrieved_chunks)

    print("\nQuestion:")
    print(question)

    print("\nDocuMind AI Answer:")
    print(answer)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
