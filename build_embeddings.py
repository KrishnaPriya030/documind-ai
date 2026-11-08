import json
import numpy as np

from app.pdf_loader import extract_pages
from app.chunker import chunk_pages
from app.embedder import embed_texts


PDF_PATH = "data/academic_regulations.pdf"
CHUNKS_PATH = "data/chunks_500.json"
EMBEDDINGS_PATH = "data/embeddings_500.npy"

CHUNK_SIZE = 500
OVERLAP = 150


def main():
    # Step 1: Extract text from the PDF.
    pages = extract_pages(PDF_PATH)

    # Step 2: Divide the extracted text into chunks.
    chunks = chunk_pages(
        pages,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP
    )

    print(f"\nTotal chunks: {len(chunks)}")

    # Step 3: Save the chunks and their metadata.
    with open(CHUNKS_PATH, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    # Step 4: Extract the text from each chunk.
    texts = [chunk["text"] for chunk in chunks]

    # Step 5: Generate the embeddings.
    embeddings = embed_texts(texts)

    # Step 6: Save the embeddings.
    np.save(EMBEDDINGS_PATH, embeddings)

    print("\n========== EMBEDDING REPORT ==========")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding data type: {embeddings.dtype}")
    print(f"Saved chunks: {CHUNKS_PATH}")
    print(f"Saved embeddings: {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()