from app.pdf_loader import extract_pages
from app.chunker import chunk_pages


PDF_PATH = "data/academic_regulations.pdf"
OVERLAP = 150

# Step 1: Extract the PDF text.
pages = extract_pages(PDF_PATH)

print("\n========== CHUNK COUNT COMPARISON ==========")

# Step 2: Generate and compare chunks of both sizes.
for size in [500, 1000]:
    chunks = chunk_pages(
        pages,
        chunk_size=size,
        overlap=OVERLAP
    )

    print(f"\nChunk size: {size} characters")
    print(f"Overlap: {OVERLAP} characters")
    print(f"Total chunks: {len(chunks)}")

    if chunks:
        print(f"First chunk document: {chunks[0]['doc']}")
        print(f"First chunk PDF page: {chunks[0]['page']}")
        print(f"First chunk actual length: {len(chunks[0]['text'])}")
        print(f"First chunk preview: {chunks[0]['text'][:200]}")

print("\n========== TEST COMPLETED ==========")
