def chunk_pages(pages, chunk_size=1000, overlap=150):
    """
    Split extracted PDF pages into smaller text chunks.

    Each chunk preserves:
    - Document filename
    - Physical PDF page number
    - Chunk text

    Chunks never cross page boundaries.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []

    for page in pages:
        text = page["text"]

        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "doc": page["doc"],
                    "page": page["page"],
                    "text": chunk_text
                })

            # Stop when the end of this page has been reached.
            if end == len(text):
                break

            # Begin the next chunk with overlapping text.
            start = end - overlap

    return chunks