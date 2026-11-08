
import pymupdf
from pathlib import Path


def clean_text(text):
    """Remove unnecessary whitespace from extracted text."""
    return " ".join(text.split())


def extract_pages(pdf_path):
    """
    Extract cleaned text from a PDF.

    Preserves the document filename and physical PDF page number.
    Skips pages without extractable text.
    """

    pages = []
    skipped_pages = []

    pdf_path = Path(pdf_path)

    with pymupdf.open(pdf_path) as document:
        # Read the page count while the document is open.
        total_pages = len(document)

        for page_number, page in enumerate(document, start=1):
            raw_text = page.get_text("text")
            text = clean_text(raw_text)

            if not text:
                skipped_pages.append(page_number)
                continue

            pages.append({
                "doc": pdf_path.name,
                "page": page_number,
                "text": text
            })

    # The document is now closed, so use the saved total_pages variable.
    print(f"Document: {pdf_path.name}")
    print(f"Total PDF pages: {total_pages}")
    print(f"Pages extracted: {len(pages)}")
    print(f"Pages skipped: {len(skipped_pages)}")

    if skipped_pages:
        print(f"Skipped PDF pages: {skipped_pages}")

    return pages
