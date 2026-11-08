
from app.pdf_loader import extract_pages

pdf_path = "data/academic_regulations.pdf"  # Replace with your actual filename

pages = extract_pages(pdf_path)

total_characters = sum(len(page["text"]) for page in pages)

print("\n========== EXTRACTION REPORT ==========")
print("Pages containing extractable text:", len(pages))
print("Total cleaned characters:", total_characters)

print("\n========== FIRST 10 EXTRACTED PAGES ==========")

for page in pages[:10]:
    print(
        f"PDF page {page['page']}: "
        f"{len(page['text'])} characters"
    )

print("\n========== SAMPLE CLEANED TEXT ==========")

if pages:
    print("Document:", pages[0]["doc"])
    print("PDF page:", pages[0]["page"])
    print("Text:")
    print(pages[0]["text"][:1000])
else:
    print("No extractable text found.")
