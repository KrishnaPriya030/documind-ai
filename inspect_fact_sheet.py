
import json

from pathlib import Path

chunks_path = Path("index/chunks_500.json")

with open(chunks_path, "r", encoding="utf-8") as file:
    chunks = json.load(file)

for i, chunk in enumerate(chunks):
    if chunk.get("page") == 21:
        print("\n" + "=" * 80)
        print(f"Chunk index: {i}")
        print(f"Document: {chunk['doc']}")
        print(f"Page: {chunk['page']}")
        print("-" * 80)
        print(chunk["text"])
