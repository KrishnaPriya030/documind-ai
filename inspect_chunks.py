import json
chunks = json.load(open("index/chunks_500.json", encoding="utf-8"))
for c in chunks:
    if c["page"] in (220, 222):
        print(c["page"], len(c["text"]), "|", c["text"], "\n---")