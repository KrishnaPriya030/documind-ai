"""Dense (FAISS), keyword (BM25) and hybrid (weighted RRF) retrieval for DocuMind AI."""
import json
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Change these paths if your files live somewhere else.
CHUNKS_PATH = Path("index/chunks_500.json")
EMBEDDINGS_PATH = Path("index/embeddings_500.npy")
MODEL_NAME = "all-MiniLM-L6-v2"


def tokenize(text):
    """Lowercase, join hyphenated words (e-mail -> email), keep word characters."""
    text = text.lower()
    text = re.sub(r"(?<=\w)-(?=\w)", "", text)
    return re.findall(r"\w+", text)


class Retriever:
    def __init__(self, chunks_path=CHUNKS_PATH, embeddings_path=EMBEDDINGS_PATH):
        with open(chunks_path, encoding="utf-8") as f:
            self.chunks = json.load(f)
        embeddings = np.load(embeddings_path).astype("float32")
        if len(self.chunks) != len(embeddings):
            raise ValueError(
                f"{len(self.chunks)} chunks but {len(embeddings)} embeddings: "
                "they must be in the same order and the same length."
            )
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    def _item(self, idx, score):
        chunk = self.chunks[idx]
        return {
            "index": idx,
            "chunk_id": chunk.get("chunk_id"),
            "doc": chunk["doc"],
            "page": chunk["page"],
            "text": chunk["text"],
            "score": score,
        }

    def dense(self, question, k=5):
        """Semantic search with FAISS. Score = cosine similarity."""
        q = self.model.encode(
            [question], normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        scores, ids = self.index.search(q, k)
        return [self._item(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def keyword(self, question, k=5):
        """Keyword search with BM25."""
        scores = self.bm25.get_scores(tokenize(question))
        top = np.argsort(scores)[::-1][:k]
        return [self._item(int(i), float(scores[i])) for i in top]

    def hybrid(self, question, k=5, depth=20, w_dense=1.0, w_bm25=1.0, c=60):
        """Weighted Reciprocal Rank Fusion.

        Take the top `depth` results from each method, give each chunk
        weight / (c + rank) from every list it appears in, add the points up,
        sort by the total (highest first) and keep the top k.
        """
        fused = {}
        lists = (
            (w_dense, self.dense(question, depth)),
            (w_bm25, self.keyword(question, depth)),
        )
        for weight, results in lists:
            for rank, item in enumerate(results, start=1):
                fused[item["index"]] = fused.get(item["index"], 0.0) + weight / (c + rank)
        ranked = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
        return [self._item(idx, score) for idx, score in ranked[:k]]