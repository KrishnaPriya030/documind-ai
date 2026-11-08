"""Embedding adapter for DocuMind AI.

Local Windows development defaults to Sentence Transformers.
Docker deployment can set EMBEDDING_BACKEND=fastembed to use ONNX Runtime
and avoid installing the large PyTorch package.

Both backends use sentence-transformers/all-MiniLM-L6-v2 (384 dimensions).
All output vectors are converted to contiguous, normalized float32 arrays.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def _backend_name() -> str:
    backend = os.getenv("EMBEDDING_BACKEND", "sentence-transformers").strip().lower()
    if backend not in {"sentence-transformers", "fastembed"}:
        raise ValueError(
            "Unsupported EMBEDDING_BACKEND. Use 'sentence-transformers' or 'fastembed'."
        )
    return backend


@lru_cache(maxsize=1)
def _load_model():
    """Load one embedding model per process, only when first needed."""
    backend = _backend_name()

    if backend == "fastembed":
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "EMBEDDING_BACKEND=fastembed, but FastEmbed is not installed. "
                "Install it with 'pip install fastembed'."
            ) from exc

        cache_dir = os.getenv("FASTEMBED_CACHE_PATH") or None
        print(f"Loading FastEmbed ONNX model: {MODEL_NAME}", flush=True)
        return TextEmbedding(model_name=MODEL_NAME, cache_dir=cache_dir)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Sentence Transformers is not installed. Install it or set "
            "EMBEDDING_BACKEND=fastembed and install FastEmbed."
        ) from exc

    print(f"Loading SentenceTransformer model: {MODEL_NAME}", flush=True)
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: Iterable[str]) -> np.ndarray:
    """Encode texts as normalized float32 vectors with shape (n, 384)."""
    text_list = list(texts or [])
    if not text_list:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    if any(not isinstance(text, str) for text in text_list):
        raise TypeError("All embedding inputs must be strings.")

    backend = _backend_name()
    model = _load_model()

    if backend == "fastembed":
        # FastEmbed returns a generator of NumPy arrays.
        vectors = np.asarray(
            list(model.embed(text_list, batch_size=32)),
            dtype=np.float32,
        )
    else:
        vectors = model.encode(
            text_list,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim != 2 or vectors.shape != (len(text_list), EMBEDDING_DIM):
        raise ValueError(
            f"Embedding model returned shape {vectors.shape}; expected "
            f"({len(text_list)}, {EMBEDDING_DIM})."
        )

    if not np.isfinite(vectors).all():
        raise ValueError("Embedding model returned NaN or infinite values.")

    # Normalize explicitly so both backends match the FAISS inner-product setup.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding model returned a zero-length vector.")
    vectors = vectors / norms

    return np.ascontiguousarray(vectors, dtype=np.float32)
