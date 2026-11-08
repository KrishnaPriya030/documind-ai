"""

DocuMind AI retrieval layer.

Supports:

1. Dense semantic retrieval using FAISS.

2. Keyword retrieval using BM25.

3. Hybrid retrieval using weighted Reciprocal Rank Fusion (RRF).

4. Improved ranking for exact-keyword and email-related questions.

"""

from __future__ import annotations

import json

import re

from pathlib import Path

from typing import Any

import faiss

import numpy as np

from rank_bm25 import BM25Okapi

from app.embedder import embed_texts

# Preserve complete email addresses as searchable tokens.

EMAIL_RE = re.compile(

    r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}",

    re.IGNORECASE,

)

# Match ordinary words and common identifiers.

TOKEN_RE = re.compile(

    r"[a-z0-9]+(?:[._%+-][a-z0-9]+)*",

    re.IGNORECASE,

)

def tokenize(text: str) -> list[str]:

    """Tokenize text while preserving complete email addresses."""

    text = text.lower()

    # Extract email addresses before tokenizing the remaining text.

    emails = EMAIL_RE.findall(text)

    without_emails = EMAIL_RE.sub(" ", text)

    tokens = TOKEN_RE.findall(without_emails)

    # Keep complete email addresses and their individual components.

    for email in emails:

        email = email.lower()

        tokens.append(email)

        local, _, domain = email.partition("@")

        tokens.extend(

            part

            for part in re.split(r"[._%+-]+", local)

            if part

        )

        tokens.extend(

            part

            for part in re.split(r"[.-]+", domain)

            if part

        )

    return tokens

class VectorStore:

    """Load document chunks and embeddings and provide retrieval methods."""

    def __init__(

        self,

        chunks_path: str | Path = "index/chunks_500.json",

        embeddings_path: str | Path = "index/embeddings_500.npy",

        expected_dimension: int = 384,

    ) -> None:

        self.chunks_path = Path(chunks_path)

        self.embeddings_path = Path(embeddings_path)

        if not self.chunks_path.exists():

            raise FileNotFoundError(

                f"Chunks file not found: {self.chunks_path}"

            )

        if not self.embeddings_path.exists():

            raise FileNotFoundError(

                f"Embeddings file not found: {self.embeddings_path}"

            )

        with self.chunks_path.open("r", encoding="utf-8") as file:

            self.chunks: list[dict[str, Any]] = json.load(file)

        if not isinstance(self.chunks, list) or not self.chunks:

            raise ValueError(

                "Chunks JSON must contain a non-empty list."

            )

        for index, chunk in enumerate(self.chunks):

            if not isinstance(chunk, dict):

                raise ValueError(

                    f"Chunk {index} must be a dictionary."

                )

            if not all(

                key in chunk for key in ("doc", "page", "text")

            ):

                raise ValueError(

                    f"Chunk {index} must contain doc, page, and text."

                )

            if not isinstance(chunk["text"], str):

                raise ValueError(

                    f"Chunk {index} text must be a string."

                )

        embeddings = np.load(self.embeddings_path)

        if embeddings.ndim != 2:

            raise ValueError(

                f"Embeddings must be 2D; got shape {embeddings.shape}."

            )

        if embeddings.shape[0] != len(self.chunks):

            raise ValueError(

                f"Chunk/embedding count mismatch: "

                f"{len(self.chunks)} chunks versus "

                f"{embeddings.shape[0]} embeddings."

            )

        if embeddings.shape[1] != expected_dimension:

            raise ValueError(

                f"Expected {expected_dimension}-dimensional embeddings; "

                f"got {embeddings.shape[1]}."

            )

        if not np.isfinite(embeddings).all():

            raise ValueError(

                "Embeddings contain NaN or infinite values."

            )

        self.embeddings = np.ascontiguousarray(

            embeddings,

            dtype=np.float32,

        )

        # Normalized embeddings make inner product equivalent to cosine

        # similarity, assuming the saved vectors are normalized.

        self.faiss_index = faiss.IndexFlatIP(

            self.embeddings.shape[1]

        )

        self.faiss_index.add(self.embeddings)

        # Build the keyword index.

        tokenized_corpus = [

            tokenize(chunk["text"])

            for chunk in self.chunks

        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

    def _embed_query(self, question: str) -> np.ndarray:

        """Generate an embedding compatible with the saved index."""

        vectors = embed_texts([question])

        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim != 2 or vectors.shape[0] != 1:

            raise ValueError(

                "embed_texts([question]) must return one embedding row."

            )

        if vectors.shape[1] != self.embeddings.shape[1]:

            raise ValueError(

                f"Query embedding dimension {vectors.shape[1]} "

                f"does not match index dimension "

                f"{self.embeddings.shape[1]}."

            )

        return np.ascontiguousarray(vectors)

    @staticmethod

    def _top_indices(

        scores: np.ndarray,

        limit: int,

    ) -> list[int]:

        """Return indices sorted by descending score."""

        limit = min(max(int(limit), 0), len(scores))

        if limit == 0:

            return []

        return sorted(

            range(len(scores)),

            key=lambda index: (

                -float(scores[index]),

                index,

            ),

        )[:limit]

    @staticmethod

    def _is_email_or_exact_lookup(question: str) -> bool:

        """Identify questions that depend heavily on exact keywords."""

        question_lower = question.lower()

        email_terms = (

            "email",

            "e-mail",

            "email address",

            "email addresses",

            "mail address",

            "contact email",

        )

        exact_lookup_terms = (

            "exact address",

            "phone number",

            "telephone number",

            "registration number",

            "website address",

        )

        return (

            any(term in question_lower for term in email_terms)

            or any(term in question_lower for term in exact_lookup_terms)

        )

    def dense_search(

        self,

        question: str,

        k: int = 5,

    ) -> list[dict[str, Any]]:

        """Search using semantic similarity only."""

        if not question.strip():

            raise ValueError("Question must not be empty.")

        k = min(max(int(k), 1), len(self.chunks))

        query_vector = self._embed_query(question)

        scores, indices = self.faiss_index.search(

            query_vector,

            k,

        )

        results: list[dict[str, Any]] = []

        for score, index in zip(scores[0], indices[0]):

            if index < 0:

                continue

            results.append(

                {

                    **self.chunks[int(index)],

                    "chunk_id": int(index),

                    "score": float(score),

                    "semantic_score": float(score),

                    "keyword_score": None,

                    "retrieval_method": "dense",

                }

            )

        return results

    def bm25_search(

        self,

        question: str,

        k: int = 5,

    ) -> list[dict[str, Any]]:

        """Search using keyword relevance only."""

        if not question.strip():

            raise ValueError("Question must not be empty.")

        scores = np.asarray(

            self.bm25.get_scores(tokenize(question)),

            dtype=np.float64,

        )

        indices = self._top_indices(scores, k)

        return [

            {

                **self.chunks[index],

                "chunk_id": index,

                "score": float(scores[index]),

                "semantic_score": None,

                "keyword_score": float(scores[index]),

                "retrieval_method": "bm25",

            }

            for index in indices

        ]

    def email_search(
        self,
        question: str,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve email-containing chunks and rank by query context."""

        if not question.strip():
            raise ValueError("Question must not be empty.")

        question_lower = question.lower()
        ignored_terms = {
            "what", "which", "are", "is", "the", "a", "an",
            "in", "for", "of", "on", "to", "and", "listed",
            "addresses", "address", "email", "emails", "mail",
            "give", "show", "find", "provide",
        }
        context_terms = {
            token
            for token in tokenize(question_lower)
            if token not in ignored_terms
        }

        results: list[dict[str, Any]] = []
        for index, chunk in enumerate(self.chunks):
            passage = chunk.get("text", "")
            emails = EMAIL_RE.findall(passage)
            if not emails:
                continue

            passage_tokens = set(tokenize(passage))
            overlap = len(context_terms & passage_tokens)

            # Rank by query-to-passage token overlap only. No institution- or
            # domain-specific email addresses receive special treatment.
            results.append({
                **chunk,
                "chunk_id": index,
                "score": float(overlap),
                "semantic_score": None,
                "keyword_score": None,
                "retrieval_method": "email_lookup",
                "email_matches": emails,
                "context_overlap": overlap,
            })

        results.sort(
            key=lambda item: (
                -item["score"],
                item["chunk_id"],
            )
        )
        return results[:max(1, int(k))]

    def hybrid_search(

        self,

        question: str,

        k: int = 5,

        candidate_k: int = 100,

        rrf_constant: int = 60,

    ) -> list[dict[str, Any]]:

        """

        Combine dense and BM25 rankings using weighted RRF.

        Ordinary questions use a higher dense-retrieval weight.

        Email and exact-lookup questions give more influence to BM25.

        For email-related questions, passages containing email addresses

        receive a small additional boost. The boost is based on the

        presence of actual email addresses in the retrieved passage,

        not merely on whether the word 'email' appears.

        """

        if not question.strip():

            raise ValueError("Question must not be empty.")

        if k < 1 or candidate_k < 1 or rrf_constant < 1:

            raise ValueError(

                "k, candidate_k, and rrf_constant must be positive."

            )

        candidate_k = min(candidate_k, len(self.chunks))

        k = min(k, len(self.chunks))

        # Retrieve semantic candidates.

        query_vector = self._embed_query(question)

        dense_scores, dense_indices = self.faiss_index.search(

            query_vector,

            candidate_k,

        )

        # Retrieve keyword candidates.

        bm25_scores_all = np.asarray(

            self.bm25.get_scores(tokenize(question)),

            dtype=np.float64,

        )

        bm25_indices = self._top_indices(

            bm25_scores_all,

            candidate_k,

        )

        # Record semantic ranks and scores.

        dense_rank: dict[int, int] = {}

        dense_score_by_id: dict[int, float] = {}

        for rank, (score, index) in enumerate(

            zip(dense_scores[0], dense_indices[0]),

            start=1,

        ):

            if index >= 0:

                index = int(index)

                dense_rank[index] = rank

                dense_score_by_id[index] = float(score)

        # Record BM25 ranks.

        bm25_rank = {

            index: rank

            for rank, index in enumerate(bm25_indices, start=1)

        }

        candidate_ids = set(dense_rank) | set(bm25_rank)

        # Choose weights based on the type of question.

        exact_lookup = self._is_email_or_exact_lookup(question)

        if exact_lookup:

            dense_weight = 1.0

            bm25_weight = 5.0

        else:

            dense_weight = 5.0

            bm25_weight = 1.0

        # Apply weighted Reciprocal Rank Fusion.

        rrf_scores: dict[int, float] = {}

        for index in candidate_ids:

            score = 0.0

            if index in dense_rank:

                score += dense_weight / (

                    rrf_constant + dense_rank[index]

                )

            if index in bm25_rank:

                score += bm25_weight / (

                    rrf_constant + bm25_rank[index]

                )

            # For email-related questions, reward passages containing

            # actual email addresses. This helps retrieve contact details.

            if exact_lookup:

                passage = self.chunks[index]["text"]

                email_matches = EMAIL_RE.findall(passage)

                if email_matches:

                    score += 0.02

            rrf_scores[index] = score

        # Sort by fused score, with deterministic tie-breaking.

        ranked_rows = sorted(

            rrf_scores.items(),

            key=lambda item: (

                -item[1],

                item[0],

            ),

        )

        results: list[dict[str, Any]] = []

        for index, fused_score in ranked_rows[:k]:

            results.append(

                {

                    **self.chunks[index],

                    "chunk_id": index,

                    "score": float(fused_score),

                    "rrf_score": float(fused_score),

                    "semantic_score": dense_score_by_id.get(index),

                    "keyword_score": (

                        float(bm25_scores_all[index])

                        if index in bm25_rank

                        else None

                    ),

                    "dense_rank": dense_rank.get(index),

                    "bm25_rank": bm25_rank.get(index),

                    "retrieval_method": "hybrid",

                }

            )

        return results

    def search(
        self,
        question: str,
        k: int = 5,
        candidate_k: int = 100,
    ) -> list[dict[str, Any]]:
        """Route email questions to email lookup and others to hybrid search."""

        if not question.strip():
            raise ValueError("Question must not be empty.")

        question_lower = question.lower()
        email_terms = (
            "email", "e-mail", "mail address", "email address", "emails",
        )
        if any(term in question_lower for term in email_terms):
            return self.email_search(question, k=k)

        return self.hybrid_search(
            question,
            k=k,
            candidate_k=candidate_k,
        )