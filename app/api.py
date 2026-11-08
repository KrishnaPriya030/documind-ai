"""FastAPI backend for DocuMind AI."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.database import (
    delete_document_record,
    get_document,
    get_document_by_hash,
    init_db,
    insert_document,
    list_documents,
)
from app.document_service import process_pdf, remove_document_files

app = FastAPI(
    title="DocuMind AI API",
    version="1.0.0",
    description="Domain-independent PDF ingestion, OCR, hybrid retrieval, and source-grounded question answering.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this to your frontend domain before public production use.
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=3000)
    document_ids: list[str] = Field(min_length=1, max_length=10)
    passages_per_document: int = Field(default=6, ge=3, le=10)


def _record_to_public(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "filename": record["filename"],
        "original_filename": record["original_filename"],
        "total_pages": record["total_pages"],
        "text_pages": record["text_pages"],
        "ocr_pages": record["ocr_pages"],
        "chunks_count": record["chunks_count"],
        "created_at": record.get("created_at"),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "documind-api"}


@app.get("/api/documents")
def documents() -> list[dict[str, Any]]:
    return list_documents()


@app.post("/api/documents", status_code=201)
def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = file.filename or "uploaded_document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    try:
        content = file.file.read(50 * 1024 * 1024 + 1)
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="PDF exceeds the 50 MiB upload limit.")
        if not content:
            raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")
        import hashlib
        content_hash = hashlib.sha256(content).hexdigest()
        existing = get_document_by_hash(content_hash)
        if existing:
            return {**_record_to_public(existing), "reused": True, "message": "This PDF was already indexed."}
        record = process_pdf(content, filename)
        insert_document(record)
        _cached_store.cache_clear()
        return {**_record_to_public(record), "reused": False, "message": "PDF indexed successfully."}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {type(exc).__name__}: {exc}") from exc
    finally:
        file.file.close()


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, str]:
    record = get_document(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found.")
    _cached_store.cache_clear()
    remove_document_files(record)
    delete_document_record(document_id)
    return {"message": f"Deleted {record['filename']}"}


@lru_cache(maxsize=24)
def _cached_store(chunks_path: str, embeddings_path: str):
    from app.vector_store import VectorStore
    return VectorStore(chunks_path, embeddings_path)


_QUERY_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "is", "are", "am", "was", "were", "be", "been", "being", "do", "does", "did",
    "can", "could", "would", "should", "will", "shall", "may", "might",
    "i", "we", "you", "they", "he", "she", "it", "me", "us", "them",
    "my", "our", "your", "their", "his", "her", "its", "this", "that", "these", "those",
    "to", "of", "for", "in", "on", "at", "by", "with", "from", "into", "about", "as",
    "according", "based", "provided", "document", "documents", "pdf", "file", "files",
    "please", "tell", "me", "explain", "give", "show", "find", "identify", "list",
    "each", "every", "all", "across", "between", "among", "following",
}


def _query_variants(question: str) -> list[str]:
    """Create small domain-neutral retrieval variants from the user's query.

    The original wording is always searched. A focused keyword query improves
    lexical matching, while intent terms are added only for generic comparison
    or summary requests. No subject-specific vocabulary is injected.
    """
    original = question.strip()
    variants = [original]
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_./%+-]*", original)
    keywords = [token for token in tokens if token.casefold() not in _QUERY_STOPWORDS]
    focused = " ".join(keywords).strip()
    if len(keywords) >= 2 and focused.casefold() != original.casefold():
        variants.append(focused)

    lower = original.casefold()
    if any(term in lower for term in ("compare", "comparison", "difference", "different", "similar")) and focused:
        variants.append(f"{focused} similarities differences changes exceptions conflicts")
    elif any(term in lower for term in ("summarize", "summary", "overview", "main points")) and focused:
        variants.append(f"{focused} overview main points key details conclusions")

    # Keep order, remove duplicates, and avoid sending blank variants.
    return list(dict.fromkeys(item for item in variants if item))


def _retrieve_across_documents(
    question: str,
    selected_records: list[dict[str, Any]],
    passages_per_document: int,
) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    query_variants = _query_variants(question)
    for record in selected_records:
        chunks_path = Path(record["chunks_path"])
        embeddings_path = Path(record["embeddings_path"])
        if not chunks_path.is_file() or not embeddings_path.is_file():
            raise HTTPException(status_code=500, detail=f"Saved index missing for {record['filename']}. Re-upload this PDF.")
        store = _cached_store(str(chunks_path), str(embeddings_path))
        fused: dict[tuple[Any, ...], dict[str, Any]] = {}
        for query in query_variants:
            results = store.search(query, k=max(passages_per_document, 8), candidate_k=100)
            for rank, result in enumerate(results, start=1):
                key = (result.get("page"), result.get("text", ""))
                entry = fused.setdefault(key, {"result": dict(result), "rrf_score": 0.0, "best_rank": rank})
                entry["rrf_score"] += 1.0 / (60 + rank)
                entry["best_rank"] = min(entry["best_rank"], rank)
        ranked = sorted(
            fused.values(),
            key=lambda item: (-item["rrf_score"], item["best_rank"], int(item["result"].get("page", 0))),
        )
        for entry in ranked[:passages_per_document]:
            item = dict(entry["result"])
            item["score"] = float(entry["rrf_score"])
            item["retrieval_method"] = "api_balanced_per_document_rrf"
            combined.append(item)
    return combined


@app.post("/api/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    records: list[dict[str, Any]] = []
    for document_id in dict.fromkeys(request.document_ids):
        record = get_document(document_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"Selected document not found: {document_id[:12]}")
        records.append(record)
    try:
        evidence = _retrieve_across_documents(question, records, request.passages_per_document)
        if not evidence:
            return {"answer": "Not found in the provided documents.", "found": False, "citations_ok": True, "evidence": [], "documents_searched": [r["filename"] for r in records]}
        from app.answer_generator import answer_question
        result = answer_question(question, evidence)
        return {
            **result,
            "evidence": evidence,
            "documents_searched": [record["filename"] for record in records],
            "evidence_by_document": {
                name: sum(1 for item in evidence if item.get("doc") == name)
                for name in dict.fromkeys(item.get("doc", "Unknown") for item in evidence)
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Question answering failed: {type(exc).__name__}: {exc}") from exc
