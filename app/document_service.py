"""PDF extraction, OCR, chunking, embedding, and per-document index storage."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DOCUMIND_DATA_DIR", str(BASE_DIR / "data")))
INDEX_DIR = Path(os.getenv("DOCUMIND_INDEX_DIR", str(BASE_DIR / "index")))
UPLOAD_DIR = DATA_DIR / "uploads"
DOCUMENT_INDEX_DIR = INDEX_DIR / "uploads"
MAX_FILE_SIZE = 50 * 1024 * 1024
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150


def safe_filename(filename: str) -> str:
    name = (filename or "uploaded_document.pdf").replace("\\", "/").rsplit("/", 1)[-1].strip()
    name = re.sub(r"[^A-Za-z0-9._() -]+", "_", name).strip(" .")
    if not name:
        name = "uploaded_document.pdf"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name[:180]


def _load_pdf_library():
    try:
        import pymupdf
        return pymupdf
    except ImportError:
        try:
            import fitz
            return fitz
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required. Install the 'pymupdf' package.") from exc


def _find_tesseract() -> tuple[str | None, str | None]:
    executable = os.getenv("TESSERACT_CMD") or shutil.which("tesseract")
    if not executable and os.name == "nt":
        candidates = [
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        ]
        executable_path = next((path for path in candidates if path.is_file()), None)
        executable = str(executable_path) if executable_path else None
    if not executable:
        return None, None
    tesseract_dir = Path(executable).resolve().parent
    candidates = [tesseract_dir / "tessdata"]
    prefix = os.getenv("TESSDATA_PREFIX")
    if prefix:
        prefix_path = Path(prefix)
        candidates.extend([prefix_path, prefix_path / "tessdata"])
    tessdata = next((path for path in candidates if (path / "eng.traineddata").is_file()), None)
    return executable, str(tessdata) if tessdata else None


def extract_pdf_pages(pdf_bytes: bytes) -> tuple[list[dict[str, Any]], int, int]:
    if not pdf_bytes or b"%PDF-" not in pdf_bytes[:1024]:
        raise ValueError("The uploaded file is not a valid PDF.")
    pdf_library = _load_pdf_library()
    pages: list[dict[str, Any]] = []
    ocr_pages = 0
    with pdf_library.open(stream=pdf_bytes, filetype="pdf") as pdf:
        page_count = len(pdf)
        if page_count == 0:
            raise ValueError("The PDF contains no pages.")
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            ocr_processed = False
            if len(text) < 20:
                executable, tessdata = _find_tesseract()
                if not executable:
                    raise RuntimeError(
                        f"Page {page_number} appears scanned, but Tesseract OCR is not installed."
                    )
                os.environ["PATH"] = str(Path(executable).resolve().parent) + os.pathsep + os.environ.get("PATH", "")
                try:
                    kwargs: dict[str, Any] = {"language": "eng", "dpi": 200, "full": True}
                    if tessdata:
                        kwargs["tessdata"] = tessdata
                    text_page = page.get_textpage_ocr(**kwargs)
                    text = page.get_text("text", textpage=text_page).strip()
                except Exception as exc:
                    raise RuntimeError(f"OCR failed on physical PDF page {page_number}: {exc}") from exc
                if text:
                    ocr_processed = True
                    ocr_pages += 1
            if text:
                pages.append({"page": page_number, "text": text, "ocr_processed": ocr_processed})
    if not pages:
        raise ValueError("No text could be extracted from this PDF, even after OCR.")
    return pages, page_count, ocr_pages


def make_chunks(pages: list[dict[str, Any]], filename: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for page in pages:
        text = str(page["text"])
        for start in range(0, len(text), step):
            piece = text[start : start + CHUNK_SIZE].strip()
            if piece:
                chunks.append({"doc": filename, "page": int(page["page"]), "text": piece})
            if start + CHUNK_SIZE >= len(text):
                break
    return chunks


def process_pdf(pdf_bytes: bytes, original_filename: str) -> dict[str, Any]:
    """Create persistent per-document chunks and normalized embeddings."""
    if len(pdf_bytes) > MAX_FILE_SIZE:
        raise ValueError("The PDF must be smaller than 50 MiB.")
    filename = safe_filename(original_filename)
    content_sha256 = hashlib.sha256(pdf_bytes).hexdigest()
    document_id = content_sha256
    upload_dir = UPLOAD_DIR / document_id
    index_dir = DOCUMENT_INDEX_DIR / document_id
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    DOCUMENT_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    upload_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = upload_dir / filename
    chunks_path = index_dir / "chunks_500.json"
    embeddings_path = index_dir / "embeddings_500.npy"

    try:
        pages, page_count, ocr_pages = extract_pdf_pages(pdf_bytes)
        chunks = make_chunks(pages, filename)
        if not chunks:
            raise ValueError("No searchable text chunks could be created from this PDF.")

        from app.embedder import embed_texts
        embeddings = np.asarray(embed_texts([item["text"] for item in chunks]), dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise RuntimeError(f"Unexpected embeddings shape: {embeddings.shape}")
        if not np.isfinite(embeddings).all():
            raise RuntimeError("The embedding model returned NaN or infinite values.")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)

        pdf_path.write_bytes(pdf_bytes)
        chunks_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        np.save(embeddings_path, embeddings.astype(np.float32))
    except Exception:
        shutil.rmtree(upload_dir, ignore_errors=True)
        shutil.rmtree(index_dir, ignore_errors=True)
        raise

    return {
        "id": document_id,
        "filename": filename,
        "original_filename": safe_filename(original_filename),
        "content_sha256": content_sha256,
        "pdf_path": str(pdf_path.resolve()),
        "chunks_path": str(chunks_path.resolve()),
        "embeddings_path": str(embeddings_path.resolve()),
        "total_pages": page_count,
        "text_pages": len(pages),
        "ocr_pages": ocr_pages,
        "chunks_count": len(chunks),
    }


def remove_document_files(record: dict[str, Any]) -> None:
    for key in ("pdf_path", "chunks_path", "embeddings_path"):
        raw = record.get(key)
        if raw:
            path = Path(raw)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
    for raw in (record.get("pdf_path"), record.get("chunks_path")):
        if raw:
            try:
                Path(raw).parent.rmdir()
            except OSError:
                pass
