"""SQLite persistence for DocuMind AI document metadata."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("DOCUMIND_DB_PATH", str(BASE_DIR / "data" / "documind.sqlite3")))


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def init_db() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                content_sha256 TEXT NOT NULL UNIQUE,
                pdf_path TEXT NOT NULL,
                chunks_path TEXT NOT NULL,
                embeddings_path TEXT NOT NULL,
                total_pages INTEGER NOT NULL,
                text_pages INTEGER NOT NULL,
                ocr_pages INTEGER NOT NULL,
                chunks_count INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC)"
        )


def insert_document(record: dict[str, Any]) -> None:
    fields = (
        "id", "filename", "original_filename", "content_sha256", "pdf_path",
        "chunks_path", "embeddings_path", "total_pages", "text_pages",
        "ocr_pages", "chunks_count",
    )
    values = [record[field] for field in fields]
    placeholders = ",".join("?" for _ in fields)
    with _connect() as connection:
        connection.execute(
            f"INSERT INTO documents ({','.join(fields)}) VALUES ({placeholders})",
            values,
        )


def get_document(document_id: str) -> dict[str, Any] | None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
    return dict(row) if row else None


def get_document_by_hash(content_sha256: str) -> dict[str, Any] | None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM documents WHERE content_sha256 = ?", (content_sha256,)
        ).fetchone()
    return dict(row) if row else None


def list_documents() -> list[dict[str, Any]]:
    with _connect() as connection:
        rows = connection.execute(
            """SELECT id, filename, original_filename, total_pages, text_pages,
                      ocr_pages, chunks_count, created_at
               FROM documents ORDER BY created_at DESC, filename COLLATE NOCASE"""
        ).fetchall()
    return [dict(row) for row in rows]


def delete_document_record(document_id: str) -> bool:
    with _connect() as connection:
        cursor = connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return cursor.rowcount > 0
