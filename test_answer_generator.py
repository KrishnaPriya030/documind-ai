"""Tests for the parts of answer generation that need no API call."""
from app.answer_generator import (
    NOT_FOUND_MESSAGE,
    format_context,
    is_not_found,
    validate_citations,
)

CHUNKS = [
    {"doc": "academic_regulations.pdf", "page": 220, "text": "Minimum 75% attendance."},
    {"doc": "academic_regulations.pdf", "page": 221, "text": "Repeat the course."},
]


def test_format_context_numbers_sources_and_keeps_pages():
    text = format_context(CHUNKS)
    assert "[Source 1]" in text and "[Source 2]" in text
    assert "Physical PDF page: 220" in text


def test_format_context_skips_empty_passages():
    chunks = CHUNKS + [{"doc": "x.pdf", "page": 1, "text": "   "}]
    assert "x.pdf" not in format_context(chunks)


def test_is_not_found_accepts_exact_and_trailing_variants():
    assert is_not_found(NOT_FOUND_MESSAGE)
    assert is_not_found(NOT_FOUND_MESSAGE.rstrip("."))
    assert is_not_found(f'"{NOT_FOUND_MESSAGE}"')
    assert not is_not_found("Students need 75% attendance.")


def test_valid_citation_passes():
    answer = "75% (Document: academic_regulations.pdf, Physical PDF page: 220)."
    result = validate_citations(answer, CHUNKS)
    assert result["valid"] and not result["invalid"]


def test_invented_page_is_flagged():
    answer = "75% (Document: academic_regulations.pdf, Physical PDF page: 999)."
    result = validate_citations(answer, CHUNKS)
    assert not result["valid"]
    assert result["invalid"] == [("academic_regulations.pdf", 999)]


def test_answer_without_citation_is_not_valid():
    result = validate_citations("Students need 75% attendance.", CHUNKS)
    assert not result["valid"] and result["cited"] == []


def test_file_name_with_parentheses_is_checked_correctly():
    chunks = [{"doc": "Exam Rules (2020).pdf", "page": 5, "text": "x"}]
    answer = "Yes (Document: Exam Rules (2020).pdf, Physical PDF page: 5)."
    result = validate_citations(answer, chunks)
    assert result["valid"] and result["cited"] == [("Exam Rules (2020).pdf", 5)]