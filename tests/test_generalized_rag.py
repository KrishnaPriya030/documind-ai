import unittest
from pathlib import Path
from unittest.mock import patch

from app.answer_generator import (
    NOT_FOUND_MESSAGE,
    SYSTEM_INSTRUCTION,
    _build_prompt,
    answer_question,
    format_context,
    is_not_found,
    validate_citations,
)
from app.api import _query_variants


class GeneralizedAnswerTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "doc": "employee_handbook.pdf",
                "page": 12,
                "text": "Employees must submit leave requests at least five working days in advance.",
            },
            {
                "doc": "technical_manual_v2.pdf",
                "page": 7,
                "text": "The device operating temperature is 5-40 degrees Celsius.",
            },
        ]

    def test_system_prompt_is_domain_independent(self):
        self.assertIn("domain-independent", SYSTEM_INSTRUCTION.lower())
        self.assertIn("business", SYSTEM_INSTRUCTION.lower())
        self.assertIn("technical", SYSTEM_INSTRUCTION.lower())
        self.assertNotIn("scholarship", SYSTEM_INSTRUCTION.lower())
        self.assertNotIn("university", SYSTEM_INSTRUCTION.lower())

    def test_context_preserves_document_names_and_pages(self):
        context = format_context(self.chunks)
        self.assertIn("Document: employee_handbook.pdf", context)
        self.assertIn("Physical PDF page: 12", context)
        self.assertIn("Document: technical_manual_v2.pdf", context)
        self.assertIn("Physical PDF page: 7", context)

    def test_prompt_includes_all_documents_and_grounding_instructions(self):
        context = format_context(self.chunks)
        prompt = _build_prompt("Compare the supplied documents.", self.chunks, context)
        self.assertIn("employee_handbook.pdf", prompt)
        self.assertIn("technical_manual_v2.pdf", prompt)
        self.assertIn("supplied passages", prompt)
        self.assertIn("Not specified in the supplied passages", prompt)

    def test_citation_with_parentheses_in_filename(self):
        chunks = [{"doc": "Safety Manual (2025).pdf", "page": 4, "text": "Safety procedure."}]
        answer = "Follow the procedure (Document: Safety Manual (2025).pdf, Physical PDF page: 4)."
        result = validate_citations(answer, chunks)
        self.assertTrue(result["valid"])
        self.assertEqual(result["invalid"], [])

    def test_page_range_requires_all_pages_to_be_supplied(self):
        chunks = [
            {"doc": "manual.pdf", "page": 4, "text": "Page four."},
            {"doc": "manual.pdf", "page": 5, "text": "Page five."},
        ]
        result = validate_citations(
            "Details (Document: manual.pdf, Physical PDF page: 4-5).", chunks
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["cited"], [("manual.pdf", 4), ("manual.pdf", 5)])

    def test_unsupplied_page_is_rejected(self):
        result = validate_citations(
            "Claim (Document: manual.pdf, Physical PDF page: 99).",
            [{"doc": "manual.pdf", "page": 4, "text": "Some detail."}],
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["invalid"], [("manual.pdf", 99)])

    def test_not_found_variants(self):
        self.assertTrue(is_not_found(NOT_FOUND_MESSAGE))
        self.assertTrue(is_not_found("Not found in provided documents."))
        self.assertFalse(is_not_found("The handbook says the device operates at 5-40 C."))

    def test_no_evidence_returns_not_found_without_api_call(self):
        result = answer_question("What are the leave rules?", [])
        self.assertEqual(result["answer"], NOT_FOUND_MESSAGE)
        self.assertFalse(result["found"])
        self.assertTrue(result["citations_ok"])

    def test_answer_question_preserves_api_result_contract(self):
        answer = "Leave requests need five working days (Document: employee_handbook.pdf, Physical PDF page: 12)."
        with patch("app.answer_generator.generate_answer", return_value=answer):
            result = answer_question("What is the leave notice?", self.chunks[:1])
        self.assertEqual(
            set(result),
            {"answer", "found", "cited", "invalid_citations", "citations_ok"},
        )
        self.assertTrue(result["found"])
        self.assertTrue(result["citations_ok"])


class GeneralizedRetrievalQueryTests(unittest.TestCase):
    def test_query_expansion_is_not_subject_specific(self):
        variants = _query_variants(
            "Compare the warranty periods and exclusions across these product manuals."
        )
        self.assertGreaterEqual(len(variants), 2)
        self.assertIn("warranty", " ".join(variants).lower())
        self.assertIn("similarities differences", variants[-1])
        self.assertFalse(any("scholarship" in value.lower() for value in variants))

    def test_summary_expansion_is_general(self):
        variants = _query_variants("Summarize the main findings of this research report.")
        self.assertTrue(any("overview main points" in value for value in variants))

    def test_original_question_is_always_preserved(self):
        question = "What is the payment deadline?"
        self.assertEqual(_query_variants(question)[0], question)


class DomainSpecificCodeRemovalTests(unittest.TestCase):
    def test_api_does_not_contain_scholarship_specific_expansion(self):
        api_source = Path("app/api.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("scholarship continuation renewal eligibility", api_source)
        self.assertNotIn("academic performance percentage cgpa", api_source)

    def test_vector_store_has_no_university_specific_retrieval_hacks(self):
        source = Path("app/vector_store.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("university of kerala", source)
        self.assertNotIn("school_count_evidence", source)
        self.assertNotIn("registrar@keralauniversity.ac.in", source)
        self.assertNotIn("regrku@gmail.com", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
