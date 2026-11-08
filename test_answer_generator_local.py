"""Offline regression tests for DocuMind AI's deterministic answer paths.

Run from the project root with the project's virtual environment active:
    python test_answer_generator_local.py

These tests do not make Gemini API requests.
"""

import unittest

from app.answer_generator import (
    NOT_FOUND_MESSAGE,
    answer_question,
    format_context,
    validate_citations,
)


class LocalAnswerGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.school_count_chunks = [
            {
                "doc": "academic_regulations.pdf",
                "page": 21,
                "text": "No. of Schools 11 No. of Departments 43",
            },
            {
                "doc": "academic_regulations.pdf",
                "page": 14,
                "text": (
                    "The forty-three departments of teaching and research "
                    "are organized as nine schools."
                ),
            },
        ]

    def test_school_count_conflict_cites_both_pages(self):
        result = answer_question(
            "How many schools does the University of Kerala have?",
            self.school_count_chunks,
        )

        self.assertTrue(result["found"])
        self.assertTrue(result["citations_ok"])
        self.assertEqual(result["invalid_citations"], [])
        self.assertIn("11 schools", result["answer"])
        self.assertIn("9 schools", result["answer"])
        self.assertIn("Physical PDF page: 21", result["answer"])
        self.assertIn("Physical PDF page: 14", result["answer"])

    def test_live_stock_price_question_is_refused_without_api(self):
        result = answer_question(
            "What is Apple's current stock price?",
            [{
                "doc": "academic_regulations.pdf",
                "page": 286,
                "text": "A university project mentions stock management.",
            }],
        )

        self.assertFalse(result["found"])
        self.assertEqual(result["answer"], NOT_FOUND_MESSAGE)
        self.assertTrue(result["citations_ok"])

    def test_live_weather_question_is_refused_without_api(self):
        result = answer_question(
            "What is the weather in Varkala today?",
            [{
                "doc": "academic_regulations.pdf",
                "page": 340,
                "text": "Varkala is a coastal town.",
            }],
        )

        self.assertFalse(result["found"])
        self.assertEqual(result["answer"], NOT_FOUND_MESSAGE)
        self.assertTrue(result["citations_ok"])

    def test_latest_academy_award_question_is_refused_without_api(self):
        result = answer_question(
            "Who won the latest Academy Award for Best Actor?",
            [{
                "doc": "academic_regulations.pdf",
                "page": 20,
                "text": "The university received several awards in earlier years.",
            }],
        )

        self.assertFalse(result["found"])
        self.assertEqual(result["answer"], NOT_FOUND_MESSAGE)
        self.assertTrue(result["citations_ok"])

    def test_unsupported_guaranteed_placement_package_is_refused_without_api(self):
        result = answer_question(
            "What guaranteed placement package will every student receive in 2027?",
            [{
                "doc": "academic_regulations.pdf",
                "page": 291,
                "text": "The University Placement Cell provides opportunities for students.",
            }],
        )

        self.assertFalse(result["found"])
        self.assertEqual(result["answer"], NOT_FOUND_MESSAGE)
        self.assertTrue(result["citations_ok"])

    def test_empty_context_is_refused_without_api(self):
        result = answer_question(
            "What is the registrar's email address?",
            [],
        )

        self.assertFalse(result["found"])
        self.assertEqual(result["answer"], NOT_FOUND_MESSAGE)
        self.assertTrue(result["citations_ok"])

    def test_format_context_includes_document_and_page(self):
        context = format_context(self.school_count_chunks[:1])

        self.assertIn("Document: academic_regulations.pdf", context)
        self.assertIn("Physical PDF page: 21", context)
        self.assertIn("No. of Schools 11", context)

    def test_citation_validator_rejects_unsupplied_page(self):
        supplied = [{
            "doc": "academic_regulations.pdf",
            "page": 21,
            "text": "No. of Schools 11",
        }]
        answer = (
            "The Fact Sheet lists 11 schools "
            "(Document: academic_regulations.pdf, Physical PDF page: 999)."
        )

        check = validate_citations(answer, supplied)

        self.assertFalse(check["valid"])
        self.assertEqual(
            check["invalid"], [("academic_regulations.pdf", 999)]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
