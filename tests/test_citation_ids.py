import unittest

from app.answer_generator import (
    _format_context_and_sources,
    resolve_source_references,
    validate_citations,
)


class CitationIdTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {"doc": "Northstar_Analytics_Employee_Policy (2).pdf", "page": 1, "text": "Annual leave is 20 days. Carry-over is 5 days."},
            {"doc": "Northstar_Analytics_Employee_Policy (2).pdf", "page": 2, "text": "Sick leave is 10 days."},
            {"doc": "AeroSense_AM200_Operation_Manual.pdf", "page": 3, "text": "Battery runtime is up to 8 hours."},
        ]

    def test_context_assigns_stable_source_ids(self):
        context, source_map = _format_context_and_sources(self.chunks)
        self.assertIn("[S1]", context)
        self.assertIn("[S2]", context)
        self.assertIn("[S3]", context)
        self.assertEqual(source_map[1], ("Northstar_Analytics_Employee_Policy (2).pdf", 1))

    def test_source_ids_resolve_to_exact_pages(self):
        _, source_map = _format_context_and_sources(self.chunks)
        answer, invalid = resolve_source_references(
            "Annual leave is 20 days [S1]. Sick leave is 10 days [S2].", source_map
        )
        self.assertEqual(invalid, [])
        self.assertIn("Physical PDF page: 1", answer)
        self.assertIn("Physical PDF page: 2", answer)
        self.assertNotIn("[S1]", answer)

    def test_grouped_references_resolve(self):
        _, source_map = _format_context_and_sources(self.chunks)
        answer, invalid = resolve_source_references("Summary [S1, S2].", source_map)
        self.assertEqual(invalid, [])
        self.assertEqual(answer.count("(Document:"), 2)

    def test_unknown_source_id_is_flagged(self):
        _, source_map = _format_context_and_sources(self.chunks)
        answer, invalid = resolve_source_references("Unsupported claim [S99].", source_map)
        self.assertEqual(invalid, [99])
        self.assertIn("Invalid source reference", answer)

    def test_validator_rejects_page_not_in_that_document(self):
        answer = (
            "Annual leave is 20 days "
            "(Document: Northstar_Analytics_Employee_Policy (2).pdf, Physical PDF page: 6)"
        )
        result = validate_citations(answer, self.chunks)
        self.assertFalse(result["valid"])
        self.assertEqual(result["invalid"], [("Northstar_Analytics_Employee_Policy (2).pdf", 6)])

    def test_valid_document_page_pairs_pass(self):
        answer = (
            "Annual leave is 20 days "
            "(Document: Northstar_Analytics_Employee_Policy (2).pdf, Physical PDF page: 1). "
            "Sick leave is 10 days "
            "(Document: Northstar_Analytics_Employee_Policy (2).pdf, Physical PDF page: 2)."
        )
        result = validate_citations(answer, self.chunks)
        self.assertTrue(result["valid"])
        self.assertEqual(result["invalid"], [])


if __name__ == "__main__":
    unittest.main()
