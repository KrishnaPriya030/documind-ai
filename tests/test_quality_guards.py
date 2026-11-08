import unittest

from app.answer_generator import (
    MAX_OUTPUT_TOKENS,
    _numeric_unit_pairs,
    find_unsupported_numeric_units,
    resolve_source_references,
)


class OutputBudgetTests(unittest.TestCase):
    def test_default_output_budget_supports_longer_summaries(self):
        self.assertGreaterEqual(MAX_OUTPUT_TOKENS, 4096)


class NumericUnitGuardTests(unittest.TestCase):
    def test_flags_unit_substitution(self):
        answer = "The standard workweek is 40 working days per week."
        context = "The standard workweek is 40 hours, Monday to Friday."
        self.assertEqual(find_unsupported_numeric_units(answer, context), [("40", "day")])

    def test_accepts_matching_numeric_units(self):
        answer = "Annual leave is 20 working days and sick leave is 10 days."
        context = "Employees receive 20 working days of annual leave and 10 paid sick days per calendar year."
        self.assertEqual(find_unsupported_numeric_units(answer, context), [])

    def test_normalizes_commas_and_plural_units(self):
        self.assertIn(("1200", "day"), _numeric_unit_pairs("INR 1,200 per travel day"))


class DuplicateCitationTests(unittest.TestCase):
    def test_deduplicates_repeated_source_ids_in_one_group(self):
        source_map = {1: ("policy.pdf", 2)}
        answer, invalid = resolve_source_references("Claim [S1, S1].", source_map)
        self.assertEqual(invalid, [])
        self.assertEqual(answer.count("(Document:"), 1)

    def test_deduplicates_adjacent_resolved_citations(self):
        source_map = {1: ("policy.pdf", 2)}
        answer, invalid = resolve_source_references("Claim [S1] [S1].", source_map)
        self.assertEqual(invalid, [])
        self.assertEqual(answer.count("(Document:"), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
