
import unittest

from benchmark import normalize_answer, check_answer


class TestLLMBenchmark(unittest.TestCase):

    def test_exact_answer(self):
        result = check_answer(
            "Paris", "Paris", "Knowledge"
        )
        self.assertEqual(result[0], 1)
        self.assertEqual(result[2], 0)

    def test_case_insensitive(self):
        result = check_answer(
            "PARIS", "Paris", "Knowledge"
        )
        self.assertEqual(result[0], 1)

    def test_sentence_answer(self):
        result = check_answer(
            "The capital of France is Paris.",
            "Paris",
            "Knowledge"
        )
        self.assertEqual(result[0], 1)
        self.assertEqual(result[2], 1)

    def test_wrong_answer(self):
        result = check_answer(
            "Berlin", "Paris", "Knowledge"
        )
        self.assertEqual(result[0], 0)
        self.assertEqual(result[2], 1)

    def test_instruction_following(self):
        result = check_answer(
            "The answer is HELLO",
            "HELLO",
            "Instructions"
        )
        self.assertEqual(result[0], 0)

    def test_normalization(self):
        self.assertEqual(
            normalize_answer("  Hello, WORLD! "),
            "hello world"
        )

    def test_negation_requires_review(self):
        result = check_answer(
            "Paris is not the capital.",
            "Paris",
            "Knowledge"
        )
        # Keyword matching is not semantic understanding.
        # This response must be flagged for review.
        self.assertEqual(result[2], 1)


if __name__ == "__main__":
    unittest.main()
