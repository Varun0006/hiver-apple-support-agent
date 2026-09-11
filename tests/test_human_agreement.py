import unittest

import pandas as pd

from evaluation.judge_human_agreement import DIMENSIONS, validate_human_ratings


def make_ratings(**overrides):
    values = {dimension: 4 for dimension in DIMENSIONS}
    values.update(overrides)
    return pd.DataFrame([{"conversation_id": "conv_1", **values}])


class HumanRatingValidationTests(unittest.TestCase):
    def test_accepts_complete_ratings(self):
        validate_human_ratings(make_ratings(), expected_conversation_ids=["conv_1"])

    def test_rejects_blank_ratings(self):
        with self.assertRaisesRegex(ValueError, "number from 1 to 5"):
            validate_human_ratings(make_ratings(correctness=""))

    def test_rejects_out_of_range_ratings(self):
        with self.assertRaisesRegex(ValueError, "number from 1 to 5"):
            validate_human_ratings(make_ratings(tone=6))

    def test_rejects_fractional_ratings(self):
        with self.assertRaisesRegex(ValueError, "number from 1 to 5"):
            validate_human_ratings(make_ratings(tone=3.5))

    def test_rejects_duplicate_conversation_ids(self):
        ratings = pd.concat([make_ratings(), make_ratings()], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_human_ratings(ratings)

    def test_rejects_unknown_conversation_ids(self):
        with self.assertRaisesRegex(ValueError, "missing from the judge results"):
            validate_human_ratings(make_ratings(), expected_conversation_ids=["conv_2"])


if __name__ == "__main__":
    unittest.main()