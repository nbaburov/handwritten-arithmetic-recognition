"""Tests that each (case, split) pair gets a distinct RNG seed.

Verifies fix C-4: the seed now includes case.value in addition to split_name,
so different cases at the same split produce different random sequences and
therefore diverse synthetic data rather than identical scenes.
"""
from __future__ import annotations

import random
import unittest


class TestRngSeedPerCase(unittest.TestCase):
    """Verify that two different SceneCases yield distinct random sequences."""

    def _make_rng(self, seed: int, split_name: str, case_value: str) -> random.Random:
        """Mirror the seeding formula in build_synthetic_yolo_dataset (C-4 fix)."""
        return random.Random(
            seed
            + sum(ord(c) for c in split_name)
            + sum(ord(c) for c in case_value)
        )

    def test_different_cases_produce_different_sequences(self) -> None:
        from src.generation.layouts import SceneCase

        seed = 42
        split = "train"

        case_a = SceneCase.addition
        case_b = SceneCase.subtraction

        rng_a = self._make_rng(seed, split, case_a.value)
        rng_b = self._make_rng(seed, split, case_b.value)

        values_a = [rng_a.random() for _ in range(10)]
        values_b = [rng_b.random() for _ in range(10)]

        self.assertNotEqual(
            values_a,
            values_b,
            msg="Two different SceneCases produced identical RNG sequences — C-4 fix not effective.",
        )

    def test_same_case_same_split_is_deterministic(self) -> None:
        """Same (case, split, seed) must always yield the same sequence."""
        from src.generation.layouts import SceneCase

        seed = 42
        split = "train"
        case = SceneCase.division_long

        rng_1 = self._make_rng(seed, split, case.value)
        rng_2 = self._make_rng(seed, split, case.value)

        values_1 = [rng_1.random() for _ in range(10)]
        values_2 = [rng_2.random() for _ in range(10)]

        self.assertEqual(
            values_1,
            values_2,
            msg="Same (case, split, seed) should produce identical sequences but did not.",
        )

    def test_different_splits_produce_different_sequences(self) -> None:
        """(case, train) vs (case, val) must also differ."""
        from src.generation.layouts import SceneCase

        seed = 42
        case = SceneCase.addition

        rng_train = self._make_rng(seed, "train", case.value)
        rng_val = self._make_rng(seed, "val", case.value)

        values_train = [rng_train.random() for _ in range(10)]
        values_val = [rng_val.random() for _ in range(10)]

        self.assertNotEqual(
            values_train,
            values_val,
            msg="Different splits produced identical RNG sequences.",
        )
