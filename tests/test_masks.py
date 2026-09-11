import os
import tempfile
import unittest

import numpy as np

from rbdp.masks import CONTROLLED_MECHANISMS
from rbdp.masks import generate_mask
from rbdp.masks import load_mask
from rbdp.masks import mask_statistics
from rbdp.masks import save_mask


class ControlledMaskTest(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(11)
        self.features = rng.normal(size=(101, 7)).astype(np.float32)

    def _generate(self, mechanism, seed=13):
        keyword = {"features": self.features} if mechanism == "feature_dependent" else {}
        return generate_mask(
            n_samples=101,
            n_views=3,
            incomplete_rate=0.37,
            observation_rate=0.5,
            mechanism=mechanism,
            seed=seed,
            **keyword
        )

    def test_every_mechanism_preserves_exact_budgets(self):
        for mechanism in CONTROLLED_MECHANISMS:
            mask, metadata = self._generate(mechanism)
            stats = mask_statistics(mask)
            self.assertEqual(mask.dtype, np.uint8)
            self.assertEqual(mask.shape, (101, 3))
            self.assertEqual(stats["n_incomplete"], 37)
            self.assertGreaterEqual(stats["min_observed_views_per_sample"], 1)
            incomplete = mask.sum(axis=1) < 3
            # round(37 * 3 * .5) by the generator's round-half-up rule.
            self.assertEqual(int(mask[incomplete].sum()), 56)
            self.assertEqual(
                metadata["target_observed_entries_within_incomplete"], 56)
            self.assertEqual(
                metadata["statistics"]["mask_hash"], stats["mask_hash"])

    def test_same_seed_is_bitwise_reproducible(self):
        first, first_meta = self._generate("correlated", seed=2)
        second, second_meta = self._generate("correlated", seed=2)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(
            first_meta["statistics"]["mask_hash"],
            second_meta["statistics"]["mask_hash"],
        )

    def test_different_seeds_change_the_mask(self):
        first, _ = self._generate("balanced_mcar", seed=2)
        second, _ = self._generate("balanced_mcar", seed=3)
        self.assertFalse(np.array_equal(first, second))

    def test_correlated_mechanism_favors_its_declared_group(self):
        balanced, _ = generate_mask(
            2001, 3, 1.0, mechanism="balanced_mcar", seed=4,
            observation_rate=0.5)
        correlated, metadata = generate_mask(
            2001, 3, 1.0, mechanism="correlated", seed=4,
            observation_rate=0.5, correlation_groups=[[0, 1]])
        self.assertEqual(metadata["correlation_groups"], [[0, 1]])
        balanced_pair = np.mean((balanced[:, 0] == 0) & (balanced[:, 1] == 0))
        correlated_pair = np.mean(
            (correlated[:, 0] == 0) & (correlated[:, 1] == 0))
        self.assertGreater(correlated_pair, balanced_pair + 0.1)

    def test_edge_rates_remain_valid(self):
        complete, _ = generate_mask(17, 2, 0.0, seed=1)
        self.assertTrue(np.all(complete == 1))
        incomplete, _ = generate_mask(17, 2, 1.0, seed=1)
        self.assertTrue(np.all(incomplete.sum(axis=1) == 1))

    def test_feature_dependent_requires_features(self):
        with self.assertRaises(ValueError):
            generate_mask(10, 3, 0.5, mechanism="feature_dependent", seed=1)

    def test_saved_mask_round_trip_verifies_hash(self):
        mask, metadata = self._generate("view_skew")
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "mask.npz")
            save_mask(path, mask, metadata)
            loaded, loaded_metadata = load_mask(path)
        np.testing.assert_array_equal(mask, loaded)
        self.assertEqual(
            metadata["statistics"]["mask_hash"], loaded_metadata["mask_hash"])


if __name__ == "__main__":
    unittest.main()
