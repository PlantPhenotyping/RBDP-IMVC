import os
import unittest

import numpy as np

from rbdp.config import PROJECT_ROOT, canonical_dataset_name, dataset_spec
from rbdp.data import load_multiview_data, make_split


class DatasetRegistryTest(unittest.TestCase):

    def test_aliases_are_canonicalized(self):
        self.assertEqual(canonical_dataset_name("Scene_15"), "Scene-15")
        self.assertEqual(canonical_dataset_name("landuse21"), "LandUse-21")
        self.assertEqual(canonical_dataset_name("caltech"), "Caltech101-20")

    def test_registry_returns_a_copy(self):
        first = dataset_spec("NoisyMNIST")
        first["samples"] = -1
        self.assertEqual(dataset_spec("NoisyMNIST")["samples"], 20000)

    def test_local_three_view_shapes(self):
        data_root = os.path.join(PROJECT_ROOT, "data")
        expected = {
            "Caltech101-20": (2386, [1984, 512, 928], 20),
            "Scene-15": (4485, [20, 59, 40], 15),
            "LandUse-21": (2100, [59, 40, 20], 21),
        }
        for name, (samples, dimensions, classes) in expected.items():
            dataset = load_multiview_data(name, data_root=data_root)
            self.assertEqual(dataset.n_samples, samples)
            self.assertEqual(dataset.view_dims, dimensions)
            self.assertEqual(dataset.n_classes, classes)
            self.assertEqual(dataset.labels.shape, (samples,))
            for view, dimension in zip(dataset.views, dimensions):
                self.assertEqual(view.shape, (samples, dimension))
                self.assertEqual(view.dtype, np.float32)

    def test_caltech_legacy_normalization_is_safe(self):
        dataset = load_multiview_data(
            "Caltech101-20", data_root=os.path.join(PROJECT_ROOT, "data"))
        for view in dataset.views:
            self.assertGreaterEqual(float(view.min()), 0.0)
            self.assertLessEqual(float(view.max()), 1.0)


class SplitTest(unittest.TestCase):

    def test_split_is_deterministic_disjoint_and_complete(self):
        first = make_split(101, seed=7)
        second = make_split(101, seed=7)
        for key in ("train", "validation", "test"):
            np.testing.assert_array_equal(first[key], second[key])
        joined = np.concatenate([first["train"], first["validation"], first["test"]])
        self.assertEqual(np.unique(joined).size, 101)
        self.assertEqual(set(joined.tolist()), set(range(101)))
        self.assertEqual(
            np.intersect1d(first["train"], first["test"]).size, 0)

    def test_split_rejects_invalid_fractions(self):
        with self.assertRaises(ValueError):
            make_split(100, seed=1, fractions=(0.5, 0.2, 0.2))


if __name__ == "__main__":
    unittest.main()
