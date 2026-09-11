import unittest

import numpy as np

from rbdp.metrics import apply_mapping
from rbdp.metrics import cluster_embeddings
from rbdp.metrics import clustering_metrics_from_labels
from rbdp.metrics import hungarian_mapping
from rbdp.metrics import pattern_metrics
from rbdp.metrics import reliability_metrics


class ClusteringMetricTest(unittest.TestCase):

    def test_permuted_clusters_score_perfectly(self):
        truth = np.asarray([0, 0, 1, 1, 2, 2])
        predicted = np.asarray([9, 9, 4, 4, 7, 7])
        mapping = hungarian_mapping(truth, predicted)
        np.testing.assert_array_equal(apply_mapping(predicted, mapping), truth)
        metrics = clustering_metrics_from_labels(truth, predicted, mapping)
        for name in ("acc", "nmi", "ari", "ami", "f1_macro"):
            self.assertAlmostEqual(metrics[name], 1.0)

    def test_pattern_metrics_reuse_the_global_mapping(self):
        truth = np.asarray([0, 0, 1, 1, 0, 1])
        predicted = np.asarray([1, 1, 0, 0, 0, 1])
        mask = np.asarray([
            [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1]
        ], dtype=np.uint8)
        result = pattern_metrics(truth, predicted, mask)
        self.assertAlmostEqual(result["overall"]["acc"], 4.0 / 6.0)
        self.assertEqual(result["patterns"]["01"]["count"], 2)
        # A per-pattern remapping would make this subgroup perfect; the global
        # mapping correctly exposes that both predictions are wrong.
        self.assertAlmostEqual(result["patterns"]["01"]["acc"], 0.0)
        self.assertAlmostEqual(result["worst_pattern_acc"], 0.0)

    def test_kmeans_seed_is_deterministic(self):
        embedding = np.asarray([
            [-2.0, -2.0], [-1.9, -2.1], [2.0, 2.0], [2.1, 1.9]
        ])
        first = cluster_embeddings(embedding, 2, seed=5, n_init=5)
        second = cluster_embeddings(embedding, 2, seed=5, n_init=5)
        np.testing.assert_array_equal(first, second)


class ReliabilityMetricTest(unittest.TestCase):

    def test_perfect_risk_ranking(self):
        values = np.asarray([0.4, 0.1, 0.3, 0.2])
        metrics = reliability_metrics(values, values)
        self.assertAlmostEqual(metrics["spearman"], 1.0)
        self.assertLess(metrics["aurc"], metrics["mean_error"])
        self.assertEqual(metrics["coverage"][-1], 1.0)


if __name__ == "__main__":
    unittest.main()
