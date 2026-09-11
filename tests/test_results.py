import os
import tempfile
import unittest

import numpy as np

from rbdp.io import config_hash, write_json, write_npz
from rbdp.masks import mask_hash, save_mask
from rbdp.results import collect_runs, expand_expected_grids
from rbdp.results import missing_expected_runs, paired_comparisons
from rbdp.results import summarize_rows, validate_run


def _array_hash(values):
    import hashlib
    value = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.view(np.uint8).tobytes())
    return digest.hexdigest()


def _make_run(root, ablation="base", seed=1, acc=0.5, worst=0.4):
    run = os.path.join(root, "%s-%d" % (ablation, seed))
    os.makedirs(run)
    split = {
        "train": np.asarray([0, 1], dtype=np.int64),
        "validation": np.asarray([2], dtype=np.int64),
        "test": np.asarray([3, 4], dtype=np.int64),
    }
    masks = {
        "train": np.asarray([[1, 1], [1, 0]], dtype=np.uint8),
        "validation": np.asarray([[1, 0]], dtype=np.uint8),
        "test": np.asarray([[1, 0], [0, 1]], dtype=np.uint8),
    }
    mask_metadata = {}
    for name, mask in masks.items():
        metadata = {"statistics": {"mask_hash": mask_hash(mask)}}
        mask_metadata[name] = metadata
        save_mask(os.path.join(run, "%s_mask.npz" % name), mask, metadata)
    write_npz(os.path.join(run, "split.npz"), **split)
    config = {
        "experiment": "pilot",
        "protocol": "shift",
        "dataset": {"name": "toy"},
        "split": {"seed": 7},
        "masks": {
            "train": {"mechanism": "balanced_mcar", "incomplete_rate": 0.5,
                      "seed": 11},
            "validation": {"mechanism": "view_skew", "incomplete_rate": 0.5,
                           "seed": 12},
            "test": {"mechanism": "view_skew", "incomplete_rate": 0.9,
                     "seed": 13},
        },
        "training": {"seed": seed, "epochs": 10},
        "loss": {
            "prediction_weight": 0.1,
            "cycle_weight": 1.0,
            "structure_weight": 0.0,
            "groupdro_eta": 0.05,
            "groupdro_update_frequency": "epoch",
        },
        "ablation": {
            "name": ablation, "description": "test", "fusion": "concat",
            "complete_only": False, "heteroscedastic": True, "cycle": True,
            "groupdro": ablation != "base", "reliability": False,
        },
        "reliability": {
            "gamma": 1.0,
            "risk_normalization": "none",
            "source_temperature": 0.1,
            "confidence_temperature": 0.1,
        },
        "resolved": {
            "dataset_summary": {"n_samples": 5, "n_views": 2,
                                "view_dims": [3, 4]},
            "dataset_sha256": "dataset-hash",
            "source_manifest": {"source_hash": "source-%s" % ablation},
            "split_hashes": {name: _array_hash(value)
                             for name, value in split.items()},
            "mask_metadata": mask_metadata,
        },
    }
    digest = config_hash(config)
    stored = dict(config)
    stored["config_hash"] = digest
    write_json(os.path.join(run, "config.json"), stored)
    write_json(os.path.join(run, "status.json"), {
        "state": "completed", "config_hash": digest})
    metrics = {
        "clustering": {"acc": acc, "nmi": acc - 0.1, "ari": acc - 0.2,
                       "ami": acc - 0.15, "f1_macro": acc - 0.05,
                       "precision_macro": acc, "recall_macro": acc},
        "pattern_robustness": {
            "worst_pattern_acc": worst, "average_pattern_acc": acc - 0.02,
            "best_pattern_acc": acc + 0.05, "pattern_gap": 0.1},
        "reliability": {
            "completion_mse": 0.2, "completion_cosine_error": 0.3,
            "path_ranking": {"spearman": 0.6, "aurc": 0.1,
                             "mean_error": 0.2}},
        "training": {"seconds": 2.0},
        "inference_seconds": 0.1,
        "model": {"parameter_count": 100},
        "provenance": {
            "config_hash": digest,
            "source_hash": "source-%s" % ablation,
            "dataset_sha256": "dataset-hash",
            "train_mask_hash": mask_hash(masks["train"]),
            "test_mask_hash": mask_hash(masks["test"]),
        },
    }
    write_json(os.path.join(run, "metrics.json"), metrics)
    return run


class ResultsTest(unittest.TestCase):

    def test_valid_run_is_extracted_only_after_artifact_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            run = _make_run(directory)
            row, audit = validate_run(run)
            self.assertTrue(audit["valid"])
            self.assertEqual(row["dataset"], "toy")
            self.assertEqual(row["risk_normalization"], "none")
            self.assertEqual(row["source_temperature"], 0.1)
            self.assertEqual(row["normalized_risk_aurc"], 0.5)

    def test_corrupt_config_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = _make_run(directory)
            status = {"state": "completed", "config_hash": "wrong"}
            write_json(os.path.join(run, "status.json"), status)
            row, audit = validate_run(run)
            self.assertIsNone(row)
            self.assertIn("mismatched_status_config_hash", audit["issues"])

    def test_collection_summary_and_paired_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            _make_run(directory, "base", 1, 0.50, 0.40)
            _make_run(directory, "base", 2, 0.60, 0.45)
            _make_run(directory, "method", 1, 0.55, 0.43)
            _make_run(directory, "method", 2, 0.70, 0.50)
            rows, audit = collect_runs([directory])
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(record["valid"] for record in audit))
            summaries = summarize_rows(rows)
            method = [row for row in summaries if row["ablation"] == "method"][0]
            self.assertAlmostEqual(method["acc_mean"], 0.625)
            self.assertAlmostEqual(method["normalized_risk_aurc_mean"], 0.5)
            comparisons = paired_comparisons(rows, "base", metrics=("acc",))
            group = [row for row in comparisons if row["scope"] == "group"][0]
            self.assertEqual(group["pair_count"], 2)
            self.assertAlmostEqual(group["mean_delta"], 0.075)
            self.assertAlmostEqual(group["median_delta"], 0.075)
            self.assertGreater(group["paired_effect_dz"], 0.0)
            self.assertGreater(group["rank_biserial_effect"], 0.0)
            self.assertEqual(group["holm_p_two_sided"],
                             group["wilcoxon_p_two_sided"])

    def test_expected_grid_reports_missing_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            _make_run(directory, "base", 1)
            rows, _ = collect_runs([directory])
            expected = expand_expected_grids({"grids": [{
                "ablation": ["base", "method"],
                "training_seed": [1, 2],
            }]})
            missing = missing_expected_runs(rows, expected)
            self.assertEqual(len(missing), 3)


if __name__ == "__main__":
    unittest.main()
