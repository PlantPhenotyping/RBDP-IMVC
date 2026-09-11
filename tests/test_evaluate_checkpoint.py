import copy
import os
import unittest

from rbdp.config import load_json_config
from tools.evaluate_checkpoint import _deep_merge, _training_signature
from tools.evaluate_checkpoint import _validate_evaluation_overlay


def _parent_config():
    return {
        "dataset": {"name": "toy"},
        "split": {"seed": 7},
        "masks": {
            "train": {"mechanism": "balanced_mcar", "seed": 11},
            "validation": {"mechanism": "view_skew", "seed": 12},
            "test": {"mechanism": "view_skew", "seed": 13},
            "observation_rate_within_incomplete": 0.5,
            "correlation_strength": 0.8,
        },
        "model": {"latent_dim": 5},
        "training": {"epochs": 10, "seed": 1},
        "loss": {"prediction_weight": 0.1},
        "ablation": {
            "name": "a4",
            "complete_only": False,
            "heteroscedastic": True,
            "cycle": True,
            "groupdro": True,
            "conflict_safe": False,
            "reliability": False,
            "fusion": "concat",
        },
        "evaluation": {"kmeans_seed": 0},
        "reliability": {"source_temperature": 0.1},
        "runtime": {"device": "cpu"},
    }


class EvaluateCheckpointTest(unittest.TestCase):

    def test_inference_overlay_preserves_training_signature(self):
        parent = _parent_config()
        overlay = {
            "evaluation": {"kmeans_seed": 3},
            "reliability": {"source_temperature": 1.0},
            "ablation": {
                "name": "a5",
                "reliability": True,
                "fusion": "concat",
            },
            "masks": {"test": {"seed": 99}},
        }
        _validate_evaluation_overlay(overlay)
        child = _deep_merge(parent, overlay)
        self.assertEqual(
            _training_signature(parent),
            _training_signature(child),
        )
        self.assertEqual(child["masks"]["test"]["seed"], 99)

    def test_training_fields_are_rejected_or_detected(self):
        with self.assertRaises(ValueError):
            _validate_evaluation_overlay(
                {"loss": {"prediction_weight": 2.0}})
        with self.assertRaises(ValueError):
            _validate_evaluation_overlay(
                {"resolved": {"source_manifest": {}}})

        parent = _parent_config()
        overlay = {"ablation": {"conflict_safe": True}}
        _validate_evaluation_overlay(overlay)
        child = _deep_merge(copy.deepcopy(parent), overlay)
        self.assertNotEqual(
            _training_signature(parent),
            _training_signature(child),
        )

    def test_non_test_mask_overlay_is_rejected(self):
        with self.assertRaises(ValueError):
            _validate_evaluation_overlay(
                {"masks": {"train": {"seed": 4}}})
        with self.assertRaises(ValueError):
            _validate_evaluation_overlay({"masks": {"test": []}})

    def test_a5_repository_overlay_is_evaluation_only(self):
        root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), os.pardir))
        experiment = load_json_config(os.path.join(
            root, "configs", "experiments", "pilot_caltech_shift.json"))
        a4 = load_json_config(os.path.join(
            root, "configs", "ablations", "a4_geometry_balanced.json"))
        a5 = load_json_config(os.path.join(
            root, "configs", "ablations", "a5_adaptive_reliability.json"))
        parent = _deep_merge(experiment, a4)
        _validate_evaluation_overlay(a5)
        child = _deep_merge(parent, a5)
        self.assertEqual(
            _training_signature(parent),
            _training_signature(child),
        )


if __name__ == "__main__":
    unittest.main()
