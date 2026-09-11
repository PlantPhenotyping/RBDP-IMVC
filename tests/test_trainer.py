import unittest

import numpy as np
import torch

from rbdp.masks import generate_mask
from rbdp.trainer import build_model, evaluate_model, fit_model
from rbdp.trainer import primary_anchored_backward, seed_everything


def trainer_config():
    return {
        "model": {
            "hidden_dims": [12],
            "latent_dim": 5,
            "predictor_hidden_dims": [7],
            "batch_norm": False,
            "min_log_variance": -8.0,
            "max_log_variance": 3.0,
        },
        "training": {
            "seed": 2,
            "epochs": 1,
            "batch_size": 10,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip": 5.0,
            "start_prediction_epoch": 0,
        },
        "loss": {
            "reconstruction_weight": 1.0,
            "pair_weight": 1.0,
            "prediction_weight": 0.1,
            "mi_alpha": 1.0,
            "cycle_weight": 1.0,
            "groupdro_eta": 0.05,
            "min_group_size": 1,
            "stop_gradient": True,
        },
        "ablation": {
            "name": "test",
            "complete_only": False,
            "heteroscedastic": True,
            "cycle": True,
            "groupdro": True,
            "reliability": True,
            "fusion": "consensus",
        },
        "reliability": {
            "gamma": 1.0,
            "source_temperature": 0.1,
            "confidence_temperature": 0.1,
        },
        "evaluation": {
            "kmeans_seed": 0,
            "kmeans_n_init": 3,
            "min_pattern_count": 1,
        },
    }


class TrainerIntegrationTest(unittest.TestCase):

    def test_primary_anchor_projects_a_conflicting_auxiliary_gradient(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        primary = parameter[0]
        auxiliary = -parameter[0] + parameter[1]
        record = primary_anchored_backward(
            primary, auxiliary, [parameter], auxiliary_weight=1.0)
        self.assertEqual(record["gradient_conflict"], 1.0)
        self.assertLess(record["gradient_cosine"], 0.0)
        self.assertTrue(torch.allclose(
            parameter.grad, torch.tensor([1.0, 1.0]), atol=1.0e-6))
        self.assertGreater(float(parameter.grad[0].item()), 0.0)

    def test_one_epoch_fit_and_evaluation(self):
        rng = np.random.RandomState(5)
        views = [rng.rand(36, 6).astype(np.float32),
                 rng.rand(36, 4).astype(np.float32),
                 rng.rand(36, 3).astype(np.float32)]
        labels = np.repeat(np.arange(3), 12)
        mask, _ = generate_mask(
            36, 3, 0.5, mechanism="view_skew", seed=8,
            observation_rate=0.5)
        config = trainer_config()
        seed_everything(config["training"]["seed"])
        model = build_model([6, 4, 3], config)
        fitted = fit_model(model, views, mask, config, torch.device("cpu"))
        self.assertEqual(len(fitted["curves"]), 1)
        self.assertTrue(np.isfinite(fitted["curves"][0]["loss"]))
        evaluated = evaluate_model(
            model,
            views,
            labels,
            mask,
            np.arange(36),
            config,
            torch.device("cpu"),
        )
        self.assertEqual(evaluated["metrics"]["embedding_shape"], [36, 5])
        self.assertIn("worst_pattern_acc",
                      evaluated["metrics"]["pattern_robustness"])
        self.assertTrue(evaluated["reliability_rows"])

    def test_epoch_callback_observes_requested_training_states(self):
        rng = np.random.RandomState(17)
        views = [rng.rand(24, 6).astype(np.float32),
                 rng.rand(24, 4).astype(np.float32)]
        mask, _ = generate_mask(
            24, 2, 0.5, mechanism="balanced_mcar", seed=3,
            observation_rate=0.5)
        config = trainer_config()
        config["training"]["epochs"] = 3
        observed = []

        def callback(epoch, model):
            observed.append((epoch, model.training))

        model = build_model([6, 4], config)
        fit_model(model, views, mask, config, torch.device("cpu"),
                  epoch_callback=callback)
        self.assertEqual(observed, [(1, False), (2, False), (3, False)])
        self.assertTrue(model.training)

    def test_conflict_safe_training_records_gradient_diagnostics(self):
        rng = np.random.RandomState(9)
        views = [rng.rand(30, 6).astype(np.float32),
                 rng.rand(30, 4).astype(np.float32),
                 rng.rand(30, 3).astype(np.float32)]
        mask, _ = generate_mask(
            30, 3, 0.5, mechanism="balanced_mcar", seed=4,
            observation_rate=0.5)
        config = trainer_config()
        config["ablation"]["conflict_safe"] = True
        config["loss"]["structure_weight"] = 0.1
        config["loss"]["auxiliary_weight"] = 1.0
        model = build_model([6, 4, 3], config)
        fitted = fit_model(model, views, mask, config, torch.device("cpu"))
        curve = fitted["curves"][0]
        for key in ("gradient_cosine", "gradient_conflict",
                    "projection_fraction", "primary_gradient_norm",
                    "auxiliary_gradient_norm", "heteroscedastic_nll"):
            self.assertTrue(np.isfinite(curve[key]))
        self.assertGreaterEqual(curve["gradient_conflict"], 0.0)
        self.assertLessEqual(curve["gradient_conflict"], 1.0)

    def test_conflict_safe_respects_delayed_prediction_start(self):
        rng = np.random.RandomState(11)
        views = [rng.rand(24, 6).astype(np.float32),
                 rng.rand(24, 4).astype(np.float32),
                 rng.rand(24, 3).astype(np.float32)]
        mask, _ = generate_mask(
            24, 3, 0.5, mechanism="balanced_mcar", seed=7,
            observation_rate=0.5)
        config = trainer_config()
        config["ablation"]["conflict_safe"] = True
        config["training"]["start_prediction_epoch"] = 2
        model = build_model([6, 4, 3], config)
        fitted = fit_model(model, views, mask, config, torch.device("cpu"))
        curve = fitted["curves"][0]
        self.assertEqual(curve["gradient_conflict"], 0.0)
        self.assertEqual(curve["primary_gradient_norm"], 0.0)
        self.assertFalse(fitted["environment_rows"])


if __name__ == "__main__":
    unittest.main()
