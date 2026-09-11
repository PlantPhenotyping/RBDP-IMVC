import unittest

import numpy as np
import torch

from rbdp.losses import EnvironmentIndexer
from rbdp.losses import GroupDROState
from rbdp.losses import masked_pair_loss
from rbdp.losses import masked_reconstruction_loss
from rbdp.losses import prediction_environment_losses
from rbdp.models import RBDPModel
from rbdp.reliability import evaluate_completion_reliability


def synthetic_batch():
    torch.manual_seed(3)
    views = [torch.rand(8, 4), torch.rand(8, 3), torch.rand(8, 5)]
    mask = torch.tensor([
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
    ], dtype=torch.long)
    return views, mask


def small_model():
    return RBDPModel(
        view_dims=[4, 3, 5],
        hidden_dims=[[7], [7], [7]],
        latent_dim=4,
        predictor_hidden_dims=[6],
        batch_norm=False,
    )


class ModelShapeTest(unittest.TestCase):

    def test_observed_encoding_and_completion_shapes(self):
        views, mask = synthetic_batch()
        model = small_model()
        latents = model.encode_observed(views, mask)
        self.assertEqual(len(latents), 3)
        for view in range(3):
            self.assertEqual(tuple(latents[view].shape), (8, 4))
            self.assertTrue(torch.all(latents[view][mask[:, view] == 0] == 0).item())
        completed = model.complete(views, mask, use_reliability=True)
        self.assertEqual(tuple(completed["representation"].shape), (8, 4))
        self.assertEqual(tuple(completed["completed_latents"].shape), (8, 3, 4))
        self.assertEqual(tuple(completed["path_risk"].shape), (8, 3, 3))
        self.assertEqual(tuple(completed["path_valid"].shape), (8, 3, 3))
        observed_confidence = completed["view_confidence"][mask > 0]
        self.assertTrue(torch.allclose(
            observed_confidence, torch.ones_like(observed_confidence)))
        self.assertTrue(torch.isfinite(completed["representation"]).all().item())

    def test_concat_fusion_shape(self):
        views, mask = synthetic_batch()
        output = small_model().complete(
            views, mask, use_reliability=False, fusion="concat")
        self.assertEqual(tuple(output["representation"].shape), (8, 12))
        gated = small_model().complete(
            views, mask, use_reliability=True, fusion="gated_concat")
        self.assertEqual(tuple(gated["representation"].shape), (8, 12))
        self.assertEqual(tuple(gated["gated_concat_representation"].shape),
                         (8, 12))
        fallback = small_model().complete(
            views, mask, use_reliability=True, fusion="fallback_concat")
        self.assertEqual(tuple(fallback["representation"].shape), (8, 12))
        self.assertEqual(tuple(fallback["fallback_concat_representation"].shape),
                         (8, 12))
        adaptive = small_model().complete(
            views, mask, use_reliability=True, source_temperature=1.0,
            risk_normalization="per_sample_range", fusion="concat")
        missing = mask == 0
        weight_sums = adaptive["path_weight"].sum(dim=1)
        self.assertTrue(torch.allclose(
            weight_sums[missing], torch.ones_like(weight_sums[missing])))


class LossTest(unittest.TestCase):

    def test_all_masked_losses_backpropagate(self):
        views, mask = synthetic_batch()
        model = small_model()
        latents = model.encode_observed(views, mask)
        reconstruction, _ = masked_reconstruction_loss(model, views, mask, latents)
        pair, _ = masked_pair_loss(latents, mask, alpha=1.0)
        indexer = EnvironmentIndexer(mask.numpy(), min_group_size=1)
        environment_ids = torch.from_numpy(indexer.ids)
        groups, _, diagnostics = prediction_environment_losses(
            model,
            latents,
            mask,
            environment_ids,
            use_heteroscedastic=True,
            cycle_weight=0.5,
            structure_weight=1.0,
        )
        state = GroupDROState(indexer.group_keys, eta=0.1)
        prediction, record = state.combine(groups, robust=True)
        total = reconstruction + pair + prediction
        total.backward()
        self.assertTrue(torch.isfinite(total).item())
        self.assertTrue(record)
        self.assertTrue(torch.isfinite(diagnostics["cycle_mse"]).item())
        self.assertEqual(set(diagnostics["per_group"]), set(groups))
        for values in diagnostics["per_group"].values():
            self.assertTrue(torch.isfinite(values["prediction_mse"]).item())
            self.assertTrue(torch.isfinite(values["cycle_mse"]).item())
            self.assertTrue(torch.isfinite(values["structure_mse"]).item())
            self.assertTrue(torch.isfinite(
                values["normalized_structure_mse"]).item())
            self.assertTrue(torch.isfinite(values["mean_log_variance"]).item())
        predictor = model.predictors[model.predictor_key(0, 1)]
        self.assertIsNotNone(predictor.mean_head.weight.grad)

    def test_empty_pair_set_is_zero_not_nan(self):
        views, _ = synthetic_batch()
        mask = torch.tensor([
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0],
            [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0],
        ], dtype=torch.long)
        model = small_model()
        latents = model.encode_observed(views, mask)
        pair, counts = masked_pair_loss(latents, mask)
        self.assertEqual(float(pair.item()), 0.0)
        self.assertEqual(sum(counts.values()), 0)
        self.assertTrue(torch.isfinite(pair).item())

    def test_groupdro_increases_weight_on_higher_risk(self):
        state = GroupDROState(["low", "high"], eta=0.5)
        low = torch.tensor(1.0, requires_grad=True)
        high = torch.tensor(3.0, requires_grad=True)
        combined, _ = state.combine({0: low, 1: high}, robust=True)
        self.assertGreater(state.probabilities()[1], state.probabilities()[0])
        combined.backward()
        self.assertIsNotNone(low.grad)

    def test_epoch_style_groupdro_update_accepts_scalar_means(self):
        state = GroupDROState(["low", "high"], eta=0.5)
        state.update({0: 1.0, 1: 3.0})
        self.assertGreater(state.probabilities()[1], state.probabilities()[0])


class ReliabilityOutputTest(unittest.TestCase):

    def test_hidden_target_diagnostics_are_traceable(self):
        views, mask = synthetic_batch()
        model = small_model()
        model.eval()
        completion = model.complete(views, mask, use_reliability=True)
        summary, records = evaluate_completion_reliability(
            model,
            views,
            mask,
            completion,
            sample_indices=np.arange(100, 108),
        )
        self.assertGreater(summary["missing_target_count"], 0)
        self.assertIsNotNone(summary["completion_mse"])
        self.assertTrue(records)
        self.assertGreaterEqual(records[0]["sample_index"], 100)
        self.assertIn("prediction_mse", records[0])


if __name__ == "__main__":
    unittest.main()
