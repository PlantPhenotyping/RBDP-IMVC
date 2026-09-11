"""Training and evaluation loops for controlled RBDP-IMVC ablations."""

import os
import random
import tempfile
import time

import numpy as np
import torch

from .losses import EnvironmentIndexer
from .losses import GroupDROState
from .losses import differentiable_zero
from .losses import masked_pair_loss
from .losses import masked_reconstruction_loss
from .losses import prediction_environment_losses
from .metrics import evaluate_embedding, pattern_metrics
from .models import RBDPModel
from .reliability import evaluate_completion_reliability


def seed_everything(seed):
    """Seed Python, NumPy, and Torch without reading dataset labels."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 3)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_model(view_dims, config):
    """Build an :class:`RBDPModel` from a resolved model configuration."""
    model_config = config["model"]
    hidden = model_config["hidden_dims"]
    if hidden and isinstance(hidden[0], int):
        hidden = [list(hidden) for _ in view_dims]
    return RBDPModel(
        view_dims=view_dims,
        hidden_dims=hidden,
        latent_dim=model_config["latent_dim"],
        predictor_hidden_dims=model_config["predictor_hidden_dims"],
        batch_norm=model_config.get("batch_norm", True),
        min_log_variance=model_config.get("min_log_variance", -8.0),
        max_log_variance=model_config.get("max_log_variance", 3.0),
    )


def parameter_count(model):
    return int(sum(parameter.numel() for parameter in model.parameters()))


def primary_anchored_backward(primary_loss, auxiliary_loss, parameters,
                              auxiliary_weight=1.0, epsilon=1.0e-12):
    """Backpropagate an auxiliary task without opposing a primary gradient.

    This is an asymmetric, primary-preserving adaptation of PCGrad.  Inputs
    are two scalar tensors and an iterable of trainable parameters.  Gradients
    are computed on their shared parameters.  If their dot product is
    negative, the auxiliary gradient is projected onto the normal plane of
    the primary gradient before the weighted gradients are added.  Parameters
    used by only one objective retain that objective's gradient.

    The function writes ``parameter.grad`` in place and returns scalar logging
    diagnostics: shared-gradient cosine, binary conflict indicator, removed
    auxiliary fraction, and both shared gradient norms.  Positive cosine means
    local agreement; a projection fraction of zero means no surgery occurred.
    It does not step or zero an optimizer.
    """
    parameters = [parameter for parameter in parameters
                  if parameter.requires_grad]
    auxiliary_weight = float(auxiliary_weight)
    if auxiliary_weight < 0:
        raise ValueError("auxiliary_weight must be nonnegative")
    primary_gradients = torch.autograd.grad(
        primary_loss, parameters, retain_graph=True, allow_unused=True)
    auxiliary_gradients = torch.autograd.grad(
        auxiliary_loss, parameters, retain_graph=False, allow_unused=True)
    shared = [(first, second) for first, second in
              zip(primary_gradients, auxiliary_gradients)
              if first is not None and second is not None]
    if shared:
        dot = sum(torch.sum(first * second) for first, second in shared)
        primary_norm_sq = sum(torch.sum(first * first) for first, _ in shared)
        auxiliary_norm_sq = sum(torch.sum(second * second) for _, second in shared)
        denominator = torch.sqrt(primary_norm_sq * auxiliary_norm_sq).clamp(
            min=float(epsilon))
        cosine = dot / denominator
        conflict = float(dot.detach().item() < 0.0 and
                         primary_norm_sq.detach().item() > float(epsilon))
        coefficient = (dot / primary_norm_sq.clamp(min=float(epsilon))
                       if conflict else dot.new_tensor(0.0))
        primary_norm = float(torch.sqrt(primary_norm_sq).detach().item())
        auxiliary_norm = float(torch.sqrt(auxiliary_norm_sq).detach().item())
        cosine_value = float(cosine.detach().item())
    else:
        conflict = 0.0
        coefficient = None
        primary_norm = 0.0
        auxiliary_norm = 0.0
        cosine_value = 0.0

    for parameter, primary_gradient, auxiliary_gradient in zip(
            parameters, primary_gradients, auxiliary_gradients):
        if primary_gradient is None and auxiliary_gradient is None:
            parameter.grad = None
            continue
        if primary_gradient is None:
            merged = auxiliary_weight * auxiliary_gradient
        elif auxiliary_gradient is None:
            merged = primary_gradient
        else:
            projected = auxiliary_gradient
            if conflict:
                projected = auxiliary_gradient - coefficient * primary_gradient
            merged = primary_gradient + auxiliary_weight * projected
        parameter.grad = merged.detach()
    return {
        "gradient_cosine": cosine_value,
        "gradient_conflict": conflict,
        "projection_fraction": max(0.0, -cosine_value) if conflict else 0.0,
        "primary_gradient_norm": primary_norm,
        "auxiliary_gradient_norm": auxiliary_norm,
    }


def _batch_tensors(views, mask, indices, device):
    batch_mask_np = np.asarray(mask[indices], dtype=np.int64)
    batch_mask = torch.from_numpy(batch_mask_np).long().to(device)
    batch_views = []
    for view_index, values in enumerate(views):
        selected = np.asarray(values[indices], dtype=np.float32)
        # Zeroing is a second safety boundary: model code cannot accidentally
        # consume hidden benchmark features even if an index check regresses.
        selected = selected * batch_mask_np[:, view_index:view_index + 1]
        batch_views.append(torch.from_numpy(selected).float().to(device))
    return batch_views, batch_mask


def fit_model(model, train_views, train_mask, config, device, log=None,
              epoch_callback=None):
    """Fit one ablation without labels and return traceable epoch records.

    Inputs are full NumPy training features ``views[v] [N_train,D_v]`` and a
    binary availability mask ``[N_train,V]``.  Hidden feature values are zeroed
    before tensors reach the model.  The optional ``epoch_callback`` receives
    ``(epoch_number, model)`` after each completed epoch; it is intended for
    read-only visualization snapshots and must not update parameters.  The
    returned tuple contains epoch curves, environment-risk rows, the final
    GroupDRO state, and optimizer state.
    """
    training = config["training"]
    losses_config = config["loss"]
    ablation = config["ablation"]
    epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    seed_everything(training["seed"])
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )

    indexer = EnvironmentIndexer(
        train_mask, min_group_size=losses_config["min_group_size"])
    if not indexer.group_keys:
        raise ValueError("training mask contains no co-observed view pair")
    dro_state = GroupDROState(
        indexer.group_keys, eta=losses_config["groupdro_eta"])
    update_frequency = losses_config.get("groupdro_update_frequency", "epoch")
    if update_frequency not in ("batch", "epoch"):
        raise ValueError("groupdro_update_frequency must be batch or epoch")

    if ablation["complete_only"]:
        pool = np.flatnonzero(np.asarray(train_mask).sum(axis=1) == model.n_views)
    else:
        pool = np.arange(train_mask.shape[0])
    if pool.size == 0:
        raise ValueError("the selected ablation has no usable training samples")
    rng = np.random.RandomState(int(training["seed"]) + 101)
    curves = []
    environment_rows = []
    start_time = time.time()

    for epoch in range(epochs):
        order = pool[rng.permutation(pool.size)]
        totals = {
            "loss": 0.0,
            "reconstruction": 0.0,
            "pair": 0.0,
            "prediction": 0.0,
            "prediction_mse": 0.0,
            "heteroscedastic_nll": 0.0,
            "cycle_mse": 0.0,
            "structure_mse": 0.0,
            "normalized_structure_mse": 0.0,
            "mean_log_variance": 0.0,
            "primary_objective": 0.0,
            "auxiliary_objective": 0.0,
            "gradient_cosine": 0.0,
            "gradient_conflict": 0.0,
            "projection_fraction": 0.0,
            "primary_gradient_norm": 0.0,
            "auxiliary_gradient_norm": 0.0,
        }
        seen = 0
        epoch_group_sums = {}
        epoch_group_counts = {}
        epoch_group_component_sums = {}
        for start in range(0, order.size, batch_size):
            indices = order[start:start + batch_size]
            batch_views, batch_mask = _batch_tensors(
                train_views, train_mask, indices, device)
            environment_ids = torch.from_numpy(indexer.ids[indices]).long().to(device)
            latents = model.encode_observed(batch_views, batch_mask)
            reconstruction, _ = masked_reconstruction_loss(
                model, batch_views, batch_mask, latents)
            pair, _ = masked_pair_loss(
                latents, batch_mask, alpha=losses_config["mi_alpha"])

            prediction = differentiable_zero(latents[0])
            base_prediction = prediction
            diagnostics = {
                "prediction_mse": prediction,
                "heteroscedastic_nll": prediction,
                "cycle_mse": prediction,
                "structure_mse": prediction,
                "normalized_structure_mse": prediction,
                "mean_log_variance": prediction,
            }
            group_losses = {}
            if epoch >= int(training.get("start_prediction_epoch", 0)):
                group_losses, group_counts, diagnostics = prediction_environment_losses(
                    model,
                    latents,
                    batch_mask,
                    environment_ids,
                    use_heteroscedastic=ablation["heteroscedastic"],
                    cycle_weight=(losses_config["cycle_weight"]
                                  if ablation["cycle"] else 0.0),
                    structure_weight=losses_config.get("structure_weight", 0.0),
                    stop_gradient=losses_config.get("stop_gradient", True),
                )
                if group_losses:
                    base_group_losses = {
                        identifier: diagnostics["per_group"][identifier][
                            "prediction_mse"]
                        for identifier in group_losses
                    }
                    base_prediction, _ = dro_state.combine(
                        base_group_losses, robust=False, update=False)
                    prediction, _ = dro_state.combine(
                        group_losses,
                        robust=ablation["groupdro"],
                        update=(update_frequency == "batch"),
                    )
                    for identifier, group_loss in group_losses.items():
                        key = indexer.key(identifier)
                        count = group_counts[identifier]
                        epoch_group_sums[key] = epoch_group_sums.get(key, 0.0) + (
                            float(group_loss.detach().item()) * count)
                        epoch_group_counts[key] = epoch_group_counts.get(key, 0) + count
                        component_sums = epoch_group_component_sums.setdefault(
                            key, {"prediction_mse": 0.0,
                                  "heteroscedastic_nll": 0.0,
                                  "cycle_mse": 0.0,
                                  "structure_mse": 0.0,
                                  "normalized_structure_mse": 0.0,
                                  "mean_log_variance": 0.0})
                        for name, value in diagnostics["per_group"][identifier].items():
                            component_sums[name] += float(
                                value.detach().item()) * count

            ordinary_total = (
                float(losses_config["reconstruction_weight"]) * reconstruction +
                float(losses_config["pair_weight"]) * pair +
                float(losses_config["prediction_weight"]) * prediction
            )
            primary_objective = ordinary_total
            auxiliary_objective = differentiable_zero(latents[0])
            gradient_record = {
                "gradient_cosine": 0.0,
                "gradient_conflict": 0.0,
                "projection_fraction": 0.0,
                "primary_gradient_norm": 0.0,
                "auxiliary_gradient_norm": 0.0,
            }
            conflict_safe = bool(ablation.get("conflict_safe", False))
            if conflict_safe and group_losses:
                primary_objective = (
                    float(losses_config["reconstruction_weight"]) * reconstruction +
                    float(losses_config["pair_weight"]) * pair +
                    float(losses_config["prediction_weight"]) * base_prediction
                )
                auxiliary_objective = (
                    float(losses_config["prediction_weight"]) * prediction)
                auxiliary_weight = float(
                    losses_config.get("auxiliary_weight", 1.0))
                total = primary_objective + auxiliary_weight * auxiliary_objective
            else:
                total = ordinary_total
            if not torch.isfinite(total).item():
                raise FloatingPointError(
                    "non-finite training loss at epoch %d batch %d" %
                    (epoch + 1, start // batch_size + 1))
            optimizer.zero_grad()
            if conflict_safe and group_losses:
                gradient_record = primary_anchored_backward(
                    primary_objective,
                    auxiliary_objective,
                    model.parameters(),
                    auxiliary_weight=losses_config.get("auxiliary_weight", 1.0),
                )
            else:
                total.backward()
            gradient_clip = float(training.get("gradient_clip", 0.0))
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            count = int(indices.size)
            seen += count
            for name, value in (
                    ("loss", total),
                    ("reconstruction", reconstruction),
                    ("pair", pair),
                    ("prediction", prediction),
                    ("prediction_mse", diagnostics["prediction_mse"]),
                    ("heteroscedastic_nll",
                     diagnostics["heteroscedastic_nll"]),
                    ("cycle_mse", diagnostics["cycle_mse"]),
                    ("structure_mse", diagnostics["structure_mse"]),
                    ("normalized_structure_mse",
                     diagnostics["normalized_structure_mse"]),
                    ("mean_log_variance", diagnostics["mean_log_variance"])):
                totals[name] += float(value.detach().item()) * count
            totals["primary_objective"] += float(
                primary_objective.detach().item()) * count
            totals["auxiliary_objective"] += float(
                auxiliary_objective.detach().item()) * count
            for name, value in gradient_record.items():
                totals[name] += float(value) * count

        elapsed = time.time() - start_time
        curve = {"epoch": epoch + 1, "elapsed_seconds": elapsed,
                 "training_samples": int(pool.size)}
        for name, value in totals.items():
            curve[name] = value / max(seen, 1)
        curves.append(curve)

        if ablation["groupdro"] and update_frequency == "epoch":
            dro_state.update({
                indexer.key_to_id[key]: epoch_group_sums[key] /
                epoch_group_counts[key]
                for key in epoch_group_sums
            })
        probabilities = dro_state.probabilities()
        for key in sorted(epoch_group_sums):
            identifier = indexer.key_to_id[key]
            environment_rows.append({
                "epoch": epoch + 1,
                "environment": key,
                "count": epoch_group_counts[key],
                "mean_loss": epoch_group_sums[key] / epoch_group_counts[key],
                "prediction_mse": epoch_group_component_sums[key][
                    "prediction_mse"] / epoch_group_counts[key],
                "heteroscedastic_nll": epoch_group_component_sums[key][
                    "heteroscedastic_nll"] / epoch_group_counts[key],
                "cycle_mse": epoch_group_component_sums[key][
                    "cycle_mse"] / epoch_group_counts[key],
                "structure_mse": epoch_group_component_sums[key][
                    "structure_mse"] / epoch_group_counts[key],
                "normalized_structure_mse": epoch_group_component_sums[key][
                    "normalized_structure_mse"] / epoch_group_counts[key],
                "mean_log_variance": epoch_group_component_sums[key][
                    "mean_log_variance"] / epoch_group_counts[key],
                "groupdro_probability": probabilities[identifier],
                "robust_enabled": int(bool(ablation["groupdro"])),
            })
        if log is not None:
            log("epoch=%d/%d loss=%.6f rec=%.6f pair=%.6f pred=%.6f" % (
                epoch + 1,
                epochs,
                curve["loss"],
                curve["reconstruction"],
                curve["pair"],
                curve["prediction"],
            ))
        if epoch_callback is not None:
            was_training = model.training
            model.eval()
            try:
                epoch_callback(epoch + 1, model)
            finally:
                model.train(was_training)
    return {
        "curves": curves,
        "environment_rows": environment_rows,
        "environment_index": indexer.summary(),
        "groupdro_state": dro_state.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "training_seconds": time.time() - start_time,
        "training_sample_count": int(pool.size),
    }


def _serializable_mapping(mapping):
    return {str(key): int(value) for key, value in mapping.items()}


def _extract_coverage_rows(reliability_summary):
    rows = []
    ranking = reliability_summary.get("path_ranking")
    if ranking is not None:
        for coverage, error in zip(ranking.pop("coverage"),
                                   ranking.pop("selective_error")):
            rows.append({"scope": "all_paths", "coverage": coverage,
                         "selective_error": error})
    for path, values in reliability_summary.get("per_path", {}).items():
        for coverage, error in zip(values.pop("coverage"),
                                   values.pop("selective_error")):
            rows.append({"scope": path, "coverage": coverage,
                         "selective_error": error})
    return rows


def evaluate_model(model, full_test_views, test_labels, test_mask, test_indices,
                   config, device):
    """Complete test views, cluster once, and compute traceable diagnostics.

    Labels are first accessed here, after optimization has finished.  They are
    used only for the final global Hungarian mapping and reported metrics.
    Hidden test features are used only by completion-error diagnostics, never
    by the representation returned from ``model.complete``.
    """
    model.eval()
    mask_np = np.asarray(test_mask, dtype=np.int64)
    mask_tensor = torch.from_numpy(mask_np).long().to(device)
    full_tensors = [torch.from_numpy(np.asarray(view, dtype=np.float32)).to(device)
                    for view in full_test_views]
    masked_tensors = [
        full_tensor * mask_tensor[:, index:index + 1].float()
        for index, full_tensor in enumerate(full_tensors)
    ]
    start = time.time()
    with torch.no_grad():
        completion = model.complete(
            masked_tensors,
            mask_tensor,
            use_reliability=config["ablation"]["reliability"],
            gamma=config["reliability"]["gamma"],
            source_temperature=config["reliability"]["source_temperature"],
            confidence_temperature=config["reliability"][
                "confidence_temperature"],
            risk_normalization=config["reliability"].get(
                "risk_normalization", "none"),
            fusion=config["ablation"]["fusion"],
        )
    inference_seconds = time.time() - start
    embedding = completion["representation"].detach().cpu().numpy()
    predicted, mapping, overall = evaluate_embedding(
        embedding,
        test_labels,
        n_clusters=np.unique(test_labels).size,
        kmeans_seed=config["evaluation"]["kmeans_seed"],
        n_init=config["evaluation"]["kmeans_n_init"],
    )
    patterns = pattern_metrics(
        test_labels,
        predicted,
        mask_np,
        mapping=mapping,
        min_count=config["evaluation"].get("min_pattern_count", 1),
    )
    reliability_summary, reliability_records = evaluate_completion_reliability(
        model,
        full_tensors,
        mask_tensor,
        completion,
        gamma=config["reliability"]["gamma"],
        sample_indices=test_indices,
    )
    coverage_rows = _extract_coverage_rows(reliability_summary)
    arrays = {
        "embedding": embedding,
        "predicted_clusters": predicted,
        "labels": np.asarray(test_labels),
        "sample_indices": np.asarray(test_indices),
        "mask": mask_np.astype(np.uint8),
    }
    for name in (
            "observed_latents", "completed_latents", "view_confidence",
            "path_risk", "path_log_variance", "path_cycle", "path_weight",
            "path_valid"):
        arrays[name] = completion[name].detach().cpu().numpy()
    metrics = {
        "clustering": overall,
        "pattern_robustness": patterns,
        "hungarian_mapping": _serializable_mapping(mapping),
        "reliability": reliability_summary,
        "inference_seconds": inference_seconds,
        "embedding_shape": list(embedding.shape),
    }
    return {
        "metrics": metrics,
        "arrays": arrays,
        "reliability_rows": reliability_records,
        "coverage_rows": coverage_rows,
    }


def save_checkpoint(path, model, optimizer_state, config_hash_value,
                    groupdro_state):
    """Atomically save model/optimizer/risk state for exact resumption."""
    destination = os.path.abspath(path)
    parent = os.path.dirname(destination)
    if not os.path.isdir(parent):
        os.makedirs(parent)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".checkpoint-", suffix=".pt", dir=parent)
    os.close(descriptor)
    try:
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer_state,
            "config_hash": config_hash_value,
            "groupdro_state": groupdro_state,
        }, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination
