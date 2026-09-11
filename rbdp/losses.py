"""Mask-aware objectives and missing-environment risk aggregation."""

import math

import numpy as np
import torch
import torch.nn.functional as functional


def differentiable_zero(reference):
    """Return a scalar zero connected to ``reference`` for empty valid sets."""
    return reference.sum() * 0.0


def mutual_information_loss(first, second, alpha=10.0, epsilon=1.0e-12):
    """DCP-style cross-view mutual-information loss.

    Both inputs are probability-simplex tensors ``[B_st, d]`` for samples
    where both views are observed.  The output is a scalar.  At least two rows
    are required by the caller; no labels enter this objective.
    """
    if first.dim() != 2 or second.shape != first.shape:
        raise ValueError("paired latents must have equal shape [B_st, d]")
    joint = torch.sum(first.unsqueeze(2) * second.unsqueeze(1), dim=0)
    joint = 0.5 * (joint + joint.t())
    joint = joint / torch.clamp(joint.sum(), min=float(epsilon))
    marginal_i = joint.sum(dim=1, keepdim=True).expand_as(joint)
    marginal_j = joint.sum(dim=0, keepdim=True).expand_as(joint)
    joint = torch.clamp(joint, min=float(epsilon))
    marginal_i = torch.clamp(marginal_i, min=float(epsilon))
    marginal_j = torch.clamp(marginal_j, min=float(epsilon))
    return torch.sum(-joint * (
        torch.log(joint) - float(alpha) * torch.log(marginal_i) -
        float(alpha) * torch.log(marginal_j)))


def masked_reconstruction_loss(model, views, mask, latents):
    """Reconstruct every observed view entry, normalized per sample/view.

    ``views[v]`` has shape ``[B,D_v]``, ``mask`` is ``[B,V]``, and
    ``latents[v]`` is ``[B,d]`` with zeros at missing positions.  Each observed
    sample-view contributes its feature-mean squared error, preventing a
    high-dimensional view from dominating solely because it has more columns.
    The return is ``(scalar_loss, per_view_dict)``.  Empty views contribute a
    differentiable zero rather than NaN.
    """
    total = differentiable_zero(latents[0])
    observed_total = 0
    per_view = {}
    for view_index in range(model.n_views):
        observed = mask[:, view_index] > 0
        count = int(observed.long().sum().item())
        if count == 0:
            loss = differentiable_zero(latents[view_index])
        else:
            reconstructed = model.autoencoders[view_index].decode(
                latents[view_index][observed])
            errors = torch.mean(
                (reconstructed - views[view_index][observed]) ** 2, dim=1)
            loss = errors.mean()
            total = total + errors.sum()
            observed_total += count
        per_view["view_%d" % view_index] = loss
    if observed_total:
        total = total / float(observed_total)
    return total, per_view


def masked_pair_loss(latents, mask, alpha=10.0):
    """Average pair consistency over all view pairs with >=2 co-observations."""
    losses = []
    counts = {}
    n_views = len(latents)
    for first in range(n_views):
        for second in range(first + 1, n_views):
            valid = (mask[:, first] > 0) & (mask[:, second] > 0)
            count = int(valid.long().sum().item())
            key = "v%d-v%d" % (first, second)
            counts[key] = count
            if count >= 2:
                losses.append(mutual_information_loss(
                    latents[first][valid], latents[second][valid], alpha=alpha))
    if not losses:
        return differentiable_zero(latents[0]), counts
    return torch.stack(losses).mean(), counts


class EnvironmentIndexer(object):
    """Assign rows to supported target/source-set missing environments.

    Input ``mask`` is a NumPy binary array ``[N,V]``.  For target ``t``, an
    environment uses the other observed source set if that exact set occurs at
    least ``min_group_size`` times while ``t`` is observed.  Rarer sets fall
    back to ``t<t>|fallback``.  ``ids [N,V]`` contains -1 where the target or
    every source is unavailable; otherwise it contains a stable integer ID.
    """

    def __init__(self, mask, min_group_size=16):
        value = np.asarray(mask)
        if value.ndim != 2 or value.shape[1] < 2:
            raise ValueError("mask must have shape [N,V] with V >= 2")
        if not np.all((value == 0) | (value == 1)):
            raise ValueError("mask must be binary")
        self.min_group_size = int(min_group_size)
        if self.min_group_size <= 0:
            raise ValueError("min_group_size must be positive")
        self.n_views = int(value.shape[1])
        raw = np.empty(value.shape, dtype=object)
        raw[:] = ""
        counts = {}
        for target in range(self.n_views):
            for row in range(value.shape[0]):
                if value[row, target] == 0:
                    continue
                source_bits = value[row].copy()
                source_bits[target] = 0
                source_count = int(source_bits.sum())
                if source_count == 0:
                    continue
                bits = "".join(str(int(bit)) for bit in source_bits)
                key = "t%d|k%d|s%s" % (target, source_count, bits)
                raw[row, target] = key
                counts[key] = counts.get(key, 0) + 1

        resolved = np.empty(value.shape, dtype=object)
        resolved[:] = ""
        keys = set()
        for target in range(self.n_views):
            for row in range(value.shape[0]):
                key = raw[row, target]
                if not key:
                    continue
                if counts[key] < self.min_group_size:
                    key = "t%d|fallback" % target
                resolved[row, target] = key
                keys.add(key)
        self.group_keys = sorted(keys)
        self.key_to_id = {key: index for index, key in enumerate(self.group_keys)}
        self.ids = np.full(value.shape, -1, dtype=np.int64)
        for key, identifier in self.key_to_id.items():
            self.ids[resolved == key] = identifier

    def key(self, identifier):
        return self.group_keys[int(identifier)]

    def summary(self):
        counts = {}
        for identifier, key in enumerate(self.group_keys):
            counts[key] = int(np.sum(self.ids == identifier))
        return {
            "min_group_size": self.min_group_size,
            "group_count": len(self.group_keys),
            "counts": counts,
        }


def prediction_environment_losses(model, latents, mask, environment_ids,
                                  use_heteroscedastic=True,
                                  cycle_weight=1.0,
                                  structure_weight=0.0,
                                  stop_gradient=True):
    """Compute directed prediction risks grouped by missing environment.

    Args:
        model: :class:`rbdp.models.RBDPModel`.
        latents: Length-``V`` list, each ``FloatTensor [B,d]``.
        mask: Binary tensor ``[B,V]``.
        environment_ids: LongTensor ``[B,V]`` from
            :class:`EnvironmentIndexer`; its second axis is target view.
        use_heteroscedastic: Use ``exp(-logvar)*MSE + logvar`` when true,
            otherwise deterministic latent MSE.
        cycle_weight: Nonnegative multiplier for returning the prediction to
            its source view.  Set zero for A0/A1.
        structure_weight: Nonnegative multiplier for preserving the pairwise
            cosine geometry of the true target latent inside each missing
            environment.  Each raw environment loss is divided by the detached
            batch mean across environments, avoiding dataset-specific tuning
            caused by different raw geometry scales.  This is label-free and
            operates only on co-observed training samples.
        stop_gradient: Detach target/source latent supervision when true.

    Returns:
        ``(group_losses, group_counts, diagnostics)``.  Group losses are scalar
        tensors retaining gradients.  Diagnostics contain batch scalar tensors
        for prediction MSE, cycle MSE, and log variance, plus ``per_group``
        tensors for the same raw components.  This separation is important:
        the heteroscedastic NLL can be negative and must not be mistaken for a
        raw completion error.  Empty valid sets return differentiable zeros
        through the caller's aggregator.
    """
    if environment_ids.shape != mask.shape:
        raise ValueError("environment_ids must have the same [B,V] shape as mask")
    if cycle_weight < 0:
        raise ValueError("cycle_weight must be nonnegative")
    if structure_weight < 0:
        raise ValueError("structure_weight must be nonnegative")
    grouped = {}
    grouped_components = {}
    grouped_representations = {}
    counts = {}
    prediction_errors = []
    heteroscedastic_losses = []
    cycle_errors = []
    log_variances = []
    for source in range(model.n_views):
        for target in range(model.n_views):
            if source == target:
                continue
            valid = ((mask[:, source] > 0) & (mask[:, target] > 0) &
                     (environment_ids[:, target] >= 0))
            if not torch.any(valid).item():
                continue
            source_latent = latents[source][valid]
            target_latent = latents[target][valid]
            target_reference = target_latent.detach() if stop_gradient else target_latent
            mean, log_variance = model.predict(source, target, source_latent)
            prediction_mse = torch.mean((mean - target_reference) ** 2, dim=1)
            if use_heteroscedastic:
                heteroscedastic_nll = (
                    torch.exp(-log_variance.reshape(-1)) * prediction_mse +
                    log_variance.reshape(-1))
            else:
                heteroscedastic_nll = prediction_mse
            path_loss = heteroscedastic_nll

            if cycle_weight > 0:
                returned, _ = model.predict(target, source, mean)
                source_reference = source_latent.detach() if stop_gradient else source_latent
                cycle_mse = torch.mean((returned - source_reference) ** 2, dim=1)
                path_loss = path_loss + float(cycle_weight) * cycle_mse
            else:
                cycle_mse = prediction_mse * 0.0

            identifiers = environment_ids[:, target][valid]
            for identifier in torch.unique(identifiers):
                identifier_value = int(identifier.item())
                selected = identifiers == identifier
                values = path_loss[selected]
                grouped.setdefault(identifier_value, []).append(values)
                components = grouped_components.setdefault(identifier_value, {
                    "prediction_mse": [],
                    "heteroscedastic_nll": [],
                    "cycle_mse": [],
                    "mean_log_variance": [],
                })
                components["prediction_mse"].append(prediction_mse[selected])
                components["heteroscedastic_nll"].append(
                    heteroscedastic_nll[selected])
                components["cycle_mse"].append(cycle_mse[selected])
                components["mean_log_variance"].append(
                    log_variance.reshape(-1)[selected])
                representations = grouped_representations.setdefault(
                    identifier_value, {"predicted": [], "target": []})
                representations["predicted"].append(mean[selected])
                representations["target"].append(target_reference[selected])
                counts[identifier_value] = counts.get(identifier_value, 0) + int(
                    selected.long().sum().item())
            prediction_errors.append(prediction_mse)
            heteroscedastic_losses.append(heteroscedastic_nll)
            cycle_errors.append(cycle_mse)
            log_variances.append(log_variance.reshape(-1))

    reference = latents[0]
    pointwise_losses = {}
    raw_structure_losses = {}
    per_group = {}
    for identifier, values in grouped.items():
        pointwise = torch.cat(values, dim=0).mean()
        predicted = torch.cat(
            grouped_representations[identifier]["predicted"], dim=0)
        target = torch.cat(
            grouped_representations[identifier]["target"], dim=0)
        if predicted.shape[0] >= 2:
            predicted_unit = functional.normalize(predicted, p=2, dim=1)
            target_unit = functional.normalize(target, p=2, dim=1)
            predicted_geometry = torch.mm(predicted_unit, predicted_unit.t())
            target_geometry = torch.mm(target_unit, target_unit.t())
            structure_mse = torch.mean(
                (predicted_geometry - target_geometry) ** 2)
        else:
            structure_mse = differentiable_zero(predicted)
        pointwise_losses[identifier] = pointwise
        raw_structure_losses[identifier] = structure_mse
        per_group[identifier] = {
            name: torch.cat(component_values).mean()
            for name, component_values in grouped_components[identifier].items()
        }
        per_group[identifier]["structure_mse"] = structure_mse
    if raw_structure_losses:
        structure_scale = torch.stack([
            value.detach() for value in raw_structure_losses.values()
        ]).mean().clamp(min=1.0e-8)
    else:
        structure_scale = reference.new_tensor(1.0)
    group_losses = {}
    for identifier, pointwise in pointwise_losses.items():
        normalized_structure = raw_structure_losses[identifier] / structure_scale
        group_losses[identifier] = (
            pointwise + float(structure_weight) * normalized_structure)
        per_group[identifier]["normalized_structure_mse"] = normalized_structure
    diagnostics = {}
    for name, values in (
            ("prediction_mse", prediction_errors),
            ("heteroscedastic_nll", heteroscedastic_losses),
            ("cycle_mse", cycle_errors),
            ("mean_log_variance", log_variances)):
        diagnostics[name] = (torch.cat(values).mean() if values
                             else differentiable_zero(reference))
    total_group_count = float(sum(counts.values()))
    diagnostics["structure_mse"] = sum(
        per_group[identifier]["structure_mse"] *
        (float(counts[identifier]) / total_group_count)
        for identifier in per_group
    ) if per_group else differentiable_zero(reference)
    diagnostics["normalized_structure_mse"] = sum(
        per_group[identifier]["normalized_structure_mse"] *
        (float(counts[identifier]) / total_group_count)
        for identifier in per_group
    ) if per_group else differentiable_zero(reference)
    diagnostics["per_group"] = per_group
    return group_losses, counts, diagnostics


class GroupDROState(object):
    """Exponentiated-gradient environment weights inspired by GroupDRO.

    The class stores Python floats, not trainable parameters.  ``combine``
    updates present groups using detached losses, normalizes weights over all
    known environments, then renormalizes over groups present in the current
    batch so batch composition does not scale the objective.
    """

    def __init__(self, group_keys, eta=0.05):
        self.group_keys = list(group_keys)
        self.eta = float(eta)
        if self.eta < 0:
            raise ValueError("GroupDRO eta must be nonnegative")
        self.log_weights = {index: 0.0 for index in range(len(self.group_keys))}

    def probabilities(self):
        if not self.log_weights:
            return {}
        maximum = max(self.log_weights.values())
        raw = {key: math.exp(value - maximum)
               for key, value in self.log_weights.items()}
        total = sum(raw.values())
        return {key: value / total for key, value in raw.items()}

    def update(self, group_losses):
        """Update log weights once from scalar/tensor environment means."""
        unknown = sorted(set(group_losses) - set(self.log_weights))
        if unknown:
            raise KeyError("unknown environment IDs: %r" % unknown)
        for identifier, loss in group_losses.items():
            value = float(loss.detach().item()) if hasattr(loss, "detach") else float(loss)
            self.log_weights[identifier] += self.eta * value
        if self.log_weights:
            maximum = max(self.log_weights.values())
            for identifier in self.log_weights:
                self.log_weights[identifier] -= maximum

    def combine(self, group_losses, robust=True, update=True):
        if not group_losses:
            raise ValueError("group_losses must not be empty")
        unknown = sorted(set(group_losses) - set(self.log_weights))
        if unknown:
            raise KeyError("unknown environment IDs: %r" % unknown)
        if robust and update:
            self.update(group_losses)

        global_probabilities = self.probabilities()
        if robust:
            active_total = sum(global_probabilities[key] for key in group_losses)
            active = {key: global_probabilities[key] / active_total
                      for key in group_losses}
        else:
            active = {key: 1.0 / len(group_losses) for key in group_losses}
        combined = None
        for identifier, loss in group_losses.items():
            term = loss * float(active[identifier])
            combined = term if combined is None else combined + term
        record = {}
        for identifier, loss in group_losses.items():
            key = self.group_keys[identifier]
            record[key] = {
                "loss": float(loss.detach().item()),
                "global_weight": float(global_probabilities[identifier]),
                "active_weight": float(active[identifier]),
            }
        return combined, record

    def state_dict(self):
        probabilities = self.probabilities()
        return {
            "eta": self.eta,
            "groups": {
                self.group_keys[index]: {
                    "log_weight": float(self.log_weights[index]),
                    "probability": float(probabilities[index]),
                }
                for index in range(len(self.group_keys))
            },
        }
