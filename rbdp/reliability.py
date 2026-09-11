"""Diagnostics for self-verifiable completion reliability."""

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import reliability_metrics


def _mask_patterns(mask):
    value = mask.detach().cpu().numpy().astype(np.uint8)
    return np.asarray(["".join(str(int(bit)) for bit in row) for row in value])


def evaluate_completion_reliability(model, full_views, mask, completion,
                                    gamma=1.0, sample_indices=None):
    """Compare self-risk with hidden benchmark latents for missing targets.

    Args:
        model: Trained :class:`rbdp.models.RBDPModel` in evaluation mode.
        full_views: Length-``V`` feature list, each ``FloatTensor [B,D_v]``.
            These complete benchmark values are used only to compute diagnostic
            targets and never enter completion or model selection.
        mask: Binary tensor ``[B,V]`` used by ``model.complete``.
        completion: Dict returned by ``model.complete`` for the same batch.
        gamma: Cycle-residual coefficient used in path risk.
        sample_indices: Optional original dataset indices ``[B]`` for CSV trace.

    Returns:
        ``(summary, records)``.  ``summary`` contains global/per-path Spearman,
        AURC, prediction MSE, fused completion MSE, and cosine error. ``records``
        is a list of scalar dictionaries suitable for ``reliability.csv``.
    """
    batch_size = int(mask.size(0))
    if sample_indices is None:
        sample_indices = np.arange(batch_size, dtype=np.int64)
    else:
        sample_indices = np.asarray(sample_indices).reshape(-1)
    if sample_indices.shape[0] != batch_size:
        raise ValueError("sample_indices must have length B")

    with torch.no_grad():
        true_latents = model.encode_all(full_views)
        observed_latents = [completion["observed_latents"][:, view]
                            for view in range(model.n_views)]
        patterns = _mask_patterns(mask)
        records = []
        all_risks = []
        all_errors = []
        path_values = {}
        for source in range(model.n_views):
            for target in range(model.n_views):
                if source == target:
                    continue
                valid = (mask[:, source] > 0) & (mask[:, target] == 0)
                if not torch.any(valid).item():
                    continue
                mean, log_variance = model.predict(
                    source, target, observed_latents[source][valid])
                returned, _ = model.predict(target, source, mean)
                cycle = torch.mean(
                    (returned - observed_latents[source][valid]) ** 2, dim=1)
                risk = torch.exp(log_variance.reshape(-1)) + float(gamma) * cycle
                error = torch.mean((mean - true_latents[target][valid]) ** 2, dim=1)
                valid_indices = torch.nonzero(valid).reshape(-1).cpu().numpy()
                risk_values = risk.cpu().numpy()
                error_values = error.cpu().numpy()
                log_values = log_variance.reshape(-1).cpu().numpy()
                cycle_values = cycle.cpu().numpy()
                key = "v%d_to_v%d" % (source, target)
                path_values[key] = (risk_values, error_values)
                all_risks.append(risk_values)
                all_errors.append(error_values)
                for offset, local_index in enumerate(valid_indices):
                    records.append({
                        "sample_index": int(sample_indices[local_index]),
                        "pattern": str(patterns[local_index]),
                        "source_view": source,
                        "target_view": target,
                        "log_variance": float(log_values[offset]),
                        "cycle_mse": float(cycle_values[offset]),
                        "risk": float(risk_values[offset]),
                        "prediction_mse": float(error_values[offset]),
                    })

        if all_risks:
            global_risk = np.concatenate(all_risks)
            global_error = np.concatenate(all_errors)
            ranking = reliability_metrics(global_risk, global_error)
            per_path = {
                key: reliability_metrics(values[0], values[1])
                for key, values in sorted(path_values.items())
            }
        else:
            ranking = None
            per_path = {}

        completed = completion["completed_latents"]
        fused_errors = []
        fused_cosine_errors = []
        per_target = {}
        for target in range(model.n_views):
            missing = mask[:, target] == 0
            count = int(missing.long().sum().item())
            if count == 0:
                continue
            predicted = completed[missing, target]
            expected = true_latents[target][missing]
            mse = torch.mean((predicted - expected) ** 2, dim=1)
            cosine_error = 1.0 - F.cosine_similarity(predicted, expected, dim=1)
            fused_errors.append(mse.cpu().numpy())
            fused_cosine_errors.append(cosine_error.cpu().numpy())
            per_target["view_%d" % target] = {
                "count": count,
                "mse": float(mse.mean().item()),
                "cosine_error": float(cosine_error.mean().item()),
                "mean_confidence": float(
                    completion["view_confidence"][missing, target].mean().item()),
            }
        summary = {
            "path_ranking": ranking,
            "per_path": per_path,
            "per_target": per_target,
            "missing_target_count": int(sum(
                value["count"] for value in per_target.values())),
            "completion_mse": (float(np.concatenate(fused_errors).mean())
                               if fused_errors else None),
            "completion_cosine_error": (
                float(np.concatenate(fused_cosine_errors).mean())
                if fused_cosine_errors else None),
        }
    return summary, records
