"""Deterministic clustering, subgroup, and reliability metrics."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import f1_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from .masks import pattern_ids


def _one_dimensional(values, name):
    result = np.asarray(values).reshape(-1)
    if result.size == 0:
        raise ValueError("%s must not be empty" % name)
    return result


def hungarian_mapping(y_true, y_pred):
    """Map predicted cluster IDs to true IDs using the full evaluation set.

    The mapping is fit once globally.  It can then be reused for every missing
    pattern so subgroup accuracy cannot benefit from a separate label matching.
    """
    truth = _one_dimensional(y_true, "y_true")
    predicted = _one_dimensional(y_pred, "y_pred")
    if truth.shape[0] != predicted.shape[0]:
        raise ValueError("y_true and y_pred must have equal length")
    true_ids = np.unique(truth)
    predicted_ids = np.unique(predicted)
    contingency = np.zeros((predicted_ids.size, true_ids.size), dtype=np.int64)
    for row, predicted_id in enumerate(predicted_ids):
        selected = truth[predicted == predicted_id]
        for column, true_id in enumerate(true_ids):
            contingency[row, column] = int(np.sum(selected == true_id))

    rows, columns = linear_sum_assignment(-contingency)
    mapping = {}
    # Define a majority fallback for unmatched clusters in rectangular cases.
    for row, predicted_id in enumerate(predicted_ids):
        mapping[predicted_id.item() if hasattr(predicted_id, "item") else predicted_id] = (
            true_ids[int(np.argmax(contingency[row]))].item()
            if contingency.shape[1] else predicted_id
        )
    for row, column in zip(rows, columns):
        predicted_id = predicted_ids[row]
        true_id = true_ids[column]
        mapping[predicted_id.item() if hasattr(predicted_id, "item") else predicted_id] = (
            true_id.item() if hasattr(true_id, "item") else true_id
        )
    return mapping


def apply_mapping(y_pred, mapping):
    """Apply a cluster-to-class mapping and return a one-dimensional array."""
    predicted = _one_dimensional(y_pred, "y_pred")
    missing = [value for value in np.unique(predicted)
               if (value.item() if hasattr(value, "item") else value) not in mapping]
    if missing:
        raise KeyError("mapping is missing predicted IDs: %r" % missing)
    return np.asarray([
        mapping[value.item() if hasattr(value, "item") else value]
        for value in predicted
    ])


def clustering_metrics_from_labels(y_true, y_pred, mapping=None):
    """Compute ACC/NMI/ARI/AMI and mapped macro precision/recall/F1."""
    truth = _one_dimensional(y_true, "y_true")
    predicted = _one_dimensional(y_pred, "y_pred")
    if truth.shape[0] != predicted.shape[0]:
        raise ValueError("y_true and y_pred must have equal length")
    fitted_mapping = hungarian_mapping(truth, predicted) if mapping is None else mapping
    mapped = apply_mapping(predicted, fitted_mapping)
    return {
        "acc": float(np.mean(mapped == truth)),
        "nmi": float(normalized_mutual_info_score(truth, predicted)),
        "ari": float(adjusted_rand_score(truth, predicted)),
        "ami": float(adjusted_mutual_info_score(truth, predicted)),
        "precision_macro": float(precision_score(
            truth, mapped, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(
            truth, mapped, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(
            truth, mapped, average="macro", zero_division=0)),
    }


def cluster_embeddings(embedding, n_clusters, seed=0, n_init=20,
                       max_iter=300):
    """Run k-means with all randomness and restart counts made explicit.

    Args:
        embedding: Finite ``float`` array shaped ``[N, D]``.
        n_clusters: Requested number of clusters.
        seed: ``random_state`` passed to scikit-learn.
        n_init: Number of k-means restarts (an integer for sklearn 0.23).

    Returns:
        Predicted cluster IDs as ``int64 [N]``.
    """
    values = np.asarray(embedding)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("embedding must have non-empty shape [N, D]")
    if not np.isfinite(values).all():
        raise ValueError("embedding contains NaN or infinity")
    n_clusters = int(n_clusters)
    if n_clusters < 2 or n_clusters > values.shape[0]:
        raise ValueError("n_clusters must lie in [2, N]")
    model = KMeans(
        n_clusters=n_clusters,
        random_state=int(seed),
        n_init=int(n_init),
        max_iter=int(max_iter),
    )
    return model.fit_predict(values).astype(np.int64, copy=False)


def evaluate_embedding(embedding, y_true, n_clusters=None, kmeans_seed=0,
                       n_init=20):
    """Cluster ``embedding`` and return labels, mapping, and metric dict."""
    truth = _one_dimensional(y_true, "y_true")
    clusters = int(np.unique(truth).size if n_clusters is None else n_clusters)
    predicted = cluster_embeddings(
        embedding, clusters, seed=kmeans_seed, n_init=n_init)
    mapping = hungarian_mapping(truth, predicted)
    return predicted, mapping, clustering_metrics_from_labels(
        truth, predicted, mapping=mapping)


def pattern_metrics(y_true, y_pred, mask, mapping=None, min_count=1):
    """Evaluate every missing pattern under one global Hungarian mapping.

    ``mask`` has shape ``[N, V]`` and follows ``1=observed`` semantics.
    Average/worst pattern accuracy are unweighted across patterns with at least
    ``min_count`` rows; overall metrics remain sample-weighted.
    """
    truth = _one_dimensional(y_true, "y_true")
    predicted = _one_dimensional(y_pred, "y_pred")
    value = np.asarray(mask)
    if value.ndim != 2 or value.shape[0] != truth.shape[0]:
        raise ValueError("mask must have shape [len(y_true), V]")
    if predicted.shape[0] != truth.shape[0]:
        raise ValueError("y_true and y_pred must have equal length")
    min_count = int(min_count)
    if min_count <= 0:
        raise ValueError("min_count must be positive")

    fitted_mapping = hungarian_mapping(truth, predicted) if mapping is None else mapping
    mapped = apply_mapping(predicted, fitted_mapping)
    identifiers = pattern_ids(value)
    groups = {}
    accuracies = []
    for identifier in sorted(np.unique(identifiers)):
        selected = identifiers == identifier
        count = int(selected.sum())
        if count < min_count:
            continue
        accuracy = float(np.mean(mapped[selected] == truth[selected]))
        groups[str(identifier)] = {
            "count": count,
            "acc": accuracy,
            "nmi": float(normalized_mutual_info_score(
                truth[selected], predicted[selected])),
            "ari": float(adjusted_rand_score(
                truth[selected], predicted[selected])),
        }
        accuracies.append(accuracy)
    if not accuracies:
        raise ValueError("no mask pattern has at least min_count=%d rows" % min_count)
    return {
        "overall": clustering_metrics_from_labels(
            truth, predicted, mapping=fitted_mapping),
        "patterns": groups,
        "average_pattern_acc": float(np.mean(accuracies)),
        "worst_pattern_acc": float(np.min(accuracies)),
        "best_pattern_acc": float(np.max(accuracies)),
        "pattern_gap": float(np.max(accuracies) - np.min(accuracies)),
    }


def reliability_metrics(risk, error):
    """Measure whether lower predicted risk selects lower completion error.

    Both inputs have shape ``[N]``.  ``aurc`` is the mean prefix error after
    sorting samples from low to high risk; lower is better.  ``spearman`` is
    positive when risk correctly ranks error.  Constant inputs produce a JSON
    ``null``-compatible ``None`` Spearman value instead of NaN.
    """
    risks = _one_dimensional(risk, "risk").astype(np.float64)
    errors = _one_dimensional(error, "error").astype(np.float64)
    if risks.shape[0] != errors.shape[0]:
        raise ValueError("risk and error must have equal length")
    if not np.isfinite(risks).all() or not np.isfinite(errors).all():
        raise ValueError("risk and error must be finite")
    order = np.argsort(risks, kind="mergesort")
    sorted_errors = errors[order]
    coverage = np.arange(1, errors.size + 1, dtype=np.float64) / errors.size
    selective_error = np.cumsum(sorted_errors) / np.arange(
        1, errors.size + 1, dtype=np.float64)
    coefficient = float(spearmanr(risks, errors).correlation)
    if not np.isfinite(coefficient):
        coefficient_value = None
    else:
        coefficient_value = coefficient
    return {
        "spearman": coefficient_value,
        "aurc": float(np.mean(selective_error)),
        "mean_error": float(errors.mean()),
        "coverage": [float(value) for value in coverage],
        "selective_error": [float(value) for value in selective_error],
    }
