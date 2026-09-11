"""Controlled and reproducible missing-view mask generation.

Mask semantics are deliberately explicit:

* ``incomplete_rate`` is the fraction of rows with at least one missing view;
* ``observation_rate`` is the requested observed-entry fraction inside those
  incomplete rows, subject to retaining at least one and missing at least one
  view per such row;
* one means observed and zero means missing.

All mechanisms exactly preserve the number of incomplete rows and the total
observation budget.  They differ only in which rows/views receive missingness.
"""

import hashlib
import json
import os
import tempfile

import numpy as np


CONTROLLED_MECHANISMS = (
    "balanced_mcar",
    "view_skew",
    "correlated",
    "feature_dependent",
)


def _rounded_count(rate, total):
    return int(np.floor(float(rate) * int(total) + 0.5))


def _validate_rates(incomplete_rate, observation_rate):
    incomplete_rate = float(incomplete_rate)
    observation_rate = float(observation_rate)
    if not 0.0 <= incomplete_rate <= 1.0:
        raise ValueError("incomplete_rate must lie in [0, 1]")
    if not 0.0 <= observation_rate <= 1.0:
        raise ValueError("observation_rate must lie in [0, 1]")
    return incomplete_rate, observation_rate


def _normalized_weights(view_weights, n_views, mechanism):
    if view_weights is None:
        if mechanism == "view_skew":
            weights = np.linspace(2.0, 0.5, n_views, dtype=np.float64)
        else:
            weights = np.ones(n_views, dtype=np.float64)
    else:
        weights = np.asarray(view_weights, dtype=np.float64)
    if weights.shape != (n_views,):
        raise ValueError("view_weights must have shape [%d]" % n_views)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("view_weights must be finite and strictly positive")
    return weights / weights.sum()


def _feature_scores(features, n_samples, rng):
    if features is None:
        raise ValueError("feature_dependent masks require features [N, D]")
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != n_samples:
        raise ValueError("features must have shape [%d, D], got %r" %
                         (n_samples, values.shape))
    if values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("features must be finite and contain at least one column")
    projection = rng.normal(size=values.shape[1])
    projection /= max(np.linalg.norm(projection), np.finfo(np.float64).eps)
    scores = np.dot(values, projection)
    scores = (scores - scores.mean()) / max(scores.std(), np.finfo(np.float64).eps)
    projection_hash = hashlib.sha256(
        np.ascontiguousarray(projection).view(np.uint8).tobytes()
    ).hexdigest()
    return scores, projection_hash


def _row_probabilities(base_weights, score, n_views):
    if score is None:
        return base_weights
    offsets = np.linspace(-0.75, 0.75, n_views, dtype=np.float64)
    logits = np.log(base_weights) + np.clip(float(score), -4.0, 4.0) * offsets
    logits -= logits.max()
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum()


def _allocate_missing(n_rows, n_views, target_missing, rng, base_weights,
                      mechanism, row_scores, correlation_strength,
                      correlation_groups):
    missing = np.zeros((n_rows, n_views), dtype=np.bool_)
    if n_rows == 0:
        return missing

    # Give every incomplete row one missing view first.
    for row in range(n_rows):
        score = None if row_scores is None else row_scores[row]
        probabilities = _row_probabilities(base_weights, score, n_views)
        view = int(rng.choice(n_views, p=probabilities))
        missing[row, view] = True

    remaining = int(target_missing - n_rows)
    while remaining > 0:
        counts = missing.sum(axis=1)
        eligible = np.flatnonzero(counts < n_views - 1)
        if eligible.size == 0:
            raise RuntimeError("unable to allocate the requested observation budget")
        row = int(eligible[rng.randint(eligible.size)])
        available = np.flatnonzero(~missing[row])
        score = None if row_scores is None else row_scores[row]
        probabilities = _row_probabilities(base_weights, score, n_views)[available]

        if mechanism == "correlated" and missing[row].any():
            group_boost = np.ones(available.shape[0], dtype=np.float64)
            missing_set = set(np.flatnonzero(missing[row]).tolist())
            for offset, view in enumerate(available):
                for group in correlation_groups:
                    if int(view) in group and missing_set.intersection(group):
                        group_boost[offset] = 1.0 + 8.0 * correlation_strength
                        break
            probabilities = probabilities * group_boost

        probabilities = probabilities / probabilities.sum()
        view = int(rng.choice(available, p=probabilities))
        missing[row, view] = True
        remaining -= 1
    return missing


def mask_hash(mask):
    """Return a shape- and dtype-aware SHA-256 digest for a binary mask."""
    value = np.ascontiguousarray(mask)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.view(np.uint8).tobytes())
    return digest.hexdigest()


def pattern_ids(mask):
    """Encode each mask row as a stable bit string such as ``"101"``."""
    value = np.asarray(mask)
    if value.ndim != 2:
        raise ValueError("mask must have shape [N, V]")
    return np.asarray(["".join(str(int(bit)) for bit in row) for row in value])


def mask_statistics(mask):
    """Return JSON-safe counts and rates for ``mask [N, V]``."""
    value = np.asarray(mask)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] < 2:
        raise ValueError("mask must have non-empty shape [N, V] with V >= 2")
    if not np.all((value == 0) | (value == 1)):
        raise ValueError("mask entries must be binary")
    observed_per_row = value.sum(axis=1)
    patterns, counts = np.unique(pattern_ids(value), return_counts=True)
    return {
        "n_samples": int(value.shape[0]),
        "n_views": int(value.shape[1]),
        "n_complete": int(np.sum(observed_per_row == value.shape[1])),
        "n_incomplete": int(np.sum(observed_per_row < value.shape[1])),
        "incomplete_rate": float(np.mean(observed_per_row < value.shape[1])),
        "overall_observation_rate": float(value.mean()),
        "per_view_observation_rate": [float(x) for x in value.mean(axis=0)],
        "min_observed_views_per_sample": int(observed_per_row.min()),
        "max_observed_views_per_sample": int(observed_per_row.max()),
        "patterns": {str(pattern): int(count)
                     for pattern, count in zip(patterns, counts)},
        "mask_hash": mask_hash(value),
    }


def validate_mask(mask, expected_incomplete_rate=None):
    """Validate binary/nonempty-row invariants and return mask statistics."""
    stats = mask_statistics(mask)
    value = np.asarray(mask)
    if np.any(value.sum(axis=1) == 0):
        raise ValueError("every sample must retain at least one observed view")
    if expected_incomplete_rate is not None:
        expected = _rounded_count(expected_incomplete_rate, value.shape[0])
        if stats["n_incomplete"] != expected:
            raise ValueError("expected %d incomplete rows, found %d" %
                             (expected, stats["n_incomplete"]))
    return stats


def generate_mask(n_samples, n_views, incomplete_rate, mechanism="balanced_mcar",
                  seed=0, observation_rate=0.5, view_weights=None, features=None,
                  correlation_strength=0.8, correlation_groups=None):
    """Generate one controlled missing-view mask and its provenance metadata.

    Args:
        n_samples: Number of samples ``N``.
        n_views: Number of views ``V``; must be at least two.
        incomplete_rate: Exact (up to integer rounding) fraction of rows that
            have between one and ``V-1`` observed views.
        mechanism: One of :data:`CONTROLLED_MECHANISMS`.
        seed: Local NumPy seed.  Global RNG state is never read or changed.
        observation_rate: Requested observed-entry fraction among incomplete
            rows.  It is clipped to the feasible interval ``[1/V, (V-1)/V]``.
        view_weights: Optional positive ``[V]`` relative missing propensities.
            ``view_skew`` uses ``linspace(2, .5, V)`` by default; other
            mechanisms use equal propensities.
        features: Required only for ``feature_dependent``; finite ``[N, D]``
            array used with a seeded random projection.  Labels must not be
            supplied here.
        correlation_strength: In ``[0, 1]``; favors members of a correlation
            group being missing together while preserving the same budget.
        correlation_groups: Optional list of view-index groups, for example
            ``[[0, 1]]``.  Defaults to consecutive disjoint pairs.  With two
            views, correlated missingness necessarily degenerates to a single
            missing view because every row must retain an observation.

    Returns:
        ``(mask, metadata)``.  ``mask`` is ``uint8 [N, V]`` with one meaning
        observed.  ``metadata`` is JSON-serializable and includes realized
        rates, mechanism parameters, and the content hash.
    """
    n_samples = int(n_samples)
    n_views = int(n_views)
    seed = int(seed)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if n_views < 2:
        raise ValueError("n_views must be at least two")
    if mechanism not in CONTROLLED_MECHANISMS:
        raise ValueError("unknown mechanism %r; choices are %s" %
                         (mechanism, ", ".join(CONTROLLED_MECHANISMS)))
    incomplete_rate, observation_rate = _validate_rates(
        incomplete_rate, observation_rate)
    correlation_strength = float(correlation_strength)
    if not 0.0 <= correlation_strength <= 1.0:
        raise ValueError("correlation_strength must lie in [0, 1]")
    if correlation_groups is None:
        correlation_groups = [list(range(start, min(start + 2, n_views)))
                              for start in range(0, n_views - 1, 2)]
    normalized_groups = []
    for group in correlation_groups:
        normalized = sorted(set(int(value) for value in group))
        if len(normalized) < 2:
            raise ValueError("each correlation group needs at least two views")
        if normalized[0] < 0 or normalized[-1] >= n_views:
            raise ValueError("correlation group index outside [0, %d]" %
                             (n_views - 1))
        normalized_groups.append(normalized)
    if mechanism == "correlated" and n_views > 2 and not normalized_groups:
        raise ValueError("correlated masks require at least one view group")

    rng = np.random.RandomState(seed)
    n_incomplete = _rounded_count(incomplete_rate, n_samples)
    feasible_observation_rate = min(
        max(observation_rate, 1.0 / n_views),
        float(n_views - 1) / n_views,
    )
    total_entries = n_incomplete * n_views
    target_observed = _rounded_count(feasible_observation_rate, total_entries)
    target_observed = min(max(target_observed, n_incomplete),
                          n_incomplete * (n_views - 1))
    target_missing = total_entries - target_observed

    scores = None
    projection_hash = None
    if mechanism == "feature_dependent":
        scores, projection_hash = _feature_scores(features, n_samples, rng)
        # Stable sorting makes tied feature rows deterministic across platforms.
        ranked = np.argsort(-scores, kind="mergesort")
        incomplete_indices = ranked[:n_incomplete]
        selected_scores = scores[incomplete_indices]
    else:
        incomplete_indices = rng.permutation(n_samples)[:n_incomplete]
        selected_scores = None

    weights = _normalized_weights(view_weights, n_views, mechanism)
    missing = _allocate_missing(
        n_incomplete,
        n_views,
        target_missing,
        rng,
        weights,
        mechanism,
        selected_scores,
        correlation_strength,
        normalized_groups,
    )
    mask = np.ones((n_samples, n_views), dtype=np.uint8)
    if n_incomplete:
        mask[incomplete_indices] = (~missing).astype(np.uint8)

    stats = validate_mask(mask, expected_incomplete_rate=incomplete_rate)
    metadata = {
        "format_version": 1,
        "semantics": "1=observed,0=missing",
        "mechanism": mechanism,
        "seed": seed,
        "requested_incomplete_rate": incomplete_rate,
        "requested_observation_rate_within_incomplete": observation_rate,
        "feasible_observation_rate_within_incomplete": feasible_observation_rate,
        "target_observed_entries_within_incomplete": int(target_observed),
        "view_missing_weights": [float(value) for value in weights],
        "correlation_strength": correlation_strength,
        "correlation_groups": normalized_groups,
        "feature_projection_hash": projection_hash,
        "statistics": stats,
    }
    return mask, metadata


def save_mask(path, mask, metadata):
    """Atomically save a mask and verified JSON metadata to a compressed NPZ."""
    value = np.asarray(mask, dtype=np.uint8)
    stats = validate_mask(value)
    details = dict(metadata)
    details["statistics"] = stats
    details["mask_hash"] = stats["mask_hash"]
    destination = os.path.abspath(path)
    parent = os.path.dirname(destination)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    descriptor, temporary = tempfile.mkstemp(prefix=".mask-", suffix=".npz",
                                             dir=parent or None)
    os.close(descriptor)
    try:
        np.savez_compressed(
            temporary,
            mask=value,
            metadata_json=np.asarray(json.dumps(details, sort_keys=True)),
        )
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def load_mask(path):
    """Load an NPZ created by :func:`save_mask` and verify its digest."""
    with np.load(path, allow_pickle=False) as payload:
        mask = np.asarray(payload["mask"], dtype=np.uint8)
        raw_metadata = np.asarray(payload["metadata_json"]).reshape(()).item()
    metadata = json.loads(str(raw_metadata))
    actual = validate_mask(mask)["mask_hash"]
    expected = metadata.get("mask_hash") or metadata.get("statistics", {}).get(
        "mask_hash")
    if expected != actual:
        raise ValueError("mask hash mismatch for %s" % path)
    return mask, metadata
