"""Integrity-aware collection and aggregation of experiment results.

The collector deliberately works from immutable per-run artifacts instead of
accepting manually entered scores.  A run contributes to a summary only when
its status, resolved configuration, provenance hashes, split file, and mask
files agree with one another.
"""

import hashlib
import itertools
import json
import os

import numpy as np

from .io import config_hash, read_json
from .masks import mask_hash


IDENTITY_FIELDS = (
    "experiment",
    "protocol",
    "ablation",
    "dataset",
    "split_seed",
    "train_mechanism",
    "train_incomplete_rate",
    "test_mechanism",
    "test_incomplete_rate",
    "training_seed",
    "epochs",
    "fusion",
    "risk_normalization",
    "reliability_gamma",
    "source_temperature",
    "confidence_temperature",
    "conflict_safe",
    "auxiliary_weight",
    "prediction_weight",
    "cycle_weight",
    "structure_weight",
    "groupdro_eta",
    "groupdro_update_frequency",
)

SUMMARY_GROUP_FIELDS = (
    "experiment",
    "protocol",
    "ablation",
    "dataset",
    "train_mechanism",
    "train_incomplete_rate",
    "test_mechanism",
    "test_incomplete_rate",
    "epochs",
    "fusion",
    "risk_normalization",
    "reliability_gamma",
    "source_temperature",
    "confidence_temperature",
    "conflict_safe",
    "auxiliary_weight",
    "prediction_weight",
    "cycle_weight",
    "structure_weight",
    "groupdro_eta",
    "groupdro_update_frequency",
)

METRIC_PATHS = (
    ("acc", ("clustering", "acc")),
    ("nmi", ("clustering", "nmi")),
    ("ari", ("clustering", "ari")),
    ("ami", ("clustering", "ami")),
    ("f1_macro", ("clustering", "f1_macro")),
    ("precision_macro", ("clustering", "precision_macro")),
    ("recall_macro", ("clustering", "recall_macro")),
    ("worst_pattern_acc", ("pattern_robustness", "worst_pattern_acc")),
    ("average_pattern_acc", ("pattern_robustness", "average_pattern_acc")),
    ("best_pattern_acc", ("pattern_robustness", "best_pattern_acc")),
    ("pattern_gap", ("pattern_robustness", "pattern_gap")),
    ("completion_mse", ("reliability", "completion_mse")),
    ("completion_cosine_error", ("reliability", "completion_cosine_error")),
    ("risk_spearman", ("reliability", "path_ranking", "spearman")),
    ("risk_aurc", ("reliability", "path_ranking", "aurc")),
    ("mean_path_error", ("reliability", "path_ranking", "mean_error")),
    ("training_seconds", ("training", "seconds")),
    ("inference_seconds", ("inference_seconds",)),
    ("parameter_count", ("model", "parameter_count")),
)

SUMMARY_METRICS = tuple(name for name, _ in METRIC_PATHS) + (
    "normalized_risk_aurc",
)

PAIR_GROUP_FIELDS = (
    "experiment",
    "protocol",
    "dataset",
    "train_mechanism",
    "train_incomplete_rate",
    "test_mechanism",
    "test_incomplete_rate",
    "epochs",
)

LOWER_IS_BETTER = {
    "pattern_gap",
    "completion_mse",
    "completion_cosine_error",
    "risk_aurc",
    "mean_path_error",
    "normalized_risk_aurc",
    "training_seconds",
    "inference_seconds",
    "parameter_count",
}

LONG_FIELDNAMES = (
    "run_path",
    "experiment",
    "protocol",
    "ablation",
    "ablation_description",
    "dataset",
    "n_samples",
    "n_views",
    "view_dims",
    "split_seed",
    "train_mechanism",
    "train_incomplete_rate",
    "train_mask_seed",
    "validation_mechanism",
    "validation_incomplete_rate",
    "validation_mask_seed",
    "test_mechanism",
    "test_incomplete_rate",
    "test_mask_seed",
    "training_seed",
    "epochs",
    "fusion",
    "risk_normalization",
    "reliability_gamma",
    "source_temperature",
    "confidence_temperature",
    "conflict_safe",
    "auxiliary_weight",
    "prediction_weight",
    "cycle_weight",
    "structure_weight",
    "groupdro_eta",
    "groupdro_update_frequency",
    "complete_only",
    "heteroscedastic",
    "cycle",
    "groupdro",
    "reliability_enabled",
    "config_hash",
    "source_hash",
    "dataset_sha256",
    "train_mask_hash",
    "validation_mask_hash",
    "test_mask_hash",
    "train_split_hash",
    "validation_split_hash",
    "test_split_hash",
    "parent_config_hash",
    "parent_checkpoint_sha256",
) + tuple(name for name, _ in METRIC_PATHS) + ("normalized_risk_aurc",)


def _nested(value, path, default=""):
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _array_hash(values):
    value = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.view(np.uint8).tobytes())
    return digest.hexdigest()


def _audit(run_path, state, valid, issues=None, warnings=None):
    return {
        "run_path": os.path.abspath(run_path),
        "state": state,
        "valid": bool(valid),
        "issues": list(issues or []),
        "warnings": list(warnings or []),
    }


def _check_equal(issues, label, first, second):
    if first in (None, "") or second in (None, ""):
        issues.append("missing_%s" % label)
    elif first != second:
        issues.append("mismatched_%s" % label)


def _validate_split_file(run_path, config, issues):
    expected = _nested(config, ("resolved", "split_hashes"), {})
    if not expected:
        issues.append("missing_split_hashes")
        return
    path = os.path.join(run_path, "split.npz")
    if not os.path.isfile(path):
        issues.append("missing_split_file")
        return
    try:
        with np.load(path, allow_pickle=False) as archive:
            for split_name in ("train", "validation", "test"):
                if split_name not in archive:
                    issues.append("missing_%s_split_array" % split_name)
                elif _array_hash(archive[split_name]) != expected.get(split_name):
                    issues.append("mismatched_%s_split_hash" % split_name)
    except (IOError, OSError, ValueError) as error:
        issues.append("unreadable_split_file:%s" % type(error).__name__)


def _validate_mask_file(run_path, split_name, config, issues):
    expected = _nested(
        config,
        ("resolved", "mask_metadata", split_name, "statistics", "mask_hash"),
        "",
    )
    if not expected:
        issues.append("missing_%s_mask_hash" % split_name)
        return
    path = os.path.join(run_path, "%s_mask.npz" % split_name)
    if not os.path.isfile(path):
        issues.append("missing_%s_mask_file" % split_name)
        return
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "mask" not in archive:
                issues.append("missing_%s_mask_array" % split_name)
                return
            actual = mask_hash(archive["mask"])
        if actual != expected:
            issues.append("mismatched_%s_mask_file_hash" % split_name)
    except (IOError, OSError, ValueError) as error:
        issues.append("unreadable_%s_mask_file:%s" %
                      (split_name, type(error).__name__))


def _make_row(run_path, config, metrics):
    ablation = config["ablation"]
    masks = config["masks"]
    resolved = config["resolved"]
    summary = resolved.get("dataset_summary", {})
    provenance = metrics["provenance"]
    reliability = config.get("reliability", {})
    mask_metadata = resolved.get("mask_metadata", {})
    split_hashes = resolved.get("split_hashes", {})
    row = {
        "run_path": os.path.abspath(run_path),
        "experiment": config["experiment"],
        "protocol": config.get("protocol", ""),
        "ablation": ablation["name"],
        "ablation_description": ablation.get("description", ""),
        "dataset": config["dataset"]["name"],
        "n_samples": summary.get("n_samples", ""),
        "n_views": summary.get("n_views", ""),
        "view_dims": "x".join(str(value) for value in
                              summary.get("view_dims", [])),
        "split_seed": config["split"]["seed"],
        "train_mechanism": masks["train"]["mechanism"],
        "train_incomplete_rate": masks["train"]["incomplete_rate"],
        "train_mask_seed": masks["train"]["seed"],
        "validation_mechanism": masks["validation"]["mechanism"],
        "validation_incomplete_rate": masks["validation"]["incomplete_rate"],
        "validation_mask_seed": masks["validation"]["seed"],
        "test_mechanism": masks["test"]["mechanism"],
        "test_incomplete_rate": masks["test"]["incomplete_rate"],
        "test_mask_seed": masks["test"]["seed"],
        "training_seed": config["training"]["seed"],
        "epochs": config["training"]["epochs"],
        "fusion": ablation["fusion"],
        "risk_normalization": reliability.get("risk_normalization", "none"),
        "reliability_gamma": reliability.get("gamma", ""),
        "source_temperature": reliability.get("source_temperature", ""),
        "confidence_temperature": reliability.get(
            "confidence_temperature", ""),
        "conflict_safe": int(bool(ablation.get("conflict_safe", False))),
        "auxiliary_weight": config["loss"].get("auxiliary_weight", 0.0),
        "prediction_weight": config["loss"]["prediction_weight"],
        "cycle_weight": config["loss"]["cycle_weight"],
        "structure_weight": config["loss"].get("structure_weight", 0.0),
        "groupdro_eta": config["loss"]["groupdro_eta"],
        "groupdro_update_frequency": config["loss"].get(
            "groupdro_update_frequency", "batch"),
        "complete_only": int(bool(ablation["complete_only"])),
        "heteroscedastic": int(bool(ablation["heteroscedastic"])),
        "cycle": int(bool(ablation["cycle"])),
        "groupdro": int(bool(ablation["groupdro"])),
        "reliability_enabled": int(bool(ablation["reliability"])),
        "config_hash": provenance["config_hash"],
        "source_hash": provenance["source_hash"],
        "dataset_sha256": provenance["dataset_sha256"],
        "train_mask_hash": provenance["train_mask_hash"],
        "validation_mask_hash": _nested(
            mask_metadata, ("validation", "statistics", "mask_hash")),
        "test_mask_hash": provenance["test_mask_hash"],
        "train_split_hash": split_hashes.get("train", ""),
        "validation_split_hash": split_hashes.get("validation", ""),
        "test_split_hash": split_hashes.get("test", ""),
        "parent_config_hash": provenance.get("parent_config_hash", ""),
        "parent_checkpoint_sha256": provenance.get(
            "parent_checkpoint_sha256", ""),
    }
    for name, path in METRIC_PATHS:
        row[name] = _nested(metrics, path)
    mean_error = row["mean_path_error"]
    if mean_error not in ("", None) and float(mean_error) > 0:
        row["normalized_risk_aurc"] = float(row["risk_aurc"]) / float(mean_error)
    else:
        row["normalized_risk_aurc"] = ""
    return row


def validate_run(run_path):
    """Validate one run directory and return ``(row, audit_record)``.

    ``row`` is ``None`` if any integrity condition fails.  Warnings document
    non-fatal conditions and never change reported metric values.
    """
    run_path = os.path.abspath(run_path)
    issues, warnings = [], []
    documents = {}
    for name in ("config", "status", "metrics"):
        path = os.path.join(run_path, "%s.json" % name)
        if not os.path.isfile(path):
            issues.append("missing_%s_json" % name)
            continue
        try:
            documents[name] = read_json(path)
        except (IOError, OSError, ValueError, json.JSONDecodeError) as error:
            issues.append("unreadable_%s_json:%s" %
                          (name, type(error).__name__))
    state = _nested(documents, ("status", "state"), "missing")
    if state != "completed":
        issues.append("status_%s" % state)
    if issues or len(documents) != 3:
        return None, _audit(run_path, state, False, issues, warnings)

    config = documents["config"]
    status = documents["status"]
    metrics = documents["metrics"]
    stored_hash = config.get("config_hash")
    unhashed = dict(config)
    unhashed.pop("config_hash", None)
    calculated_hash = config_hash(unhashed)
    _check_equal(issues, "config_hash_content", stored_hash, calculated_hash)
    _check_equal(issues, "status_config_hash", stored_hash,
                 status.get("config_hash"))
    _check_equal(issues, "metrics_config_hash", stored_hash,
                 _nested(metrics, ("provenance", "config_hash")))

    source_hash = _nested(config, ("resolved", "source_manifest", "source_hash"))
    _check_equal(issues, "source_hash", source_hash,
                 _nested(metrics, ("provenance", "source_hash")))
    dataset_hash = _nested(config, ("resolved", "dataset_sha256"))
    _check_equal(issues, "dataset_hash", dataset_hash,
                 _nested(metrics, ("provenance", "dataset_sha256")))
    evaluation_parent = _nested(config, ("resolved", "evaluation_parent"), {})
    if evaluation_parent:
        _check_equal(
            issues, "parent_config_hash",
            evaluation_parent.get("config_hash"),
            _nested(metrics, ("provenance", "parent_config_hash")))
        _check_equal(
            issues, "parent_checkpoint_hash",
            evaluation_parent.get("checkpoint_sha256"),
            _nested(metrics, ("provenance", "parent_checkpoint_sha256")))
    for split_name in ("train", "test"):
        expected = _nested(
            config,
            ("resolved", "mask_metadata", split_name, "statistics", "mask_hash"),
        )
        _check_equal(issues, "%s_mask_hash" % split_name, expected,
                     _nested(metrics, ("provenance", "%s_mask_hash" % split_name)))

    _validate_split_file(run_path, config, issues)
    for split_name in ("train", "validation", "test"):
        _validate_mask_file(run_path, split_name, config, issues)

    required_metric_paths = (
        ("clustering", "acc"),
        ("clustering", "nmi"),
        ("clustering", "ari"),
        ("pattern_robustness", "worst_pattern_acc"),
    )
    for path in required_metric_paths:
        if _nested(metrics, path, None) is None:
            issues.append("missing_metric_%s" % "_".join(path))
    if issues:
        return None, _audit(run_path, state, False, issues, warnings)
    try:
        row = _make_row(run_path, config, metrics)
    except (KeyError, TypeError, ValueError) as error:
        issues.append("malformed_run_schema:%s" % type(error).__name__)
        return None, _audit(run_path, state, False, issues, warnings)
    return row, _audit(run_path, state, True, issues, warnings)


def _logical_identity(row):
    return tuple(row[field] for field in IDENTITY_FIELDS) + (
        row["train_mask_hash"], row["test_mask_hash"])


def collect_runs(roots):
    """Scan roots, reject ambiguous/invalid runs, and return rows plus audit."""
    status_paths = []
    for root in roots:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            status_paths.append(os.path.join(root, "status.json"))
            continue
        for directory, names, filenames in os.walk(root):
            names[:] = sorted(name for name in names if name != "__pycache__")
            if "status.json" in filenames:
                status_paths.append(os.path.join(directory, "status.json"))
    rows, audit = [], []
    for status_path in sorted(set(status_paths)):
        run_path = os.path.dirname(status_path)
        row, record = validate_run(run_path)
        audit.append(record)
        if row is not None:
            rows.append(row)

    # Exact copies are counted once.  Distinct configurations claiming the
    # same logical cell are ambiguous and all such rows are excluded.
    exact_seen = {}
    exact_unique = []
    for row in sorted(rows, key=lambda value: value["run_path"]):
        digest = row["config_hash"]
        if digest in exact_seen:
            audit.append(_audit(
                row["run_path"], "completed", False,
                ["duplicate_config_hash"],
                ["same_as:%s" % exact_seen[digest]],
            ))
        else:
            exact_seen[digest] = row["run_path"]
            exact_unique.append(row)

    by_identity = {}
    for row in exact_unique:
        by_identity.setdefault(_logical_identity(row), []).append(row)
    conflicts = set()
    for candidates in by_identity.values():
        if len(candidates) > 1:
            paths = [candidate["run_path"] for candidate in candidates]
            conflicts.update(paths)
            for candidate in candidates:
                audit.append(_audit(
                    candidate["run_path"], "completed", False,
                    ["conflicting_logical_run"],
                    ["candidates:%s" % "|".join(paths)],
                ))
    accepted = [row for row in exact_unique if row["run_path"] not in conflicts]
    return sorted(accepted, key=lambda value: tuple(
        str(value.get(field, "")) for field in IDENTITY_FIELDS)), audit


def _mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    std = "" if array.size < 2 else float(array.std(ddof=1))
    return mean, std


def _paired_effects(deltas, lower_is_better=False):
    """Return paired effect sizes with positive values meaning improvement.

    ``deltas`` stores comparison minus baseline in the metric's native units.
    Cohen's ``d_z`` and the matched-pairs rank-biserial correlation are
    sign-oriented so that positive always favors the comparison, including
    error metrics for which a smaller value is better.
    """
    raw = np.asarray(deltas, dtype=np.float64)
    oriented = -raw if lower_is_better else raw
    if oriented.size < 2 or np.isclose(oriented.std(ddof=1), 0.0):
        effect_dz = ""
    else:
        effect_dz = float(oriented.mean() / oriented.std(ddof=1))

    nonzero = oriented[~np.isclose(oriented, 0.0)]
    if nonzero.size == 0:
        rank_biserial = 0.0
    else:
        absolute = np.abs(nonzero)
        order = np.argsort(absolute, kind="mergesort")
        ranks = np.empty(nonzero.size, dtype=np.float64)
        start = 0
        while start < nonzero.size:
            end = start + 1
            while (end < nonzero.size and
                   np.isclose(absolute[order[end]], absolute[order[start]])):
                end += 1
            average_rank = 0.5 * ((start + 1) + end)
            ranks[order[start:end]] = average_rank
            start = end
        total = float(ranks.sum())
        positive = float(ranks[nonzero > 0].sum())
        negative = float(ranks[nonzero < 0].sum())
        rank_biserial = (positive - negative) / total
    return effect_dz, float(rank_biserial)


def _add_holm_adjustment(records):
    """Add Holm-adjusted p-values within each method/metric family."""
    families = {}
    for index, record in enumerate(records):
        if record["scope"] != "group":
            continue
        p_value = record["wilcoxon_p_two_sided"]
        if p_value in ("", None):
            continue
        key = (record["baseline"], record["comparison"], record["metric"])
        families.setdefault(key, []).append((index, float(p_value)))
    for members in families.values():
        ordered = sorted(members, key=lambda item: item[1])
        running = 0.0
        count = len(ordered)
        for rank, (index, p_value) in enumerate(ordered):
            adjusted = min(1.0, float(count - rank) * p_value)
            running = max(running, adjusted)
            records[index]["holm_p_two_sided"] = running
    for record in records:
        if record["scope"] == "overall":
            record["holm_p_two_sided"] = record["wilcoxon_p_two_sided"]
        elif "holm_p_two_sided" not in record:
            record["holm_p_two_sided"] = ""
    return records


def summarize_rows(rows, metrics=SUMMARY_METRICS):
    """Return wide mean/std summaries grouped independently of train seed."""
    grouped = {}
    for row in rows:
        key = tuple(row[field] for field in SUMMARY_GROUP_FIELDS)
        grouped.setdefault(key, []).append(row)
    summaries = []
    for key in sorted(grouped, key=lambda value: tuple(str(x) for x in value)):
        members = grouped[key]
        summary = dict(zip(SUMMARY_GROUP_FIELDS, key))
        summary["run_count"] = len(members)
        summary["training_seeds"] = "|".join(
            str(value) for value in sorted(set(
                int(member["training_seed"]) for member in members)))
        summary["source_hash_count"] = len(set(
            member["source_hash"] for member in members))
        for metric in metrics:
            values = [member[metric] for member in members
                      if member.get(metric) not in ("", None)]
            if values:
                summary[metric + "_mean"], summary[metric + "_std"] = (
                    _mean_std(values))
            else:
                summary[metric + "_mean"] = ""
                summary[metric + "_std"] = ""
        summaries.append(summary)
    return summaries


def _pair_key(row):
    return (
        row["experiment"], row["protocol"], row["dataset"], row["split_seed"],
        row["train_mechanism"], row["train_incomplete_rate"],
        row["test_mechanism"], row["test_incomplete_rate"],
        row["train_mask_hash"], row["test_mask_hash"],
        row["training_seed"], row["epochs"],
    )


def paired_comparisons(rows, baseline, metrics=SUMMARY_METRICS):
    """Compare methods per dataset/protocol and overall on exact run pairs."""
    methods = sorted(set(row["ablation"] for row in rows))
    if baseline not in methods:
        raise ValueError("baseline %r is absent from collected results" % baseline)
    output = []
    group_values = sorted(set(
        tuple(row[field] for field in PAIR_GROUP_FIELDS) for row in rows),
        key=lambda value: tuple(str(x) for x in value))
    scopes = [("group", values, [
        row for row in rows
        if tuple(row[field] for field in PAIR_GROUP_FIELDS) == values
    ]) for values in group_values]
    scopes.append(("overall", tuple("" for _ in PAIR_GROUP_FIELDS), rows))
    for scope, group_values, scope_rows in scopes:
        lookup = {}
        for row in scope_rows:
            key = (row["ablation"], _pair_key(row))
            if key in lookup:
                raise ValueError("duplicate paired cell for %s" % row["ablation"])
            lookup[key] = row
        baseline_keys = set(key for method, key in lookup if method == baseline)
        for method in methods:
            if method == baseline:
                continue
            method_keys = set(key for candidate, key in lookup
                              if candidate == method)
            common = sorted(baseline_keys.intersection(method_keys),
                            key=lambda value: tuple(str(x) for x in value))
            for metric in metrics:
                pairs = []
                sources = set()
                for key in common:
                    first = lookup[(baseline, key)]
                    second = lookup[(method, key)]
                    if (first.get(metric) in ("", None) or
                            second.get(metric) in ("", None)):
                        continue
                    pairs.append((float(first[metric]), float(second[metric])))
                    sources.add(first["source_hash"])
                    sources.add(second["source_hash"])
                if not pairs:
                    continue
                base_values = np.asarray(
                    [pair[0] for pair in pairs], dtype=np.float64)
                method_values = np.asarray(
                    [pair[1] for pair in pairs], dtype=np.float64)
                deltas = method_values - base_values
                p_value = ""
                if deltas.size >= 2:
                    if np.allclose(deltas, 0.0):
                        p_value = 1.0
                    else:
                        try:
                            import warnings
                            from scipy.stats import wilcoxon
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", UserWarning)
                                p_value = float(wilcoxon(deltas).pvalue)
                        except (ImportError, ValueError, ZeroDivisionError):
                            p_value = ""
                delta_mean, delta_std = _mean_std(deltas)
                oriented = -deltas if metric in LOWER_IS_BETTER else deltas
                effect_dz, rank_biserial = _paired_effects(
                    deltas, lower_is_better=metric in LOWER_IS_BETTER)
                record = {
                    "scope": scope,
                    "baseline": baseline,
                    "comparison": method,
                    "metric": metric,
                    "higher_is_better": int(metric not in LOWER_IS_BETTER),
                    "pair_count": int(deltas.size),
                    "baseline_mean": float(base_values.mean()),
                    "comparison_mean": float(method_values.mean()),
                    "mean_delta": delta_mean,
                    "median_delta": float(np.median(deltas)),
                    "delta_std": delta_std,
                    "paired_effect_dz": effect_dz,
                    "rank_biserial_effect": rank_biserial,
                    "wins": int(np.sum(oriented > 0)),
                    "ties": int(np.sum(np.isclose(deltas, 0.0))),
                    "losses": int(np.sum(oriented < 0)),
                    "wilcoxon_p_two_sided": p_value,
                    "source_hash_count": len(sources),
                }
                record.update(dict(zip(PAIR_GROUP_FIELDS, group_values)))
                output.append(record)
    return _add_holm_adjustment(output)


def _normalized_grid_value(value):
    if isinstance(value, float):
        return "%.12g" % value
    return str(value)


def expand_expected_grids(document):
    """Expand ``{"grids": [{field: [values]}]}`` into expected run cells."""
    if not isinstance(document, dict) or not isinstance(document.get("grids"), list):
        raise ValueError("expected grid must contain a 'grids' list")
    expanded = []
    for index, grid in enumerate(document["grids"]):
        if not isinstance(grid, dict) or not grid:
            raise ValueError("grid %d must be a non-empty object" % index)
        unknown = sorted(set(grid).difference(IDENTITY_FIELDS))
        if unknown:
            raise ValueError("grid %d has unknown fields: %s" %
                             (index, ", ".join(unknown)))
        fields = sorted(grid)
        dimensions = []
        for field in fields:
            values = grid[field]
            if not isinstance(values, list):
                values = [values]
            if not values:
                raise ValueError("grid field %s cannot be empty" % field)
            dimensions.append(values)
        for values in itertools.product(*dimensions):
            expanded.append(dict(zip(fields, values)))
    return expanded


def missing_expected_runs(rows, expected):
    """Return expected partial identities that match no collected valid row."""
    missing = []
    for candidate in expected:
        found = False
        for row in rows:
            if all(_normalized_grid_value(row.get(field)) ==
                   _normalized_grid_value(value)
                   for field, value in candidate.items()):
                found = True
                break
        if not found:
            missing.append(candidate)
    return missing


def summary_fieldnames(metrics=SUMMARY_METRICS):
    fields = list(SUMMARY_GROUP_FIELDS) + [
        "run_count", "training_seeds", "source_hash_count"]
    for metric in metrics:
        fields.extend([metric + "_mean", metric + "_std"])
    return fields


PAIRED_FIELDNAMES = (
    "scope",
) + PAIR_GROUP_FIELDS + (
    "baseline", "comparison", "metric", "higher_is_better", "pair_count",
    "baseline_mean",
    "comparison_mean", "mean_delta", "median_delta", "delta_std",
    "paired_effect_dz", "rank_biserial_effect", "wins", "ties", "losses",
    "wilcoxon_p_two_sided", "holm_p_two_sided", "source_hash_count",
)
