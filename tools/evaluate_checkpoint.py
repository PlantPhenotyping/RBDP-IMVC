#!/usr/bin/env python
"""Re-evaluate a verified checkpoint without repeating label-free training."""

import argparse
import copy
import os
import re
import shutil
import sys
import traceback

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rbdp.config import load_json_config  # noqa: E402
from rbdp.data import load_multiview_data  # noqa: E402
from rbdp.io import RunDirectory, environment_snapshot, file_sha256  # noqa: E402
from rbdp.io import read_json, source_manifest, write_csv, write_npz  # noqa: E402
from rbdp.masks import generate_mask, save_mask  # noqa: E402
from rbdp.results import validate_run  # noqa: E402
from rbdp.trainer import build_model, evaluate_model, parameter_count  # noqa: E402


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True,
                        help="completed parent training run")
    parser.add_argument("--evaluation-ablation", required=True,
                        help="JSON overlay that may change evaluation only")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-tag", default="reevaluated")
    parser.add_argument("--test-mechanism", default=None)
    parser.add_argument("--test-incomplete-rate", type=float, default=None)
    parser.add_argument("--test-mask-seed", type=int, default=None)
    return parser.parse_args()


def _deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _validate_evaluation_overlay(overlay):
    """Reject fields outside the documented evaluation-only surface."""
    allowed_top_level = {
        "evaluation", "reliability", "ablation", "runtime", "masks",
    }
    unknown = sorted(set(overlay) - allowed_top_level)
    if unknown:
        raise ValueError(
            "evaluation overlay contains forbidden top-level fields: %s" %
            ", ".join(unknown))
    for section in allowed_top_level:
        if section in overlay and not isinstance(overlay[section], dict):
            raise ValueError(
                "evaluation overlay section %s must be an object" % section)

    allowed_ablation = {
        "name", "description", "complete_only", "heteroscedastic", "cycle",
        "groupdro", "conflict_safe", "reliability", "fusion",
    }
    unknown_ablation = sorted(
        set(overlay.get("ablation", {})) - allowed_ablation)
    if unknown_ablation:
        raise ValueError(
            "evaluation overlay contains unknown ablation fields: %s" %
            ", ".join(unknown_ablation))

    mask_overlay = overlay.get("masks", {})
    unknown_masks = sorted(set(mask_overlay) - {"test"})
    if unknown_masks:
        raise ValueError(
            "evaluation overlay may change only masks.test, not: %s" %
            ", ".join(unknown_masks))
    allowed_test_mask = {
        "mechanism", "incomplete_rate", "seed", "view_weights",
        "correlation_groups",
    }
    test_mask_overlay = mask_overlay.get("test", {})
    if not isinstance(test_mask_overlay, dict):
        raise ValueError("evaluation overlay masks.test must be an object")
    unknown_test_mask = sorted(set(test_mask_overlay) - allowed_test_mask)
    if unknown_test_mask:
        raise ValueError(
            "evaluation overlay contains unknown masks.test fields: %s" %
            ", ".join(unknown_test_mask))


def _training_signature(config):
    ablation = config["ablation"]
    return {
        "dataset": config["dataset"],
        "split": config["split"],
        "train_mask": config["masks"]["train"],
        "validation_mask": config["masks"]["validation"],
        "observation_rate_within_incomplete": config["masks"][
            "observation_rate_within_incomplete"],
        "correlation_strength": config["masks"]["correlation_strength"],
        "model": config["model"],
        "training": config["training"],
        "loss": config["loss"],
        "training_ablation": {
            key: ablation.get(key, False) for key in (
                "complete_only", "heteroscedastic", "cycle", "groupdro",
                "conflict_safe")
        },
    }


def _resolve_device(name):
    name = str(name)
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def _safe(value, name):
    value = str(value)
    if not re.match(r"^[A-Za-z0-9_.-]+$", value):
        raise ValueError("%s contains unsafe path characters: %r" % (name, value))
    return value


def _rate_name(value):
    return ("%.4f" % float(value)).rstrip("0").rstrip(".")


def _load_npz_array(path, name):
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive[name]).copy()


def _make_test_mask(dataset, indices, config):
    mask_config = config["masks"]["test"]
    shared = config["masks"]
    parameters = {
        "n_samples": int(indices.size),
        "n_views": dataset.n_views,
        "incomplete_rate": mask_config["incomplete_rate"],
        "mechanism": mask_config["mechanism"],
        "seed": mask_config["seed"],
        "observation_rate": shared["observation_rate_within_incomplete"],
        "correlation_strength": shared["correlation_strength"],
    }
    for optional in ("view_weights", "correlation_groups"):
        if optional in mask_config:
            parameters[optional] = mask_config[optional]
    if mask_config["mechanism"] == "feature_dependent":
        parameters["features"] = dataset.views[0][indices]
    return generate_mask(**parameters)


def _copy_training_artifacts(parent, destination):
    for filename in (
            "split.npz", "train_mask.npz", "validation_mask.npz",
            "curves.csv", "environment_risks.csv"):
        source = os.path.join(parent, filename)
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(destination, filename))


def _write_evaluation_artifacts(run, evaluated):
    write_csv(os.path.join(run.path, "reliability.csv"),
              evaluated["reliability_rows"],
              ["sample_index", "pattern", "source_view", "target_view",
               "log_variance", "cycle_mse", "risk", "prediction_mse"])
    write_csv(os.path.join(run.path, "risk_coverage.csv"),
              evaluated["coverage_rows"],
              ["scope", "coverage", "selective_error"])
    write_npz(os.path.join(run.path, "embedding.npz"), **evaluated["arrays"])


def main():
    arguments = _arguments()
    parent = os.path.abspath(arguments.run_dir)
    _, parent_audit = validate_run(parent)
    if not parent_audit["valid"]:
        raise ValueError("parent run failed integrity checks: %s" %
                         ",".join(parent_audit["issues"]))
    parent_stored = read_json(os.path.join(parent, "config.json"))
    parent_hash = parent_stored.pop("config_hash")
    parent_metrics = read_json(os.path.join(parent, "metrics.json"))
    overlay_path = os.path.abspath(arguments.evaluation_ablation)
    overlay = load_json_config(overlay_path)
    _validate_evaluation_overlay(overlay)
    config = _deep_merge(parent_stored, overlay)
    if _training_signature(config) != _training_signature(parent_stored):
        raise ValueError("evaluation overlay changes the parent training signature")

    test_overridden = any(value is not None for value in (
        arguments.test_mechanism,
        arguments.test_incomplete_rate,
        arguments.test_mask_seed,
    ))
    if arguments.test_mechanism is not None:
        config["masks"]["test"]["mechanism"] = arguments.test_mechanism
    if arguments.test_incomplete_rate is not None:
        config["masks"]["test"]["incomplete_rate"] = float(
            arguments.test_incomplete_rate)
    if arguments.test_mask_seed is not None:
        config["masks"]["test"]["seed"] = int(arguments.test_mask_seed)

    device = _resolve_device(arguments.device)
    config["runtime"]["device"] = str(device)
    config["runtime"]["save_checkpoint"] = False
    torch.set_num_threads(int(config["runtime"].get("torch_num_threads", 1)))
    data_root = (os.path.abspath(arguments.data_root)
                 if arguments.data_root else
                 parent_stored["resolved"]["data_root"])
    dataset = load_multiview_data(
        config["dataset"]["name"], data_root=data_root,
        view_indices=config["dataset"].get("view_indices"),
        normalization=config["dataset"].get("normalization", "legacy"))
    split_path = os.path.join(parent, "split.npz")
    split = {name: _load_npz_array(split_path, name)
             for name in ("train", "validation", "test")}
    if test_overridden:
        test_mask, test_metadata = _make_test_mask(dataset, split["test"], config)
        config["resolved"]["mask_metadata"]["test"] = test_metadata
    else:
        test_mask = _load_npz_array(os.path.join(parent, "test_mask.npz"), "mask")
        test_metadata = config["resolved"]["mask_metadata"]["test"]

    checkpoint_path = os.path.join(parent, "checkpoint.pt")
    checkpoint_hash = file_sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("config_hash") != parent_hash:
        raise ValueError("checkpoint does not belong to the parent configuration")
    model = build_model(dataset.view_dims, config).to(device)
    model.load_state_dict(checkpoint["model_state"])

    old_manifest = config["resolved"].get("source_manifest")
    manifest = source_manifest(PROJECT_ROOT, [
        "rbdp", "tools/evaluate_checkpoint.py", overlay_path])
    config["resolved"]["training_source_manifest"] = old_manifest
    config["resolved"]["source_manifest"] = manifest
    config["resolved"]["environment"] = environment_snapshot(PROJECT_ROOT)
    config["resolved"]["environment"].pop("captured_at", None)
    config["resolved"]["evaluation_parent"] = {
        "run_path": parent,
        "config_hash": parent_hash,
        "source_hash": parent_metrics["provenance"]["source_hash"],
        "checkpoint_sha256": checkpoint_hash,
        "evaluation_ablation_config": overlay_path,
    }
    config["resolved"]["device"] = str(device)
    config["resolved"]["data_root"] = data_root

    test_config = config["masks"]["test"]
    experiment_name = "%s__%s__%s" % (
        _safe(config["experiment"], "experiment"),
        _safe(config["ablation"]["name"], "ablation"),
        _safe(arguments.run_tag, "run tag"),
    )
    run_path = os.path.join(
        os.path.abspath(arguments.output_root), experiment_name,
        _safe(dataset.name, "dataset"),
        _safe(test_config["mechanism"], "test mechanism"),
        "r%s" % _rate_name(test_config["incomplete_rate"]),
        "seed%d" % int(config["training"]["seed"]),
    )
    run = RunDirectory(run_path, config)
    if run.reusable():
        print("Reusing completed evaluation: %s" % run.path)
        return 0

    started = False
    try:
        run.start()
        started = True
        _copy_training_artifacts(parent, run.path)
        if test_overridden:
            save_mask(os.path.join(run.path, "test_mask.npz"),
                      test_mask, test_metadata)
        else:
            shutil.copy2(os.path.join(parent, "test_mask.npz"),
                         os.path.join(run.path, "test_mask.npz"))
        evaluated = evaluate_model(
            model, [view[split["test"]] for view in dataset.views],
            dataset.labels[split["test"]], test_mask, split["test"],
            config, device)
        _write_evaluation_artifacts(run, evaluated)
        metrics = evaluated["metrics"]
        metrics["training"] = parent_metrics["training"]
        metrics["model"] = {
            "parameter_count": parameter_count(model),
            "ablation": config["ablation"],
        }
        metrics["provenance"] = {
            "config_hash": run.hash,
            "source_hash": manifest["source_hash"],
            "dataset_sha256": config["resolved"]["dataset_sha256"],
            "train_mask_hash": config["resolved"]["mask_metadata"][
                "train"]["statistics"]["mask_hash"],
            "test_mask_hash": test_metadata["statistics"]["mask_hash"],
            "parent_config_hash": parent_hash,
            "parent_checkpoint_sha256": checkpoint_hash,
        }
        run.complete(metrics)
        with open(os.path.join(run.path, "stdout.log"), "w",
                  encoding="utf-8") as handle:
            handle.write("parent=%s\n" % parent)
            handle.write("parent_config_hash=%s\n" % parent_hash)
            handle.write("checkpoint_sha256=%s\n" % checkpoint_hash)
            handle.write("completed acc=%.6f nmi=%.6f ari=%.6f worst=%.6f\n" % (
                metrics["clustering"]["acc"], metrics["clustering"]["nmi"],
                metrics["clustering"]["ari"],
                metrics["pattern_robustness"]["worst_pattern_acc"]))
        print("Completed evaluation: %s" % run.path)
        return 0
    except Exception as error:
        if started:
            run.fail(error)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    sys.exit(main())
