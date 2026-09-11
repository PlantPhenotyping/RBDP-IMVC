#!/usr/bin/env python
"""Run one fixed-mask RBDP-IMVC ablation with structured outputs."""

import argparse
import copy
import hashlib
import os
import re
import sys
import time
import traceback

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rbdp.config import canonical_dataset_name, dataset_spec  # noqa: E402
from rbdp.config import load_json_config  # noqa: E402
from rbdp.data import load_multiview_data, make_split  # noqa: E402
from rbdp.io import RunDirectory, environment_snapshot, file_sha256  # noqa: E402
from rbdp.io import source_manifest, write_csv, write_npz  # noqa: E402
from rbdp.masks import generate_mask, save_mask  # noqa: E402
from rbdp.trainer import build_model, evaluate_model, fit_model  # noqa: E402
from rbdp.trainer import parameter_count, save_checkpoint, seed_everything  # noqa: E402

DEFAULT_EXPERIMENT = os.path.join(
    PROJECT_ROOT, "configs", "experiments", "pilot_caltech_shift.json")
DEFAULT_ABLATION = os.path.join(
    PROJECT_ROOT, "configs", "ablations", "a5_rbdp.json")


def _deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _array_hash(values):
    value = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.view(np.uint8).tobytes())
    return digest.hexdigest()


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--ablation", default=DEFAULT_ABLATION)
    parser.add_argument("--data-root", default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--output-root", default=os.path.join(PROJECT_ROOT, "outputs"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--training-seed", type=int, default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--observation-rate", type=float, default=None,
                        help="observed-entry rate inside incomplete rows")
    for split_name in ("train", "validation", "test"):
        parser.add_argument("--%s-mechanism" % split_name, default=None)
        parser.add_argument("--%s-incomplete-rate" % split_name,
                            type=float, default=None)
        parser.add_argument("--%s-mask-seed" % split_name,
                            type=int, default=None)
    parser.add_argument("--groupdro-eta", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-tag", default=None)
    return parser.parse_args()


def _validate_ablation(config):
    required = ["name", "complete_only", "heteroscedastic", "cycle",
                "groupdro", "reliability", "fusion"]
    missing = [key for key in required if key not in config.get("ablation", {})]
    if missing:
        raise ValueError("ablation is missing fields: %s" % ", ".join(missing))
    if config["ablation"]["fusion"] not in (
            "concat", "consensus", "gated_concat", "fallback_concat"):
        raise ValueError("unsupported ablation fusion")


def _resolve_device(name):
    requested = str(name)
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def _make_mask(dataset, indices, mask_config, shared_config):
    parameters = {
        "n_samples": int(indices.size),
        "n_views": dataset.n_views,
        "incomplete_rate": mask_config["incomplete_rate"],
        "mechanism": mask_config["mechanism"],
        "seed": mask_config["seed"],
        "observation_rate": shared_config["observation_rate_within_incomplete"],
        "correlation_strength": shared_config["correlation_strength"],
    }
    for optional in ("view_weights", "correlation_groups"):
        if optional in mask_config:
            parameters[optional] = mask_config[optional]
    if mask_config["mechanism"] == "feature_dependent":
        parameters["features"] = dataset.views[0][indices]
    return generate_mask(**parameters)


def _safe(value, name):
    value = str(value)
    if not re.match(r"^[A-Za-z0-9_.-]+$", value):
        raise ValueError("%s contains unsafe path characters: %r" % (name, value))
    return value


def _rate_name(value):
    return ("%.4f" % float(value)).rstrip("0").rstrip(".")


class RunLogger(object):
    def __init__(self, path):
        self.handle = open(path, "a", encoding="utf-8")

    def __call__(self, message):
        line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), message)
        print(line, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()

    def close(self):
        self.handle.close()


def _resolve_run(arguments):
    experiment_path = os.path.abspath(arguments.config)
    ablation_path = os.path.abspath(arguments.ablation)
    config = _deep_merge(
        load_json_config(experiment_path), load_json_config(ablation_path))
    if arguments.epochs is not None:
        config["training"]["epochs"] = int(arguments.epochs)
    if arguments.training_seed is not None:
        config["training"]["seed"] = int(arguments.training_seed)
    if arguments.dataset is not None:
        selected_dataset = dataset_spec(arguments.dataset)
        config["dataset"]["name"] = selected_dataset["name"]
        config["dataset"]["view_indices"] = selected_dataset[
            "default_view_indices"]
    if arguments.split_seed is not None:
        config["split"]["seed"] = int(arguments.split_seed)
    if arguments.observation_rate is not None:
        config["masks"]["observation_rate_within_incomplete"] = float(
            arguments.observation_rate)
    for split_name in ("train", "validation", "test"):
        mechanism = getattr(arguments, "%s_mechanism" % split_name)
        rate = getattr(arguments, "%s_incomplete_rate" % split_name)
        mask_seed = getattr(arguments, "%s_mask_seed" % split_name)
        if mechanism is not None:
            config["masks"][split_name]["mechanism"] = mechanism
        if rate is not None:
            config["masks"][split_name]["incomplete_rate"] = float(rate)
        if mask_seed is not None:
            config["masks"][split_name]["seed"] = int(mask_seed)
    if arguments.groupdro_eta is not None:
        config["loss"]["groupdro_eta"] = float(arguments.groupdro_eta)
    if arguments.device is not None:
        config["runtime"]["device"] = arguments.device
    _validate_ablation(config)
    torch.set_num_threads(int(config["runtime"].get("torch_num_threads", 1)))

    dataset_name = canonical_dataset_name(config["dataset"]["name"])
    dataset = load_multiview_data(
        dataset_name, data_root=arguments.data_root,
        view_indices=config["dataset"].get("view_indices"),
        normalization=config["dataset"].get("normalization", "legacy"))
    split = make_split(dataset.n_samples, seed=config["split"]["seed"],
                       fractions=config["split"]["fractions"])
    masks, metadata = {}, {}
    for split_name in ("train", "validation", "test"):
        masks[split_name], metadata[split_name] = _make_mask(
            dataset, split[split_name], config["masks"][split_name],
            config["masks"])

    environment = environment_snapshot(PROJECT_ROOT)
    environment.pop("captured_at", None)
    manifest = source_manifest(PROJECT_ROOT, [
        "rbdp", experiment_path, ablation_path, "tools/run_rbdp.py"])
    config["dataset"]["name"] = dataset_name
    config["resolved"] = {
        "experiment_config": experiment_path,
        "ablation_config": ablation_path,
        "data_root": os.path.abspath(arguments.data_root),
        "dataset_summary": dataset.summary(),
        "dataset_sha256": file_sha256(dataset.source_path),
        "split_hashes": {key: _array_hash(value) for key, value in split.items()},
        "mask_metadata": metadata,
        "environment": environment,
        "source_manifest": manifest,
    }
    seed_everything(config["training"]["seed"])
    model = build_model(dataset.view_dims, config)
    config["resolved"]["parameter_count"] = parameter_count(model)
    device = _resolve_device(config["runtime"]["device"])
    config["resolved"]["device"] = str(device)

    experiment_name = "%s__%s" % (
        _safe(config["experiment"], "experiment"),
        _safe(config["ablation"]["name"], "ablation name"))
    if arguments.run_tag:
        experiment_name += "__" + _safe(arguments.run_tag, "run tag")
    test_config = config["masks"]["test"]
    path = os.path.join(
        os.path.abspath(arguments.output_root), experiment_name,
        _safe(dataset_name, "dataset"),
        _safe(test_config["mechanism"], "mask mechanism"),
        "r%s" % _rate_name(test_config["incomplete_rate"]),
        "seed%d" % int(config["training"]["seed"]))
    return config, dataset, split, masks, metadata, model, device, path, manifest


def main():
    arguments = _arguments()
    (config, dataset, split, masks, metadata, model, device,
     run_path, manifest) = _resolve_run(arguments)
    run = RunDirectory(run_path, config)
    if run.reusable():
        print("Reusing completed run: %s" % run.path)
        return 0

    started, logger = False, None
    try:
        run.start()
        started = True
        logger = RunLogger(os.path.join(run.path, "stdout.log"))
        logger("run=%s" % run.path)
        logger("config_hash=%s" % run.hash)
        logger("dataset=%s dims=%r device=%s parameters=%d" % (
            dataset.name, dataset.view_dims, device, parameter_count(model)))
        for name in ("train", "validation", "test"):
            save_mask(os.path.join(run.path, "%s_mask.npz" % name),
                      masks[name], metadata[name])
        write_npz(os.path.join(run.path, "split.npz"), **split)

        trained = fit_model(
            model, [view[split["train"]] for view in dataset.views],
            masks["train"], config, device, log=logger)
        evaluated = evaluate_model(
            model, [view[split["test"]] for view in dataset.views],
            dataset.labels[split["test"]], masks["test"], split["test"],
            config, device)
        _write_artifacts(run, config, model, trained, evaluated)

        metrics = evaluated["metrics"]
        metrics["training"] = {
            "seconds": trained["training_seconds"],
            "sample_count": trained["training_sample_count"],
            "final_curve": trained["curves"][-1],
            "environment_index": trained["environment_index"],
            "groupdro_state": trained["groupdro_state"],
        }
        metrics["model"] = {"parameter_count": parameter_count(model),
                            "ablation": config["ablation"]}
        metrics["provenance"] = {
            "config_hash": run.hash,
            "source_hash": manifest["source_hash"],
            "dataset_sha256": config["resolved"]["dataset_sha256"],
            "train_mask_hash": metadata["train"]["statistics"]["mask_hash"],
            "test_mask_hash": metadata["test"]["statistics"]["mask_hash"],
        }
        run.complete(metrics)
        logger("completed acc=%.6f nmi=%.6f ari=%.6f worst=%.6f" % (
            metrics["clustering"]["acc"], metrics["clustering"]["nmi"],
            metrics["clustering"]["ari"],
            metrics["pattern_robustness"]["worst_pattern_acc"]))
        logger.close()
        print("Completed run: %s" % run.path)
        return 0
    except Exception as error:
        if logger is not None:
            logger("FAILED %s: %s" % (type(error).__name__, error))
            logger.handle.write(traceback.format_exc())
            logger.close()
        if started:
            run.fail(error)
        raise


def _write_artifacts(run, config, model, trained, evaluated):
    write_csv(os.path.join(run.path, "curves.csv"), trained["curves"])
    write_csv(os.path.join(run.path, "environment_risks.csv"),
              trained["environment_rows"],
              ["epoch", "environment", "count", "mean_loss",
               "prediction_mse", "heteroscedastic_nll", "cycle_mse",
               "structure_mse",
               "normalized_structure_mse",
               "mean_log_variance",
               "groupdro_probability", "robust_enabled"])
    write_csv(os.path.join(run.path, "reliability.csv"),
              evaluated["reliability_rows"],
              ["sample_index", "pattern", "source_view", "target_view",
               "log_variance", "cycle_mse", "risk", "prediction_mse"])
    write_csv(os.path.join(run.path, "risk_coverage.csv"),
              evaluated["coverage_rows"],
              ["scope", "coverage", "selective_error"])
    write_npz(os.path.join(run.path, "embedding.npz"), **evaluated["arrays"])
    if config["runtime"].get("save_checkpoint", True):
        save_checkpoint(os.path.join(run.path, "checkpoint.pt"), model,
                        trained["optimizer_state"], run.hash,
                        trained["groupdro_state"])


if __name__ == "__main__":
    sys.exit(main())
