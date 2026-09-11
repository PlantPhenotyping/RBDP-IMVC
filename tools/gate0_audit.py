#!/usr/bin/env python
"""Audit deterministic loading, splits, and masks for all local datasets."""

import argparse
import gc
import hashlib
import os
import sys

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rbdp.config import canonical_dataset_name, load_json_config  # noqa: E402
from rbdp.data import load_multiview_data, make_split  # noqa: E402
from rbdp.io import config_hash, environment_snapshot, file_sha256  # noqa: E402
from rbdp.io import source_manifest, write_json  # noqa: E402
from rbdp.masks import CONTROLLED_MECHANISMS, generate_mask, save_mask  # noqa: E402


DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "experiments", "gate0.json")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "outputs", "gate0", "audit.json")


def _array_hash(values):
    value = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.view(np.uint8).tobytes())
    return digest.hexdigest()


def _parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="Gate-0 JSON configuration")
    parser.add_argument("--data-root", default=os.path.join(PROJECT_ROOT, "data"),
                        help="directory containing MATLAB datasets")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="atomic JSON audit output")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="optional dataset subset or aliases")
    parser.add_argument("--mechanisms", nargs="+", default=None,
                        choices=CONTROLLED_MECHANISMS)
    parser.add_argument("--rates", nargs="+", default=None, type=float,
                        help="override incomplete rates")
    parser.add_argument("--mask-seed", default=None, type=int)
    parser.add_argument("--save-mask-dir", default=None,
                        help="optional directory for generated NPZ masks")
    parser.add_argument("--skip-file-hash", action="store_true",
                        help="skip potentially slow dataset SHA-256")
    return parser.parse_args()


def _resolved_config(arguments):
    config = load_json_config(arguments.config)
    if arguments.datasets is not None:
        config["datasets"] = [canonical_dataset_name(name)
                              for name in arguments.datasets]
    else:
        config["datasets"] = [canonical_dataset_name(name)
                              for name in config["datasets"]]
    if arguments.mechanisms is not None:
        config["mechanisms"] = list(arguments.mechanisms)
    if arguments.rates is not None:
        config["incomplete_rates"] = [float(value) for value in arguments.rates]
    if arguments.mask_seed is not None:
        config["mask_seed"] = int(arguments.mask_seed)
    if arguments.skip_file_hash:
        config["hash_dataset_files"] = False
    config["data_root"] = os.path.abspath(arguments.data_root)
    config["config_path"] = os.path.abspath(arguments.config)
    return config


def _split_record(n_samples, config):
    split = make_split(
        n_samples,
        seed=config["split_seed"],
        fractions=config["split_fractions"],
    )
    concatenated = np.concatenate([
        split["train"], split["validation"], split["test"]
    ])
    if np.unique(concatenated).size != n_samples:
        raise AssertionError("split indices are not a disjoint partition")
    return {
        "seed": int(config["split_seed"]),
        "fractions": [float(value) for value in config["split_fractions"]],
        "sizes": {key: int(value.size) for key, value in split.items()},
        "hashes": {key: _array_hash(value) for key, value in split.items()},
    }


def _mask_record(dataset, mechanism, rate, config, save_mask_dir):
    parameters = {
        "n_samples": dataset.n_samples,
        "n_views": dataset.n_views,
        "incomplete_rate": float(rate),
        "mechanism": mechanism,
        "seed": int(config["mask_seed"]),
        "observation_rate": float(
            config["observation_rate_within_incomplete"]),
        "correlation_strength": float(config["correlation_strength"]),
    }
    if mechanism == "feature_dependent":
        parameters["features"] = dataset.views[0]
    mask, metadata = generate_mask(**parameters)
    repeated, repeated_metadata = generate_mask(**parameters)
    if not np.array_equal(mask, repeated):
        raise AssertionError("mask generation is not deterministic")
    if metadata["statistics"]["mask_hash"] != repeated_metadata[
            "statistics"]["mask_hash"]:
        raise AssertionError("repeated mask hashes disagree")

    record = dict(metadata)
    record["deterministic_repeat_verified"] = True
    if save_mask_dir:
        rate_name = ("%.4f" % float(rate)).rstrip("0").rstrip(".")
        filename = "%s__%s__r%s__seed%d.npz" % (
            dataset.name.replace("/", "_"),
            mechanism,
            rate_name,
            int(config["mask_seed"]),
        )
        path = os.path.join(save_mask_dir, filename)
        save_mask(path, mask, metadata)
        record["saved_path"] = os.path.abspath(path)
    return record


def main():
    arguments = _parse_arguments()
    config = _resolved_config(arguments)
    result = {
        "schema_version": 1,
        "state": "running",
        "config": config,
        "config_hash": config_hash(config),
        "environment": environment_snapshot(PROJECT_ROOT),
        "source": source_manifest(PROJECT_ROOT, [
            "rbdp",
            "configs/datasets",
            "configs/experiments/gate0.json",
            "tools/gate0_audit.py",
            "scripts/run_gate0.sh",
        ]),
        "datasets": {},
    }

    for name in config["datasets"]:
        dataset = load_multiview_data(
            name,
            data_root=config["data_root"],
            normalization=config["normalization"],
        )
        record = dataset.summary()
        record["source_sha256"] = (
            file_sha256(dataset.source_path)
            if config.get("hash_dataset_files", True) else None
        )
        record["split"] = _split_record(dataset.n_samples, config)
        record["masks"] = {}
        for mechanism in config["mechanisms"]:
            for rate in config["incomplete_rates"]:
                key = "%s/r%.4f" % (mechanism, float(rate))
                record["masks"][key] = _mask_record(
                    dataset, mechanism, rate, config, arguments.save_mask_dir)
        result["datasets"][dataset.name] = record
        del dataset
        gc.collect()

    result["state"] = "completed"
    write_json(arguments.output, result)
    print("Gate 0 audit completed: %s" % os.path.abspath(arguments.output))
    print("Configuration hash: %s" % result["config_hash"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
