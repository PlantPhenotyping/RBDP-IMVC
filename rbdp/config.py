"""Dataset registry and JSON configuration helpers.

This module only uses the Python standard library so it remains compatible
with the project's Python 3.7 ``imvc`` environment.
"""

import copy
import json
import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


DATASET_SPECS = {
    "Caltech101-20": {
        "file": "Caltech101-20.mat",
        "samples": 2386,
        "classes": 20,
        "all_view_dims": [48, 40, 254, 1984, 512, 928],
        "default_view_indices": [3, 4, 5],
        "default_view_names": ["hog", "gist", "lbp"],
        "legacy_normalize": True,
    },
    "Scene-15": {
        "file": "Scene-15.mat",
        "samples": 4485,
        "classes": 15,
        "all_view_dims": [20, 59, 40],
        "default_view_indices": [0, 1, 2],
        "default_view_names": ["gist", "phog", "lbp"],
        "legacy_normalize": False,
    },
    "LandUse-21": {
        "file": "LandUse-21.mat",
        "samples": 2100,
        "classes": 21,
        "all_view_dims": [20, 59, 40],
        "default_view_indices": [1, 2, 0],
        "default_view_names": ["phog", "lbp", "gist"],
        "legacy_normalize": False,
    },
    "NoisyMNIST": {
        "file": "NoisyMNIST.mat",
        "samples": 20000,
        "classes": 10,
        "all_view_dims": [784, 784],
        "default_view_indices": [0, 1],
        "default_view_names": ["clean", "noisy"],
        "legacy_normalize": False,
        "subset": "tune+test",
    },
}


_ALIASES = {
    "caltech101-20": "Caltech101-20",
    "caltech101_20": "Caltech101-20",
    "caltech": "Caltech101-20",
    "scene-15": "Scene-15",
    "scene_15": "Scene-15",
    "scene15": "Scene-15",
    "landuse-21": "LandUse-21",
    "landuse_21": "LandUse-21",
    "landuse21": "LandUse-21",
    "noisymnist": "NoisyMNIST",
    "noisy_mnist": "NoisyMNIST",
}


def canonical_dataset_name(name):
    """Return the canonical dataset name used by new experiment outputs."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("dataset name must be a non-empty string")
    if name in DATASET_SPECS:
        return name
    key = name.strip().lower()
    if key not in _ALIASES:
        raise KeyError("unknown dataset %r; choices are %s" %
                       (name, ", ".join(sorted(DATASET_SPECS))))
    return _ALIASES[key]


def dataset_spec(name):
    """Return a defensive copy of one registry entry."""
    canonical = canonical_dataset_name(name)
    spec = copy.deepcopy(DATASET_SPECS[canonical])
    spec["name"] = canonical
    return spec


def resolve_data_file(name, data_root=None):
    """Resolve and validate a local MATLAB dataset file."""
    spec = dataset_spec(name)
    root = os.path.abspath(data_root or os.path.join(PROJECT_ROOT, "data"))
    path = os.path.join(root, spec["file"])
    if not os.path.isfile(path):
        raise IOError("dataset file does not exist: %s" % path)
    return path


def load_json_config(path):
    """Read a JSON object from ``path`` without requiring PyYAML."""
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a JSON object: %s" % path)
    return value
