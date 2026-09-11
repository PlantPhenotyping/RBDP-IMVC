"""Atomic, hash-addressed result recording for reproducible experiments."""

import hashlib
import csv
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime

import numpy as np


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "__fspath__"):
        return os.fspath(value)
    raise TypeError("object of type %s is not JSON serializable" %
                    type(value).__name__)


def canonical_json(value):
    """Serialize JSON deterministically for hashing and provenance."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    )


def config_hash(config):
    """Return the SHA-256 digest of a fully resolved configuration object."""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def file_sha256(path, block_size=1024 * 1024):
    """Stream a file into SHA-256 without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(int(block_size))
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def source_manifest(project_root, paths):
    """Hash selected source/config paths and return a combined code digest."""
    root = os.path.abspath(project_root)
    selected = []
    for path in paths:
        absolute = os.path.abspath(os.path.join(root, path))
        if os.path.commonpath([root, absolute]) != root:
            raise ValueError("manifest path escapes project root: %s" % path)
        if os.path.isdir(absolute):
            for directory, names, filenames in os.walk(absolute):
                names[:] = sorted(name for name in names if name != "__pycache__")
                for filename in sorted(filenames):
                    if filename.endswith((".py", ".json", ".sh")):
                        selected.append(os.path.join(directory, filename))
        elif os.path.isfile(absolute):
            selected.append(absolute)
        else:
            raise IOError("manifest path does not exist: %s" % absolute)
    files = {}
    for absolute in sorted(set(selected)):
        relative = os.path.relpath(absolute, root).replace(os.sep, "/")
        files[relative] = file_sha256(absolute)
    return {
        "files": files,
        "source_hash": config_hash(files),
    }


def write_json(path, value):
    """Atomically write a UTF-8, human-readable JSON document."""
    destination = os.path.abspath(path)
    parent = os.path.dirname(destination)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    descriptor, temporary = tempfile.mkstemp(prefix=".json-", suffix=".tmp",
                                             dir=parent or None)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
                default=_json_default,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path, rows, fieldnames=None):
    """Atomically write a list of scalar dictionaries as UTF-8 CSV."""
    rows = list(rows)
    if fieldnames is None:
        keys = set()
        for row in rows:
            keys.update(row.keys())
        fieldnames = sorted(keys)
    fieldnames = list(fieldnames)
    if not fieldnames:
        raise ValueError("fieldnames are required for an empty CSV")
    destination = os.path.abspath(path)
    parent = os.path.dirname(destination)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    descriptor, temporary = tempfile.mkstemp(prefix=".csv-", suffix=".tmp",
                                             dir=parent or None)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames,
                                    extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def write_npz(path, **arrays):
    """Atomically write compressed NumPy arrays."""
    destination = os.path.abspath(path)
    parent = os.path.dirname(destination)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    descriptor, temporary = tempfile.mkstemp(prefix=".npz-", suffix=".npz",
                                             dir=parent or None)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def utc_timestamp():
    """Return an ISO-8601 UTC timestamp without Python 3.11 dependencies."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _git_value(project_root, arguments):
    try:
        output = subprocess.check_output(
            ["git"] + list(arguments),
            cwd=project_root,
            stderr=subprocess.DEVNULL,
        )
        return output.decode("utf-8").strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_snapshot(project_root):
    """Capture versions and immutable Git identifiers without changing state."""
    try:
        import scipy
    except ImportError:
        scipy = None
    try:
        import sklearn
    except ImportError:
        sklearn = None
    try:
        import torch
    except ImportError:
        torch = None
    git_status = _git_value(project_root, ["status", "--short"])
    return {
        "captured_at": utc_timestamp(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": None if scipy is None else scipy.__version__,
        "scikit_learn": None if sklearn is None else sklearn.__version__,
        "torch": None if torch is None else torch.__version__,
        "cuda_available": None if torch is None else bool(torch.cuda.is_available()),
        "git_commit": _git_value(project_root, ["rev-parse", "HEAD"]),
        "git_branch": _git_value(project_root, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": None if git_status is None else bool(git_status),
    }


class RunDirectory(object):
    """Manage ``config/status/metrics.json`` with resume-safety checks.

    ``start`` writes status ``running``.  ``complete`` first writes metrics and
    only then switches status to ``completed``.  An existing completed run is
    reusable only when its stored configuration hash matches exactly.
    """

    def __init__(self, path, config):
        self.path = os.path.abspath(path)
        self.config = dict(config)
        self.hash = config_hash(self.config)

    def _path(self, filename):
        return os.path.join(self.path, filename)

    def reusable(self):
        config_path = self._path("config.json")
        status_path = self._path("status.json")
        if not os.path.isfile(config_path) or not os.path.isfile(status_path):
            return False
        stored_config = read_json(config_path)
        status = read_json(status_path)
        return (status.get("state") == "completed" and
                stored_config.get("config_hash") == self.hash)

    def start(self):
        config_path = self._path("config.json")
        status_path = self._path("status.json")
        if os.path.isfile(config_path):
            existing = read_json(config_path)
            if existing.get("config_hash") != self.hash:
                raise ValueError(
                    "run directory already belongs to a different configuration")
        if os.path.isfile(status_path):
            status = read_json(status_path)
            if status.get("state") == "completed" and self.reusable():
                raise RuntimeError("matching run is already completed; reuse it")
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        resolved = dict(self.config)
        resolved["config_hash"] = self.hash
        write_json(self._path("config.json"), resolved)
        write_json(self._path("status.json"), {
            "state": "running",
            "config_hash": self.hash,
            "started_at": utc_timestamp(),
        })

    def complete(self, metrics):
        write_json(self._path("metrics.json"), metrics)
        write_json(self._path("status.json"), {
            "state": "completed",
            "config_hash": self.hash,
            "completed_at": utc_timestamp(),
        })

    def fail(self, error):
        write_json(self._path("status.json"), {
            "state": "failed",
            "config_hash": self.hash,
            "failed_at": utc_timestamp(),
            "error_type": type(error).__name__,
            "error": str(error),
        })
