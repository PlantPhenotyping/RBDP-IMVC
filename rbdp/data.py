"""Deterministic loaders for the four datasets already present in the repo.

The returned views are dense ``float32`` arrays with shapes ``[N, D_v]`` and
the label vector has shape ``[N]``.  Labels are metadata for evaluation only;
model training APIs must not consume them.  No row is shuffled by this module.
"""

import os

import numpy as np
import scipy.io as sio
from scipy import sparse

from .config import canonical_dataset_name, dataset_spec, resolve_data_file


class MultiViewData(object):
    """A validated in-memory multi-view dataset.

    Attributes:
        name: Canonical dataset name.
        views: List of ``float32`` arrays, each shaped ``[N, D_v]``.
        labels: Integer array shaped ``[N]``; evaluation metadata only.
        view_indices: Indices into the original MATLAB view cell.
        view_names: Human-readable feature names when known.
        source_path: Absolute path of the loaded ``.mat`` file.
    """

    def __init__(self, name, views, labels, view_indices, view_names, source_path):
        self.name = canonical_dataset_name(name)
        self.views = [np.asarray(view, dtype=np.float32) for view in views]
        self.labels = np.asarray(labels).reshape(-1).astype(np.int64, copy=False)
        self.view_indices = [int(index) for index in view_indices]
        self.view_names = list(view_names)
        self.source_path = os.path.abspath(source_path)
        self._validate()

    def _validate(self):
        if not self.views:
            raise ValueError("a multi-view dataset must contain at least one view")
        n_samples = self.labels.shape[0]
        for index, view in enumerate(self.views):
            if view.ndim != 2:
                raise ValueError("view %d must have shape [N, D], got %r" %
                                 (index, view.shape))
            if view.shape[0] != n_samples:
                raise ValueError("view %d has %d rows but labels have %d" %
                                 (index, view.shape[0], n_samples))
            if view.shape[1] <= 0:
                raise ValueError("view %d has no features" % index)
            if not np.isfinite(view).all():
                raise ValueError("view %d contains NaN or infinity" % index)
        if len(self.view_indices) != len(self.views):
            raise ValueError("view_indices length does not match views")
        if len(self.view_names) != len(self.views):
            raise ValueError("view_names length does not match views")

    @property
    def n_samples(self):
        return int(self.labels.shape[0])

    @property
    def n_views(self):
        return len(self.views)

    @property
    def view_dims(self):
        return [int(view.shape[1]) for view in self.views]

    @property
    def n_classes(self):
        return int(np.unique(self.labels).shape[0])

    def summary(self):
        """Return a JSON-serializable shape/meaning summary."""
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "n_views": self.n_views,
            "view_dims": self.view_dims,
            "view_indices": list(self.view_indices),
            "view_names": list(self.view_names),
            "n_classes": self.n_classes,
            "label_min": int(self.labels.min()),
            "label_max": int(self.labels.max()),
            "source_path": self.source_path,
            "dtype": "float32",
        }


def _as_dense_2d(value):
    if sparse.issparse(value):
        value = value.toarray()
    value = np.asarray(value)
    if value.ndim != 2:
        raise ValueError("expected a two-dimensional feature matrix, got %r" %
                         (value.shape,))
    return value


def _global_minmax(value):
    value = np.asarray(value, dtype=np.float32)
    lower = float(np.min(value))
    upper = float(np.max(value))
    width = upper - lower
    if width <= np.finfo(np.float32).eps:
        return np.zeros_like(value, dtype=np.float32)
    return ((value - lower) / width).astype(np.float32, copy=False)


def _cell_views(mat):
    if "X" not in mat:
        raise KeyError("MATLAB file does not contain variable 'X'")
    cells = np.asarray(mat["X"], dtype=object).reshape(-1)
    return [_as_dense_2d(cell) for cell in cells]


def _load_noisy_mnist(mat):
    required = ["XV1", "XV2", "XTe1", "XTe2", "tuneLabel", "testLabel"]
    missing = [key for key in required if key not in mat]
    if missing:
        raise KeyError("NoisyMNIST file is missing variables: %s" %
                       ", ".join(missing))
    views = [
        np.concatenate([mat["XV1"], mat["XTe1"]], axis=0),
        np.concatenate([mat["XV2"], mat["XTe2"]], axis=0),
    ]
    labels = np.concatenate([
        np.asarray(mat["tuneLabel"]).reshape(-1),
        np.asarray(mat["testLabel"]).reshape(-1),
    ])
    return views, labels


def load_multiview_data(name, data_root=None, view_indices=None,
                        normalization="legacy"):
    """Load one dataset through a deterministic, unified entry point.

    Args:
        name: Canonical name or a documented alias.
        data_root: Directory containing local ``.mat`` files.
        view_indices: Optional original-view indices and ordering.  Defaults to
            the three-view DCP protocol (or both NoisyMNIST views).
        normalization: ``"legacy"`` normalizes Caltech globally per view,
            ``"none"`` leaves all feature scales untouched, and ``"minmax"``
            normalizes every selected view globally to ``[0, 1]``.

    Returns:
        :class:`MultiViewData`.  Every view is ``float32 [N, D_v]``; labels are
        ``int64 [N]`` and are never used to choose or transform samples.
    """
    canonical = canonical_dataset_name(name)
    spec = dataset_spec(canonical)
    path = resolve_data_file(canonical, data_root=data_root)
    mat = sio.loadmat(path)

    if canonical == "NoisyMNIST":
        all_views, labels = _load_noisy_mnist(mat)
    else:
        all_views = _cell_views(mat)
        if "Y" not in mat:
            raise KeyError("MATLAB file does not contain variable 'Y'")
        labels = np.asarray(mat["Y"]).reshape(-1)

    selected = list(spec["default_view_indices"] if view_indices is None
                    else view_indices)
    if not selected:
        raise ValueError("view_indices must contain at least one view")
    if len(set(selected)) != len(selected):
        raise ValueError("view_indices contains duplicates: %r" % selected)
    if min(selected) < 0 or max(selected) >= len(all_views):
        raise IndexError("view index outside [0, %d]: %r" %
                         (len(all_views) - 1, selected))

    if normalization not in ("legacy", "none", "minmax"):
        raise ValueError("normalization must be legacy, none, or minmax")
    normalize = normalization == "minmax" or (
        normalization == "legacy" and spec["legacy_normalize"])
    views = []
    for index in selected:
        value = _as_dense_2d(all_views[index])
        if normalize:
            value = _global_minmax(value)
        else:
            value = np.asarray(value, dtype=np.float32)
        views.append(value)

    names_by_index = {
        int(index): name for index, name in zip(
            spec["default_view_indices"], spec["default_view_names"])
    }
    view_names = [names_by_index.get(int(index), "view_%d" % int(index))
                  for index in selected]
    result = MultiViewData(canonical, views, labels, selected, view_names, path)

    if result.n_samples != int(spec["samples"]):
        raise ValueError("%s expected %d samples, found %d" %
                         (canonical, spec["samples"], result.n_samples))
    if result.n_classes != int(spec["classes"]):
        raise ValueError("%s expected %d classes, found %d" %
                         (canonical, spec["classes"], result.n_classes))
    expected_dims = [int(spec["all_view_dims"][index]) for index in selected]
    if result.view_dims != expected_dims:
        raise ValueError("%s expected dimensions %r, found %r" %
                         (canonical, expected_dims, result.view_dims))
    return result


def make_split(n_samples, seed, fractions=(0.7, 0.1, 0.2)):
    """Create a deterministic, label-free train/validation/test split.

    The output is a dict of disjoint ``int64`` arrays.  Fractions must sum to
    one.  Labels are intentionally not accepted, preventing accidental use of
    test class information during experimental setup.
    """
    n_samples = int(n_samples)
    if n_samples < 3:
        raise ValueError("at least three samples are required")
    values = np.asarray(fractions, dtype=np.float64)
    if values.shape != (3,) or np.any(values <= 0):
        raise ValueError("fractions must contain three positive values")
    if not np.isclose(values.sum(), 1.0, atol=1e-10):
        raise ValueError("fractions must sum to one")

    rng = np.random.RandomState(int(seed))
    order = rng.permutation(n_samples).astype(np.int64, copy=False)
    train_end = int(np.floor(values[0] * n_samples))
    valid_end = train_end + int(np.floor(values[1] * n_samples))
    if train_end == 0 or valid_end == train_end or valid_end == n_samples:
        raise ValueError("fractions produce an empty split for N=%d" % n_samples)
    return {
        "train": np.sort(order[:train_end]),
        "validation": np.sort(order[train_end:valid_end]),
        "test": np.sort(order[valid_end:]),
    }
