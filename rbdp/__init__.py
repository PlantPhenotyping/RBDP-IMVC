"""Reproducible infrastructure for the RBDP-IMVC research branch.

The original COMPLETER and DCP entry points intentionally do not import this
package.  New experiments use the APIs here so that legacy behavior remains
available as a regression baseline.
"""

from .config import DATASET_SPECS, canonical_dataset_name, dataset_spec
from .data import MultiViewData, load_multiview_data, make_split

__all__ = [
    "DATASET_SPECS",
    "MultiViewData",
    "canonical_dataset_name",
    "dataset_spec",
    "load_multiview_data",
    "make_split",
]

__version__ = "0.1.0"
