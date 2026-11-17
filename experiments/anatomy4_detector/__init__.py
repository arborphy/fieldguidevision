"""Anatomy4 four-class detector: validation, training, evaluation, export.

Contract ``anatomy4-v1``. Pure modules (contracts, dataset, metrics, render)
never import torch/ultralytics so they stay unit-testable and import-safe.
"""

from .contracts import CLASS_IDS, CLASS_NAMES, CONTRACT

__all__ = ["CLASS_IDS", "CLASS_NAMES", "CONTRACT"]
