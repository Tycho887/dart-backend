"""Small numerical seam for stacking heterogeneous forward-model blocks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForwardModelBlock:
    """One evaluated measurement model in its native unit."""

    kind: str
    unit: str
    observed: np.ndarray
    predicted: np.ndarray
    sigma: np.ndarray
    jacobian: np.ndarray

    def normalized_residuals(self) -> np.ndarray:
        return normalize_residuals(self.predicted, self.observed, self.sigma)

    def normalized_jacobian(self) -> np.ndarray:
        return normalize_jacobian(self.jacobian, self.sigma)


def normalize_residuals(
    predicted: np.ndarray, observed: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """Normalize one native-unit residual block by per-row uncertainty."""

    if predicted.shape != observed.shape or observed.shape != sigma.shape:
        raise ValueError("predicted, observed, and sigma vectors must have equal shapes")
    if np.any(sigma <= 0.0):
        raise ValueError("measurement sigma values must be positive")
    return (predicted - observed) / sigma


def normalize_jacobian(jacobian: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Apply the same row normalization to a forward-model Jacobian."""

    if jacobian.ndim != 2 or jacobian.shape[0] != len(sigma):
        raise ValueError("Jacobian rows must align with measurement sigma values")
    if np.any(sigma <= 0.0):
        raise ValueError("measurement sigma values must be positive")
    return jacobian / sigma[:, None]


def stack_residual_blocks(blocks: list[np.ndarray]) -> np.ndarray:
    """Vertically concatenate residual blocks selected for one solve."""

    if not blocks:
        raise ValueError("At least one forward-model residual block is required")
    return np.concatenate(blocks)


def stack_jacobian_blocks(blocks: list[np.ndarray]) -> np.ndarray:
    """Vertically concatenate Jacobian blocks selected for one solve."""

    if not blocks:
        raise ValueError("At least one forward-model Jacobian block is required")
    if any(block.ndim != 2 for block in blocks):
        raise ValueError("Every Jacobian block must be a matrix")
    column_counts = {block.shape[1] for block in blocks}
    if len(column_counts) != 1:
        raise ValueError("All Jacobian blocks must be matrices with equal column counts")
    return np.vstack(blocks)
