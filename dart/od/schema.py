"""Public data contracts for orbit-determination optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, TypeAlias

import numpy as np
import satkit
from numpy.typing import NDArray

from dart.io import EphemerisMetadata, ForwardModelContext

LossKind: TypeAlias = Literal["linear", "soft_l1", "huber", "cauchy", "arctan"]
FloatArray: TypeAlias = NDArray[np.float64]


class OrbitModel(StrEnum):
    """Forward model selected by :func:`dart.od.fit`."""

    SGP4 = "sgp4"
    FULL_STATE = "full_state"


class ParameterRole(StrEnum):
    """How an optimizer parameter participates in estimation."""

    ESTIMATE = "estimate"
    CONSIDER = "consider"
    FIXED = "fixed"


class PriorSource(StrEnum):
    """Prior representation actually used to initialize a fit."""

    TLE = "tle"
    FULL_STATE = "full_state"
    TLE_DERIVED_FULL_STATE = "tle_derived_full_state"


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """One fitted parameter in physical units."""

    name: str
    initial: float
    lower_bound: float
    upper_bound: float
    scale: float
    role: ParameterRole = ParameterRole.ESTIMATE


@dataclass(frozen=True, slots=True)
class OptimizerContext:
    """Reusable least-squares configuration, independent of observations."""

    model: OrbitModel
    parameters: tuple[ParameterSpec, ...]
    loss: LossKind = "linear"
    loss_scale: float = 1.0
    max_evaluations: int = 1000
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float = 1e-8


@dataclass(frozen=True, slots=True)
class PriorStateData:
    """Observations and source-orbit information for one spacecraft."""

    observations: ForwardModelContext
    ephemeris: EphemerisMetadata
    epoch: satkit.time
    nominal_state_gcrf_si: FloatArray | None = None


@dataclass(frozen=True, slots=True)
class OptimizerOutput:
    """Common result contract for the explicit OD entry points.

    Parameter-vector and covariance columns, as well as Jacobian columns,
    follow ``parameter_names``.
    """

    model_kind: OrbitModel
    prior_source: PriorSource
    parameter_names: tuple[str, ...]
    parameter_roles: tuple[ParameterRole, ...]
    parameters: FloatArray
    residuals: FloatArray
    jacobian: FloatArray
    loss: LossKind
    cost: float
    optimality: float
    success: bool
    status: int
    message: str
    function_evaluations: int
    covariance: FloatArray | None = None
    covariance_rank: int | None = None
    jacobian_evaluations: int | None = None


__all__ = [
    "OrbitModel",
    "OptimizerContext",
    "OptimizerOutput",
    "ParameterRole",
    "ParameterSpec",
    "PriorSource",
    "PriorStateData",
]
