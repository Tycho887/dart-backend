"""Causal contact grouping and Doppler-only local information selection."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Literal

import numpy as np

from dart.forward_models import _native
from dart.io import ContactMetadata
from dart.od import _canonical_parameter_names, _sgp4_evaluator
from dart.od.schema import OptimizerContext, OptimizerOutput, OrbitModel, PriorStateData


@dataclass(frozen=True)
class PassInformation:
    singular_values: tuple[float, ...]
    rank: int
    fim_condition: float | None
    inverse_information_trace: float | None
    weighting: str


@dataclass(frozen=True)
class ContactGroup:
    contact_ids: tuple[str, ...]
    status: str
    reasons: Mapping[str, str]


def build_contact_groups(
    contacts: Sequence[ContactMetadata],
    anchor: ContactMetadata,
    window_sizes: Sequence[int] = (1, 3, 5, 8),
) -> dict[int, ContactGroup]:
    """Input contacts have already passed quality gates; exclude future contacts."""
    if len({c.contact_id for c in contacts}) != len(contacts):
        raise ValueError("duplicate contact IDs")
    if any(n < 1 for n in window_sizes):
        raise ValueError("window sizes must be positive")
    available = sorted(
        (
            c
            for c in contacts
            if c.spacecraft_id == anchor.spacecraft_id
            and (c.stop, c.contact_id) <= (anchor.stop, anchor.contact_id)
        ),
        key=lambda c: (c.stop, c.contact_id),
    )
    if anchor.contact_id not in {c.contact_id for c in available}:
        return {
            n: ContactGroup(
                (),
                "screening_failed",
                {anchor.contact_id: "anchor failed quality gates"},
            )
            for n in window_sizes
        }
    ids = tuple(c.contact_id for c in available)
    return {
        n: ContactGroup(ids[-n:], "ready" if len(ids) >= n else "warm_up", {})
        for n in window_sizes
    }


def assess_pass_information(
    data: PriorStateData,
    pilot: OptimizerOutput,
    optimizer: OptimizerContext,
    orbit_scales: Sequence[float],
) -> PassInformation:
    """Full six-orbit information at an L pilot; not a calibrated covariance.

    Existing raw whitened Jacobians remain untouched. The Rust kernel applies
    Soft-L1 curvature and removes bias sensitivity before its rank-revealing SVD.
    """
    if not pilot.success or len(data.observations.contacts) != 1:
        raise ValueError("information assessment requires one successful pilot")
    if optimizer.model != OrbitModel.SGP4 or pilot.model_kind != OrbitModel.SGP4:
        raise ValueError("information assessment requires SGP4")
    if optimizer.loss not in ("linear", "soft_l1"):
        raise ValueError("information supports linear and soft_l1 losses")
    names = _canonical_parameter_names(data, optimizer.model)
    values = dict(zip(pilot.parameter_names, pilot.parameters, strict=True))
    evaluate, _ = _sgp4_evaluator(data)
    evaluated = evaluate(np.array([values.get(n, 0.0) for n in names]))
    singular, rank, condition, trace = _native.orbit_information(
        evaluated.jacobian[:, [0, 1, 2, 3, 4, 5, 9]].tolist(),
        evaluated.residuals.tolist(),
        list(orbit_scales),
        optimizer.loss_scale if optimizer.loss == "soft_l1" else 0.0,
    )
    return PassInformation(tuple(singular), rank, condition, trace, optimizer.loss)


def select_contacts(
    candidates: Sequence[ContactMetadata],
    metrics: Mapping[str, PassInformation],
    metric: Literal["trace", "condition"] = "trace",
    *,
    fraction: float = 0.5,
) -> ContactGroup:
    if metric not in ("trace", "condition") or not 0 < fraction <= 1:
        raise ValueError("invalid selection metric or retained fraction")
    field = {"trace": "inverse_information_trace", "condition": "fim_condition"}[metric]
    valid = [
        c
        for c in candidates
        if c.contact_id in metrics
        and metrics[c.contact_id].rank == 6
        and getattr(metrics[c.contact_id], field) is not None
    ]
    ordered = sorted(
        valid,
        key=lambda c: (
            getattr(metrics[c.contact_id], field),
            -c.stop.timestamp(),
            c.contact_id,
        ),
    )
    selected = {c.contact_id for c in ordered[: ceil(fraction * len(valid))]}
    reasons = {
        c.contact_id: _selection_reason(c.contact_id, selected, metrics)
        for c in candidates
    }
    ids = tuple(
        c.contact_id
        for c in sorted(candidates, key=lambda c: (c.stop, c.contact_id))
        if c.contact_id in selected
    )
    return ContactGroup(ids, "ready" if len(ids) >= 2 else "selection_failed", reasons)


def _selection_reason(
    cid: str, selected: set[str], metrics: Mapping[str, PassInformation]
) -> str:
    if cid not in metrics:
        return "pilot failed"
    if metrics[cid].rank < 6:
        return "rank deficient"
    return "selected" if cid in selected else "below retained fraction"
