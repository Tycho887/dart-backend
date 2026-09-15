"""Post-fit diagnostics from saved solutions, without refitting or OEM selection."""

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import satkit as sk

from dart.forward_models import FloatArray, ForwardModelEvaluation, evaluate_sgp4_epoch
from dart.io import ForwardModelContext
from dart.io.doppler import prepare_doppler, select_time_offset_doppler
from dart.od import OrbitModel, PriorStateData, _canonical_parameter_names
from experiments._benchmark_io import _contact, _ephemeris, _read_snapshot

Record = dict[str, Any]


@dataclass(frozen=True)
class QualityGate:
    min_samples: int = 250
    max_condition_number: float = 1e6
    sample_scope: Literal["fit", "pass"] = "fit"

    def __post_init__(self) -> None:
        if self.min_samples < 1:
            raise ValueError("minimum samples must be positive")
        if not np.isfinite(self.max_condition_number) or self.max_condition_number < 1:
            raise ValueError("maximum condition number must be finite and >= 1")
        if self.sample_scope not in {"fit", "pass"}:
            raise ValueError("sample scope must be fit or pass")


def jacobian_diagnostics(
    evaluation: ForwardModelEvaluation, scales: FloatArray, loss: str, loss_scale: float
) -> Record:
    """Condition the profile-scaled Jacobian, with SciPy's robust curvature weights.

    For soft-L1, sqrt(rho'(z) + 2*z*rho''(z)) = (1+z)**(-3/4).
    This diagnoses local identifiability, not calibrated parameter uncertainty.
    """
    jacobian = evaluation.jacobian * scales
    if loss == "soft_l1":
        weights = (1 + np.square(evaluation.residuals / loss_scale)) ** -0.75
        jacobian = jacobian * weights[:, None]
    elif loss != "linear":
        raise ValueError("quality diagnostics support linear and soft_l1 loss")
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("nonfinite fit Jacobian")
    singular = np.linalg.svd(jacobian, compute_uv=False)
    tolerance = np.finfo(float).eps * max(jacobian.shape) * singular.max(initial=0)
    rank = int(np.count_nonzero(singular > tolerance))
    parameters = jacobian.shape[1]
    condition = (
        float(singular[0] / singular[-1]) if rank == parameters and parameters else None
    )
    return {
        "quality_condition_number": condition,
        "quality_rank": rank,
        "quality_parameter_count": parameters,
        "quality_degrees_of_freedom": jacobian.shape[0] - parameters,
        "quality_diagnostic_unavailable_reason": "",
    }


def _replay_context(run: Record, measurements: pl.DataFrame) -> ForwardModelContext:
    metadata = run["metadata"]
    contacts = [_contact(c) for c in metadata["contacts"]]
    selected = measurements.filter(pl.col("contact_id").is_in(run["contact_ids"]))
    context, counts = prepare_doppler(
        contacts,
        selected,
        center_frequency_hz=metadata["center_frequency_hz"],
        variance_hz2=metadata["variance_hz2"],
        min_samples=metadata["min_samples"],
        min_ebn0_db=metadata["min_ebn0_db"],
        max_abs_offset_hz=metadata["max_abs_offset_hz"],
        selector=(
            select_time_offset_doppler
            if metadata.get("selection_policy") == "forest_time_offset"
            else None
        ),
    )
    if [c.retained_samples for c in counts] != [
        c["retained_samples"] for c in metadata["selection"]
    ]:
        raise ValueError("quality replay sample counts differ from saved fit")
    times = [o.time.as_unixtime() for o in context.observations]
    if times != run["doppler"]["timestamp_unix_s"]:
        raise ValueError("quality replay observation timestamps differ from saved fit")
    return context


def _replay_evaluation(
    run: Record, measurements: pl.DataFrame
) -> tuple[ForwardModelEvaluation, FloatArray]:
    metadata, output = run["metadata"], run["metadata"]["output"]
    context = _replay_context(run, measurements)
    prior = PriorStateData(
        context,
        _ephemeris(metadata["initial_ephemeris"]),
        sk.time.from_unixtime(metadata["epoch_unix_s"]),
    )
    names = _canonical_parameter_names(prior, OrbitModel.SGP4)
    vector = np.zeros(len(names))
    for name, value in zip(
        output["parameter_names"], output["parameters"], strict=True
    ):
        vector[names.index(name)] = value
    prepared = metadata["sgp4_preparation"]
    evaluation = evaluate_sgp4_epoch(vector, tuple(prepared["tle_lines"]), context)
    expected = np.asarray(run["doppler"]["residual_hz"]) / np.sqrt(
        metadata["variance_hz2"]
    )
    if not np.allclose(evaluation.residuals, expected, rtol=1e-10, atol=1e-5):
        raise ValueError("quality replay residuals differ from saved fit")
    estimated = [
        p for p in metadata["optimizer"]["parameters"] if p["role"] == "estimate"
    ]
    indices = [names.index(p["name"]) for p in estimated]
    return ForwardModelEvaluation(
        evaluation.residuals, evaluation.jacobian[:, indices]
    ), np.array([p["scale"] for p in estimated])


def _diagnostics(run: Record, measurements: pl.DataFrame) -> Record:
    unavailable = {
        "quality_condition_number": None,
        "quality_rank": None,
        "quality_parameter_count": None,
        "quality_degrees_of_freedom": None,
    }
    if not run["metadata"]["output"]["success"]:
        return {
            **unavailable,
            "quality_diagnostic_unavailable_reason": "fit/preparation unsuccessful",
        }
    if "optimizer" not in run["metadata"] or not run["metadata"].get(
        "sgp4_preparation"
    ):
        return {
            **unavailable,
            "quality_diagnostic_unavailable_reason": "saved fit lacks replay metadata",
        }
    evaluation, scales = _replay_evaluation(run, measurements)
    optimizer = run["metadata"]["optimizer"]
    return jacobian_diagnostics(
        evaluation, scales, optimizer["loss"], optimizer["loss_scale"]
    )


def _sample_counts(case: Record, run: Record) -> dict[str, int]:
    selection = run["metadata"].get("selection", case.get("inventory", []))
    counts = {c["contact_id"]: c["retained_samples"] for c in selection}
    return {cid: counts[cid] for cid in run["contact_ids"] if cid in counts}


def _conditioning_rejections(diagnostic: Record, gate: QualityGate) -> list[str]:
    if diagnostic["quality_diagnostic_unavailable_reason"]:
        return [diagnostic["quality_diagnostic_unavailable_reason"]]
    reasons = []
    if diagnostic["quality_parameter_count"] == 0:
        reasons.append("no estimated parameters")
    if diagnostic["quality_rank"] < diagnostic["quality_parameter_count"]:
        reasons.append("rank-deficient scaled Jacobian")
    if diagnostic["quality_degrees_of_freedom"] <= 0:
        reasons.append("no residual degrees of freedom")
    condition = diagnostic["quality_condition_number"]
    if condition is not None and condition > gate.max_condition_number:
        reasons.append(f"condition number exceeds {gate.max_condition_number:g}")
    return reasons


def quality_decision(
    run: Record, counts: dict[str, int], diagnostic: Record, gate: QualityGate
) -> Record:
    reasons = _conditioning_rejections(diagnostic, gate)
    total = sum(counts.values()) if counts else None
    minimum = min(counts.values()) if counts else None
    selected = total if gate.sample_scope == "fit" else minimum
    if selected is None or len(counts) != len(run["contact_ids"]):
        reasons.append("retained observation counts unavailable")
    elif selected < gate.min_samples:
        reasons.append(
            f"fewer than {gate.min_samples} retained samples per {gate.sample_scope}"
        )
    if run.get("active_bounds"):
        reasons.append("parameters at active bounds")
    if not run["metadata"]["output"]["success"]:
        reasons.append("fit/preparation unsuccessful")
    return {
        **diagnostic,
        "fit_sample_count": total,
        "minimum_pass_sample_count": minimum,
        "fit_samples_by_contact": json.dumps(counts, sort_keys=True),
        "quality_accepted": not reasons,
        "quality_rejection_reasons": "; ".join(dict.fromkeys(reasons)),
    }


def case_quality(case: Record, gate: QualityGate) -> dict[str, Record]:
    """Evaluate each saved low-fidelity solution; raw snapshots remain immutable."""
    snapshot = _read_snapshot(Path(case["snapshot_dir"]))
    measurements = pl.read_parquet(io.BytesIO(snapshot["raw-measurements.parquet"]))
    results = {}
    for run in case["runs"]:
        if run["stage"] not in {"timing", "sgp4_L+n"}:
            continue
        diagnostic = _diagnostics(run, measurements)
        results[run["run_id"]] = quality_decision(
            run, _sample_counts(case, run), diagnostic, gate
        )
    return results
