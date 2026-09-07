"""Synthetic numerical experiments; no live data or antenna access.

Run with ``uv run python tests/cross_model_validation.py --sweep --output report.json``.
Fixtures describe controlled geometry, not visibility-selected real contacts.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import NotRequired, TypedDict

import numpy as np
import satkit as sk
from numpy.typing import NDArray

from dart.forward_models import (
    ForwardModelEvaluation,
    evaluate_full_state_augmented,
    evaluate_sgp4_augmented,
    tle_state_gcrf,
)
from dart.io import EphemerisMetadata, ForwardModelContext, ForwardObservation
from dart.od import (
    OptimizerContext,
    OptimizerOutput,
    OrbitModel,
    ParameterRole,
    ParameterSpec,
    PriorStateData,
    fit,
)
from dart.od.schema import LossKind

FloatArray = NDArray[np.float64]
ISS_TLE = (
    "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
)
REGIMES = {
    "LEO": (15.5, 51.6, 0.001),
    "MEO": (2.0, 55.0, 0.01),
    "GEO": (1.0027, 0.1, 0.001),
}
ORBIT_NAMES = {
    OrbitModel.SGP4: (
        "mean_motion_rev_per_day",
        "equinoctial_f",
        "equinoctial_g",
        "equinoctial_h",
        "equinoctial_k",
        "mean_longitude_deg",
    ),
    OrbitModel.FULL_STATE: (
        "position_x_m",
        "position_y_m",
        "position_z_m",
        "velocity_x_m_s",
        "velocity_y_m_s",
        "velocity_z_m_s",
    ),
}
ORBIT_SCALES = {
    OrbitModel.SGP4: (1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 0.1),
    OrbitModel.FULL_STATE: (1000.0, 1000.0, 1000.0, 1.0, 1.0, 1.0),
}
STATE_ERROR = np.array([1000.0, -800.0, 600.0, 1.0, -0.8, 0.5])


class Diagnostics(TypedDict):
    scaled_singular_values: list[float]
    rank: int
    estimated_parameters: int
    scaled_condition: float | None
    bound_hits: list[str]


class RecoveryMetrics(TypedDict, total=False):
    position_error_m: float
    velocity_error_m_s: float
    scaled_parameter_error: list[float]


class FitMetrics(Diagnostics, RecoveryMetrics):
    success: bool
    status: int
    message: str
    function_evaluations: int
    initial_cost: float
    final_cost: float
    optimality: float
    parameter_names: list[str]
    parameters: list[float]
    initial_training_rms_hz: float
    final_training_rms_hz: float
    initial_validation_rms_hz: float
    final_validation_rms_hz: float
    shared_state_model_mismatch_rms_hz: float


class ExperimentRecord(TypedDict):
    configuration: dict[str, object]
    metrics: NotRequired[FitMetrics]
    error: NotRequired[dict[str, str]]


@dataclass(frozen=True)
class CaseConfig:
    regime: str = "LEO"
    truth_model: OrbitModel = OrbitModel.FULL_STATE
    fit_model: OrbitModel = OrbitModel.SGP4
    sigma_hz: float = 0.1
    seed: int = 42
    noisy: bool = True
    initial_error: float = 1.0
    arc_fraction: float = 1.0
    receivers: int = 2
    parameters: str = "orbit"
    loss: LossKind = "linear"
    outliers: bool = False
    varying_variance: bool = False
    passes: int = 1
    max_evaluations: int = 100


@dataclass(frozen=True)
class SyntheticCase:
    config: CaseConfig
    prior: PriorStateData
    optimizer: OptimizerContext
    clean_hz: FloatArray
    noise_hz: FloatArray
    train: NDArray[np.bool_]
    truth_parameters: dict[str, float]
    truth_state_gcrf_si: FloatArray


def synthetic_tle(regime: str) -> tuple[str, str]:
    """Use satkit's serializer for valid checksums; these are NOT real TLEs."""
    tle = sk.TLE.from_lines(list(ISS_TLE))
    assert isinstance(tle, sk.TLE)
    tle.mean_motion, tle.inclination, tle.eccen = REGIMES[regime]
    tle.bstar = 0.0
    tle.mean_motion_dot = 0.0
    tle.mean_motion_dot_dot = 0.0
    line1, line2 = tle.to_2line()
    return line1, line2


def canonical_names(model: OrbitModel, context: ForwardModelContext) -> tuple[str, ...]:
    orbit = ORBIT_NAMES[model]
    if model == OrbitModel.SGP4:
        orbit += ("bstar",)
    contacts = sorted(
        context.contact_to_pass_idx, key=context.contact_to_pass_idx.__getitem__
    )
    return (
        orbit
        + ("time_offset_s", "center_frequency_offset_hz")
        + tuple(f"pass_bias_hz:{contact}" for contact in contacts)
    )


def evaluate(
    model: OrbitModel,
    values: dict[str, float],
    data: PriorStateData,
) -> ForwardModelEvaluation:
    x = np.array(
        [values.get(name, 0.0) for name in canonical_names(model, data.observations)]
    )
    if model == OrbitModel.FULL_STATE:
        assert data.nominal_state_gcrf_si is not None
        return evaluate_full_state_augmented(
            x,
            data.nominal_state_gcrf_si,
            data.epoch,
            data.observations,
        )
    assert data.ephemeris.tle is not None
    line1, line2 = data.ephemeris.tle.splitlines()
    return evaluate_sgp4_augmented(x, (line1, line2), data.observations)


def predictions(
    model: OrbitModel, values: dict[str, float], data: PriorStateData
) -> FloatArray:
    observations = data.observations.observations
    observed = np.array([obs.observed[0] for obs in observations])
    sigma = np.sqrt([obs.noise_cov[0][0] for obs in observations])
    return observed + evaluate(model, values, data).residuals * sigma


def subset(data: PriorStateData, mask: NDArray[np.bool_]) -> PriorStateData:
    selected = [
        obs
        for obs, keep in zip(data.observations.observations, mask, strict=True)
        if keep
    ]
    return replace(data, observations=replace(data.observations, observations=selected))


def observation_context(config: CaseConfig, epoch: sk.time) -> ForwardModelContext:
    period = 86400.0 / REGIMES[config.regime][0]
    times = np.linspace(60.0, period * config.arc_fraction, 40)
    if config.passes == 2:
        times = (
            np.concatenate((np.linspace(0.1, 0.3, 20), np.linspace(0.7, 0.9, 20)))
            * period
        )
    receivers = [
        sk.itrfcoord(latitude_deg=63.0, longitude_deg=10.0, altitude=0.0),
        sk.itrfcoord(latitude_deg=-20.0, longitude_deg=130.0, altitude=0.0),
    ][: config.receivers]
    # Deliberately non-lexical names: the pass index defines parameter order.
    contacts = ("pass-z", "pass-a")[: config.passes]
    rows = []
    for index, offset in enumerate(np.repeat(times, config.receivers)):
        pass_index = (index // config.receivers) // (40 // config.passes)
        sigma = config.sigma_hz
        if config.varying_variance:
            sigma *= (0.5, 1.0, 2.0)[index % 3]
        rows.append(
            ForwardObservation.from_scalar(
                epoch.as_unixtime() + float(offset),
                0.0,
                sigma**2,
                receiver_id=index % config.receivers,
                pass_index=pass_index,
                contact_id=contacts[pass_index],
            )
        )
    return ForwardModelContext(
        center_frequency_hz=400e6,
        receivers=receivers,
        contact_to_pass_idx={contact: index for index, contact in enumerate(contacts)},
        observations=rows,
    )


def injected_parameters(config: CaseConfig) -> dict[str, float]:
    biases = {"pass_bias_hz:pass-z": 8.0, "pass_bias_hz:pass-a": -5.0}
    groups = {
        "orbit": {},
        "bstar": {},
        "timing": {"time_offset_s": 0.35},
        "frequency_bias": {"center_frequency_offset_hz": 1e5, **biases},
        "pass_bias": biases,
        "augmented": {
            "time_offset_s": 0.35,
            "center_frequency_offset_hz": 1e5,
            **biases,
        },
    }
    return groups[config.parameters]


def optimizer_context(
    config: CaseConfig, context: ForwardModelContext
) -> OptimizerContext:
    orbit = ORBIT_NAMES[config.fit_model]
    biases = canonical_names(config.fit_model, context)[-context.num_passes :]
    groups = {
        "orbit": orbit,
        "bstar": orbit + ("bstar",),
        "timing": ("time_offset_s",),
        "frequency_bias": ("center_frequency_offset_hz",) + biases,
        "pass_bias": biases,
        "augmented": orbit + ("time_offset_s", "center_frequency_offset_hz") + biases,
    }
    scales = dict(zip(orbit, ORBIT_SCALES[config.fit_model], strict=True))
    scales.update(bstar=1e-5, time_offset_s=1.0, center_frequency_offset_hz=1e5)
    scales.update({name: 10.0 for name in biases})
    parameters = []
    for name in groups[config.parameters]:
        initial = 0.2 * config.initial_error if name == "mean_longitude_deg" else 0.0
        # Keep every full-state time derivative node after the state epoch.
        bound = 10.0 if name == "time_offset_s" else 100.0 * scales[name]
        parameters.append(ParameterSpec(name, initial, -bound, bound, scales[name]))
    return OptimizerContext(
        config.fit_model,
        tuple(parameters),
        loss=config.loss,
        max_evaluations=config.max_evaluations,
    )


def make_case(config: CaseConfig) -> SyntheticCase:
    line1, line2 = synthetic_tle(config.regime)
    tle = sk.TLE.from_lines([line1, line2])
    assert isinstance(tle, sk.TLE)
    state = tle_state_gcrf((line1, line2), tle.epoch)
    metadata = EphemerisMetadata(
        ephemeris_id="synthetic-ephemeris",
        spacecraft_id="synthetic-spacecraft",
        kind="TLE",
        origin="synthetic-test",
        tenant_id=None,
        epoch=None,
        last_usable_at=None,
        submitted_at=None,
        submitted_by=None,
        tle=f"{line1}\n{line2}",
        omm=None,
        oem=None,
        is_cui=False,
        payload=None,
    )
    context = observation_context(config, tle.epoch)
    truth = PriorStateData(context, metadata, tle.epoch, state)
    injected = injected_parameters(config)
    clean = predictions(config.truth_model, injected, truth)
    sigma = np.sqrt([obs.noise_cov[0][0] for obs in context.observations])
    noise = (
        np.random.default_rng(config.seed).normal(size=clean.size)
        * sigma
        * config.noisy
    )
    if config.outliers:
        noise[::11] += 30.0 * sigma[::11]
    observed = [
        replace(obs, observed=[float(value)])
        for obs, value in zip(context.observations, clean + noise, strict=True)
    ]
    nominal = state.copy()
    if config.parameters in {"orbit", "augmented", "bstar"}:
        nominal += STATE_ERROR * config.initial_error
    prior = replace(
        truth,
        observations=replace(context, observations=observed),
        nominal_state_gcrf_si=nominal,
    )
    train = (np.arange(clean.size) // config.receivers) % 4 != 3
    return SyntheticCase(
        config,
        prior,
        optimizer_context(config, context),
        clean,
        noise,
        train,
        injected,
        state,
    )


def shared_state_parameters(case: SyntheticCase) -> dict[str, float]:
    """Same physical epoch state, not a claim that opposite trajectories agree."""
    values = dict(case.truth_parameters)
    if case.config.fit_model == OrbitModel.FULL_STATE:
        assert case.prior.nominal_state_gcrf_si is not None
        correction = case.truth_state_gcrf_si - case.prior.nominal_state_gcrf_si
        values.update(zip(ORBIT_NAMES[OrbitModel.FULL_STATE], correction, strict=True))
    return values


def diagnostics(result: OptimizerOutput, optimizer: OptimizerContext) -> Diagnostics:
    estimated = np.array(
        [p.role == ParameterRole.ESTIMATE for p in optimizer.parameters]
    )
    scales = np.array([p.scale for p in optimizer.parameters])
    scaled = result.jacobian[:, estimated] * scales[estimated]
    singular = np.linalg.svd(scaled, compute_uv=False)
    rank = int(np.linalg.matrix_rank(scaled))
    condition = None
    if rank == int(estimated.sum()) and singular.size:
        condition = float(singular[0] / singular[-1])
    hits = [
        p.name
        for p, value in zip(optimizer.parameters, result.parameters, strict=True)
        if p.role == ParameterRole.ESTIMATE
        and min(value - p.lower_bound, p.upper_bound - value) <= 1e-5 * p.scale
    ]
    return {
        "scaled_singular_values": singular.tolist(),
        "rank": rank,
        "estimated_parameters": int(estimated.sum()),
        "scaled_condition": condition,
        "bound_hits": hits,
    }


def rms(values: FloatArray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def recovery_metrics(case: SyntheticCase, result: OptimizerOutput) -> RecoveryMetrics:
    values = dict(zip(result.parameter_names, result.parameters, strict=True))
    metrics: RecoveryMetrics = {}
    if case.config.fit_model == OrbitModel.FULL_STATE:
        correction = np.array(
            [values.get(name, 0.0) for name in ORBIT_NAMES[OrbitModel.FULL_STATE]]
        )
        assert case.prior.nominal_state_gcrf_si is not None
        error = case.prior.nominal_state_gcrf_si + correction - case.truth_state_gcrf_si
        metrics.update(
            position_error_m=float(np.linalg.norm(error[:3])),
            velocity_error_m_s=float(np.linalg.norm(error[3:])),
        )
    if case.config.truth_model == case.config.fit_model:
        expected = shared_state_parameters(case)
        metrics["scaled_parameter_error"] = [
            float((value - expected.get(p.name, 0.0)) / p.scale)
            for p, value in zip(
                case.optimizer.parameters, result.parameters, strict=True
            )
        ]
    return metrics


def run_case(case: SyntheticCase) -> tuple[OptimizerOutput, FitMetrics]:
    training = subset(case.prior, case.train)
    fixed = tuple(
        replace(p, role=ParameterRole.FIXED) for p in case.optimizer.parameters
    )
    initial = fit(training, replace(case.optimizer, parameters=fixed))
    result = fit(training, case.optimizer)
    start = predictions(
        case.config.fit_model,
        dict(zip(initial.parameter_names, initial.parameters, strict=True)),
        case.prior,
    )
    final = predictions(
        case.config.fit_model,
        dict(zip(result.parameter_names, result.parameters, strict=True)),
        case.prior,
    )
    shared = predictions(
        case.config.fit_model, shared_state_parameters(case), case.prior
    )
    observed = case.clean_hz + case.noise_hz
    metrics: FitMetrics = {
        "success": bool(result.success),
        "status": result.status,
        "message": result.message,
        "function_evaluations": result.function_evaluations,
        "initial_cost": initial.cost,
        "final_cost": result.cost,
        "optimality": result.optimality,
        "parameter_names": list(result.parameter_names),
        "parameters": result.parameters.tolist(),
        "initial_training_rms_hz": rms((start - observed)[case.train]),
        "final_training_rms_hz": rms((final - observed)[case.train]),
        "initial_validation_rms_hz": rms((start - case.clean_hz)[~case.train]),
        "final_validation_rms_hz": rms((final - case.clean_hz)[~case.train]),
        "shared_state_model_mismatch_rms_hz": rms(shared - case.clean_hz),
        **diagnostics(result, case.optimizer),
        **recovery_metrics(case, result),
    }
    return result, metrics


def sweep_variants(base: CaseConfig) -> list[CaseConfig]:
    """One factor at a time; avoid a combinatorial Cartesian sweep."""
    variants = [base]
    variants.extend(replace(base, sigma_hz=sigma) for sigma in (1.0, 5.0))
    variants.extend(replace(base, initial_error=scale) for scale in (0.1, 10.0))
    variants.extend((replace(base, arc_fraction=0.1), replace(base, receivers=1)))
    variants.extend(
        replace(base, parameters=group)
        for group in ("timing", "frequency_bias", "augmented")
    )
    variants.append(replace(base, parameters="pass_bias", passes=2))
    variants.extend(
        replace(base, loss=loss) for loss in ("soft_l1", "huber", "cauchy", "arctan")
    )
    variants.extend(
        replace(base, outliers=True, loss=loss)
        for loss in ("linear", "soft_l1", "huber", "cauchy", "arctan")
    )
    if base.fit_model == OrbitModel.SGP4:
        variants.append(replace(base, parameters="bstar"))
    return variants


def sweep_configs(regimes: list[str], seeds: list[int]) -> Iterator[CaseConfig]:
    for regime, truth, model, seed in product(regimes, OrbitModel, OrbitModel, seeds):
        yield from sweep_variants(CaseConfig(regime, truth, model, seed=seed))


def experiment_record(config: CaseConfig) -> ExperimentRecord:
    try:
        _, metrics = run_case(make_case(config))
        record: ExperimentRecord = {"configuration": asdict(config), "metrics": metrics}
        # Refuse NaN/Infinity rather than producing non-standard JSON or hiding divergence.
        json.dumps(record, allow_nan=False)
        return record
    except Exception as exc:
        return {
            "configuration": asdict(config),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--regime", nargs="+", choices=tuple(REGIMES), default=list(REGIMES)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument(
        "--limit", type=int, help="Run only the first N cases for a smoke check"
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    records = []
    for index, config in enumerate(sweep_configs(args.regime, args.seeds)):
        if args.limit is not None and index >= args.limit:
            break
        record = experiment_record(config)
        records.append(record)
        # Checkpoint each completed case; an interrupted long sweep retains its results.
        args.output.write_text(
            json.dumps(
                {"schema_version": 1, "cases": records}, indent=2, allow_nan=False
            )
            + "\n"
        )
        print(
            f"{index + 1}: {config.regime} {config.truth_model}->{config.fit_model} {config.parameters}: {record.get('error', 'recorded')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
