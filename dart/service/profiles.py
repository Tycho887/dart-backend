"""Built-in profile definitions and deterministic override resolution."""

from __future__ import annotations

from copy import deepcopy

from dart.schema import Sgp4FitOptions
from dart.time_solver import TimeSolverConfig

from .models import MeanElementsSolver, SolveJobRequest

MEAN_ELEMENTS_PROFILE = "sgp4-production"
TIME_SHIFT_PROFILE = "time-shift-production"
PROFILE_VERSION = 1


def profile_documents() -> list[dict]:
    fit = Sgp4FitOptions()
    time = TimeSolverConfig()
    common = {
        "loss": fit.loss,
        "loss_scale": fit.loss_scale,
        "doppler_sigma_hz": fit.doppler_sigma_hz,
        "max_evaluations": fit.max_evaluations,
    }
    return [
        {
            "name": MEAN_ELEMENTS_PROFILE,
            "version": PROFILE_VERSION,
            "solver_kind": "sgp4_mean_elements",
            "settings": {**common, "ftol_rel": fit.ftol_rel, "xtol_rel": fit.xtol_rel},
            "physical_settings": {
                "mean_anomaly": fit.mean_anomaly.__dict__,
                "mean_motion": fit.mean_motion.__dict__,
                "center_frequency": fit.center_frequency.__dict__,
                "pass_bias": {
                    "initial": 0.0,
                    "lower": -15_000.0,
                    "upper": 15_000.0,
                    "scale": 2_000.0,
                    "finite_difference_step": 1e-2,
                },
            },
        },
        {
            "name": TIME_SHIFT_PROFILE,
            "version": PROFILE_VERSION,
            "solver_kind": "sgp4_time_shift",
            "settings": {
                **common,
                "use_qmc": time.use_qmc,
                "qmc_samples": time.qmc_samples,
            },
            "physical_settings": {
                "time_shift_bounds_s": list(time.time_shift_bounds_s),
                "pointing_penalty_weight": time.pointing_penalty_weight,
                "pointing_n_degrees": time.pointing_n_degrees,
                "reg_weights": list(time.reg_weights),
                "center_frequency": fit.center_frequency.__dict__,
                "pass_bias": {
                    "initial": 0.0,
                    "lower": -15_000.0,
                    "upper": 15_000.0,
                    "scale": 2_000.0,
                    "finite_difference_step": 1e-2,
                },
            },
        },
    ]


def resolve_profile(request: SolveJobRequest) -> tuple[dict, dict]:
    solver = request.solver
    expected = (
        MEAN_ELEMENTS_PROFILE
        if isinstance(solver, MeanElementsSolver)
        else TIME_SHIFT_PROFILE
    )
    if (
        solver.optimizer.profile != expected
        or solver.optimizer.version != PROFILE_VERSION
    ):
        raise KeyError(
            f"unknown profile {solver.optimizer.profile!r} version "
            f"{solver.optimizer.version} for {solver.kind}"
        )
    profile = next(
        p
        for p in profile_documents()
        if p["name"] == expected and p["version"] == PROFILE_VERSION
    )
    effective = deepcopy(profile["settings"])
    effective.update(
        solver.optimizer.overrides.model_dump(exclude_none=True, mode="json")
    )
    return profile, effective


def capabilities_document() -> dict:
    return {
        "api_version": "v1",
        "strategies": [
            {"name": "single_contact", "available": True, "contact_count": 1},
            {
                "name": "joint",
                "available": False,
                "minimum_contacts": 2,
                "requires_ephemeris_id": True,
            },
        ],
        "solvers": [
            {
                "kind": "sgp4_mean_elements",
                "available": True,
                "parameterizations": [
                    "mean_anomaly",
                    "mean_anomaly_mean_motion",
                    "mean_anomaly_mean_motion_frequency",
                ],
                "profile": {"name": MEAN_ELEMENTS_PROFILE, "version": PROFILE_VERSION},
                "overrides": {
                    "loss": ["linear", "huber", "soft_l1", "log_cosh"],
                    "loss_scale": {"minimum": 0.01, "maximum": 100.0},
                    "doppler_sigma_hz": {
                        "minimum": 1.0,
                        "maximum": 1_000_000.0,
                        "unit": "Hz",
                    },
                    "max_evaluations": {"minimum": 10, "maximum": 10_000},
                    "ftol_rel": {"minimum": 1e-12, "maximum": 1e-2},
                    "xtol_rel": {"minimum": 1e-12, "maximum": 1e-2},
                },
            },
            {
                "kind": "sgp4_time_shift",
                "available": True,
                "parameterizations": [
                    "time_shift",
                    "time_shift_bias",
                    "time_shift_bias_frequency",
                ],
                "profile": {"name": TIME_SHIFT_PROFILE, "version": PROFILE_VERSION},
                "overrides": {
                    "loss": ["linear", "huber", "soft_l1"],
                    "loss_scale": {"minimum": 0.01, "maximum": 100.0},
                    "doppler_sigma_hz": {
                        "minimum": 1.0,
                        "maximum": 1_000_000.0,
                        "unit": "Hz",
                    },
                    "max_evaluations": {"minimum": 10, "maximum": 10_000},
                    "use_qmc": {"type": "boolean"},
                    "qmc_samples": {"minimum": 1, "maximum": 1024},
                },
                "limitations": [
                    "log_cosh is not exposed because scipy approximates it as soft_l1"
                ],
            },
            {"kind": "rk89", "available": False, "parameterizations": []},
        ],
        "frequency": {
            "unit": "Hz",
            "minimum": 1_000_000.0,
            "maximum": 100_000_000_000.0,
            "precedence": ["request", "control_config_v2"],
            "control_config_link": "s_band_downlink_p1_1",
        },
        "telemetry_filter": {
            "doppler_semantics": "inclusive_absolute_magnitude",
            "min_elevation_deg": {"minimum": -90.0, "maximum": 90.0, "unit": "deg"},
            "min_doppler_hz": {
                "minimum": 0.0,
                "maximum": 1_000_000_000.0,
                "unit": "Hz",
            },
            "max_doppler_hz": {
                "minimum_exclusive": 0.0,
                "maximum": 1_000_000_000.0,
                "unit": "Hz",
            },
        },
    }
