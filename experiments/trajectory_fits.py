"""Cacheable Doppler-only fitting shared by archive comparisons and tuning."""

import hashlib
import json
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path

import numpy as np
import polars as pl
import satkit as sk

from dart.io import ContactMetadata
from dart.io.doppler import prepare_doppler, select_quality_doppler
from dart.io.orbit import save_orbit
from dart.od import OrbitModel, PriorStateData, fit, resolve_prior, resolve_solution
from dart.od.initialization import initialize_sgp4_phase
from dart.od.profiles import ORBIT_SCALES, Sgp4ParameterSet, sgp4_bias_profile
from dart.od.schema import OptimizerOutput, ParameterRole, PriorSource
from dart.od.selection import PassInformation, assess_pass_information
from experiments.archived_data import ArchivedExperiment, sha256
from experiments.forest_passes import screening_reason
from experiments.live_data_report import fit_diagnostics, save_json


@dataclass(frozen=True)
class StudySettings:
    loss_scale_hz: float = 200.0
    min_ebn0_db: float = 3.0
    retained_fraction: float = 0.5
    max_evaluations: int = 1000

    def __post_init__(self) -> None:
        if not isfinite(self.loss_scale_hz) or self.loss_scale_hz <= 0:
            raise ValueError("loss scale must be positive finite")
        if not isfinite(self.min_ebn0_db) or not 0 < self.retained_fraction <= 1:
            raise ValueError("invalid quality or selection setting")
        if self.max_evaluations < 1:
            raise ValueError("max evaluations must be positive")


def screened_contacts(
    archive: ArchivedExperiment, settings: StudySettings
) -> tuple[dict[str, pl.DataFrame], dict[str, str]]:
    selected, reasons = {}, {}
    for contact in archive.contacts:
        raw = archive.measurements.filter(pl.col("contact_id") == contact.contact_id)
        selected[contact.contact_id] = select_quality_doppler(
            raw, min_ebn0_db=settings.min_ebn0_db
        )
        reasons[contact.contact_id] = screening_reason(
            selected[contact.contact_id], "robust", 20
        )
    return selected, reasons


def prior_data(
    archive: ArchivedExperiment, contacts: list[ContactMetadata], frame: pl.DataFrame
) -> PriorStateData:
    context, _ = prepare_doppler(
        contacts,
        frame,
        center_frequency_hz=archive.settings.center_frequency_hz,
        variance_hz2=1.0,
        min_samples=20,
    )
    epoch = min(o.time for o in context.observations) - sk.duration(seconds=1)
    return PriorStateData(context, archive.prior, epoch)


def read_output(path: Path) -> OptimizerOutput:
    values = json.loads(path.read_text())
    values["model_kind"] = OrbitModel(values["model_kind"])
    values["prior_source"] = PriorSource(values["prior_source"])
    values["parameter_roles"] = tuple(
        ParameterRole(v) for v in values["parameter_roles"]
    )
    values["parameter_names"] = tuple(values["parameter_names"])
    for key in ("parameters", "residuals", "jacobian"):
        values[key] = np.array(values[key], dtype=float)
    return OptimizerOutput(**values)


def fit_group(
    cache: Path,
    archive: ArchivedExperiment,
    contacts: list[ContactMetadata],
    frame: pl.DataFrame,
    parameter_set: Sgp4ParameterSet,
    settings: StudySettings,
    runtime_key: str,
) -> Path:
    profile = replace(
        sgp4_bias_profile(
            parameter_set,
            [c.contact_id for c in contacts],
            robust=True,
            max_evaluations=settings.max_evaluations,
        ),
        loss_scale=settings.loss_scale_hz,
    )
    data = prior_data(archive, contacts, frame)
    # Include both normalized measurement values and source metadata, never GPS.
    identity = repr(
        (
            runtime_key,
            archive.prior,
            contacts,
            profile,
            archive.settings.center_frequency_hz,
        )
    )
    key = hashlib.sha256(
        identity.encode() + frame.hash_rows(seed=42).to_numpy().tobytes()
    ).hexdigest()
    directory = cache / key
    if (directory / "status.json").exists():
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    save_json(directory / "profile.json", profile)
    save_json(directory / "contacts.json", contacts)
    save_json(directory / "prior.json", archive.prior)
    save_json(directory / "observations.json", data.observations.observations)
    frame.write_parquet(directory / "measurements.parquet")
    save_orbit(directory / "prior-orbit.json", resolve_prior(data, OrbitModel.SGP4))
    try:
        seeded, scan = initialize_sgp4_phase(data, profile)
        save_json(directory / "seeded-profile.json", seeded)
        np.save(directory / "phase-scan.npy", scan)
        output = fit(data, seeded)
        save_json(directory / "output.json", output)
        save_json(directory / "diagnostics.json", fit_diagnostics(output, seeded))
        status = {
            "status": "converged" if output.success else "nonconverged",
            "reason": output.message,
        }
        if output.success:
            save_orbit(directory / "orbit.json", resolve_solution(data, output))
        if output.success and parameter_set == "L":
            _save_information(directory, data, output, seeded)
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        status = {"status": "fit_error", "reason": f"{type(exc).__name__}: {exc}"}
    save_json(directory / "status.json", status)
    return directory


def _save_information(directory, data, output, profile):
    try:
        metrics = assess_pass_information(
            data, output, profile, ORBIT_SCALES[OrbitModel.SGP4]
        )
        save_json(directory / "information.json", metrics)
    except (ValueError, RuntimeError) as exc:
        save_json(directory / "information-failure.json", {"reason": str(exc)})


def load_information(directory: Path) -> PassInformation | None:
    path = directory / "information.json"
    if not path.exists():
        return None
    values = json.loads(path.read_text())
    values["singular_values"] = tuple(values["singular_values"])
    return PassInformation(**values)


def fitting_runtime_key() -> str:
    from dart.forward_models import _native

    if _native.__file__ is None:
        raise RuntimeError("numerical extension has no provenance file")
    paths = [Path(_native.__file__), Path(__file__), Path("dart/io/doppler.py")]
    paths += sorted(Path("dart/od").glob("*.py"))
    return hashlib.sha256("".join(sha256(p) for p in paths).encode()).hexdigest()
