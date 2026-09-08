"""Local artifacts for live-data experiments; no provider or solver access."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import satkit as sk

from dart.evaluation import OrbitError
from dart.io import ContactMetadata, EphemerisMetadata
from dart.io.doppler import ContactSelection
from dart.io.oem import OemEphemeris, OemMetadata, write_oem
from dart.od import OptimizerContext, OptimizerOutput, ParameterRole
from experiments.references import ReferenceMetadata

if TYPE_CHECKING:
    from experiments.live_data import ExperimentResult, ExperimentSettings, WindowScore


def _json_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, sk.time):
        return value.as_unixtime()
    if isinstance(value, (datetime, Path)):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def save_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, default=_json_value, indent=2, allow_nan=False) + "\n"
    )


def fit_diagnostics(
    output: OptimizerOutput, optimizer: OptimizerContext
) -> dict[str, object]:
    scales = {p.name: p.scale for p in optimizer.parameters}
    estimated = [
        i
        for i, role in enumerate(output.parameter_roles)
        if role == ParameterRole.ESTIMATE
    ]
    matrix = output.jacobian[:, estimated] * [
        scales[output.parameter_names[i]] for i in estimated
    ]
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = max(matrix.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    condition = float(singular[0] / singular[-1]) if rank == len(estimated) else None
    values = dict(zip(output.parameter_names, output.parameters, strict=True))
    hits = [
        p.name
        for p in optimizer.parameters
        if min(abs(values[p.name] - p.lower_bound), abs(values[p.name] - p.upper_bound))
        <= p.scale * 1e-6
    ]
    return {
        "estimated_parameters": len(estimated),
        "rank": rank,
        "scaled_singular_values": singular.tolist(),
        "scaled_condition": condition,
        "bound_hits": hits,
    }


def save_inventory(
    directory: Path,
    contacts: list[ContactMetadata],
    frame: pl.DataFrame,
    prior: EphemerisMetadata,
    counts: tuple[ContactSelection, ...],
    settings: ExperimentSettings,
    reference: OemEphemeris,
    reference_metadata: ReferenceMetadata | None = None,
) -> None:
    save_json(directory / "initial-ephemeris.json", prior)
    save_json(directory / "contacts.json", contacts)
    frame.write_parquet(directory / "raw-measurements.parquet")
    reference_name = "reference-source" + "".join(reference.path.suffixes)
    (directory / reference_name).write_bytes(reference.raw)
    if reference_metadata is not None:
        quality_raw = reference_metadata.quality_report.read_bytes()
        if (
            hashlib.sha256(quality_raw).hexdigest()
            != reference_metadata.quality_report_sha256
        ):
            raise ValueError("reference quality report changed after loading")
        (directory / "reference-quality.json").write_bytes(quality_raw)
    native_file = import_module("dart._forward_models").__file__
    if native_file is None:
        raise RuntimeError("numerical extension has no file for provenance hashing")
    native_path = Path(native_file)
    manifest = {
        "initial_ephemeris_id": prior.ephemeris_id,
        "initial_ephemeris_sha256": hashlib.sha256(
            (directory / "initial-ephemeris.json").read_bytes()
        ).hexdigest(),
        "reference_source": str(reference.path),
        "reference_artifact": reference_name,
        "reference_sha256": reference.sha256,
        "reference_metadata": reference_metadata,
        "reference_quality_artifact": (
            "reference-quality.json" if reference_metadata is not None else None
        ),
        "numerical_core_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
        "settings": settings,
        "selection": counts,
        "excluded_contacts": [
            c.contact_id for c in counts if c.retained_samples < settings.min_samples
        ],
        "packages": {
            name: version(name) for name in ("dart", "satkit", "oem", "numpy", "scipy")
        },
        "propagation": "Rust satkit; SGP4 WGS72/improved or full-state PropSettings::default()",
        "epochs": "UTC Unix seconds",
        "frame": "GCRF",
        "state_units": ["m", "m/s"],
    }
    save_json(directory / "manifest.json", manifest)


def _metrics(error: OrbitError | None) -> dict[str, object]:
    if error is None:
        return {}
    return {
        field.name: getattr(error, field.name)
        for field in fields(error)
        if field.name != "differences"
    }


def _save_window(directory: Path, score: WindowScore, metadata: OemMetadata) -> None:
    if score.error is None:
        return
    write_oem(
        score.predicted, directory / f"{score.window.name}.oem", metadata=metadata
    )
    write_oem(
        score.baseline, directory / f"{score.window.name}-prior.oem", metadata=metadata
    )
    np.savez_compressed(
        directory / f"{score.window.name}-states.npz",
        epochs_unix=[t.as_unixtime() for h in score.reference for t in h.epochs],
        segment_lengths=[len(h.epochs) for h in score.reference],
        reference_gcrf_si=np.concatenate([h.states for h in score.reference]),
        fitted_gcrf_si=np.concatenate([h.states for h in score.predicted]),
        prior_gcrf_si=np.concatenate([h.states for h in score.baseline]),
        difference_gcrf_si=score.error.differences,
    )


def save_result(directory: Path, result: ExperimentResult) -> None:
    directory.mkdir()
    observations = result.prior.observations
    save_json(
        directory / "input.json",
        {
            "initial_ephemeris_id": result.prior.ephemeris.ephemeris_id,
            "epoch_unix": result.prior.epoch,
            "center_frequency_hz": observations.center_frequency_hz,
            "contact_ids": result.contact_ids,
            "contact_to_pass_idx": observations.contact_to_pass_idx,
            "system_to_receiver_idx": observations.system_to_receiver_idx,
            "receivers": [
                (r.latitude_deg, r.longitude_deg, r.altitude)
                for r in observations.receivers
            ],
            "observations": observations.observations,
        },
    )
    save_json(
        directory / "fit.json",
        {
            "settings": result.settings,
            "optimizer": result.optimizer,
            "output": result.output,
            "diagnostics": fit_diagnostics(result.output, result.optimizer),
            "doppler_rms_hz": float(
                np.sqrt(
                    np.mean(result.output.residuals**2) * result.settings.variance_hz2
                )
            ),
            "selection": result.selection,
            "reference_metadata": result.reference_metadata,
            "scores": [
                {
                    "window": s.window,
                    "error": _metrics(s.error),
                    "baseline_error": _metrics(s.baseline_error),
                    "unavailable_reason": s.unavailable_reason,
                }
                for s in result.scores
            ],
        },
    )
    contact = next(iter(observations.contacts.values()))
    metadata = OemMetadata(
        contact.spacecraft, contact.cospar, "DART", datetime.now(UTC)
    )
    for score in result.scores:
        _save_window(directory, score, metadata)


def summary_rows(results: list[ExperimentResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        base = {
            "model": result.output.model_kind,
            "contact_count": len(result.contact_ids),
            "contact_ids": " ".join(result.contact_ids),
            "success": result.output.success,
            "message": result.output.message,
            **reference_summary(result.reference_metadata),
        }
        if not result.scores:
            rows.append(
                {
                    **base,
                    "window": "unavailable",
                    "reason": "optimizer did not converge",
                }
            )
        for score in result.scores:
            rows.append(
                {
                    **base,
                    "window": score.window.name,
                    "reason": score.unavailable_reason,
                    **_metrics(score.error),
                    **{
                        f"prior_{k}": v
                        for k, v in _metrics(score.baseline_error).items()
                    },
                }
            )
    return rows


def reference_summary(metadata: ReferenceMetadata | None) -> dict[str, object]:
    """Keep assessment and limitations visible even for nonconverged fits."""
    if metadata is None:
        return {"reference_status": "unverified"}
    return {
        f"reference_{field.name}": getattr(metadata, field.name)
        for field in fields(metadata)
    }


def save_summary(directory: Path, results: list[ExperimentResult]) -> None:
    rows = summary_rows(results)
    save_json(directory / "summary.json", rows)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
