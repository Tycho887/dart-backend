"""Data-only replay artifacts and result projections in explicit native units."""

import json
import math
import os
from dataclasses import fields, is_dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import satkit as sk
from pydantic import TypeAdapter

from dart.io import ContactMetadata, EphemerisMetadata, ForwardModelContext
from dart.od import (
    OptimizerContext,
    OptimizerOutput,
    PriorStateData,
)

from .models import ResolvedEstimateConfiguration


def _json_default(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: getattr(value, f.name) for f in fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, sk.time):
        return value.as_unixtime()
    raise TypeError(f"unsupported artifact type {type(value).__name__}")


def json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def document(value: object) -> dict:
    result = json.loads(json_bytes(value))
    if not isinstance(result, dict):
        raise TypeError("expected an object document")
    return result


def software_versions() -> dict[str, str]:
    versions = {name: version(name) for name in ("dart", "satkit", "numpy", "scipy")}
    build_file = os.getenv("DART_BUILD_ID_FILE")
    if build_file:
        versions["build_sha256"] = Path(build_file).read_text().split()[0]
    return versions


def prior_document(prior: PriorStateData) -> dict:
    context = prior.observations
    return document(
        {
            "format_version": 1,
            "epoch_unix_s": prior.epoch,
            "ephemeris": prior.ephemeris,
            "nominal_state_gcrf_si": prior.nominal_state_gcrf_si,
            "derived_tle_lines": prior.derived_tle_lines,
            "center_frequency_hz": context.center_frequency_hz,
            "contacts": list(context.contacts.values()),
            "observations": context.observations,
        }
    )


def _ephemeris(data: dict) -> EphemerisMetadata:
    data = data.copy()
    for name in ("epoch", "last_usable_at", "submitted_at"):
        if data[name] is not None:
            data[name] = datetime.fromisoformat(data[name])
    return EphemerisMetadata(**data)


def restore_prior(data: dict) -> PriorStateData:
    if data["format_version"] != 1:
        raise ValueError("unsupported prior artifact version")
    context = ForwardModelContext(data["center_frequency_hz"])
    for source in data["contacts"]:
        item = {**source, "ephemeris": _ephemeris(source["ephemeris"])}
        item.update(
            start=datetime.fromisoformat(item["start"]),
            stop=datetime.fromisoformat(item["stop"]),
            ecef=tuple(item["ecef"]),
        )
        context.register_contact(ContactMetadata(**item))
    for item in data["observations"]:
        contact = context.contacts[item["contact_id"]]
        context.add_observation(
            item["time"],
            item["observed"][0],
            item["noise_cov"][0][0],
            contact.system_id,
            contact.contact_id,
        )
    nominal = data["nominal_state_gcrf_si"]
    lines = data["derived_tle_lines"]
    return PriorStateData(
        context,
        _ephemeris(data["ephemeris"]),
        sk.time.from_unixtime(data["epoch_unix_s"]),
        None if nominal is None else np.array(nominal),
        None if lines is None else tuple(lines),
    )


def restore_optimizer(data: dict) -> OptimizerContext:
    return TypeAdapter(OptimizerContext).validate_python(data)


def _validate_output(
    optimizer: OptimizerContext, output: OptimizerOutput, count: int
) -> None:
    names = tuple(p.name for p in optimizer.parameters)
    if names != output.parameter_names or output.parameters.shape != (len(names),):
        raise ValueError("optimizer parameter order differs from resolved profile")
    if tuple(p.role for p in optimizer.parameters) != output.parameter_roles:
        raise ValueError("optimizer parameter roles differ from resolved profile")
    if output.residuals.shape != (count,):
        raise ValueError("residual count differs from stored observations")
    _validate_covariance(output)


def _validate_covariance(output: OptimizerOutput) -> None:
    covariance = output.covariance
    if covariance is None:
        return
    if (
        covariance.shape != (len(output.parameter_names),) * 2
        or not np.isfinite(covariance).all()
    ):
        raise ValueError("invalid parameter covariance")
    if np.any(np.diag(covariance) < 0):
        raise ValueError("negative parameter variance")


def _parameter_rows(
    configuration: ResolvedEstimateConfiguration,
    optimizer: OptimizerContext,
    output: OptimizerOutput,
) -> list[dict]:
    covariance = output.covariance
    definitions = {p.name: p for p in configuration.parameters}
    parameters = []
    for index, (spec, value) in enumerate(
        zip(optimizer.parameters, output.parameters, strict=True)
    ):
        variance = None if covariance is None else float(covariance[index, index])
        parameters.append(
            {
                "parameter_name": spec.name,
                "ordinal": index,
                "role": spec.role.value,
                "value": float(value),
                "unit": definitions[spec.name].unit,
                "initial_value": spec.initial,
                "lower_bound": spec.lower_bound,
                "upper_bound": spec.upper_bound,
                "scale": spec.scale,
                "standard_uncertainty": None
                if variance is None
                else math.sqrt(variance),
                "contact_id": spec.name.split(":", 1)[1]
                if spec.name.startswith("pass_bias_hz:")
                else None,
            }
        )
    return parameters


def _diagnostics(prior: PriorStateData, output: OptimizerOutput) -> dict:
    covariance = output.covariance
    sigmas = np.sqrt([o.noise_cov[0][0] for o in prior.observations.observations])
    warnings = [] if output.success else ["Optimizer did not converge."]
    if output.covariance_method and output.loss != "linear":
        warnings.append(
            "Classical consider covariance uses the unweighted final Jacobian; "
            "it is not a robust sandwich covariance."
        )
    diagnostics = {
        "success": output.success,
        "optimizer_status": output.status,
        "message": output.message,
        "objective": output.cost,
        "optimality": output.optimality,
        "function_evaluations": output.function_evaluations,
        "jacobian_evaluations": output.jacobian_evaluations,
        "observation_count": len(prior.observations.observations),
        "whitened_residual_rms": float(np.sqrt(np.mean(output.residuals**2))),
        "residual_rms_hz": float(np.sqrt(np.mean((output.residuals * sigmas) ** 2))),
        "covariance": None if covariance is None else covariance.tolist(),
        "covariance_rank": output.covariance_rank,
        "covariance_method": output.covariance_method,
        "parameter_order": output.parameter_names,
        "warnings": warnings,
    }
    return diagnostics


def result_rows(
    prior: PriorStateData,
    configuration: ResolvedEstimateConfiguration,
    optimizer: OptimizerContext,
    output: OptimizerOutput,
) -> tuple[list[dict], dict]:
    _validate_output(optimizer, output, len(prior.observations.observations))
    expected_covariance = output.success and configuration.forward_model.version >= 2
    if (output.covariance is not None) != expected_covariance:
        raise ValueError(
            "parameter covariance availability differs from fit profile/status"
        )
    expected_method = "classical_consider_v1" if expected_covariance else None
    if output.covariance_method != expected_method:
        raise ValueError("parameter covariance method differs from fit profile/status")
    parameters = _parameter_rows(configuration, optimizer, output)
    diagnostics = _diagnostics(prior, output)
    json_bytes([parameters, diagnostics])
    return parameters, diagnostics
