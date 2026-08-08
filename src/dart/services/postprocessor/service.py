"""HTTP postprocessor service for scoring and selection metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from math import log, pi
from xml.etree import ElementTree

import numpy as np
import satkit as sk
from fastapi import Depends, FastAPI
from pydantic import ValidationError

from ... import contract_projection as wire
from ...contracts import (
    EvidenceClass,
    InformationCriterion,
    MeanElementsTwoParameterParameters,
    MetricGroup,
    ObservableChannel,
    OemDocument,
    OemEncoding,
    QualityRequest,
    QualityResult,
    SelectionScore,
)
from ...frames import state_to_gcrf, tle_state_gcrf
from ..http import (
    domain_problem,
    install_problem_handlers,
    problem_responses,
    require_internal_bearer,
)
from ..openapi import install_contract_openapi_rules


@dataclass(frozen=True, slots=True)
class OemState:
    epoch: sk.time
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    reference_frame: str


def _epoch(value: str, time_system: str):
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if time_system.upper() != "UTC":
        raise ValueError(f"unsupported OEM TIME_SYSTEM {time_system!r}; DART v0 requires UTC")
    return sk.time.from_datetime(parsed)


def _validate_frame(value: str) -> str:
    frame = value.strip().upper()
    if frame in {"GCRF", "EME2000"} or frame.startswith("ITRF"):
        return frame
    raise ValueError(f"unsupported Earth-centered OEM REF_FRAME {value!r}")


def _kvn(content: str) -> list[OemState]:
    metadata: dict[str, str] = {}
    result = []
    in_covariance = False
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("COMMENT"):
            continue
        if line == "COVARIANCE_START":
            in_covariance = True
            continue
        if line == "COVARIANCE_STOP":
            in_covariance = False
            continue
        if in_covariance:
            continue
        if "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            metadata[key] = value
            continue
        fields = line.split()
        if len(fields) != 7 or not fields[0][:4].isdigit():
            continue
        values = np.asarray([float(value) for value in fields[1:]], dtype=float)
        result.append(
            OemState(
                _epoch(fields[0], metadata.get("TIME_SYSTEM", "")),
                values[:3] * 1_000.0,
                values[3:] * 1_000.0,
                _validate_frame(metadata.get("REF_FRAME", "")),
            )
        )
    return result


def _local_name(element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _child_text(parent, name: str) -> str:
    for child in parent.iter():
        if _local_name(child) == name and child.text:
            return child.text.strip()
    raise ValueError(f"OEM XML is missing {name}")


def _xml(content: str) -> list[OemState]:
    root = ElementTree.fromstring(content)
    result = []
    for segment in (item for item in root.iter() if _local_name(item) == "segment"):
        metadata = next((item for item in segment if _local_name(item) == "metadata"), None)
        data = next((item for item in segment if _local_name(item) == "data"), None)
        if metadata is None or data is None:
            raise ValueError("each OEM XML segment requires metadata and data")
        frame = _validate_frame(_child_text(metadata, "REF_FRAME"))
        time_system = _child_text(metadata, "TIME_SYSTEM")
        for vector in (item for item in data.iter() if _local_name(item) == "stateVector"):
            values = [
                float(_child_text(vector, name))
                for name in ("X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT")
            ]
            result.append(
                OemState(
                    _epoch(_child_text(vector, "EPOCH"), time_system),
                    np.asarray(values[:3]) * 1_000.0,
                    np.asarray(values[3:]) * 1_000.0,
                    frame,
                )
            )
    return result


def parse_oem(document: OemDocument) -> list[OemState]:
    if sha256(document.content.encode()).hexdigest() != document.sha256:
        raise ValueError("OEM SHA-256 does not match content")
    states = (
        _kvn(document.content) if document.encoding is OemEncoding.KVN else _xml(document.content)
    )
    if not states:
        raise ValueError("OEM contains no state vectors")
    return states


def state_gcrf(state: OemState) -> tuple[np.ndarray, np.ndarray]:
    return state_to_gcrf(
        state.reference_frame,
        state.epoch,
        state.position_m,
        state.velocity_m_s,
    )


def _residual_metrics(values: np.ndarray) -> dict:
    if not len(values):
        return {"count": 0, "rmse": None, "mean": None, "acf_lag1": None, "is_white_noise": None}
    mean = float(np.mean(values))
    rmse = float(np.sqrt(np.mean(values**2)))
    if len(values) < 2:
        return {
            "count": len(values),
            "rmse": rmse,
            "mean": mean,
            "acf_lag1": 0.0,
            "is_white_noise": False,
        }
    centered = values - mean
    denominator = float(centered @ centered)
    acf = 0.0 if denominator == 0.0 else float(centered[1:] @ centered[:-1] / denominator)
    threshold = float(1.96 / np.sqrt(len(values)))
    return {
        "count": len(values),
        "rmse": rmse,
        "mean": mean,
        "acf_lag1": acf,
        "white_noise_threshold": threshold,
        "is_white_noise": abs(acf) < threshold,
    }


def assess_quality(request: QualityRequest) -> QualityResult:
    result = request.result
    residuals = np.asarray(_doppler_residuals(request), dtype=float)
    convergence = {
        "success": result.diagnostics.success,
        "healthy": result.diagnostics.healthy,
        "observations_used": result.diagnostics.observations_used,
        "jacobian_rank": result.diagnostics.jacobian_rank,
        "jacobian_condition": result.diagnostics.jacobian_condition,
        "at_bound": result.diagnostics.at_bound,
    }
    truth = None
    evidence = EvidenceClass.RESIDUAL_ONLY
    if request.reference_oem is not None:
        states = parse_oem(request.reference_oem)
        tle_data = result.corrected_tle or result.reference_tle
        tle = sk.TLE.from_lines([tle_data.name, tle_data.line1, tle_data.line2])
        offset_s = _time_offset_s(result.parameters)
        errors = []
        for state in states:
            predicted_position, _ = tle_state_gcrf(tle, state.epoch, offset_s)
            reference_position, _ = state_gcrf(state)
            errors.append(float(np.linalg.norm(predicted_position - reference_position) / 1_000.0))
        values = np.asarray(errors)
        truth = MetricGroup(
            metrics={
                "fixes": len(values),
                "best_position_error_km": float(np.min(values)),
                "median_position_error_km": float(np.median(values)),
                "p90_position_error_km": float(np.percentile(values, 90.0)),
                "worst_position_error_km": float(np.max(values)),
                "rms_position_error_km": float(np.sqrt(np.mean(values**2))),
            }
        )
        evidence = EvidenceClass.REFERENCE_EPHEMERIS
    return QualityResult(
        evidence_class=evidence,
        convergence=MetricGroup(metrics=convergence),
        residuals=MetricGroup(metrics=_residual_metrics(residuals)),
        selection=_selection_score(result, residuals, request.selection_criterion),
        truth=truth,
    )


def _doppler_residuals(request: QualityRequest) -> list[float]:
    values = []
    for record in request.result.residuals:
        for residual in record.channels:
            if (
                residual.channel is ObservableChannel.DOPPLER
                and residual.consumed
                and residual.residual is not None
            ):
                values.append(residual.residual)
    return values


def _selection_score(
    result,
    residuals: np.ndarray,
    criterion: InformationCriterion,
) -> SelectionScore:
    """Calculate a Gaussian information criterion from consumed Doppler residuals."""

    observations = len(residuals)
    parameter_count = len(result.covariance.parameter_order)
    if observations == 0:
        return SelectionScore(
            criterion=criterion,
            eligible=False,
            observations=0,
            fitted_parameter_count=parameter_count,
            residual_sum_squares_hz2=0.0,
        )
    residual_sum_squares = float(residuals @ residuals)
    variance = max(residual_sum_squares / observations, np.finfo(float).tiny)
    deviance = observations * (log(2.0 * pi) + 1.0 + log(variance))
    aic = deviance + 2.0 * parameter_count
    if criterion is InformationCriterion.AIC:
        score = aic
    elif criterion is InformationCriterion.BIC:
        score = deviance + parameter_count * log(observations)
    else:
        denominator = observations - parameter_count - 1
        if denominator <= 0:
            return SelectionScore(
                criterion=criterion,
                eligible=False,
                observations=observations,
                fitted_parameter_count=parameter_count,
                residual_sum_squares_hz2=residual_sum_squares,
            )
        score = aic + (2.0 * parameter_count * (parameter_count + 1) / denominator)
    return SelectionScore(
        criterion=criterion,
        eligible=True,
        score=score,
        observations=observations,
        fitted_parameter_count=parameter_count,
        residual_sum_squares_hz2=residual_sum_squares,
    )


def _time_offset_s(parameters) -> float:
    if isinstance(parameters, MeanElementsTwoParameterParameters):
        return 0.0
    return parameters.time_offset_s


app = FastAPI(title="DART Postprocessor", version="0.1")
install_problem_handlers(app)
install_contract_openapi_rules(app, "postprocessor-v0.1.json")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/v0/postprocess",
    response_model=wire.QualityResult,
    dependencies=[Depends(require_internal_bearer)],
    responses=problem_responses(401, 422, 503),
)
def postprocess(request: wire.QualityRequest) -> QualityResult:
    """Validate one request and calculate its metrics."""

    try:
        semantic_request = QualityRequest.model_validate(request.model_dump(mode="json"))
        return assess_quality(semantic_request)
    except (ValidationError, ValueError) as exc:
        raise domain_problem(str(exc)) from exc


def main() -> None:
    import uvicorn

    uvicorn.run("dart.services.postprocessor.service:app", host="0.0.0.0", port=8002)
