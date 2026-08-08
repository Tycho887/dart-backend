"""Python projections of the language-neutral DART v0 contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "0.1"


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonnegativeFiniteFloat = Annotated[float, Field(allow_inf_nan=False, ge=0.0)]


def _canonical_uuid(value: str) -> str:
    """Validate one externally visible identifier while retaining string storage values."""

    try:
        return str(UUID(value))
    except ValueError as exc:
        raise ValueError("must be a UUID") from exc


UuidString = Annotated[
    str,
    AfterValidator(_canonical_uuid),
    Field(json_schema_extra={"format": "uuid"}),
]


class Cartesian3(ContractModel):
    x: FiniteFloat
    y: FiniteFloat
    z: FiniteFloat

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.z]


class Measurement(ContractModel):
    """One self-contained production Doppler observation."""

    measurement_id: str = Field(min_length=1, max_length=200)
    pass_id: str = Field(min_length=1, max_length=200)
    spacecraft_id: str = Field(min_length=1, max_length=200)
    station_id: str = Field(min_length=1, max_length=200)
    time_tag: datetime
    doppler_hz: FiniteFloat
    ebn0_db: FiniteFloat | None = None
    station_position_itrf_m: Cartesian3
    valid: bool = True
    quality_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_measurement(self) -> Measurement:
        if self.time_tag.tzinfo is None or self.time_tag.utcoffset() is None:
            raise ValueError("time_tag must include a timezone")
        object.__setattr__(self, "time_tag", self.time_tag.astimezone(UTC))
        return self


class TLEData(ContractModel):
    name: str = Field(default="DART-OBJECT", min_length=1, max_length=200)
    line1: str = Field(min_length=60, max_length=100)
    line2: str = Field(min_length=60, max_length=100)


class ObservableChannel(StrEnum):
    DOPPLER = "doppler"


class OptimizerModel(StrEnum):
    """One concrete production forward model selected by an optimizer request."""

    TIME_OFFSET = "time_offset"
    TIME_OFFSET_PASS_BIAS = "time_offset_pass_bias"
    TIME_OFFSET_FREQUENCY_PASS_BIAS = "time_offset_frequency_pass_bias"
    MEAN_ELEMENTS_TWO_PARAMETER = "mean_elements_two_parameter"


class InformationCriterion(StrEnum):
    BIC = "bic"
    AIC = "aic"
    AICC = "aicc"


class _OptimizerMetaparametersBase(ContractModel):
    """Numerical controls shared by every explicit production solver."""

    doppler_standard_deviation_hz: FiniteFloat = Field(default=500.0, gt=0.0)
    robust_loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = "cauchy"
    robust_scale_hz: FiniteFloat = Field(default=500.0, gt=0.0)
    qmc_samples: int = Field(default=64, ge=0, le=4096)


class TimeOffsetMetaparameters(_OptimizerMetaparametersBase):
    """Metaparameters for a single global time-offset fit."""

    model: Literal[OptimizerModel.TIME_OFFSET]
    time_offset_bounds_s: tuple[FiniteFloat, FiniteFloat] = (-120.0, 120.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> TimeOffsetMetaparameters:
        if self.time_offset_bounds_s[0] >= self.time_offset_bounds_s[1]:
            raise ValueError("invalid time offset bounds")
        return self


class TimeOffsetPassBiasMetaparameters(_OptimizerMetaparametersBase):
    """Metaparameters for global time offset and one Doppler bias per pass."""

    model: Literal[OptimizerModel.TIME_OFFSET_PASS_BIAS]
    time_offset_bounds_s: tuple[FiniteFloat, FiniteFloat] = (-120.0, 120.0)
    pass_bias_bounds_hz: tuple[FiniteFloat, FiniteFloat] = (-100_000.0, 100_000.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> TimeOffsetPassBiasMetaparameters:
        if self.time_offset_bounds_s[0] >= self.time_offset_bounds_s[1]:
            raise ValueError("invalid time offset bounds")
        if self.pass_bias_bounds_hz[0] >= self.pass_bias_bounds_hz[1]:
            raise ValueError("invalid pass bias bounds")
        return self


class TimeOffsetFrequencyPassBiasMetaparameters(_OptimizerMetaparametersBase):
    """Metaparameters for time, shared carrier correction, and pass biases."""

    model: Literal[OptimizerModel.TIME_OFFSET_FREQUENCY_PASS_BIAS]
    time_offset_bounds_s: tuple[FiniteFloat, FiniteFloat] = (-120.0, 120.0)
    pass_bias_bounds_hz: tuple[FiniteFloat, FiniteFloat] = (-100_000.0, 100_000.0)
    center_frequency_correction_bounds_hz: tuple[FiniteFloat, FiniteFloat] = (
        -100_000.0,
        100_000.0,
    )

    @model_validator(mode="after")
    def validate_bounds(self) -> TimeOffsetFrequencyPassBiasMetaparameters:
        if self.time_offset_bounds_s[0] >= self.time_offset_bounds_s[1]:
            raise ValueError("invalid time offset bounds")
        if self.pass_bias_bounds_hz[0] >= self.pass_bias_bounds_hz[1]:
            raise ValueError("invalid pass bias bounds")
        if (
            self.center_frequency_correction_bounds_hz[0]
            >= self.center_frequency_correction_bounds_hz[1]
        ):
            raise ValueError("invalid center frequency correction bounds")
        return self


class MeanElementsTwoParameterMetaparameters(_OptimizerMetaparametersBase):
    """Metaparameters for mean anomaly, mean motion, and pass biases."""

    model: Literal[OptimizerModel.MEAN_ELEMENTS_TWO_PARAMETER]
    pass_bias_bounds_hz: tuple[FiniteFloat, FiniteFloat] = (-100_000.0, 100_000.0)
    mean_anomaly_half_width_rad: FiniteFloat = Field(default=0.5, gt=0.0)
    mean_motion_half_width_rad_min: FiniteFloat = Field(default=0.002, gt=0.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> MeanElementsTwoParameterMetaparameters:
        if self.pass_bias_bounds_hz[0] >= self.pass_bias_bounds_hz[1]:
            raise ValueError("invalid pass bias bounds")
        if self.mean_anomaly_half_width_rad > 3.141592653589793:
            raise ValueError("mean anomaly half width must not exceed pi radians")
        return self


OptimizerMetaparameters = Annotated[
    TimeOffsetMetaparameters
    | TimeOffsetPassBiasMetaparameters
    | TimeOffsetFrequencyPassBiasMetaparameters
    | MeanElementsTwoParameterMetaparameters,
    Field(discriminator="model"),
]


class OptimizerConfiguration(ContractModel):
    """Shared optimizer inputs that stay constant across a durable run."""

    reference_tle: TLEData
    spacecraft_id: str = Field(min_length=1, max_length=200)
    nominal_carrier_frequency_hz: FiniteFloat = Field(gt=0.0)


class OptimizerData(OptimizerConfiguration):
    """One stateless optimizer invocation and its explicit forward model."""

    metaparameters: OptimizerMetaparameters


class BatchRequest(ContractModel):
    schema_version: Literal["0.1"] = SCHEMA_VERSION
    measurements: list[Measurement] = Field(min_length=1)
    optimizer_data: OptimizerData

    @model_validator(mode="after")
    def validate_measurements(self) -> BatchRequest:
        measurement_ids = [item.measurement_id for item in self.measurements]
        if len(set(measurement_ids)) != len(measurement_ids):
            raise ValueError("measurement IDs must be unique within a batch request")
        spacecraft_ids = {item.spacecraft_id for item in self.measurements}
        if spacecraft_ids != {self.optimizer_data.spacecraft_id}:
            raise ValueError("measurements must match the optimizer reference spacecraft")
        station_positions: dict[str, Cartesian3] = {}
        for item in self.measurements:
            previous = station_positions.setdefault(item.station_id, item.station_position_itrf_m)
            if previous != item.station_position_itrf_m:
                raise ValueError("a station_id must have one fixed ITRF position per request")
        valid_doppler = [item for item in self.measurements if item.valid]
        if not valid_doppler:
            model = self.optimizer_data.metaparameters.model.value
            raise ValueError(f"{model} requires at least one valid Doppler observation")
        if self.optimizer_data.metaparameters.model is OptimizerModel.MEAN_ELEMENTS_TWO_PARAMETER:
            if len({item.station_id for item in valid_doppler}) != 1:
                raise ValueError("mean_elements_two_parameter requires one station")
            if len({item.pass_id for item in valid_doppler}) < 2:
                raise ValueError(
                    "mean_elements_two_parameter requires at least two distinct passes"
                )
        return self


class Covariance(ContractModel):
    parameter_order: list[str] = Field(min_length=1)
    matrix: list[list[FiniteFloat]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_shape(self) -> Covariance:
        size = len(self.parameter_order)
        if len(self.matrix) != size or any(len(row) != size for row in self.matrix):
            raise ValueError("covariance shape must match parameter_order")
        if len(set(self.parameter_order)) != size:
            raise ValueError("covariance parameter_order must be unique")
        return self


class ObservableResidual(ContractModel):
    channel: ObservableChannel
    consumed: bool
    predicted: FiniteFloat | None = None
    residual: FiniteFloat | None = None
    robust_weight: NonnegativeFiniteFloat | None = None

    @model_validator(mode="after")
    def validate_values(self) -> ObservableResidual:
        if self.consumed and (self.predicted is None or self.residual is None):
            raise ValueError("a consumed observable requires predicted and residual values")
        if not self.consumed and any(
            value is not None for value in (self.predicted, self.residual, self.robust_weight)
        ):
            raise ValueError("an unconsumed observable cannot include a prediction or residual")
        return self


class ResidualRecord(ContractModel):
    measurement_id: str = Field(min_length=1, max_length=200)
    channels: list[ObservableResidual] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_channels(self) -> ResidualRecord:
        channels = [item.channel for item in self.channels]
        if len(set(channels)) != len(channels):
            raise ValueError("a measurement may contain one residual per observable channel")
        return self


class PassDopplerBias(ContractModel):
    pass_id: str = Field(min_length=1, max_length=200)
    bias_hz: FiniteFloat


class TimeOffsetParameters(ContractModel):
    model: Literal[OptimizerModel.TIME_OFFSET] = OptimizerModel.TIME_OFFSET
    time_offset_s: FiniteFloat


class TimeOffsetPassBiasParameters(ContractModel):
    model: Literal[OptimizerModel.TIME_OFFSET_PASS_BIAS] = OptimizerModel.TIME_OFFSET_PASS_BIAS
    time_offset_s: FiniteFloat
    pass_biases: list[PassDopplerBias] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_pass_biases(self) -> TimeOffsetPassBiasParameters:
        _validate_unique_pass_biases(self.pass_biases)
        return self


class TimeOffsetFrequencyPassBiasParameters(ContractModel):
    model: Literal[OptimizerModel.TIME_OFFSET_FREQUENCY_PASS_BIAS] = (
        OptimizerModel.TIME_OFFSET_FREQUENCY_PASS_BIAS
    )
    time_offset_s: FiniteFloat
    center_frequency_correction_hz: FiniteFloat
    pass_biases: list[PassDopplerBias] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_pass_biases(self) -> TimeOffsetFrequencyPassBiasParameters:
        _validate_unique_pass_biases(self.pass_biases)
        return self


class MeanElementsTwoParameterParameters(ContractModel):
    model: Literal[OptimizerModel.MEAN_ELEMENTS_TWO_PARAMETER] = (
        OptimizerModel.MEAN_ELEMENTS_TWO_PARAMETER
    )
    mean_anomaly_rad: FiniteFloat
    mean_motion_rad_min: FiniteFloat = Field(gt=0.0)
    pass_biases: list[PassDopplerBias] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_pass_biases(self) -> MeanElementsTwoParameterParameters:
        _validate_unique_pass_biases(self.pass_biases)
        return self


def _validate_unique_pass_biases(pass_biases: list[PassDopplerBias]) -> None:
    pass_ids = [item.pass_id for item in pass_biases]
    if len(set(pass_ids)) != len(pass_ids):
        raise ValueError("pass biases must contain distinct pass IDs in model order")


FitParameters = Annotated[
    TimeOffsetParameters
    | TimeOffsetPassBiasParameters
    | TimeOffsetFrequencyPassBiasParameters
    | MeanElementsTwoParameterParameters,
    Field(discriminator="model"),
]


class FitDiagnostics(ContractModel):
    success: bool
    healthy: bool
    message: str
    observations_used: int = Field(ge=0)
    weighted_ssr: NonnegativeFiniteFloat | None = None
    robust_cost: NonnegativeFiniteFloat | None = None
    jacobian_rank: int | None = Field(default=None, ge=0)
    jacobian_condition: FiniteFloat | None = Field(default=None, ge=1.0)
    at_bound: bool | None = None


class BatchResult(ContractModel):
    schema_version: Literal["0.1"] = SCHEMA_VERSION
    model: OptimizerModel
    effective_carrier_frequency_hz: FiniteFloat = Field(gt=0.0)
    reference_tle: TLEData
    parameters: FitParameters
    covariance: Covariance
    corrected_tle: TLEData | None = None
    residuals: list[ResidualRecord] = Field(min_length=1)
    diagnostics: FitDiagnostics
    consumed_channels: list[ObservableChannel] = Field(min_length=1)
    pass_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_result(self) -> BatchResult:
        if self.parameters.model is not self.model:
            raise ValueError("result model must match the parameter discriminator")
        if len(set(self.pass_ids)) != len(self.pass_ids) or any(
            not value for value in self.pass_ids
        ):
            raise ValueError("result pass_ids must be non-empty and unique in model order")
        if len(set(self.consumed_channels)) != len(self.consumed_channels):
            raise ValueError("result consumed_channels must be unique")
        if self.consumed_channels != [ObservableChannel.DOPPLER]:
            raise ValueError("result consumed_channels for current models must equal [doppler]")
        residual_ids = [record.measurement_id for record in self.residuals]
        if len(set(residual_ids)) != len(residual_ids):
            raise ValueError("result residual measurement IDs must be unique")
        consumed_by_residuals = {
            residual.channel
            for record in self.residuals
            for residual in record.channels
            if residual.consumed
        }
        if consumed_by_residuals != set(self.consumed_channels):
            raise ValueError("result consumed_channels must match consumed residual channels")
        consumed_measurement_count = sum(
            any(channel.consumed for channel in record.channels) for record in self.residuals
        )
        if self.diagnostics.observations_used != consumed_measurement_count:
            raise ValueError("diagnostics observations_used must match consumed measurements")
        expected_order = _parameter_order(self.parameters)
        if self.covariance.parameter_order != expected_order:
            raise ValueError("covariance parameter_order must match the concrete parameter model")
        if self.diagnostics.healthy and not self.diagnostics.success:
            raise ValueError("a healthy result must report solver success")
        if self.diagnostics.healthy and self.diagnostics.observations_used <= len(expected_order):
            raise ValueError("a healthy result requires more observations than fitted parameters")
        if self.diagnostics.healthy:
            jacobian_rank = self.diagnostics.jacobian_rank
            jacobian_condition = self.diagnostics.jacobian_condition
            at_bound = self.diagnostics.at_bound
            if any(
                value is None
                for value in (
                    self.diagnostics.weighted_ssr,
                    self.diagnostics.robust_cost,
                    self.diagnostics.jacobian_rank,
                    self.diagnostics.jacobian_condition,
                    self.diagnostics.at_bound,
                )
            ):
                raise ValueError("a healthy result requires complete solver diagnostics")
            assert jacobian_rank is not None
            assert jacobian_condition is not None
            assert at_bound is not None
            if jacobian_rank != len(expected_order):
                raise ValueError("a healthy result requires full Jacobian rank")
            if jacobian_condition > 1e12:
                raise ValueError("a healthy result requires a well-conditioned Jacobian")
            if at_bound:
                raise ValueError("a healthy result cannot be at a fit bound")
        parameter_pass_ids = _parameter_pass_ids(self.parameters)
        if parameter_pass_ids is not None and self.pass_ids != parameter_pass_ids:
            raise ValueError("result pass_ids must match ordered pass biases")
        if self.model is OptimizerModel.MEAN_ELEMENTS_TWO_PARAMETER:
            if self.corrected_tle is None:
                raise ValueError("mean_elements_two_parameter requires corrected_tle")
        elif self.corrected_tle is not None:
            raise ValueError("only mean_elements_two_parameter may return corrected_tle")
        return self


def _parameter_order(parameters: FitParameters) -> list[str]:
    if isinstance(parameters, TimeOffsetParameters):
        return ["time_offset_s"]
    if isinstance(parameters, TimeOffsetPassBiasParameters):
        return ["time_offset_s", *_pass_bias_parameter_names(parameters.pass_biases)]
    if isinstance(parameters, TimeOffsetFrequencyPassBiasParameters):
        return [
            "time_offset_s",
            "center_frequency_correction_hz",
            *_pass_bias_parameter_names(parameters.pass_biases),
        ]
    return [
        "mean_anomaly_rad",
        "mean_motion_rad_min",
        *_pass_bias_parameter_names(parameters.pass_biases),
    ]


def _parameter_pass_ids(parameters: FitParameters) -> list[str] | None:
    if isinstance(parameters, TimeOffsetParameters):
        return None
    return [item.pass_id for item in parameters.pass_biases]


def _pass_bias_parameter_names(pass_biases: list[PassDopplerBias]) -> list[str]:
    return [f"pass_bias_hz:{item.pass_id}" for item in pass_biases]


class OemEncoding(StrEnum):
    KVN = "kvn"
    XML = "xml"


class OemDocument(ContractModel):
    encoding: OemEncoding
    content: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_digest(self) -> OemDocument:
        if sha256(self.content.encode()).hexdigest() != self.sha256:
            raise ValueError("OEM SHA-256 does not match content")
        return self


class TdmDocument(ContractModel):
    """CCSDS TDM KVN artifact used as the reproducible observation input."""

    media_type: Literal["application/ccsds-tdm-kvn"] = "application/ccsds-tdm-kvn"
    content: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_digest(self) -> TdmDocument:
        if sha256(self.content.encode()).hexdigest() != self.sha256:
            raise ValueError("TDM SHA-256 does not match content")
        if not self.content.startswith("CCSDS_TDM_VERS = "):
            raise ValueError("TDM content must be CCSDS KVN")
        return self


class EvidenceClass(StrEnum):
    RESIDUAL_ONLY = "residual_only"
    REFERENCE_EPHEMERIS = "reference_ephemeris"


class QualityRequest(ContractModel):
    schema_version: Literal["0.1"] = SCHEMA_VERSION
    result: BatchResult
    selection_criterion: InformationCriterion
    reference_oem: OemDocument | None = None


class MetricGroup(ContractModel):
    metrics: dict[str, FiniteFloat | int | bool | None]


class SelectionScore(ContractModel):
    """Comparable information-criterion score for one candidate fit."""

    criterion: InformationCriterion
    eligible: bool
    score: FiniteFloat | None = None
    observations: int = Field(ge=0)
    fitted_parameter_count: int = Field(ge=1)
    residual_sum_squares_hz2: NonnegativeFiniteFloat

    @model_validator(mode="after")
    def validate_eligibility(self) -> SelectionScore:
        if self.eligible and self.score is None:
            raise ValueError("an eligible selection score requires a finite score")
        if not self.eligible and self.score is not None:
            raise ValueError("an ineligible selection score cannot carry a score")
        return self


class QualityResult(ContractModel):
    schema_version: Literal["0.1"] = SCHEMA_VERSION
    evidence_class: EvidenceClass
    convergence: MetricGroup
    residuals: MetricGroup
    selection: SelectionScore
    truth: MetricGroup | None = None


class DatasetQuery(ContractModel):
    spacecraft_id: str | None = None
    contact_ids: list[str] = Field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None
    require_lock: bool = False
    minimum_elevation_deg: FiniteFloat = 1.0
    minimum_ebn0_db: FiniteFloat = 0.0
    minimum_doppler_hz: FiniteFloat = -100_000.0
    maximum_doppler_hz: FiniteFloat = 100_000.0

    @model_validator(mode="after")
    def validate_selection(self) -> DatasetQuery:
        interval = self.spacecraft_id and self.start_time and self.end_time
        if not interval and not self.contact_ids:
            raise ValueError("select by contact_ids or spacecraft_id and time interval")
        if bool(self.start_time) != bool(self.end_time):
            raise ValueError("start_time and end_time must be supplied together")
        for name in ("start_time", "end_time"):
            value = getattr(self, name)
            if value is None:
                continue
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must include a timezone")
            object.__setattr__(self, name, value.astimezone(UTC))
        if self.start_time and self.end_time and self.start_time >= self.end_time:
            raise ValueError("start_time must precede end_time")
        if self.minimum_doppler_hz >= self.maximum_doppler_hz:
            raise ValueError("invalid Doppler interval")
        return self


class DatasetPacket(ContractModel):
    schema_version: Literal["0.1"] = SCHEMA_VERSION
    tdm: TdmDocument
    measurements: list[Measurement]
    stations: dict[str, Cartesian3]
    query: DatasetQuery
    raw_count: int = Field(ge=0)
    presented_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    provenance: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_measurements(self) -> DatasetPacket:
        measurement_ids = [measurement.measurement_id for measurement in self.measurements]
        if len(set(measurement_ids)) != len(measurement_ids):
            raise ValueError("dataset measurement IDs must be unique")
        return self


class RunRequest(ContractModel):
    schema_version: Literal["0.1"] = SCHEMA_VERSION
    pipeline: Literal["batch_od"] = "batch_od"
    query: DatasetQuery
    optimizer_configuration: OptimizerConfiguration
    candidate_metaparameters: list[OptimizerMetaparameters] = Field(min_length=1)
    selection_criterion: InformationCriterion = InformationCriterion.BIC
    reference_oem: OemDocument | None = None

    @model_validator(mode="after")
    def validate_candidates(self) -> RunRequest:
        models = [item.model for item in self.candidate_metaparameters]
        if len(set(models)) != len(models):
            raise ValueError("candidate metaparameters must name distinct models")
        return self


class RunStatus(StrEnum):
    QUEUED = "queued"
    ACQUIRING = "acquiring"
    OPTIMIZING = "optimizing"
    POSTPROCESSING = "postprocessing"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"


class CandidateStatus(StrEnum):
    PENDING = "pending"
    OPTIMIZING = "optimizing"
    POSTPROCESSING = "postprocessing"
    SUCCEEDED = "succeeded"
    UNHEALTHY = "unhealthy"
    INELIGIBLE = "ineligible"
    FAILED = "failed"


class PipelineStage(StrEnum):
    ACQUISITION = "acquisition"
    OPTIMIZATION = "optimization"
    POSTPROCESSING = "postprocessing"
    SELECTION = "selection"
    PERSISTENCE = "persistence"


class StageError(ContractModel):
    stage: PipelineStage
    error_type: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=2_000)


class CandidateRecord(ContractModel):
    candidate_id: UuidString
    candidate_index: int = Field(ge=0)
    metaparameters: OptimizerMetaparameters
    status: CandidateStatus
    result: BatchResult | None = None
    quality: QualityResult | None = None
    error: StageError | None = None

    @model_validator(mode="after")
    def validate_artifacts(self) -> CandidateRecord:
        if self.quality is not None and self.result is None:
            raise ValueError("candidate quality requires a solver result")
        if self.result is not None and self.result.model is not self.metaparameters.model:
            raise ValueError("candidate result model must match candidate metaparameters")
        if self.status is CandidateStatus.SUCCEEDED and (
            self.result is None
            or self.quality is None
            or not self.result.diagnostics.healthy
            or not self.quality.selection.eligible
        ):
            raise ValueError("a succeeded candidate requires healthy result and quality")
        if self.status is CandidateStatus.UNHEALTHY and (
            self.result is None or self.quality is None or self.result.diagnostics.healthy
        ):
            raise ValueError("an unhealthy candidate requires an unhealthy scored result")
        if self.status is CandidateStatus.INELIGIBLE:
            if self.result is None or self.quality is None or not self.result.diagnostics.healthy:
                raise ValueError("an ineligible candidate requires a healthy scored result")
            if self.quality.selection.eligible:
                raise ValueError("an ineligible candidate cannot have a selectable score")
        if self.status is CandidateStatus.FAILED and self.error is None:
            raise ValueError("a failed candidate requires a stage error")
        return self


class RunCreated(ContractModel):
    run_id: UuidString
    status: RunStatus = RunStatus.QUEUED


class RunRecord(ContractModel):
    run_id: UuidString
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    attempts: int = Field(ge=0)
    request_sha256: str
    errors: list[StageError] = Field(default_factory=list)
    candidates: list[CandidateRecord] = Field(min_length=1)
    selected_candidate_id: UuidString | None = None

    @model_validator(mode="after")
    def validate_selection(self) -> RunRecord:
        candidates = {candidate.candidate_id: candidate for candidate in self.candidates}
        if self.selected_candidate_id is not None and self.selected_candidate_id not in candidates:
            raise ValueError("selected_candidate_id must identify a run candidate")
        if (
            self.status in {RunStatus.SUCCEEDED, RunStatus.PARTIAL}
            and self.selected_candidate_id is None
        ):
            raise ValueError("a successful or partial run requires a selected candidate")
        if self.status is RunStatus.FAILED and self.selected_candidate_id is not None:
            raise ValueError("a failed run cannot select a candidate")
        if (
            self.status not in {RunStatus.SUCCEEDED, RunStatus.PARTIAL, RunStatus.FAILED}
            and self.selected_candidate_id is not None
        ):
            raise ValueError("a non-terminal run cannot select a candidate")
        if self.status in {RunStatus.SUCCEEDED, RunStatus.PARTIAL}:
            terminal = {
                CandidateStatus.SUCCEEDED,
                CandidateStatus.UNHEALTHY,
                CandidateStatus.INELIGIBLE,
                CandidateStatus.FAILED,
            }
            if any(candidate.status not in terminal for candidate in candidates.values()):
                raise ValueError("a completed run requires terminal candidate states")
            if self.status is RunStatus.SUCCEEDED and any(
                candidate.status is not CandidateStatus.SUCCEEDED
                for candidate in candidates.values()
            ):
                raise ValueError("a succeeded run requires every candidate to succeed")
            if self.status is RunStatus.PARTIAL and all(
                candidate.status is CandidateStatus.SUCCEEDED for candidate in candidates.values()
            ):
                raise ValueError("a partial run requires at least one non-succeeded candidate")
        if self.selected_candidate_id is not None:
            selected = candidates[self.selected_candidate_id]
            if selected.status is not CandidateStatus.SUCCEEDED:
                raise ValueError("selected candidate must have succeeded")
            if (
                selected.result is None
                or selected.quality is None
                or not selected.result.diagnostics.healthy
                or not selected.quality.selection.eligible
            ):
                raise ValueError("selected candidate must be healthy and score-eligible")
        return self


class RunResult(ContractModel):
    """Concrete selected-result response for a completed multi-candidate run."""

    run_id: UuidString
    status: RunStatus
    selected_candidate_id: UuidString
    selected_candidate: CandidateRecord
    candidates: list[CandidateRecord] = Field(min_length=1)
    errors: list[StageError] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_selected_candidate(self) -> RunResult:
        if self.status not in {RunStatus.SUCCEEDED, RunStatus.PARTIAL}:
            raise ValueError("a result response requires a successful or partial run")
        if self.selected_candidate.candidate_id != self.selected_candidate_id:
            raise ValueError("selected candidate payload must match selected_candidate_id")
        candidates = {candidate.candidate_id: candidate for candidate in self.candidates}
        if candidates.get(self.selected_candidate_id) != self.selected_candidate:
            raise ValueError("selected candidate must be included in candidates")
        if self.selected_candidate.status is not CandidateStatus.SUCCEEDED:
            raise ValueError("result response must select a succeeded candidate")
        return self
