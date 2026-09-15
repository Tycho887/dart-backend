"""Strict HTTP contracts, independent of numerical and legacy wire schemas."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, str_strip_whitespace=True
    )


class ProfileRef(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    version: int = Field(default=1, ge=1)


class MeasurementSelection(StrictModel):
    min_elevation_deg: float = Field(default=1, ge=-90, le=90)
    min_ebn0_db: float | None = Field(default=None, ge=-100, le=100)
    min_abs_doppler_hz: float = Field(default=0, ge=0, le=1e9)
    max_abs_doppler_hz: float = Field(default=100000, gt=0, le=1e9)
    min_samples_per_contact: int = Field(default=20, ge=1, le=1000000)
    doppler_sigma_hz: float = Field(default=1, gt=0, le=1e6)

    @model_validator(mode="after")
    def ordered_bounds(self) -> "MeasurementSelection":
        if self.min_abs_doppler_hz > self.max_abs_doppler_hz:
            raise ValueError("minimum Doppler magnitude exceeds maximum")
        return self


class OptimizerOverrides(StrictModel):
    max_evaluations: int | None = Field(default=None, ge=10, le=10000)
    ftol: float | None = Field(default=None, ge=1e-12, le=1e-2)
    xtol: float | None = Field(default=None, ge=1e-12, le=1e-2)
    gtol: float | None = Field(default=None, ge=1e-12, le=1e-2)
    loss_scale: float | None = Field(default=None, ge=0.01, le=10000)


class EstimateRequest(StrictModel):
    contact_ids: list[UUID] = Field(min_length=1, max_length=100)
    ephemeris_id: UUID | None = None
    forward_model: ProfileRef
    optimizer: ProfileRef = Field(
        default_factory=lambda: ProfileRef(name="least-squares")
    )
    measurement_selection: MeasurementSelection = Field(
        default_factory=MeasurementSelection
    )
    optimizer_overrides: OptimizerOverrides = Field(default_factory=OptimizerOverrides)
    nominal_center_frequency_hz: float | None = Field(default=None, ge=1e6, le=1e11)
    label: str | None = Field(default=None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def unique_contacts(self) -> "EstimateRequest":
        if len(set(self.contact_ids)) != len(self.contact_ids):
            raise ValueError("contact IDs must be unique")
        return self


class ParameterDefinition(StrictModel):
    name: str
    unit: str
    initial: float = 0
    lower_bound: float
    upper_bound: float
    scale: float
    role: Literal["estimate", "consider", "fixed"] = "estimate"
    prior_standard_uncertainty: float | None = Field(default=None, gt=0)


class ForwardModelProfile(StrictModel):
    name: str
    version: int = 1
    label: str
    model: Literal["sgp4", "full_state"]
    description: str
    parameters: list[ParameterDefinition]
    pass_bias: ParameterDefinition


class OptimizerProfile(StrictModel):
    name: str
    version: int = 1
    label: str
    compatible_models: list[str]
    loss: Literal["linear", "soft_l1"] = "linear"
    loss_scale: float = 1
    max_evaluations: int = 1000
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float = 1e-8
    initialization: Literal["none", "timing_scan", "phase_scan"] = "none"
    regularization: Literal["none"] = "none"


class ResolvedEstimateConfiguration(StrictModel):
    request: EstimateRequest
    forward_model: ForwardModelProfile
    optimizer: OptimizerProfile
    parameters: list[ParameterDefinition]


class ValidationResponse(StrictModel):
    valid: Literal[True] = True
    configuration: ResolvedEstimateConfiguration
    input_validation: Literal["deferred_to_worker"] = "deferred_to_worker"


JobStatus = Literal[
    "queued",
    "resolving_inputs",
    "loading_telemetry",
    "running",
    "succeeded",
    "failed",
    "canceled",
]


class EstimateAccepted(StrictModel):
    job_id: UUID
    estimate_uuid: UUID
    status: JobStatus
    idempotent_replay: bool


class CancelResponse(StrictModel):
    job_id: UUID
    status: JobStatus
    cancel_requested: bool


ActorID = Annotated[str, Field(min_length=1, max_length=200)]
