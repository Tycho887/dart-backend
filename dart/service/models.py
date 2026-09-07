# """Strict HTTP models for the versioned DART service contract.

# These models are deliberately independent of :mod:`dart.schema`, which is the
# Python/Rust MessagePack contract. Changing this module does not require a wire
# schema version bump.
# """

# from __future__ import annotations

# from enum import StrEnum
# from typing import Annotated, Literal
# from uuid import UUID

# from pydantic import BaseModel, ConfigDict, Field, model_validator


# class StrictModel(BaseModel):
#     model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# class Strategy(StrEnum):
#     SINGLE_CONTACT = "single_contact"
#     JOINT = "joint"


# class ActorType(StrEnum):
#     HUMAN = "human"
#     SERVICE = "service"


# class RobustLoss(StrEnum):
#     LINEAR = "linear"
#     HUBER = "huber"
#     SOFT_L1 = "soft_l1"
#     LOG_COSH = "log_cosh"


# class OptimizerProfileRef(StrictModel):
#     profile: str = Field(min_length=1, max_length=100)
#     version: int = Field(ge=1)


# class TdmProfileRef(StrictModel):
#     name: str = Field(min_length=1, max_length=100)
#     version: int = Field(ge=1)


# class CommonOptimizerOverrides(StrictModel):
#     loss: RobustLoss | None = None
#     loss_scale: float | None = Field(default=None, ge=0.01, le=100.0)
#     doppler_sigma_hz: float | None = Field(default=None, ge=1.0, le=1_000_000.0)
#     max_evaluations: int | None = Field(default=None, ge=10, le=10_000)


# class MeanElementsOverrides(CommonOptimizerOverrides):
#     ftol_rel: float | None = Field(default=None, ge=1e-12, le=1e-2)
#     xtol_rel: float | None = Field(default=None, ge=1e-12, le=1e-2)


# class TimeShiftOverrides(CommonOptimizerOverrides):
#     use_qmc: bool | None = None
#     qmc_samples: int | None = Field(default=None, ge=1, le=1024)


# class MeanElementsOptimizer(OptimizerProfileRef):
#     overrides: MeanElementsOverrides = Field(default_factory=MeanElementsOverrides)


# class TimeShiftOptimizer(OptimizerProfileRef):
#     overrides: TimeShiftOverrides = Field(default_factory=TimeShiftOverrides)


# MeanElementsParameterization = Literal[
#     "mean_anomaly",
#     "mean_anomaly_mean_motion",
#     "mean_anomaly_mean_motion_frequency",
# ]
# TimeShiftParameterization = Literal[
#     "time_shift",
#     "time_shift_bias",
#     "time_shift_bias_frequency",
# ]


# class MeanElementsSolver(StrictModel):
#     kind: Literal["sgp4_mean_elements"]
#     parameterization: MeanElementsParameterization
#     optimizer: MeanElementsOptimizer
#     nominal_center_frequency_hz: float | None = Field(
#         default=None, ge=1_000_000.0, le=100_000_000_000.0
#     )


# class TimeShiftSolver(StrictModel):
#     kind: Literal["sgp4_time_shift"]
#     parameterization: TimeShiftParameterization
#     optimizer: TimeShiftOptimizer
#     nominal_center_frequency_hz: float | None = Field(
#         default=None, ge=1_000_000.0, le=100_000_000_000.0
#     )


# SolverRequest = Annotated[
#     MeanElementsSolver | TimeShiftSolver,
#     Field(discriminator="kind"),
# ]


# class TelemetryFilter(StrictModel):
#     require_lock: bool = False
#     min_elevation_deg: float = Field(default=1.0, ge=-90.0, le=90.0)
#     min_doppler_hz: float = Field(default=1.0, ge=0.0, le=1_000_000_000.0)
#     max_doppler_hz: float = Field(default=100_000.0, gt=0.0, le=1_000_000_000.0)
#     min_pass_measurements: int = Field(default=250, ge=1, le=1_000_000)

#     @model_validator(mode="after")
#     def ordered_doppler_bounds(self) -> TelemetryFilter:
#         if self.min_doppler_hz > self.max_doppler_hz:
#             raise ValueError("min_doppler_hz must not exceed max_doppler_hz")
#         return self


# class ClientContext(StrictModel):
#     label: str | None = Field(default=None, max_length=200)
#     tags: list[str] = Field(default_factory=list, max_length=32)

#     @model_validator(mode="after")
#     def validate_tags(self) -> ClientContext:
#         if any(not tag or len(tag) > 64 for tag in self.tags):
#             raise ValueError("tags must contain 1 to 64 characters")
#         if len(set(self.tags)) != len(self.tags):
#             raise ValueError("tags must be unique")
#         return self


# class SolveJobRequest(StrictModel):
#     contact_ids: list[UUID] = Field(min_length=1, max_length=100)
#     strategy: Strategy = Strategy.SINGLE_CONTACT
#     ephemeris_id: UUID | None = None
#     solver: SolverRequest
#     telemetry_filter: TelemetryFilter = Field(default_factory=TelemetryFilter)
#     client_context: ClientContext = Field(default_factory=ClientContext)

#     @model_validator(mode="after")
#     def validate_strategy(self) -> SolveJobRequest:
#         if self.strategy == Strategy.SINGLE_CONTACT and len(self.contact_ids) != 1:
#             raise ValueError("single_contact requires exactly one contact_id")
#         return self


# class TdmJobRequest(StrictModel):
#     contact_id: UUID
#     product: Literal["track", "angle"]
#     profile: TdmProfileRef
#     client_context: ClientContext = Field(default_factory=ClientContext)


# class SolveJobAccepted(StrictModel):
#     job_id: UUID
#     status: Literal["queued"] = "queued"
#     idempotent_replay: bool = False


# class ValidationResponse(StrictModel):
#     valid: Literal[True] = True
#     resolved_profile: dict
#     effective_settings: dict
#     frequency_source: Literal["request", "control_config_deferred"]


# class TdmValidationResponse(StrictModel):
#     valid: Literal[True] = True
#     resolved_profile: dict


# class CancelResponse(StrictModel):
#     job_id: UUID
#     status: str
#     cancel_requested: bool


# class Problem(StrictModel):
#     type: str
#     title: str
#     status: int
#     detail: str
#     code: str
#     retryable: bool = False
#     instance: str | None = None
