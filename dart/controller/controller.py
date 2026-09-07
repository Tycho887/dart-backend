"""Guarded slow-PI control around an absolute UKF time-offset target."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Protocol

from .acquisition import AppliedOffset
from .UKF import FilterIdentity, TimeOffsetEstimate


class ControlState(StrEnum):
    IDLE = "idle"
    TRACKING = "tracking"
    HOLD = "hold"
    RESETTING = "resetting"
    COMPLETE = "complete"
    FAULT = "fault"


class DecisionAction(StrEnum):
    APPLY = "apply"
    DRY_RUN = "dry_run"
    HOLD = "hold"
    REJECT = "reject"


@dataclass(frozen=True)
class ControllerConfig:
    live_writes_enabled: bool
    absolute_semantics_confirmed: bool
    kp: float
    ki: float
    maximum_abs_offset_s: float
    maximum_step_s: float
    maximum_rate_s_per_s: float
    deadband_s: float
    minimum_update_interval_s: float
    maximum_dt_s: float
    confirmation_timeout_s: float
    maximum_estimate_age_s: float
    maximum_covariance_s2: float
    maximum_nis: float
    minimum_samples: int
    authorization_max_age_s: float = 300.0
    readback_tolerance_s: float = 1e-6

    def validate(self) -> None:
        positive = (
            self.maximum_abs_offset_s,
            self.maximum_step_s,
            self.maximum_rate_s_per_s,
            self.minimum_update_interval_s,
            self.maximum_dt_s,
            self.confirmation_timeout_s,
            self.maximum_estimate_age_s,
            self.maximum_covariance_s2,
            self.maximum_nis,
            self.authorization_max_age_s,
        )
        if not all(math.isfinite(item) and item > 0 for item in positive):
            raise ValueError("controller safety limits must be finite and positive")
        gains = (self.kp, self.ki, self.deadband_s, self.readback_tolerance_s)
        if not all(math.isfinite(item) and item >= 0 for item in gains):
            raise ValueError("PI gains and deadband must be non-negative")
        if self.minimum_samples < 1:
            raise ValueError("sample count and tolerance are invalid")
        if self.live_writes_enabled and not self.absolute_semantics_confirmed:
            raise ValueError("live writes require confirmed absolute endpoint semantics")


@dataclass(frozen=True)
class OffsetWrite:
    command_id: str
    identity: FilterIdentity
    requested_offset_s: float
    issued_at: datetime
    expires_at: datetime
    reason: str


@dataclass(frozen=True)
class LiveWriteAuthorization:
    actor: str
    identity: FilterIdentity
    approved_at: datetime
    expires_at: datetime


@dataclass(frozen=True)
class WriteAcknowledgement:
    command_id: str
    acknowledged_at: datetime
    accepted: bool
    detail: str = ""


@dataclass(frozen=True)
class PendingWrite:
    request: OffsetWrite
    acknowledgement: WriteAcknowledgement
    deadline_monotonic_s: float


@dataclass(frozen=True)
class ControlDecision:
    action: DecisionAction
    reason: str
    target_offset_s: float
    requested_offset_s: float | None
    confirmed_applied_offset_s: float


@dataclass(frozen=True)
class ControlEvent:
    recorded_at: datetime
    state: ControlState
    action: str
    reason: str
    requested_offset_s: float | None
    acknowledged: bool | None
    confirmed_applied_offset_s: float


class OffsetWriter(Protocol):
    def write_offset(self, request: OffsetWrite) -> WriteAcknowledgement: ...


class TimeOffsetController:
    """Pass-owned controller that waits for ADX before issuing another write."""

    def __init__(
        self,
        identity: FilterIdentity,
        profile_version: str,
        model_version: str,
        config: ControllerConfig,
        writer: OffsetWriter,
    ) -> None:
        config.validate()
        self.identity = identity
        self.profile_version = profile_version
        self.model_version = model_version
        self.config = config
        self.writer = writer
        self.state = ControlState.IDLE
        self.confirmed_applied_offset_s = 0.0
        self.pending: PendingWrite | None = None
        self.events: list[ControlEvent] = []
        self._integral_s = 0.0
        self._last_update_monotonic_s: float | None = None
        self._authorization: LiveWriteAuthorization | None = None

    def arm_live_writes(
        self, authorization: LiveWriteAuthorization, now: datetime
    ) -> None:
        if not self.config.live_writes_enabled:
            raise ValueError("controller is configured for dry-run")
        if authorization.identity != self.identity or not authorization.actor.strip():
            raise ValueError("live authorization identity and actor are required")
        if authorization.approved_at.tzinfo is None or authorization.expires_at.tzinfo is None:
            raise ValueError("live authorization timestamps must be timezone-aware")
        now_utc = now.astimezone(UTC)
        age = now_utc - authorization.approved_at.astimezone(UTC)
        lifetime = authorization.expires_at.astimezone(UTC) - now_utc
        if age < timedelta(0) or age > timedelta(seconds=self.config.authorization_max_age_s):
            raise ValueError("live authorization is not recent")
        if lifetime <= timedelta(0):
            raise ValueError("live authorization has expired")
        self._authorization = authorization
        self._event(now, "live_armed", authorization.actor, None, None)

    def enable_tracking(
        self,
        estimate: TimeOffsetEstimate,
        confirmed_applied_offset_s: float,
        monotonic_s: float,
    ) -> None:
        self._require_estimate_identity(estimate)
        if self._identity_rejection(estimate):
            raise ValueError("estimate version does not match controller")
        self.confirmed_applied_offset_s = confirmed_applied_offset_s
        error = estimate.target_offset_s - confirmed_applied_offset_s
        self._integral_s = confirmed_applied_offset_s - estimate.target_offset_s - self.config.kp * error
        self._last_update_monotonic_s = monotonic_s
        self.state = ControlState.TRACKING

    def update(
        self,
        estimate: TimeOffsetEstimate,
        monotonic_s: float,
        now: datetime,
    ) -> ControlDecision:
        reason = self._rejection_reason(estimate, monotonic_s, now)
        if reason:
            return self._decision(DecisionAction.REJECT, reason, estimate, None, now)
        error = estimate.target_offset_s - self.confirmed_applied_offset_s
        if abs(error) <= self.config.deadband_s:
            return self._decision(DecisionAction.HOLD, "deadband", estimate, None, now)
        requested = self._pi_request(estimate.target_offset_s, error, monotonic_s)
        if not self.config.live_writes_enabled:
            self._last_update_monotonic_s = monotonic_s
            return self._decision(DecisionAction.DRY_RUN, "dry_run", estimate, requested, now)
        request = self._request(requested, now, "tracking")
        try:
            acknowledgement = self.writer.write_offset(request)
        except Exception as exc:
            self.state = ControlState.FAULT
            self._event(now, "write_failed", type(exc).__name__, requested, None)
            return self._decision(
                DecisionAction.REJECT, "transport_failure", estimate, None, now
            )
        if not acknowledgement.accepted or acknowledgement.command_id != request.command_id:
            self.state = ControlState.FAULT
            self._event(now, "write_failed", acknowledgement.detail, requested, False)
            return self._decision(DecisionAction.REJECT, "write_not_acknowledged", estimate, None, now)
        self.pending = PendingWrite(
            request,
            acknowledgement,
            monotonic_s + self.config.confirmation_timeout_s,
        )
        self._last_update_monotonic_s = monotonic_s
        self._event(now, "api_acknowledged", "tracking", requested, True)
        return self._decision(DecisionAction.APPLY, "awaiting_adx", estimate, requested, now)

    def confirm_applied(self, readback: AppliedOffset, now: datetime) -> bool:
        self._require_readback_identity(readback)
        if self.pending is None:
            return False
        if readback.effective_command_id != self.pending.request.command_id:
            return False
        if not _fresh_readback(readback, now, self.config.maximum_estimate_age_s):
            return False
        expected = self.pending.request.requested_offset_s
        if abs(readback.applied_offset_s - expected) > self.config.readback_tolerance_s:
            return False
        self.confirmed_applied_offset_s = readback.applied_offset_s
        self.pending = None
        self._event(now, "adx_confirmed", "applied", expected, True)
        if self.state == ControlState.RESETTING and abs(expected) <= self.config.readback_tolerance_s:
            self.state = ControlState.COMPLETE
        return True

    def hold(self, now: datetime, reason: str) -> None:
        self.state = ControlState.HOLD
        self._integral_s = 0.0
        self._last_update_monotonic_s = None
        self._event(now, "hold", reason, None, None)

    def tick(self, monotonic_s: float, now: datetime) -> None:
        if self.pending and monotonic_s > self.pending.deadline_monotonic_s:
            self.state = ControlState.FAULT
            self._integral_s = 0.0
            self._event(
                now,
                "confirmation_timeout",
                "ADX did not confirm the acknowledged write",
                self.pending.request.requested_offset_s,
                True,
            )

    def terminate(self, monotonic_s: float, now: datetime) -> OffsetWrite | None:
        self._integral_s = 0.0
        self.state = ControlState.RESETTING
        if self.pending is not None:
            self.state = ControlState.FAULT
            self._event(now, "reset_blocked", "unconfirmed write", 0.0, None)
            return None
        if abs(self.confirmed_applied_offset_s) <= self.config.readback_tolerance_s:
            self.state = ControlState.COMPLETE
            self._event(now, "reset_complete", "already_zero", 0.0, True)
            return None
        request = self._request(0.0, now, "pass_complete")
        if not self.config.live_writes_enabled:
            self._event(now, "dry_run_reset", "pass_complete", 0.0, None)
            self.state = ControlState.COMPLETE
            return request
        try:
            acknowledgement = self.writer.write_offset(request)
        except Exception as exc:
            self.state = ControlState.FAULT
            self._event(now, "reset_failed", type(exc).__name__, 0.0, None)
            return request
        if not acknowledgement.accepted or acknowledgement.command_id != request.command_id:
            self.state = ControlState.FAULT
            self._event(now, "reset_failed", acknowledgement.detail, 0.0, False)
            return request
        self.pending = PendingWrite(
            request,
            acknowledgement,
            monotonic_s + self.config.confirmation_timeout_s,
        )
        self._event(now, "reset_acknowledged", "awaiting_adx", 0.0, True)
        return request

    def _rejection_reason(
        self,
        estimate: TimeOffsetEstimate,
        monotonic_s: float,
        now: datetime,
    ) -> str | None:
        if self.state != ControlState.TRACKING:
            return "controller_not_tracking"
        if self.pending is not None:
            return "write_unconfirmed"
        if self.config.live_writes_enabled and not self._authorized(now):
            return "live_write_not_authorized"
        identity_reason = self._identity_rejection(estimate)
        if identity_reason:
            return identity_reason
        quality_reason = self._quality_rejection(estimate)
        if quality_reason:
            return quality_reason
        return self._timing_rejection(estimate, monotonic_s, now)

    def _identity_rejection(self, estimate: TimeOffsetEstimate) -> str | None:
        if estimate.identity != self.identity:
            self.state = ControlState.FAULT
            return "identity_mismatch"
        if (estimate.profile_version, estimate.model_version) != (
            self.profile_version,
            self.model_version,
        ):
            self.state = ControlState.FAULT
            return "estimator_version_mismatch"
        return None

    def _authorized(self, now: datetime) -> bool:
        if self._authorization is None or now.tzinfo is None:
            return False
        return now.astimezone(UTC) <= self._authorization.expires_at.astimezone(UTC)

    def _quality_rejection(self, estimate: TimeOffsetEstimate) -> str | None:
        if not _estimate_finite(estimate):
            return "nonfinite_estimate"
        if (
            not estimate.converged
            or estimate.accepted_sample_count < self.config.minimum_samples
        ):
            return "not_converged"
        if estimate.covariance[0][0] > self.config.maximum_covariance_s2:
            return "excessive_covariance"
        if estimate.nis > self.config.maximum_nis:
            return "excessive_nis"
        return None

    def _timing_rejection(
        self,
        estimate: TimeOffsetEstimate,
        monotonic_s: float,
        now: datetime,
    ) -> str | None:
        if now.tzinfo is None or estimate.source_event_at.tzinfo is None:
            return "invalid_timestamp"
        age = now.astimezone(UTC) - estimate.source_event_at.astimezone(UTC)
        if age < timedelta(0) or age > timedelta(seconds=self.config.maximum_estimate_age_s):
            return "stale_estimate"
        if self._last_update_monotonic_s is not None:
            elapsed = monotonic_s - self._last_update_monotonic_s
            if elapsed < self.config.minimum_update_interval_s:
                return "cadence"
        return None

    def _pi_request(self, target_s: float, error_s: float, monotonic_s: float) -> float:
        previous = self._last_update_monotonic_s
        elapsed_s = self.config.maximum_dt_s if previous is None else monotonic_s - previous
        elapsed_s = min(max(elapsed_s, 0.0), self.config.maximum_dt_s)
        tentative_integral = self._integral_s + self.config.ki * error_s * elapsed_s
        raw = target_s + self.config.kp * error_s + tentative_integral
        bounded = _clamp(raw, -self.config.maximum_abs_offset_s, self.config.maximum_abs_offset_s)
        maximum_move = min(
            self.config.maximum_step_s,
            self.config.maximum_rate_s_per_s * elapsed_s,
        )
        requested = _clamp(
            bounded,
            self.confirmed_applied_offset_s - maximum_move,
            self.confirmed_applied_offset_s + maximum_move,
        )
        if requested == raw or (raw > requested and error_s < 0) or (raw < requested and error_s > 0):
            self._integral_s = tentative_integral
        return requested

    def _request(self, requested_s: float, now: datetime, reason: str) -> OffsetWrite:
        now = now.astimezone(UTC)
        return OffsetWrite(
            command_id=str(uuid.uuid4()),
            identity=self.identity,
            requested_offset_s=requested_s,
            issued_at=now,
            expires_at=now + timedelta(seconds=self.config.confirmation_timeout_s),
            reason=reason,
        )

    def _decision(
        self,
        action: DecisionAction,
        reason: str,
        estimate: TimeOffsetEstimate,
        requested_s: float | None,
        now: datetime,
    ) -> ControlDecision:
        self._event(now, action.value, reason, requested_s, None)
        return ControlDecision(
            action,
            reason,
            estimate.target_offset_s,
            requested_s,
            self.confirmed_applied_offset_s,
        )

    def _event(
        self,
        now: datetime,
        action: str,
        reason: str,
        requested_s: float | None,
        acknowledged: bool | None,
    ) -> None:
        self.events.append(
            ControlEvent(
                now.astimezone(UTC),
                self.state,
                action,
                reason,
                requested_s,
                acknowledged,
                self.confirmed_applied_offset_s,
            )
        )

    def _require_estimate_identity(self, estimate: TimeOffsetEstimate) -> None:
        if estimate.identity != self.identity:
            raise ValueError("estimate identity does not match controller")

    def _require_readback_identity(self, readback: AppliedOffset) -> None:
        actual = FilterIdentity(
            readback.contact_id,
            readback.antenna_id,
            readback.ephemeris_id,
        )
        if actual != self.identity:
            self.state = ControlState.FAULT
            raise ValueError("readback identity does not match controller")


def _estimate_finite(estimate: TimeOffsetEstimate) -> bool:
    return all(
        math.isfinite(value)
        for value in (
            estimate.target_offset_s,
            estimate.drift_s_per_s,
            estimate.innovation_s,
            estimate.nis,
            estimate.covariance[0][0],
        )
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _fresh_readback(readback: AppliedOffset, now: datetime, maximum_age_s: float) -> bool:
    if readback.source_event_at.tzinfo is None or now.tzinfo is None:
        return False
    age = now.astimezone(UTC) - readback.source_event_at.astimezone(UTC)
    return timedelta(0) <= age <= timedelta(seconds=maximum_age_s)
