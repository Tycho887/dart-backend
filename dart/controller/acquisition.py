"""Binary-lock acquisition with delayed command confirmation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum


class AcquisitionState(StrEnum):
    PLANNED = "planned"
    PRECHECK = "precheck"
    WAITING = "waiting"
    ACQUIRING = "acquiring"
    TRACKING = "tracking"
    HOLD = "hold"
    REACQUIRING = "reacquiring"
    RESETTING = "resetting"
    COMPLETE = "complete"
    ABORTED = "aborted"
    FAULT = "fault"


@dataclass(frozen=True)
class AcquisitionConfig:
    sweep_min_s: float
    sweep_max_s: float
    sweep_step_s: float
    settle_s: float
    confirmation_timeout_s: float
    maximum_sample_age_s: float
    consecutive_samples: int
    readback_tolerance_s: float = 1e-6

    def validate(self) -> None:
        values = (
            self.sweep_min_s,
            self.sweep_max_s,
            self.sweep_step_s,
            self.settle_s,
            self.confirmation_timeout_s,
            self.maximum_sample_age_s,
            self.readback_tolerance_s,
        )
        if not all(math.isfinite(item) for item in values):
            raise ValueError("acquisition limits must be finite")
        if self.sweep_min_s >= self.sweep_max_s or self.sweep_step_s <= 0:
            raise ValueError("sweep bounds and step are invalid")
        if min(self.settle_s, self.confirmation_timeout_s) <= 0:
            raise ValueError("timing limits must be positive")
        if self.maximum_sample_age_s < 0 or self.readback_tolerance_s < 0:
            raise ValueError("age and tolerance must be non-negative")
        if self.consecutive_samples < 1:
            raise ValueError("consecutive_samples must be positive")


@dataclass(frozen=True)
class OffsetCommand:
    command_id: int
    contact_id: str
    antenna_id: str
    ephemeris_id: str
    requested_offset_s: float
    issued_monotonic_s: float
    expires_monotonic_s: float
    reason: str


@dataclass(frozen=True)
class AppliedOffset:
    contact_id: str
    antenna_id: str
    ephemeris_id: str
    effective_command_id: str
    applied_offset_s: float
    observed_at: datetime
    source_event_at: datetime


@dataclass(frozen=True)
class LockSample:
    contact_id: str
    antenna_id: str
    ephemeris_id: str
    locked: bool
    observed_at: datetime
    source_event_at: datetime


@dataclass(frozen=True)
class Probe:
    requested_offset_s: float
    applied_offset_s: float
    locked: bool
    confirmed_at: datetime
    scored_at: datetime


class BinaryLockAcquisition:
    """Pass-scoped sweep that never has more than one unconfirmed command."""

    def __init__(
        self,
        contact_id: str,
        antenna_id: str,
        ephemeris_id: str,
        config: AcquisitionConfig,
    ) -> None:
        config.validate()
        self.contact_id = contact_id
        self.antenna_id = antenna_id
        self.ephemeris_id = ephemeris_id
        self.config = config
        self.state = AcquisitionState.PLANNED
        self.probes: list[Probe] = []
        self.outstanding: OffsetCommand | None = None
        self.api_acknowledged = False
        self.applied_offset_s = 0.0
        self._deadline = 0.0
        self._settled_after = 0.0
        self._command_sequence = 0
        self._sweep = _sweep_values(config)
        self._sweep_index = 0
        self._sample_value: bool | None = None
        self._sample_count = 0
        self._left_edge_s: float | None = None
        self._right_edge_s: float | None = None
        self._confirming_midpoint = False
        self._resume_after_reset = False

    def precheck(self) -> None:
        self._transition(AcquisitionState.PLANNED, AcquisitionState.PRECHECK)

    def wait(self) -> None:
        self._transition(AcquisitionState.PRECHECK, AcquisitionState.WAITING)

    def start(self, monotonic_s: float, deadline_monotonic_s: float) -> OffsetCommand:
        self._transition(AcquisitionState.WAITING, AcquisitionState.ACQUIRING)
        if deadline_monotonic_s <= monotonic_s:
            raise ValueError("pass deadline must be in the future")
        self._deadline = deadline_monotonic_s
        return self._issue(self._sweep[0], monotonic_s, "sweep")

    def acknowledge(self, command_id: int) -> None:
        if self.outstanding is None or self.outstanding.command_id != command_id:
            raise ValueError("acknowledgement does not match the outstanding command")
        self.api_acknowledged = True

    def confirm_applied(
        self, readback: AppliedOffset, monotonic_s: float
    ) -> OffsetCommand | None:
        self._require_identity(
            readback.contact_id, readback.antenna_id, readback.ephemeris_id
        )
        if self.outstanding is None:
            return None
        if readback.effective_command_id != str(self.outstanding.command_id):
            return None
        if not _fresh(
            readback.source_event_at,
            readback.observed_at,
            self.config.maximum_sample_age_s,
        ):
            return None
        expected = self.outstanding.requested_offset_s
        if abs(readback.applied_offset_s - expected) > self.config.readback_tolerance_s:
            return None
        if not self.api_acknowledged:
            self.state = AcquisitionState.HOLD
            return None
        self.applied_offset_s = readback.applied_offset_s
        self.outstanding = None
        self.api_acknowledged = False
        self._settled_after = monotonic_s + self.config.settle_s
        self._sample_value = None
        self._sample_count = 0
        if self.state == AcquisitionState.RESETTING and expected == 0.0:
            if self._resume_after_reset and monotonic_s < self._deadline:
                self._restart_sweep()
                return self._issue(self._sweep[0], monotonic_s, "reacquire")
            self.state = AcquisitionState.COMPLETE
        return None

    def observe_lock(
        self, sample: LockSample, monotonic_s: float, now: datetime
    ) -> OffsetCommand | None:
        self._require_identity(sample.contact_id, sample.antenna_id, sample.ephemeris_id)
        if self.state == AcquisitionState.TRACKING:
            return self._observe_tracking(sample, monotonic_s, now)
        active = {AcquisitionState.ACQUIRING, AcquisitionState.REACQUIRING}
        if self.state not in active or self.outstanding is not None:
            return None
        if monotonic_s < self._settled_after:
            return None
        if not _fresh(sample.source_event_at, now, self.config.maximum_sample_age_s):
            return None
        self._count_sample(sample.locked)
        if self._sample_count < self.config.consecutive_samples:
            return None
        self._record_probe(sample)
        return self._advance(monotonic_s)

    def tick(self, monotonic_s: float) -> OffsetCommand | None:
        if self.state in {AcquisitionState.COMPLETE, AcquisitionState.FAULT}:
            return None
        if self.outstanding and monotonic_s > self.outstanding.expires_monotonic_s:
            self.state = AcquisitionState.FAULT
            return None
        if monotonic_s >= self._deadline and self.state != AcquisitionState.RESETTING:
            return self.reset(monotonic_s, "pass_deadline")
        return None

    def abort(self, monotonic_s: float) -> OffsetCommand | None:
        self.state = AcquisitionState.ABORTED
        return self.reset(monotonic_s, "abort")

    def reset(self, monotonic_s: float, reason: str) -> OffsetCommand | None:
        self.state = AcquisitionState.RESETTING
        if self.outstanding is not None:
            self.state = AcquisitionState.FAULT
            return None
        if abs(self.applied_offset_s) <= self.config.readback_tolerance_s:
            self.state = AcquisitionState.COMPLETE
            return None
        return self._issue(0.0, monotonic_s, reason)

    def _advance(self, monotonic_s: float) -> OffsetCommand | None:
        probe = self.probes[-1]
        if self._confirming_midpoint:
            if probe.locked:
                self.state = AcquisitionState.TRACKING
                return None
            self.state = AcquisitionState.HOLD
            return self.reset(monotonic_s, "midpoint_not_locked")
        if self._left_edge_s is None and probe.locked:
            if self._sweep_index == 0:
                self.state = AcquisitionState.HOLD
                return self.reset(monotonic_s, "left_edge_censored")
            self._left_edge_s = probe.applied_offset_s
        elif self._left_edge_s is not None and not probe.locked:
            self._right_edge_s = self.probes[-2].applied_offset_s
            return self._command_midpoint(monotonic_s)
        self._sweep_index += 1
        if self._sweep_index >= len(self._sweep):
            self.state = AcquisitionState.HOLD
            reason = (
                "right_edge_censored"
                if self._left_edge_s is not None
                else "no_lock_window"
            )
            return self.reset(monotonic_s, reason)
        return self._issue(self._sweep[self._sweep_index], monotonic_s, "sweep")

    def _command_midpoint(self, monotonic_s: float) -> OffsetCommand:
        assert self._left_edge_s is not None and self._right_edge_s is not None
        self._confirming_midpoint = True
        midpoint = (self._left_edge_s + self._right_edge_s) / 2.0
        return self._issue(midpoint, monotonic_s, "lock_window_midpoint")

    def _observe_tracking(
        self, sample: LockSample, monotonic_s: float, now: datetime
    ) -> OffsetCommand | None:
        if not _fresh(sample.source_event_at, now, self.config.maximum_sample_age_s):
            return None
        if sample.locked:
            self._sample_count = 0
            return None
        self._sample_count += 1
        if self._sample_count < self.config.consecutive_samples:
            return None
        self.state = AcquisitionState.REACQUIRING
        self._resume_after_reset = True
        return self.reset(monotonic_s, "lock_lost")

    def _restart_sweep(self) -> None:
        self.state = AcquisitionState.REACQUIRING
        self._sweep_index = 0
        self._left_edge_s = None
        self._right_edge_s = None
        self._confirming_midpoint = False
        self._resume_after_reset = False

    def _count_sample(self, locked: bool) -> None:
        if locked != self._sample_value:
            self._sample_value = locked
            self._sample_count = 1
        else:
            self._sample_count += 1

    def _record_probe(self, sample: LockSample) -> None:
        self.probes.append(
            Probe(
                requested_offset_s=self.applied_offset_s,
                applied_offset_s=self.applied_offset_s,
                locked=sample.locked,
                confirmed_at=sample.observed_at,
                scored_at=sample.source_event_at,
            )
        )
        self._sample_count = 0
        self._sample_value = None

    def _issue(self, value_s: float, monotonic_s: float, reason: str) -> OffsetCommand:
        if self.outstanding is not None:
            raise RuntimeError("only one unconfirmed command is permitted")
        self._command_sequence += 1
        command = OffsetCommand(
            command_id=self._command_sequence,
            contact_id=self.contact_id,
            antenna_id=self.antenna_id,
            ephemeris_id=self.ephemeris_id,
            requested_offset_s=value_s,
            issued_monotonic_s=monotonic_s,
            expires_monotonic_s=monotonic_s + self.config.confirmation_timeout_s,
            reason=reason,
        )
        self.outstanding = command
        return command

    def _require_identity(self, contact: str, antenna: str, ephemeris: str) -> None:
        if (contact, antenna, ephemeris) != (
            self.contact_id,
            self.antenna_id,
            self.ephemeris_id,
        ):
            self.state = AcquisitionState.FAULT
            raise ValueError("telemetry identity does not match this acquisition")

    def _transition(
        self, expected: AcquisitionState, target: AcquisitionState
    ) -> None:
        if self.state != expected:
            raise RuntimeError(f"expected {expected}, found {self.state}")
        self.state = target


def _sweep_values(config: AcquisitionConfig) -> tuple[float, ...]:
    span = config.sweep_max_s - config.sweep_min_s
    count = int(math.floor(span / config.sweep_step_s))
    values = [
        config.sweep_min_s + index * config.sweep_step_s
        for index in range(count + 1)
    ]
    if values[-1] < config.sweep_max_s - config.readback_tolerance_s:
        values.append(config.sweep_max_s)
    return tuple(values)


def _fresh(source_event_at: datetime, now: datetime, maximum_age_s: float) -> bool:
    if source_event_at.tzinfo is None or now.tzinfo is None:
        return False
    age = now.astimezone(UTC) - source_event_at.astimezone(UTC)
    return timedelta(0) <= age <= timedelta(seconds=maximum_age_s)
