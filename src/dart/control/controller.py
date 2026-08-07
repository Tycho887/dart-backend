"""Safety-bounded dither acquisition and UKF tracking controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..estimation.ukf import PassiveRFUKF, UKFConfig
from ..types import Estimate, TLEContext
from .acquisition import AcquisitionResult, DitherConfig, acquire
from .interfaces import AntennaBackend, OffsetConvention, SignConvertingBackend


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    dither: DitherConfig = DitherConfig()
    ukf: UKFConfig = UKFConfig()
    phase_capable: bool = False
    max_abs_command_s: float = 120.0
    max_command_step_s: float = 5.0
    invalid_before_reacquire: int = 30
    max_tracking_reads: int = 5_000
    minimum_updates_for_health: int = 20
    minimum_acceptance_fraction: float = 0.25


@dataclass(frozen=True, slots=True)
class ControllerResult:
    acquisition: AcquisitionResult
    estimates: tuple[Estimate, ...]
    accepted_updates: int
    rejected_updates: int
    reacquisitions: int
    final_offset_s: float | None
    healthy: bool
    reason: str | None = None


class _SafeCommander:
    def __init__(self, backend: AntennaBackend, maximum: float, maximum_step: float):
        self.backend = backend
        self.maximum = float(maximum)
        self.maximum_step = float(maximum_step)
        self.last: float | None = None

    def apply(self, requested: float, *, immediate: bool = False) -> float:
        command = float(np.clip(requested, -self.maximum, self.maximum))
        if self.last is not None and not immediate:
            command = float(
                np.clip(command, self.last - self.maximum_step, self.last + self.maximum_step)
            )
        self.backend.apply_offset(command)
        self.last = command
        return command


class LEOPController:
    """Run acquisition, sequential tracking, and beam-loss recovery."""

    def __init__(
        self,
        context: TLEContext,
        backend: AntennaBackend,
        config: ControllerConfig = ControllerConfig(),
    ) -> None:
        self.context = context
        self.backend = (
            backend
            if backend.convention is OffsetConvention.PROPAGATION_ADVANCE
            else SignConvertingBackend(backend)
        )
        self.config = config
        self.commander = _SafeCommander(
            self.backend, config.max_abs_command_s, config.max_command_step_s
        )

    def run(self) -> ControllerResult:
        acquisition = acquire(self.backend, self.config.dither)
        if not acquisition.locked or acquisition.offset_s is None:
            return ControllerResult(
                acquisition, (), 0, 0, 0, None, False, acquisition.reason
            )
        self.commander.apply(acquisition.offset_s, immediate=True)
        ukf = PassiveRFUKF(
            self.context,
            acquisition.offset_s,
            phase_capable=self.config.phase_capable,
            config=self.config.ukf,
        )
        estimates: list[Estimate] = []
        accepted = rejected = reacquisitions = invalid_run = 0

        for _ in range(self.config.max_tracking_reads):
            observation = self.backend.read()
            if observation is None:
                break
            estimate = ukf.step(observation)
            estimates.append(estimate)
            if estimate.accepted and estimate.healthy:
                accepted += 1
                invalid_run = 0
                self.commander.apply(estimate.offset_s)
            else:
                rejected += 1
                above_horizon_guard = (
                    observation.commanded_el_deg is not None
                    and observation.commanded_el_deg > self.config.dither.horizon_guard_deg
                )
                invalid_run = (
                    invalid_run + 1
                    if not observation.valid and above_horizon_guard
                    else 0
                )

            if invalid_run > self.config.invalid_before_reacquire:
                reacquired = acquire(self.backend, self.config.dither)
                reacquisitions += 1
                invalid_run = 0
                if not reacquired.locked or reacquired.offset_s is None:
                    return ControllerResult(
                        acquisition,
                        tuple(estimates),
                        accepted,
                        rejected,
                        reacquisitions,
                        self.commander.last,
                        False,
                        "beam lost and reacquisition failed",
                    )
                self.commander.apply(reacquired.offset_s, immediate=True)
                ukf.reset_offset(reacquired.offset_s, self.config.dither.coarse_step_s)

        attempted = accepted + rejected
        acceptance_fraction = accepted / attempted if attempted else 0.0
        healthy = (
            accepted >= self.config.minimum_updates_for_health
            and acceptance_fraction >= self.config.minimum_acceptance_fraction
        )
        return ControllerResult(
            acquisition=acquisition,
            estimates=tuple(estimates),
            accepted_updates=accepted,
            rejected_updates=rejected,
            reacquisitions=reacquisitions,
            final_offset_s=self.commander.last,
            healthy=healthy,
            reason=(
                None
                if healthy
                else "insufficient accepted tracking updates or acceptance fraction"
            ),
        )
