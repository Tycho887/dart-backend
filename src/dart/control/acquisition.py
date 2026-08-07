"""Coarse/fine dither acquisition using visibility rather than RF truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import RFObservation
from .interfaces import AntennaBackend, OffsetConvention, SignConvertingBackend


@dataclass(frozen=True, slots=True)
class DitherConfig:
    sweep_min_s: float = -100.0
    sweep_max_s: float = 100.0
    coarse_step_s: float = 10.0
    fine_steps_s: tuple[float, ...] = (2.0, -2.0, 1.0, -1.0)
    lock_elevation_deg: float = 18.0
    horizon_guard_deg: float = 2.0
    max_delivery_reads: int = 5
    max_sweeps: int = 60
    sweep_start_s: float | None = None
    require_offset_tag: bool = True

    def __post_init__(self) -> None:
        if self.coarse_step_s <= 0.0:
            raise ValueError("coarse_step_s must be positive")
        if self.sweep_min_s >= self.sweep_max_s:
            raise ValueError("sweep bounds are reversed")
        if self.max_delivery_reads < 1 or self.max_sweeps < 1:
            raise ValueError("read and sweep limits must be positive")


@dataclass(frozen=True, slots=True)
class Probe:
    offset_s: float
    observation: RFObservation | None
    status: str


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    locked: bool
    offset_s: float | None
    probes: tuple[Probe, ...]
    reason: str | None = None


def _candidate_order(config: DitherConfig) -> np.ndarray:
    count = int(round((config.sweep_max_s - config.sweep_min_s) / config.coarse_step_s))
    candidates = config.sweep_min_s + np.arange(count + 1) * config.coarse_step_s
    if config.sweep_start_s is not None:
        start = int(np.argmin(np.abs(candidates - config.sweep_start_s)))
        candidates = np.concatenate((candidates[start:], candidates[:start]))
    return candidates


def acquire(backend: AntennaBackend, config: DitherConfig = DitherConfig()) -> AcquisitionResult:
    """Find a visible offset while respecting delayed measurement tags."""

    if backend.convention is not OffsetConvention.PROPAGATION_ADVANCE:
        backend = SignConvertingBackend(backend)
    probes: list[Probe] = []
    last_epoch_s: float | None = None

    def probe(offset_s: float) -> tuple[str, RFObservation | None]:
        nonlocal last_epoch_s
        backend.apply_offset(float(offset_s))
        for _ in range(config.max_delivery_reads):
            observation = backend.read()
            if observation is None:
                probes.append(Probe(offset_s, None, "pass_over"))
                return "pass_over", None
            epoch_s = float(observation.epoch.as_unixtime())
            matching_tag = (
                observation.applied_offset_s is not None
                and np.isclose(observation.applied_offset_s, offset_s, atol=1e-9)
            ) or (
                observation.applied_offset_s is None and not config.require_offset_tag
            )
            fresh = last_epoch_s is None or epoch_s > last_epoch_s
            if matching_tag and fresh:
                last_epoch_s = epoch_s
                probes.append(Probe(offset_s, observation, "ok"))
                return "ok", observation
        probes.append(Probe(offset_s, None, "delivery_timeout"))
        return "delivery_timeout", None

    candidates = _candidate_order(config)
    for _ in range(config.max_sweeps):
        run: list[tuple[float, RFObservation]] = []
        for offset in candidates:
            status, observation = probe(float(offset))
            if status == "pass_over":
                return AcquisitionResult(False, None, tuple(probes), "pass ended")
            if observation is None:
                continue
            if observation.valid:
                run.append((float(offset), observation))
            elif run:
                break
        if not run:
            continue

        center = (run[0][0] + run[-1][0]) / 2.0
        best_offset, center_observation = min(run, key=lambda item: abs(item[0] - center))
        elevation = center_observation.commanded_el_deg
        if elevation is None or elevation <= config.lock_elevation_deg:
            continue

        for delta in config.fine_steps_s:
            candidate = best_offset + delta
            status, observation = probe(candidate)
            if status == "pass_over":
                return AcquisitionResult(False, None, tuple(probes), "pass ended during fine dither")
            if (
                observation is not None
                and observation.valid
                and observation.commanded_el_deg is not None
                and observation.commanded_el_deg > config.horizon_guard_deg
            ):
                best_offset = candidate
        backend.apply_offset(best_offset)
        return AcquisitionResult(True, best_offset, tuple(probes))

    return AcquisitionResult(False, None, tuple(probes), "maximum sweep count reached")
