"""Reusable live contact-list experiments; numerical work stays in dart.

Run ``python -m experiments.live_data --help`` for the FOREST comparison CLI.
No provider calls or experiment execution occur at import time.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import satkit as sk
from azure.kusto.data import KustoClient

from dart.evaluation import OrbitError, compare_states
from dart.io import ContactMetadata, EphemerisMetadata, kogs, load_passes
from dart.io.doppler import (
    ContactSelection,
    prepare_doppler,
    select_doppler,
    selection_counts,
)
from dart.io.load import _contact_ids
from dart.io.oem import OemEphemeris, reference_samples
from dart.od import (
    OptimizerContext,
    OptimizerOutput,
    OrbitModel,
    PriorStateData,
    fit,
    resolve_prior,
    resolve_solution,
)
from dart.od.profiles import orbit_bias_profile
from dart.orbit import StateHistory, propagate
from experiments.references import ReferenceMetadata, bind_reference, load_reference


@dataclass(frozen=True)
class EvaluationWindow:
    name: str
    start: sk.time
    stop: sk.time
    after_start: bool = False


@dataclass(frozen=True)
class ExperimentSettings:
    model: OrbitModel
    center_frequency_hz: float
    variance_hz2: float = 1.0
    min_samples: int = 20
    max_evaluations: int = 1000
    timeout_seconds: float = 30.0
    # Absent for a standalone solve; resolved once for comparison matrices.
    epoch: sk.time | None = None
    windows: tuple[EvaluationWindow, ...] = ()

    def __post_init__(self) -> None:
        positive = (self.center_frequency_hz, self.variance_hz2, self.timeout_seconds)
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise ValueError(
                "frequency, variance and timeout must be finite and positive"
            )
        if self.min_samples < 1 or self.max_evaluations < 1:
            raise ValueError("sample and evaluation limits must be positive")


@dataclass(frozen=True)
class WindowScore:
    window: EvaluationWindow
    reference: tuple[StateHistory, ...]
    predicted: tuple[StateHistory, ...]
    baseline: tuple[StateHistory, ...]
    error: OrbitError | None
    baseline_error: OrbitError | None
    unavailable_reason: str = ""


@dataclass(frozen=True)
class ExperimentResult:
    contact_ids: tuple[str, ...]
    settings: ExperimentSettings
    prior: PriorStateData
    optimizer: OptimizerContext
    output: OptimizerOutput
    selection: tuple[ContactSelection, ...]
    scores: tuple[WindowScore, ...]
    reference_metadata: ReferenceMetadata | None = None


async def load_experiment(
    contact_ids: Sequence[str],
    *,
    ephemeris_id: str,
    kogs_api_key: str,
    adx_client: KustoClient,
    timeout_seconds: float = 30.0,
) -> tuple[list[ContactMetadata], pl.DataFrame, EphemerisMetadata]:
    """Fetch the explicitly chosen prior once; contact ephemerides never select it."""
    ids = _contact_ids(contact_ids)
    if not ephemeris_id.strip():
        raise ValueError("an explicit initial ephemeris_id is required")
    prior = await asyncio.to_thread(
        kogs.get_ephemeris, kogs_api_key, ephemeris_id, timeout_seconds=timeout_seconds
    )
    if prior.ephemeris_id != ephemeris_id:
        raise ValueError("returned initial ephemeris ID differs from requested ID")
    if prior.tle is None or not prior.tle.strip():
        raise ValueError("selected initial ephemeris must contain a TLE")
    contacts, frame = await load_passes(
        ids,
        kogs_api_key=kogs_api_key,
        adx_client=adx_client,
        timeout_seconds=timeout_seconds,
        allow_empty=True,
    )
    if {c.spacecraft_id for c in contacts} != {prior.spacecraft_id}:
        raise ValueError("selected prior and contact spacecraft identities differ")
    return contacts, frame, prior


def common_settings(
    contacts: Sequence[ContactMetadata],
    frame: pl.DataFrame,
    settings: ExperimentSettings,
    reference: OemEphemeris,
) -> ExperimentSettings:
    """Choose epochs once from Doppler/contact coverage, never from GPS states."""
    selected = select_doppler(frame)
    if selected.is_empty():
        raise ValueError("no locked Doppler samples")
    first_sample = cast(datetime, selected["timestamp"].min())
    start = sk.time.from_datetime(first_sample)
    stop = sk.time.from_datetime(selected["timestamp"].max())
    last_contact_stop = sk.time.from_datetime(max(c.stop for c in contacts))
    reference_stop = max(t for segment in reference.segments for t in segment.epochs)
    epoch = settings.epoch
    if epoch is None:
        epoch = sk.time.from_datetime(first_sample - timedelta(seconds=1))
    windows = settings.windows or (
        EvaluationWindow("contact_span", start, stop),
        EvaluationWindow("future", last_contact_stop, reference_stop, after_start=True),
    )
    return replace(settings, epoch=epoch, windows=windows)


def _combined(histories: tuple[StateHistory, ...]) -> StateHistory:
    first = histories[0]
    return StateHistory(
        first.object_id,
        first.source_id,
        tuple(t for h in histories for t in h.epochs),
        np.concatenate([h.states for h in histories]),
    )


def evaluate_window(
    prior: PriorStateData,
    output: OptimizerOutput,
    reference: OemEphemeris,
    window: EvaluationWindow,
) -> WindowScore:
    samples = reference_samples(
        reference, window.start, window.stop, after_start=window.after_start
    )
    if not samples:
        return WindowScore(
            window, (), (), (), None, None, "no reference samples in this window"
        )
    solution = resolve_solution(prior, output)
    initial = resolve_prior(prior, output.model_kind)
    predicted = tuple(propagate(solution, segment.epochs) for segment in samples)
    baseline = tuple(propagate(initial, segment.epochs) for segment in samples)
    truth = _combined(samples)
    return WindowScore(
        window,
        samples,
        predicted,
        baseline,
        compare_states(_combined(predicted), truth),
        compare_states(_combined(baseline), truth),
    )


def solve_loaded(
    contacts: Sequence[ContactMetadata],
    frame: pl.DataFrame,
    ephemeris: EphemerisMetadata,
    *,
    settings: ExperimentSettings,
    reference: OemEphemeris,
    reference_metadata: ReferenceMetadata | None = None,
) -> ExperimentResult:
    """Fit one exact contact group using already acquired data and a pinned prior."""
    reference = bind_reference(reference, contacts, reference_metadata)
    context, counts = prepare_doppler(
        contacts,
        frame,
        center_frequency_hz=settings.center_frequency_hz,
        variance_hz2=settings.variance_hz2,
        min_samples=settings.min_samples,
    )
    effective = common_settings(contacts, frame, settings, reference)
    assert effective.epoch is not None  # common_settings always supplies the epoch
    prior = PriorStateData(context, ephemeris, effective.epoch)
    # Validate/materialize the chosen TLE before invoking optimization.
    resolve_prior(prior, settings.model)
    ids = tuple(c.contact_id for c in contacts)
    optimizer = orbit_bias_profile(
        settings.model, ids, max_evaluations=settings.max_evaluations
    )
    output = fit(prior, optimizer)
    scores = (
        tuple(
            evaluate_window(prior, output, reference, window)
            for window in effective.windows
        )
        if output.success
        else ()
    )
    return ExperimentResult(
        ids, effective, prior, optimizer, output, counts, scores, reference_metadata
    )


async def solve_contacts(
    contact_ids: Sequence[str],
    *,
    ephemeris_id: str,
    settings: ExperimentSettings,
    reference: OemEphemeris,
    kogs_api_key: str,
    adx_client: KustoClient,
    reference_metadata: ReferenceMetadata | None = None,
) -> ExperimentResult:
    """Load, fit and score any explicit same-spacecraft list of contact IDs."""
    contacts, frame, prior = await load_experiment(
        contact_ids,
        ephemeris_id=ephemeris_id,
        kogs_api_key=kogs_api_key,
        adx_client=adx_client,
        timeout_seconds=settings.timeout_seconds,
    )
    return solve_loaded(
        contacts,
        frame,
        prior,
        settings=settings,
        reference=reference,
        reference_metadata=reference_metadata,
    )


def contact_groups(contacts: Sequence[ContactMetadata]) -> tuple[tuple[str, ...], ...]:
    ids = tuple(
        c.contact_id for c in sorted(contacts, key=lambda c: (c.start, c.contact_id))
    )
    return tuple((cid,) for cid in ids) + tuple(
        ids[:count] for count in range(2, len(ids) + 1)
    )


def fit_comparison(
    contacts: Sequence[ContactMetadata],
    frame: pl.DataFrame,
    prior: EphemerisMetadata,
    *,
    settings: ExperimentSettings,
    reference: OemEphemeris,
    reference_metadata: ReferenceMetadata | None = None,
) -> Iterator[ExperimentResult]:
    """Fit an already selected usable inventory without acquisition or artifact IO."""
    common = common_settings(contacts, frame, settings, reference)
    lookup = {c.contact_id: c for c in contacts}
    for model in OrbitModel:
        for group in contact_groups(contacts):
            yield solve_loaded(
                [lookup[cid] for cid in group],
                frame.filter(pl.col("contact_id").is_in(group)),
                prior,
                settings=replace(common, model=model),
                reference=reference,
                reference_metadata=reference_metadata,
            )


async def run_comparison(
    contact_ids: Sequence[str],
    *,
    ephemeris_id: str,
    spacecraft_id: str,
    settings: ExperimentSettings,
    reference: OemEphemeris,
    output_dir: Path,
    kogs_api_key: str,
    adx_client: KustoClient,
    reference_metadata: ReferenceMetadata | None = None,
) -> list[ExperimentResult]:
    """Run singles and growing groups for both models, sharing acquisition/prior."""
    from experiments.live_data_report import save_inventory, save_result, save_summary

    contacts, frame, prior = await load_experiment(
        contact_ids,
        ephemeris_id=ephemeris_id,
        kogs_api_key=kogs_api_key,
        adx_client=adx_client,
        timeout_seconds=settings.timeout_seconds,
    )
    if prior.spacecraft_id != spacecraft_id:
        raise ValueError("experiment spacecraft differs from selected prior")
    bind_reference(reference, contacts, reference_metadata)
    output_dir.mkdir(parents=True, exist_ok=False)
    counts = selection_counts(contacts, frame)
    save_inventory(
        output_dir,
        contacts,
        frame,
        prior,
        counts,
        settings,
        reference,
        reference_metadata,
    )
    usable_ids = {
        c.contact_id for c in counts if c.retained_samples >= settings.min_samples
    }
    usable = [c for c in contacts if c.contact_id in usable_ids]
    if not usable:
        raise ValueError("no contacts meet the minimum locked sample count")
    filtered = frame.filter(pl.col("contact_id").is_in(usable_ids))
    results = []
    cases = fit_comparison(
        usable,
        filtered,
        prior,
        settings=settings,
        reference=reference,
        reference_metadata=reference_metadata,
    )
    for index, result in enumerate(cases):
        save_result(output_dir / f"{result.output.model_kind}-{index:03d}", result)
        results.append(result)
        save_summary(output_dir, results)
    return results


def main() -> None:
    import argparse
    import os
    import runpy

    from dotenv import load_dotenv

    from dart.io.adx import client_from_env

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--ephemeris-id", required=True)
    parser.add_argument(
        "--reference-oem", type=Path, help="override the case GPS snapshot"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(
        os.getenv("DART_SECRETS_ENV", "/opt/dart/secrets/test.env"), override=False
    )
    case = runpy.run_path(str(args.case))
    reference, reference_metadata = load_reference(
        case["DEFAULT_REFERENCE_OEM"],
        case["REFERENCE_OBJECT_ID"],
        case["SPACECRAFT_ID"],
        args.reference_oem,
    )
    settings = ExperimentSettings(OrbitModel.SGP4, case["CENTER_FREQUENCY_HZ"])
    with client_from_env() as client:
        results = asyncio.run(
            run_comparison(
                case["CONTACT_IDS"],
                ephemeris_id=args.ephemeris_id,
                spacecraft_id=case["SPACECRAFT_ID"],
                settings=settings,
                reference=reference,
                reference_metadata=reference_metadata,
                output_dir=args.output,
                kogs_api_key=os.environ["KOGS_API_KEY"],
                adx_client=client,
            )
        )
    print(f"Saved {len(results)} fits to {args.output}")


if __name__ == "__main__":
    main()
