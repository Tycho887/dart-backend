"""One joint Doppler fit, scored against actual OEM samples.

See docs/benchmark.md for acquisition, snapshot replay, and CSV export.
Numerical work is delegated to the existing DART library.
"""

from dataclasses import dataclass, fields, replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import polars as pl
import satkit as sk

from dart.evaluation import compare_states
from dart.io import ContactMetadata, ForwardModelContext
from dart.io.doppler import prepare_doppler
from dart.io.load import _contact_ids
from dart.io.oem import OemEphemeris, read_oem
from dart.od import (
    OptimizerContext,
    OptimizerOutput,
    PriorStateData,
    fit,
    resolve_prior,
    resolve_solution,
)
from dart.od.initialization import initialize_sgp4_time
from dart.orbit import OrbitSolution, StateHistory, propagate
from experiments._benchmark_io import load_inputs, save_json

_ERROR_COLUMNS = ("dx_m", "dy_m", "dz_m", "dvx_m_s", "dvy_m_s", "dvz_m_s")
_STATE_SCHEMA = {
    "timestamp_unix_s": pl.Float64,
    "segment": pl.Int64,
    "solution": pl.String,
    **dict.fromkeys(_ERROR_COLUMNS, pl.Float64),
}


@dataclass(frozen=True)
class BenchmarkResult:
    states: pl.DataFrame
    doppler: pl.DataFrame
    epoch: sk.time
    output: OptimizerOutput
    metadata: dict[str, object]

    def save(self, directory: Path) -> None:
        """Write two CSV tables and a JSON run record; never overwrite a run."""
        directory.mkdir(parents=True, exist_ok=False)
        self.states.write_csv(directory / "states.csv")
        self.doppler.write_csv(directory / "doppler.csv")
        save_json(directory / "run.json", self.metadata)


def _bind_reference(
    reference: OemEphemeris,
    contacts: list[ContactMetadata],
    spacecraft_id: str | None,
) -> OemEphemeris:
    """Bind comparison histories only; preserve source OEM bytes and metadata."""
    identities = {c.cospar for c in contacts}
    if len(identities) != 1 or not next(iter(identities)).strip():
        raise ValueError("reference comparison requires one contact COSPAR")
    cospar = next(iter(identities))
    if spacecraft_id is not None and {c.spacecraft_id for c in contacts} != {
        spacecraft_id
    }:
        raise ValueError("reference binding and contact spacecraft identities differ")
    expected = cospar if spacecraft_id is None else reference.object_id
    source_ids = {s.metadata["OBJECT_ID"] for s in reference.document}
    if source_ids != {expected} or {s.object_id for s in reference.segments} != {
        expected
    }:
        raise ValueError("reference OEM object ID differs from COSPAR or binding")
    return replace(
        reference,
        segments=tuple(replace(s, object_id=cospar) for s in reference.segments),
    )


def _state_errors(
    solution: OrbitSolution,
    reference: OemEphemeris,
    epoch: sk.time,
    label: str,
) -> pl.DataFrame:
    tables = []
    for index, segment in enumerate(reference.segments):
        selected = [i for i, t in enumerate(segment.epochs) if t >= epoch]
        if not selected:
            continue
        truth = StateHistory(
            segment.object_id,
            segment.source_id,
            tuple(segment.epochs[i] for i in selected),
            segment.states[selected],
        )
        errors = compare_states(propagate(solution, truth.epochs), truth).differences
        tables.append(
            pl.DataFrame(
                {
                    "timestamp_unix_s": [t.as_unixtime() for t in truth.epochs],
                    "segment": [index] * len(selected),
                    "solution": [label] * len(selected),
                    **dict(zip(_ERROR_COLUMNS, errors.T, strict=True)),
                },
                schema=_STATE_SCHEMA,
            )
        )
    return pl.concat(tables) if tables else pl.DataFrame(schema=_STATE_SCHEMA)


def _doppler_residuals(
    context: ForwardModelContext, output: OptimizerOutput
) -> pl.DataFrame:
    observations = context.observations
    return pl.DataFrame(
        {
            "timestamp_unix_s": [o.time.as_unixtime() for o in observations],
            "contact_id": [o.contact_id for o in observations],
            "system_id": [
                context.contacts[str(o.contact_id)].system_id for o in observations
            ],
            "observed_hz": [o.observed[0] for o in observations],
            "residual_hz": output.residuals
            * np.sqrt([o.noise_cov[0][0] for o in observations]),
        }
    )


async def benchmark(
    contact_ids: list[str],
    oem_path: Path,
    *,
    optimizer: OptimizerContext,
    ephemeris_id: str,
    center_frequency_hz: float,
    snapshot_dir: Path | None = None,
    reference_spacecraft_id: str | None = None,
    variance_hz2: float = 1.0,
    min_samples: int = 20,
    epoch: sk.time | None = None,
    derived_tle_lines: tuple[str, str] | None = None,
    initialize_time: bool = False,
) -> BenchmarkResult:
    """Fit exactly these contacts once and return signed residuals in SI/Hz.

    A new snapshot_dir freezes inputs; an existing one replays without providers.
    A snapshot may supply subsets, but its prior and OEM must match this request.
    Nonconvergence retains Doppler diagnostics and prior state errors, without
    publishing fitted state errors. Contract and provider failures raise.
    An explicit epoch must be finite and no later than the first observation.
    """
    ids = _contact_ids(contact_ids)
    if not ephemeris_id.strip():
        raise ValueError("an explicit initial ephemeris_id is required")
    reference = read_oem(oem_path)
    contacts, measurements, ephemeris, hashes = await load_inputs(
        ids, ephemeris_id, reference, snapshot_dir
    )
    reference = _bind_reference(reference, contacts, reference_spacecraft_id)
    context, counts = prepare_doppler(
        contacts,
        measurements,
        center_frequency_hz=center_frequency_hz,
        variance_hz2=variance_hz2,
        min_samples=min_samples,
    )
    epoch = _initial_epoch(context, epoch)
    prior = PriorStateData(
        context, ephemeris, epoch, derived_tle_lines=derived_tle_lines
    )
    initial = resolve_prior(prior, optimizer.model)
    output, optimizer, timing = _fit_with_initialization(
        prior, optimizer, initialize_time
    )
    states = _state_errors(initial, reference, epoch, "prior")
    if output.success:
        states = pl.concat(
            [
                states,
                _state_errors(
                    resolve_solution(prior, output), reference, epoch, "fitted"
                ),
            ]
        )
    metadata: dict[str, object] = {
        "format_version": 1,
        "created_at": datetime.now(UTC),
        "contact_ids": ids,
        "contacts": contacts,
        "initial_ephemeris": ephemeris,
        "derived_tle_lines": derived_tle_lines,
        "epoch_unix_s": epoch.as_unixtime(),
        "optimizer": optimizer,
        "output": {
            f.name: getattr(output, f.name)
            for f in fields(output)
            if f.name not in {"residuals", "jacobian"}
        },
        "center_frequency_hz": center_frequency_hz,
        "variance_hz2": variance_hz2,
        "min_samples": min_samples,
        "selection": counts,
        "input_sha256": hashes,
        "reference_path": oem_path,
        "reference_object_id": reference.object_id,
        "reference_spacecraft_id": reference_spacecraft_id,
        "reference_sha256": reference.sha256,
        "state_unavailable_reason": (
            "no OEM samples at or after initialization epoch"
            if states.is_empty()
            else ""
        ),
        "frame": "GCRF",
        "time_system": "UTC Unix seconds",
        "residual_sign": "predicted minus reference/observed",
        "packages": {
            name: version(name)
            for name in ("dart", "satkit", "oem", "numpy", "scipy", "polars")
        },
    }
    metadata.update(timing)
    return BenchmarkResult(
        states, _doppler_residuals(context, output), epoch, output, metadata
    )


def _initial_epoch(context: ForwardModelContext, epoch: sk.time | None) -> sk.time:
    first_observation = min(o.time for o in context.observations)
    if epoch is None:
        epoch = first_observation - sk.duration(seconds=1)
    if not isinstance(epoch, sk.time):
        raise TypeError("initialization epoch must be a satkit.time")
    if not np.isfinite(epoch.as_unixtime()) or epoch > first_observation:
        raise ValueError("initialization epoch must be finite and precede observations")
    return epoch


def _fit_with_initialization(
    prior: PriorStateData, optimizer: OptimizerContext, initialize_time: bool
) -> tuple[OptimizerOutput, OptimizerContext, dict[str, object]]:
    if not initialize_time:
        return fit(prior, optimizer), optimizer, {}
    optimizer, scan = initialize_sgp4_time(prior, optimizer)
    timing = _timing_scan_metadata(prior, optimizer, scan)
    output = fit(prior, optimizer)
    timing.update(_timing_fit_metadata(optimizer, output))
    return output, optimizer, {"timing_initialization": timing}


def _timing_scan_metadata(
    prior: PriorStateData, optimizer: OptimizerContext, scan: np.ndarray
) -> dict[str, object]:
    mapping = prior.observations.contact_to_pass_idx
    columns = [
        "offset_s",
        *[f"pass_bias_hz:{cid}" for cid in sorted(mapping, key=mapping.__getitem__)],
        "cost",
    ]
    spec = next(p for p in optimizer.parameters if p.name == "time_offset_s")
    best = scan[np.argmin(scan[:, -1])]
    return {
        "columns": columns,
        "scan": scan,
        "step_s": 10.0,
        "zero_offset_cost": float(scan[scan[:, 0] == 0, -1][0]),
        "coarse_offset_s": float(best[0]),
        "coarse_cost": float(best[-1]),
        "lower_bound_s": spec.lower_bound,
        "upper_bound_s": spec.upper_bound,
    }


def _timing_fit_metadata(
    optimizer: OptimizerContext, output: OptimizerOutput
) -> dict[str, object]:
    spec = next(p for p in optimizer.parameters if p.name == "time_offset_s")
    offset = float(output.parameters[output.parameter_names.index("time_offset_s")])
    return {
        "refined_offset_s": offset,
        "final_cost": output.cost,
        "at_bound": bool(
            np.any(
                np.isclose(
                    offset, [spec.lower_bound, spec.upper_bound], rtol=0, atol=1e-6
                )
            )
        ),
        "success": output.success,
        "status": output.status,
        "message": output.message,
    }
