# Dart

Dart is a Python prototype for passive-RF spacecraft tracking and orbit estimation during LEOP. It combines antenna acquisition, bounded live steering, pass-constant Doppler/phase estimation, real FOREST replay against independent GPS, and multi-pass TLE mean-element refinement.

The source TLE is immutable during tracking. A correction is represented as `SGP4(t + offset_s)`, where a positive offset advances propagation along the TLE trajectory.

## Documentation

| Document | Contents |
| --- | --- |
| [Codebase guide](docs/codebase.md) | Package data flow, source-module map, estimator states, CLI commands, and tests. |
| [Architecture](docs/architecture.md) | Estimation/control boundaries, validation meaning, and operational risks. |
| [Dependencies](docs/dependencies.md) | Python environment, direct/optional dependencies, external data, and reproducibility. |
| [Reports guide](docs/reports.md) | Report catalog, reading order, interpretation boundaries, and reproduction commands. |
| [Migration provenance](docs/migration.md) | Legacy repository origins and intentional behavior changes. |

For the current real-data result, read the [Doppler-only batch-LS protocol](reports/production/protocol.md) before the [batch-LS evaluation](reports/production/doppler_batch_ls.md). The [report map](reports/README.md) separates that evidence from the UKF and Henault-style experiments.

## Supported workflows

| Workflow | Data status | Main implementation |
| --- | --- | --- |
| Post-pass Doppler-only pass estimation | Recorded FOREST Doppler + raw BESTXYZ scoring | Robust full-pass batch LS |
| Static UKF replay | Recorded Doppler and closed-loop simulation | Experimental only |
| Doppler plus interferometric phase difference | Simulation only | Henault-style measurement-model research |
| Antenna acquisition and steering | Closed-loop simulation; HTTP backend available | Experimental dither controller |
| Mean anomaly/mean motion refinement | Held-out synthetic multi-pass study | Experimental batch mean-element estimator |

Real phase results require a calibrated ENU baseline, phase-chain delay characterization, continuity/cycle-slip handling, and suitable reference passes. Synthetic complete-phase results are idealized information studies, not real phase validation.

## Quick start

Requirements are Python 3.13 and `uv`.

```bash
uv sync
uv run pytest
uv run dart --help
uv run dart simulate --mode both
```

Optional environments:

```bash
uv sync --extra api
uv sync --extra plot
```

See [dependencies.md](docs/dependencies.md) for what each package is used for. The full transitive environment is pinned in `uv.lock`.

## Current real-data post-pass batch-LS result

The real-Doppler cohort inventories 61 raw contacts, 15 contacts with at least 301 presented Doppler samples, and 11 passes with at least five same-pass GPS fixes.

| Product | Best pass median | Median | Worst pass median |
| --- | ---: | ---: | ---: |
| Source TLE | 2.437 km | 9.310 km | 21.391 km |
| Full-pass batch backcast | 0.494 km | 3.857 km | 24.856 km |

This is an 11-pass cohort with at least five same-pass GPS fixes and at least 301 presented Doppler samples. It is a real-data, post-pass, Doppler-only batch-LS result. The observed 0.494 km best pass establishes sub-kilometre performance in favourable conditions; it is not a guaranteed accuracy level. A 0.307 km pass with only three GPS fixes is retained as supplementary context rather than included in the primary cohort.

The production result and its limitations are in the [batch-LS report](reports/production/doppler_batch_ls.md). The following remain separate experimental branches:

- [static-UKF replay on real Doppler](reports/experimental/ukf/ukf_replay.md);
- [Henault-style phase-difference experiments](reports/experimental/henault_phase/); and
- [report map and input/audit artifacts](reports/README.md).

## Repository layout

```text
src/dart/
├── control/       acquisition, backend interfaces, live controller, HTTP client
├── estimation/    robust batch, static UKF, mean-element refinement
├── io/            FOREST telemetry and NovAtel BESTXYZ loaders
├── validation/    inventory, replay, paired simulation, model ablation
├── geometry.py    propagation, frames, station and topocentric geometry
├── measurements.py Doppler and wrapped-phase forward model
├── simulation.py  truth generation and in-process antenna backend
├── types.py       public observations, contexts, estimates and configuration types
└── cli.py         command-line entry points and report writers

docs/              architecture, codebase, dependencies, reports, provenance
reports/           production, experimental, reference, and archived evidence bundles
tests/             deterministic unit and integration tests
deprecated/        read-only legacy implementations and provenance
skillset/          local analysis references; not part of the runtime package
```

The detailed module-by-module description is in [docs/codebase.md](docs/codebase.md).

## Commands

| Command | Purpose |
| --- | --- |
| `uv run dart simulate` | Closed-loop dither acquisition and tracking simulation. |
| `uv run dart replay-forest` | Sequential real Doppler replay and GPS scoring. |
| `uv run dart inventory-forest` | Complete raw/filter/eligibility contact inventory. |
| `uv run dart simulate-windows` | Paired Doppler and idealized complete-phase trials on recorded windows. |
| `uv run dart model-ablation` | Held-out scalar-offset versus mean-element comparison. |

Run `uv run dart <command> --help` for complete options.

### Real Doppler replay

```bash
uv run dart replay-forest \
  --data-dir deprecated/dart-v1/data \
  --raw-gps-dir deprecated/dart-v1/data/Ororatech-HFS-GNSS-data-raw \
  --satellites 16 17 18 19 \
  --output reports/reference/forest_replay_audit.json \
  --batch-report-output reports/production/doppler_batch_ls.json \
  --ukf-report-output reports/experimental/ukf/ukf_replay.json
```

### Experimental Henault-style analysis

```bash
uv run dart inventory-forest \
  --data-dir deprecated/dart-v1/data \
  --raw-gps-dir deprecated/dart-v1/data/Ororatech-HFS-GNSS-data-raw \
  --output reports/reference/observation_inventory.json

uv run dart simulate-windows \
  --data-dir deprecated/dart-v1/data \
  --tiers closure independent_dynamics \
  --seeds 0 \
  --output reports/experimental/henault_phase/window_simulation.json

uv run dart model-ablation \
  --seeds 0 1 2 \
  --output reports/experimental/henault_phase/model_ablation.json
```

The [reports guide](docs/reports.md) contains the empirical-residual command, report status rules, and the complete output map.

## Data and provenance

Bulk FOREST telemetry and raw GNSS exports are external research inputs and are not included in the Python wheel. Commands accept alternate data roots. The current study expects the legacy data layout under `deprecated/dart-v1/data`.

Verify the recorded inputs with:

```bash
sha256sum -c reports/reference/data_manifest.sha256
```

FOREST antenna encoders are treated as control/visibility metadata, not independent angular observations. BESTXYZ evaluation uses the receiver measurement epoch embedded in the raw NovAtel data rather than packet arrival time.

## Conventions

- Positions and velocities are GCRF metres and metres/second internally.
- Station coordinates are geodetic degrees and metres above the ellipsoid.
- Doppler measurements are carrier-frequency offsets in hertz.
- Phase baselines are antenna 1 minus antenna 2 in local ENU metres.
- Observation timestamps are UTC measurement epochs.
- Time offset, transmitter-frequency bias, and phase bias are constant within the declared static-pass model; the UKF defaults to identity transition with `Q=0`.

## Validation boundaries

- Current FOREST results are retrospective and exploratory; the contacts have already been inspected.
- Historical replay validates estimation on recorded signals, not counterfactual acquisition or beam retention.
- Closure simulation uses estimator-identical shifted-TLE truth and is an implementation check.
- UKF covariance and NIS are not calibrated on real residuals, which are correlated and include model error.
- A scalar time offset cannot repair mean-motion, altitude, plane, drag, or cross-track errors.
- Mean-motion claims require separated fit/evaluation passes and observability diagnostics.

See [architecture.md](docs/architecture.md), the [production protocol](reports/production/protocol.md), and the [experimental phase protocol](reports/experimental/henault_phase/protocol.md) for the complete boundaries.

## Development

```bash
uv run pytest
uv run pytest --cov=dart
uv run python -m compileall -q src
```

Small deterministic fixtures belong under `tests/`. Large intermediate analysis products belong outside the package or under ignored artifact directories. Checked reports should include their configuration, seeds, commands, and input provenance.
