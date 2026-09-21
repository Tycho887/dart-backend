# DART

DART is a passive-RF processing toolkit with three pillars:

- **Orbit determination:** estimate time offsets, correct SGP4/TLE orbits, and
  fit six-degree-of-freedom Cartesian states from tracking data.
- **Real-time control:** estimate and validate antenna time-offset corrections,
  then pass approved targets to an antenna's existing open-loop controller.
- **Data aggregation and interchange:** normalize provider data, preserve
  provenance, and produce tracking deliveries and derived orbit products.

The current core interfaces are `dart.io` for typed provider access,
`dart.forward_models` for the Rust numerical core, and `dart.od` for batch
fitting. Integration of some older consumers into these interfaces is ongoing.

## Features and development status

**Implemented** means the capability exists in the current code and interfaces;
it does not establish operational certification or deployment readiness.
**Partial integration** means components exist but need migration or runtime
wiring. **Planned** describes capabilities that are not currently available.

| Feature | Status | Current capability and limits |
| --- | --- | --- |
| [Provider access and replay](docs/io.md) | Implemented | Typed KOGS/ADX access, contact metadata, normalization, and Parquet replay. Live sources require credentials and configuration; MEOS uses reviewed calibration data pending live integration. |
| [Shared numerical models](docs/forward-models.md) | Implemented | Rust/satkit SGP4 and Cartesian propagation, Doppler predictions, whitened residuals, and Jacobians. |
| [Time-offset estimation](docs/orbit-determination.md) | Implemented | Batch fitting of measurement time offsets or physical TLE epoch corrections, with optional RF nuisance terms. These corrections have different meanings. |
| [SGP4/TLE correction](docs/orbit-determination.md) | Implemented | Bounded fits of selected mean-element corrections, with optional B*, timing, and frequency parameters. |
| [Cartesian full-state estimation](docs/orbit-determination.md) | Implemented | Six-component state fitting with high-precision propagation, using a supplied GCRF state or a TLE-derived prior. The OD wrapper still requires source TLE metadata. |
| [Classical consider covariance analysis (CCA)](docs/orbit-determination.md#consider-covariance) | Implemented | Rust covariance analysis, automatic calculation for eligible successful fits, and version-2 service profile integration. See qualifications below. |
| [Estimate service and Grafana](docs/async-api.md) | Implemented | Queued batch fits, separate worker, durable results/provenance in PostgreSQL/TimescaleDB, and Grafana submission and inspection. |
| [UKF offset/drift monitoring](dart/controller/UKF.py) | Partial integration | Tested two-state Python filter; consumes offset measurements, with no integrated live telemetry-to-estimator runner. |
| [Acquisition and automatic offset control](dart/controller/) | Partial integration | Tested acquisition and guarded PI controller components plus a separate Orbital transport. Live runtime wiring and durable controller audit integration remain. |
| [KSAT tracking TDM](docs/ksat-tdm-export.md) | Partial integration | TRACK mode-4 and ANGLE AZEL exporters exist, but retain obsolete IO imports. Current end-to-end export requires migration. |
| [Shadow-contact scheduling](dart/shadow_scheduler.py) | Partial integration | Planning, booking, and compensation code exists but retains obsolete IO dependencies. Booking a shadow contact is separate from controller dry-run operation. |
| [Orbit products and OEM](docs/orbit-determination.md) | Implemented | Typed fitted/prior orbits, Rust-backed state sampling, and CCSDS OEM reading/writing through `dart.io.oem`. |
| [Benchmarking and GPS reference workflows](docs/benchmark.md) | Implemented | Reference-orbit comparisons, frozen-input replay, and export; GMAT/GPS studies require their external runtime and recorded inputs. |

### UKF and automatic control

The current `TimeOffsetUKF` estimates **absolute offset and drift** from supplied
offset measurements and variances. It includes irregular-cadence prediction,
innovation gating, convergence checks, and resets on identity changes or
excessive gaps. It is a Python component, not yet the proposed Rust estimator
driven directly by Doppler. The four-state offset/drift/Doppler-bias/frequency
filter in the architecture roadmap remains **planned**.

The acquisition component searches for a carrier-lock window using confirmed
offsets. The controller combines the UKF target with slow PI feedback and
guards for identity, freshness, uncertainty, innovation, command magnitude,
and slew. It supports dry-run decisions, explicitly authorized live writes,
one outstanding command awaiting fresh ADX readback, and confirmed zero-reset
handling. An HTTP acknowledgement alone does not confirm application.

These pieces have [UKF](tests/test_controller_ukf.py),
[acquisition](tests/test_controller_acquisition.py), and
[control](tests/test_offset_controller.py) tests. Completing live operation
still requires telemetry-to-offset estimation, wiring the components to the
[Orbital adapter](dart/io/orbital.py), a separate controller runtime, durable
audit storage, and operational shadow validation. The batch worker does not
run this loop.

### CCA and uncertainty

CCA accounts for uncertainty in parameters held at their configured initial
values while other parameters are estimated. Its numerical implementation is
in Rust; Python provides validation and parameter-role mapping. It is already
connected to OD and service results, rather than being a standalone prototype.

Successful fits automatically receive covariance when every configured
parameter has a prior standard uncertainty and none has the `FIXED` role.
Version-2 service profiles provide these priors and persist the covariance,
rank, method, and parameter uncertainties. Version-1 profiles retain their
historical behavior without automatic covariance. CCA does not change the
point fit or optimize the considered parameters.

For robust-loss fits, CCA uses the final whitened Jacobian **without robust-loss
weights**; it is a classical linearized analysis, not a robust sandwich
covariance. See the [CCA contract](docs/orbit-determination.md#consider-covariance)
and [behavior tests](tests/test_cca.py).

### Remaining tracking-product work

TDM migration must replace references to removed IO modules before the export
workflow can be advertised as usable through the current interfaces. TRACK
range modes 1 and 3 await an authoritative raw delay source; mode 2 is unsupported
by the supplied KSAT profile. Other angle types await confirmed site mappings,
and SIGMET export is not implemented. Raw tracking deliveries remain distinct
from legacy solver diagnostics and derived OEM orbit products.

## IO quick start

```python
import asyncio

from dart.io import load_passes
from dart.io.adx import client_from_env
from dart.io.kogs import api_key_from_env


async def main() -> None:
    contacts, measurements = await load_passes(
        ["contact-uuid"],
        kogs_api_key=api_key_from_env(),
        adx_client=client_from_env(),
    )
    print(contacts[0])
    print(measurements)


asyncio.run(main())
```

`load_passes` resolves each KOGS contact, antenna, spacecraft, and ephemeris,
then retrieves the contact-bounded ADX measurements. It returns metadata in
the requested order and one timestamp-sorted Polars DataFrame. It does not
apply optimizer filters or select a high- or low-fidelity model.

The canonical measurement columns are:

```text
timestamp, contact_id, spacecraft_id, system_id, antenna_name,
tracking_epoch_offset_s, azimuth_deg, elevation_deg, carrier_lock,
ebn0, doppler_hz
```

For callers of the shared forward-model interface,
`dart.io.load_forward_context` converts this result into a
`ForwardModelContext` using explicit center-frequency and Doppler-variance
arguments.

## Forward models

`dart.forward_models` exposes the authoritative Rust SGP4 and Cartesian
full-state Doppler models to Python. Both return whitened residual and Jacobian
NumPy arrays that can be passed directly to SciPy `least_squares` or consumed
by another estimator. See [Forward models](docs/forward-models.md) for the
equations, parameter ordering, units, examples, and test strategy.

## Orbit determination

`dart.od.fit` runs bounded SciPy least squares around the Rust SGP4 or
Cartesian full-state evaluator. A full-state fit can use a supplied GCRF prior
or derive one from the selected source TLE. See [Orbit determination](docs/orbit-determination.md)
for the input contract, fallback behavior, and canonical parameters.

## Provider modules

The complete API and use-case reference is in [DART IO](docs/io.md).

- `dart.io.kogs`: KOGS authentication, typed reads, contact metadata, and
  guarded scheduling mutations.
- `dart.io.adx`: ADX clients, bounded raw queries, and canonical measurements.
- `dart.io.orbital`: fail-closed absolute time-offset writes.
- `dart.io.ctrl_config`: read-only ctrl-config access.
- `dart.io.meos`: reviewed calibration data pending a live MEOS integration.
- `dart.io.parquet`: canonical replay of recorded ADX measurements.
- `dart.io.load`: asynchronous multi-provider workflows.

Credentials are loaded only through provider helpers and are passed explicitly
to orchestration functions. They are never stored in returned metadata.

## Development

```bash
uv sync
uv run pytest
uv run ruff check dart/io tests
uv run ty check dart/io
cargo test --manifest-path crates/forward-models/Cargo.toml
```

See [the suite architecture](docs/dart-suite-architecture.md) for the target
module boundaries and proposed designs. Parts of its roadmap describe an older
implementation state, including full-state estimation and UKF work; use the
feature overview above and the current IO, forward-model, OD, and service
references for implemented behavior.
The [estimate service](docs/async-api.md) now connects the current OD interface
to TimescaleDB and Grafana: `scripts/start_results.sh` starts the local stack,
and `/d/dart-estimates/estimates` opens the result workflow on port 3001.
See [results stack operations](docs/results-stack-operations.md) for the
container lifecycle, protected configuration, local `.venv`, dependency
updates, rebuilds, migrations, and dashboard workflow.

## GPS orbit fitting and CCSDS OEM

For Doppler fitting against a reference OEM, use the single
[`benchmark` function](docs/benchmark.md). It accepts contact IDs, a selected
prior and optimizer, returns raw state/Doppler residual tables, and supports
frozen-input replay and CSV export. All current study runners, plotting tools,
and tests remain available. See [reference inventories](experiments/notes.md)
for recorded inputs and [the experiment archive](docs/experiment-archive.md)
for V5–V7 results and restoration instructions.

`uv run python scripts/smooth_gps.py run` fits the FOREST GPS observations with
GMAT over May 3–5, 2026 (noon UTC endpoints). It produces one-minute EME2000 OEMs
and independent withheld-GPS validation reports. See
[the GMAT workflow](scripts/gmat/README.md) for runtime installation, preparation,
resuming interrupted runs, and the output quality gates.

The recorded FOREST-16 through FOREST-19 OEMs and their quality reports are in
[the May 2026 reference data](reports/forest-gps/20260504/README.md).
