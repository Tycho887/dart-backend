# Python IO code map

The supported Python surface begins with `dart.io` for acquisition and
normalization, then passes model-independent contexts to
`dart.forward_models` and `dart.od`. See [DART IO](io.md) for provider APIs,
[Forward models](forward-models.md) for numerical evaluation, and
[Orbit determination](orbit-determination.md) for fitting.

```text
KOGS metadata ─┐
               ├─ dart.io.load ──► ContactMetadata + canonical Polars frame
ADX telemetry ─┘                                      │
                                                      └─► ForwardModelContext
recorded Parquet ── dart.io.parquet ──► canonical Polars frame

ForwardModelContext ── dart.forward_models ──► Rust residuals + Jacobian
ForwardModelContext + EphemerisMetadata ── dart.od ──► OptimizerOutput
```

## Package responsibilities

| File | Responsibility |
| --- | --- |
| `dart/io/contact.py` | Provider-independent contact, ephemeris, observation, and forward-context types. |
| `dart/io/kogs.py` | KOGS authentication, typed response parsing, metadata resolution, and scheduling endpoints. |
| `dart/io/adx.py` | ADX client construction, bounded raw queries, and canonical measurement naming. |
| `dart/io/measurement.py` | Canonical measurement columns, types, and validation. |
| `dart/io/load.py` | Async orchestration for complete passes and forward contexts. |
| `dart/io/parquet.py` | Recorded measurement discovery and canonical replay. |
| `dart/io/orbital.py` | Orbital contract validation and absolute offset writes. |
| `dart/io/ctrl_config.py` | Read-only spacecraft and system configuration. |
| `dart/io/meos.py` | Reviewed calibration lookup pending a live MEOS API. |
| `dart/forward_models.py` | Typed adapter for Rust SGP4 and full-state residual/Jacobian evaluation. |
| `dart/od/` | Prior resolution, optimizer validation, and SGP4/full-state least-squares fitting. |

Provider functions remain synchronous because the KOGS/Orbital clients and
ADX SDK are synchronous. `load.py` moves independent blocking calls to worker
threads and awaits them together.

## Boundary rules

- A pass is identified by its KOGS contact ID.
- `ContactMetadata` includes the raw TLE/OMM/OEM source through nested
  `EphemerisMetadata`; downstream modules decide how to use it.
- ADX frames are contact- and time-bounded but otherwise unfiltered.
- Empty data, identity mismatches, malformed metadata, and partial provider
  failures fail the complete load.
- Provider credentials are explicit inputs and never become result fields.
- TDM serialization, optimization, and controller policy do not belong in IO.

The older schema/codec/loaders/solver and analysis-script interfaces were
removed during the pre-1.0 refactor. Remaining downstream packages will be
migrated independently to this boundary.
