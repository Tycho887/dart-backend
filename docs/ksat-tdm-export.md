# KSAT TDM export service

The KSAT TDM export service produces CCSDS 503.0-B-2 KVN delivery files for
one KOGS contact and a bounded ADX time interval. It currently supports
`TRACK`, `ANGLE`, and `SIGMET`; `METEO` remains unavailable until its normative
KSAT definition and a weather source are available.

The service is an in-process Python orchestration API with a thin CLI, not a
long-running daemon or HTTP endpoint.

This is a delivery pipeline, not an orbit-determination pipeline. It reads raw
telemetry and authoritative contact metadata directly, then writes KSAT-profile
TDM files. It does not construct `Sgp4Input` or `Rk89Input`, invoke either
solver, use the MessagePack wire format, or pass through `dart.io.tdm`.

## Architecture and data flow

```text
scripts/write_tdm.py
        │
        ├── load_ksat_export_config() ── strict TOML validation
        │
        └── export_ksat_contact()
                │
                ├── KOGS contact ─────── verify contact, spacecraft,
                │   antenna/spacecraft   system, and station identities
                │          │
                │          ├── antenna WGS-84 ── satkit ── ITRF/ECEF metres
                │          └── WGS-84 ── reverse geocoder ── place name
                │
                ├── bounded ADX query ── Polars DataFrame
                │          │
                │          └── product builders and declared unit conversion
                │
                └── typed KSAT documents ── KVN renderer ── *.tdm files
```

The service deliberately performs KOGS validation before contacting ADX. A
contact can therefore never be exported with header metadata from a different
spacecraft, antenna system, or station. The ADX query still independently
requires exactly one contact, explicit UTC bounds, and one antenna identifier
matching `PARTICIPANT_1`.

The WGS-84 antenna coordinates come from KOGS. `satkit.itrfcoord` converts them
directly to ITRF/ECEF coordinates, rendered to three decimal metres. No
measurement epoch is needed because ITRF/ECEF is Earth-fixed. Reverse
geocoding supplies only the human-readable locality, region, and country; it
does not affect the coordinate values.

## Module responsibilities

| Component | Responsibility |
| --- | --- |
| `scripts/write_tdm.py` | Parses command-line arguments, calls the service, and reports generated, skipped, and warning results. It contains no KQL, TDM models, or filename logic. |
| `dart.io.ksat_export` | Loads strict TOML, discovers configured products, normalizes CLI selections, and orchestrates metadata enrichment plus the bounded export. |
| `dart.io.ksat_metadata` | Reads and validates KOGS metadata, derives ECEF coordinates with satkit, performs reverse geocoding, and constructs runtime site/spacecraft header values. |
| `dart.io.kogs` | Owns KOGS HTTP requests and payload normalization shared with the solver loaders. |
| `dart.io.ksat_adx` | Builds the validated ADX query, converts declared source units, constructs requested products, writes files, and returns structured result details. |
| `dart.io.ksat_tdm` | Defines the typed KSAT profile, validates metadata and observations, renders ASCII KVN, and creates standard filenames. |

The public orchestration interface is:

```python
from dart.io.ksat_export import (
    export_ksat_contact,
    load_ksat_export_config,
    parse_utc_datetime,
)

config = load_ksat_export_config("/path/to/ksat-tdm.toml")
result = export_ksat_contact(
    config,
    contact_id="CONTACT_UUID",
    start_time=parse_utc_datetime("2026-08-25T10:00:00Z"),
    stop_time=parse_utc_datetime("2026-08-25T10:15:00Z"),
    output_dir="/path/to/delivery",
    products=("angle",),  # omit to select every configured product
)

for generated in result.generated.values():
    print(generated.path)
for product, reason in result.skipped.items():
    print(f"{product} skipped: {reason}")
for warning in result.warnings:
    print(f"warning: {warning}")
```

Call `dart.io.ksat_tdm` directly when observations and authoritative header
metadata are already available in typed form. Call `dart.io.ksat_adx` directly
when a caller owns metadata enrichment itself. Use `dart.io.ksat_export` for
the normal configuration-backed KOGS plus ADX workflow.

## Configuration and metadata ownership

Start with [`config/ksat-tdm.example.toml`](../config/ksat-tdm.example.toml).
Unknown keys, missing required fields, incomplete mappings, invalid units, and
unsupported product metadata fail during configuration loading.

| TOML section | Purpose and authority |
| --- | --- |
| `[kogs]` | Required expected spacecraft, system, and station UUIDs, plus an optional KOGS timeout. These values constrain the selected contact; they are not header display names. |
| `[site]` | Reviewed facts not exposed by KOGS: optional distinct antenna name, pedestal offset, and paired TLT band/date. Location and coordinates are intentionally not configurable here. |
| `[spacecraft]` | TDM participant/filename identifier and optional COSPAR/catalog fallback. KOGS supplies the runtime common name and catalog value when available and mismatches are rejected. |
| `[geocoder]` | Optional reverse-geocoder URL, user-agent, and timeout. Defaults to the OpenStreetMap Nominatim reverse endpoint. |
| `[header]` | Optional delivery summary and additional validated ASCII comments. |
| `[adx]` | Source columns and source units. Its `station_id` column must contain the operational antenna identifier returned by KOGS, such as `SG221`, not the internal system UUID. |
| `[track]`, `[angle]`, `[sigmet]` | Product metadata and default product enablement. A product is selected by default when its section exists. |

Credentials never belong in TOML. The service reads `KOGS_API_KEY` and the
existing ADX variables `AZURE_ADX_CLUSTER_ENDPOINT`, `AZURE_CLIENT_ID`,
`AZURE_CLIENT_SECRET`, and `AZURE_TENANT_ID` from the environment or existing
`.env` behavior. `HTTP_PROXY` remains optional.

The CLI's `--timeout-seconds` controls the ADX request. KOGS and geocoder
timeouts are independently configured as `kogs.timeout_seconds` and
`geocoder.timeout_seconds`.

## Command-line operation

```bash
uv run python scripts/write_tdm.py \
  --config /path/to/ksat-tdm.toml \
  --contact-id CONTACT_UUID \
  --start 2026-08-25T10:00:00Z \
  --stop 2026-08-25T10:15:00Z \
  --output-dir /path/to/delivery
```

Both timestamps must be offset-aware and are normalized to UTC. Repeat
`--product track|angle|sigmet` to request a subset. Without `--product`, all
products having TOML sections are requested. Use `--overwrite` only to replace
an existing standard filename.

Files are written directly below `--output-dir` as
`<TYPE>_<GSID>_<SVID>_<CREATION_DATE>.tdm`. The ground-station identifier comes
from the validated KOGS antenna, while the spacecraft identifier comes from
the reviewed `[spacecraft]` configuration.

## Results and failure behavior

`KsatExportResult` separates three outcomes:

- `generated`: products successfully rendered, with filename, text, and output
  path;
- `skipped`: requested products that could not be constructed, keyed by
  product with a reason;
- `warnings`: non-product metadata degradation, such as an unavailable place
  name or missing reviewed calibration facts.

KOGS credentials, requests, identity mismatches, incomplete antenna
coordinates, invalid configuration, ADX failures, and unsafe overwrite
attempts are fatal. Reverse-geocoder failures are non-fatal: the header uses
`UNKNOWN` and reports a warning. Missing pedestal offset or TLT calibration
also produces normative `UNKNOWN` comments and warnings.

The CLI exits `0` when at least one requested product was written, including
partial success. It exits `1` for fatal configuration/runtime failures or when
no requested product was generated. Invalid command-line usage retains
argparse exit status `2`.

## Relationship to the solver pipeline

The KSAT delivery exporter and solver pipeline share backend clients and CCSDS
terminology, but their contracts are intentionally separate:

| KSAT delivery export | Solver pipeline |
| --- | --- |
| Reads one bounded ADX contact plus KOGS metadata. | Loaders construct versioned `Sgp4Input` or `Rk89Input`. |
| Preserves raw measurements and declared correction terms. | Fits orbit/time/frequency parameters. |
| Uses `dart.io.ksat_tdm` and KSAT participant conventions. | Uses MessagePack, Rust/Python solvers, and `dart.io.tdm`. |
| Writes TRACK, ANGLE, or SIGMET delivery products. | Writes diagnostic solver input/result TDM records. |
| Does not change or cross the Python/Rust schema boundary. | Depends on the mirrored Python/Rust schema contract. |

Do not route KSAT delivery products through `dart.io.tdm`: that writer is for
DART solver records, uses different participant semantics, and supports
DART-specific `USER_DEFINED_*` fields.

## Testing and extension points

The export tests are split along the same boundaries as the implementation:

- `tests/test_ksat_metadata.py`: KOGS identity checks, geocoder parsing and
  failure behavior, and satkit ECEF values;
- `tests/test_ksat_export.py`: TOML loading, product discovery, and orchestration;
- `tests/test_ksat_adx.py`: bounded KQL, conversions, builders, skips, and writes;
- `tests/test_ksat_tdm.py`: typed profile validation, filenames, and KVN output;
- `tests/test_write_tdm.py`: CLI forwarding, reporting, and exit statuses;
- `tests/test_ksat_examples.py`: offline validation of checked-in AWESAT-1 output.

When adding a telemetry field, keep backend column/unit conversion in
`ksat_adx` and serialization in `ksat_tdm`. When adding header metadata, define
its authority and fallback in `ksat_metadata`; do not infer it from unrelated
ADX fields. Adding a product requires a normative KSAT definition, a typed
document/serializer, a confirmed data source, strict configuration, and tests
at each boundary.
