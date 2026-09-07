# DART IO

`dart.io` is DART's boundary for external data. It provides typed metadata,
canonical measurement frames, direct provider calls, and two asynchronous
loading workflows. It deliberately does not choose an orbit model, configure
an optimizer, serialize a TDM, or apply controller policy.

Use the highest-level function that fits the task:

| Need | Interface |
| --- | --- |
| Complete metadata and measurements for one or more passes | `load_passes` |
| A Doppler-ready shared forward-model input | `load_forward_context` |
| One KOGS resource or scheduling operation | `dart.io.kogs` |
| Raw selected ADX columns | `adx.fetch_columns` |
| Canonical ADX measurements for known metadata | `adx.fetch_measurements` |
| Recorded measurements | `parquet.read_measurements` |
| Control-config values | `dart.io.ctrl_config` |
| Reviewed station calibration | `dart.io.meos` |
| A confirmed absolute offset write | `orbital.write_offset` |

The package root re-exports `ContactMetadata`, `EphemerisMetadata`,
`ForwardObservation`, `ForwardModelContext`, `MeasurementKind`, `LoadError`,
`load_passes`, and `load_forward_context`. Provider-specific records and
functions remain under their named modules so imports show which external
system is being used.

## Loading passes

The normal entry point is:

```python
async def load_passes(
    contact_ids: Sequence[str],
    *,
    kogs_api_key: str,
    adx_client: KustoClient,
    timeout_seconds: float = 30.0,
) -> tuple[list[ContactMetadata], pl.DataFrame]
```

It resolves each contact, antenna, spacecraft, and ephemeris through KOGS,
then queries ADX using that contact's exact start and stop times. Independent
blocking calls run concurrently without making the provider modules
themselves asynchronous.

```python
import asyncio

from dart.io import load_passes
from dart.io.adx import client_from_env
from dart.io.kogs import api_key_from_env


async def main() -> None:
    with client_from_env() as adx_client:
        contacts, measurements = await load_passes(
            ["first-contact", "second-contact"],
            kogs_api_key=api_key_from_env(),
            adx_client=adx_client,
        )

    first_tle = contacts[0].ephemeris.tle
    first_pass = measurements.filter(
        measurements["contact_id"] == contacts[0].contact_id
    )
```

Contact metadata preserves the input order. The combined measurement frame is
sorted by timestamp. Contact IDs must be non-empty and unique. Missing
metadata, malformed identities, empty telemetry, or a failure for any one
contact raises `LoadError`; partial results are not returned.

`load_forward_context` calls `load_passes` and converts its Doppler rows into a
`ForwardModelContext`:

```python
context = await load_forward_context(
    ["contact-id"],
    kogs_api_key=api_key,
    adx_client=adx_client,
    center_frequency_hz=2_269_750_000.0,
    doppler_variance_hz2=25.0,
)
```

The center frequency and measurement variance are mandatory because IO must
not infer model or estimator policy. Non-Doppler forward-context loading is a
future extension; `load_passes` itself remains propagation-model agnostic.

## Common data contracts

The types below live in `dart.io.contact` and are re-exported by `dart.io`.

### `EphemerisMetadata`

Preserves the KOGS source-orbit record without selecting a propagator. It
contains the ephemeris and spacecraft IDs, kind, origin, tenant, UTC source
timestamps, submitter, CUI marker, serialized payload, and optional raw TLE,
OMM, and OEM text. Missing OMM or OEM is an expected state. Downstream low- and
high-fidelity modules choose the appropriate source; IO does not convert it to
a model input.

### `ContactMetadata`

Represents one pass. It contains:

- contact, spacecraft, system, station, and ephemeris IDs;
- antenna, station-location, spacecraft, COSPAR, and catalog names/identities;
- contact start and stop as aware UTC datetimes;
- WGS-84 latitude/longitude in degrees and altitude in metres;
- ITRF ECEF coordinates in metres; and
- the complete nested `EphemerisMetadata`.

`to_itrfcoord()` converts the station coordinates to `satkit.itrfcoord` for a
forward model. No API access occurs in this class.

### `MeasurementKind` and `ForwardObservation`

`MeasurementKind` identifies Doppler, true-range, and pseudorange-phase
observations. `ForwardObservation` carries a satkit time, observed vector,
native-unit covariance, receiver index, pass index, kind, and optional contact
ID. `ForwardObservation.from_scalar(...)` validates finite values and positive
variance while accepting a satkit time, aware datetime, or Unix timestamp.

### `ForwardModelContext`

Groups the center frequency, unique ITRF receivers, contact/pass mappings, and
observations required by shared forward models. Its methods are:

- `register_contact(contact) -> (receiver_id, pass_index)`: deduplicates
  receivers by `system_id` and indexes passes by their real `contact_id`.
- `add_observation(...)`: adds a scalar observation using registered system
  and contact identities.
- `validate()`: checks frequency, receiver/observation presence, and index
  bounds before numerical code is called.
- `num_passes`: number of registered contact IDs.

This structure is model-independent. Model selection, state initialization,
parameter bounds, and optimizer options belong in downstream modules.

### `LoadError`

Identifies the failed `provider` and `contact_id`. The original exception is
retained as its cause. It represents external loading failures; invalid caller
arguments raise `ValueError` directly.

## Canonical measurement frame

`dart.io.measurement.MEASUREMENT_COLUMNS` defines the public column order:

| Column | Type/unit | Meaning |
| --- | --- | --- |
| `timestamp` | UTC datetime, microsecond resolution | Measurement epoch |
| `contact_id` | string | KOGS pass/contact identity |
| `spacecraft_id` | string | KOGS spacecraft identity |
| `system_id` | string | KOGS antenna-system identity |
| `antenna_name` | string, nullable | Source antenna name |
| `tracking_epoch_offset_s` | float seconds, nullable | Reported tracking epoch offset |
| `azimuth_deg` | float degrees, nullable | Reported antenna azimuth |
| `elevation_deg` | float degrees, nullable | Reported antenna elevation |
| `carrier_lock` | string, nullable | Receiver lock state |
| `ebn0` | float, nullable | Provider-reported Eb/N0 value |
| `doppler_hz` | float hertz, nullable | Measured carrier-frequency offset |

`canonical_measurements(frame)` selects and types these columns, converts the
timestamp to UTC, validates required identity columns, and sorts by timestamp.
It does not drop rows or apply quality thresholds. Filtering by elevation,
lock, Doppler magnitude, Eb/N0, or sample count belongs to the consumer.

## Provider modules

### `dart.io.kogs`

KOGS credentials and payload interpretation live only in this module.

Data classes:

- `Contact`: scheduling/contact identity, interval, state, and duration.
- `Antenna`: antenna and station identity, coordinates, and supported bands.
- `Spacecraft`: spacecraft identity, name, and normalized catalog number.
- `GroundStation`: station identity and WGS-84 coordinates.
- `BookingPlan`: protocol accepted by `book_shadow`.
- `KogsError`: invalid payload, identity, or unsafe-mutation error.

Read and authentication functions:

- `api_key_from_env(variable="KOGS_API_KEY")`: reads and validates one key.
- `headers(api_key, json_content=False)`: creates canonical KOGS headers.
- `validate_credentials(...)`: performs a minimal authenticated contact query.
- `get_contact(...)`, `get_spacecraft(...)`, `get_antenna(...)`,
  `get_station(...)`, and `get_ephemeris(...)`: fetch one typed resource.
- `load_contact_metadata(...)`: combines and cross-checks all resources needed
  for one self-contained `ContactMetadata`.
- `list_contacts(...)`: returns typed contacts in an aware UTC interval with
  optional station/system filters.

Mutation functions:

- `book_shadow(...)`
- `assign_ephemeris(...)`
- `cancel_contact(...)`

Every mutation defaults to `mutation_contract_confirmed=False` and fails
before network access. Set it to true only after the deployed endpoint,
payload, and compensation behavior have been operationally confirmed.

### `dart.io.adx`

- `get_client(endpoint, client_id, client_secret, tenant_id, proxy="")` builds
  a client from explicit credentials.
- `client_from_env()` uses `AZURE_ADX_CLUSTER_ENDPOINT`, `AZURE_CLIENT_ID`,
  `AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID`, and optional `HTTP_PROXY`.
- `fetch_columns(...)` performs a safe, contact- and UTC-time-bounded query for
  an explicit tuple of Kusto columns. Use it for source-specific consumers
  such as future TDM builders that need columns outside the common schema.
- `fetch_measurements(client, contact, ...)` selects the standard tracking
  columns and returns a canonical frame. It deliberately applies no quality
  gates.

The caller owns the passed `KustoClient` and should close it after use.

### `dart.io.parquet`

- `files(source)` resolves one file, directory, or glob into sorted Parquet
  paths and rejects an empty match.
- `read_measurements(source)` concatenates the files, recognizes canonical or
  legacy ADX column names, and returns the same canonical frame as ADX.

Use this module for deterministic replay and fixtures, not for constructing
optimizer-specific inputs.

### `dart.io.ctrl_config`

- `spacecraft_config_path(...)` and `system_config_path(...)` construct safe
  paths below a supplied ctrl-config root.
- `has_spacecraft_config(...)` and `has_system_config(...)` test for those
  files.
- `qradio_rest_url(...)` reads the configured qradio REST endpoint.
- `get_link_frequency(...)` returns a validated positive frequency for an
  exact link and direction.
- `get_observed_frequency(...)` selects the conventional primary S-band
  downlink.

These functions read deployment configuration only. They do not contact a
remote API or choose an optimizer parameterization.

### `dart.io.meos`

`TrackCalibration` stores a reviewed pedestal offset, TLT calibration date,
and Doppler correction. `get_track_calibration(...)` retrieves a value from
`TRACK_CALIBRATIONS` and fails closed if no reviewed entry exists or its date
is later than the contact. This is a temporary local data source until a live
MEOS integration exists.

### `dart.io.orbital`

- `OrbitalContract` describes a confirmed endpoint, timeout, sign convention,
  idempotency header, and acknowledgement fields. `validate()` fails closed
  unless absolute-offset semantics have been explicitly confirmed.
- `write_offset(contract, command_id, target_offset_s)` sends an absolute
  target and requires a correlated JSON acknowledgement.
- `OffsetAcknowledgement` contains the command ID, acknowledgement time,
  accepted flag, and response detail.
- `OrbitalError` reports malformed or uncorrelated acknowledgements.

The Orbital module is transport only. Authorization timing, safety gates,
retry policy, and decisions about whether to command remain controller
responsibilities.

## Ownership and extension rules

- Add an external provider in one correspondingly named module.
- Keep provider functions synchronous and explicit about credentials/clients.
- Put multi-provider async workflows only in `load.py`.
- Add common measurement columns in `measurement.py` and update ADX/Parquet
  parity tests together.
- Preserve raw source-orbit content and provenance in metadata.
- Never add optimizer, propagation, TDM rendering, or controller policy to IO.
