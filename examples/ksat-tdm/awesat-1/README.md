# AWESAT-1 KSAT ANGLE examples

These files demonstrate `scripts/write_tdm.py` with live ADX telemetry for
AWESAT-1:

- KOGS spacecraft UUID: `2cd1ce1c-3090-4a5f-b621-2e651c872245`
- spacecraft name: `AWESAT-1`
- NORAD/catalog identifier: `60543`
- contact: `802cbf44-8597-461a-865b-9bca3ac7cb51`
- antenna: `SG221` at SVALSAT
- KOGS station UUID: `381776b6-0867-4e9d-8589-0fcd76ee565f`
- service: S-band up/down

The examples were generated on 2026-08-26. They contain real controller
pointing telemetry, but they are examples rather than approved customer
deliveries. Confirm the site mappings and header metadata with the site owner
before operational use.

## Why these examples contain ANGLE only

ADX contained many AWESAT-1 contacts. Its frequent SG109 contacts used L-band,
which the documented KSAT TDM profile does not support. The selected SG221
contact used S-band and contained controller azimuth/elevation readback while
`antenna1_tracking_status` was `Programtrack`.

TRACK was not emitted: this contact had no range measurements, and its absolute
transmit/receive frequency semantics and calibration terms were not confirmed.
SIGMET was not emitted because no source was confirmed as KSAT `CARRIER_POWER`,
`PC_N0`, or `PR_N0`. In particular, the populated Eb/N0 signal is not a valid
substitute.

## Generated files

All windows are subsets of the same contact and stay inside its continuous
programmed-tracking interval.

| Requested window (UTC) | Observations | Checked-in output |
| --- | ---: | --- |
| `11:36:00`–`11:36:30` | 30 | [`output/30-second/ANGLE_SG221_60543_2026-08-26T10-23-05.tdm`](output/30-second/ANGLE_SG221_60543_2026-08-26T10-23-05.tdm) |
| `11:38:00`–`11:40:00` | 120 | [`output/2-minute/ANGLE_SG221_60543_2026-08-26T10-25-25.tdm`](output/2-minute/ANGLE_SG221_60543_2026-08-26T10-25-25.tdm) |
| `11:34:02`–`11:45:35` | 693 | [`output/full-program-track/ANGLE_SG221_60543_2026-08-26T10-25-28.tdm`](output/full-program-track/ANGLE_SG221_60543_2026-08-26T10-25-28.tdm) |

The timestamp in each filename is its `CREATION_DATE`, not the observation
window. A rerun therefore normally creates a differently named file.

## Prerequisites

Install the project and expose the existing ADX credentials:

```bash
uv sync
set -a
source /opt/dart/secrets/test.env
set +a
```

Equivalently, export `KOGS_API_KEY`, `AZURE_ADX_CLUSTER_ENDPOINT`,
`AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, and `AZURE_TENANT_ID`. `HTTP_PROXY`
is optional. The CLI validates the contact and obtains antenna metadata from
KOGS; the TOML file contains expected resource UUIDs but no credentials.

## Reproduce the examples

Run these commands from the repository root:

```bash
uv run python scripts/write_tdm.py \
  --config examples/ksat-tdm/awesat-1/sg221-angle.toml \
  --contact-id 802cbf44-8597-461a-865b-9bca3ac7cb51 \
  --start 2026-08-25T11:36:00Z \
  --stop 2026-08-25T11:36:30Z \
  --output-dir /tmp/awesat-angle-30s

uv run python scripts/write_tdm.py \
  --config examples/ksat-tdm/awesat-1/sg221-angle.toml \
  --contact-id 802cbf44-8597-461a-865b-9bca3ac7cb51 \
  --start 2026-08-25T11:38:00Z \
  --stop 2026-08-25T11:40:00Z \
  --output-dir /tmp/awesat-angle-2m

uv run python scripts/write_tdm.py \
  --config examples/ksat-tdm/awesat-1/sg221-angle.toml \
  --contact-id 802cbf44-8597-461a-865b-9bca3ac7cb51 \
  --start 2026-08-25T11:34:02Z \
  --stop 2026-08-25T11:45:35Z \
  --output-dir /tmp/awesat-angle-full
```

The timestamps must include `Z` or an explicit UTC offset. A successful run
prints the generated path and exits 0. A configuration/query failure or a
window with no usable ANGLE rows exits 1. Use `--overwrite` only when replacing
the exact standard filename already present in the selected output directory.

## Find another contact for the spacecraft

The CLI intentionally exports exactly one contact; the spacecraft UUID is used
to discover a contact first. In ADX, use a bounded query such as:

```kusto
contacts
| where timestamp between (
    datetime(2026-08-20T00:00:00Z) .. datetime(2026-08-27T00:00:00Z)
  )
| where spacecraft_id == "2cd1ce1c-3090-4a5f-b621-2e651c872245"
| summarize
    start_time=min(timestamp),
    stop_time=max(timestamp),
    rows=count()
  by contact_id, system_id, antenna_name
| order by start_time asc
```

Then verify the contact's mission band, site identity, available observables,
and tracking mode. Update the TOML `[kogs]` identities when selecting another
antenna. Do not combine contacts or stations in one TDM export.

For this configuration, choose only a window known to contain programmed
tracking. It declares `tracking_mode = "PROGRAM"` because ADX uses the
site-specific value `Programtrack`; including idle rows would mislabel them.

## Offline validation

The checked-in files can be validated without credentials or network access:

```bash
uv run pytest tests/test_ksat_examples.py -q
```

That test checks ASCII encoding, standard filenames, header/participant fields,
balanced metadata and data blocks, paired ANGLE observations, chronological
epochs, requested time bounds, and the expected observation counts.
