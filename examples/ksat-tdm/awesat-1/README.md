# AWESAT-1 KSAT ANGLE examples

These files demonstrate `scripts/write_tdm.py` with live ADX telemetry for
AWESAT-1:

- KOGS spacecraft UUID: `2cd1ce1c-3090-4a5f-b621-2e651c872245`
- spacecraft name: `AWESAT-1`
- NORAD/catalog identifier: `60543`
- COSPAR identifier: `2024-149CD` (derived from the contact ephemeris)
- contact: `802cbf44-8597-461a-865b-9bca3ac7cb51`
- antenna: `SG221` at SVALSAT
- KOGS station UUID: `381776b6-0867-4e9d-8589-0fcd76ee565f`
- service: S-band up/down

The examples were generated on 2026-08-26. They contain real controller
pointing telemetry, but they are examples rather than approved customer
deliveries. Confirm the site mappings and header metadata with the site owner
before operational use.

Equivalent ANGLE configurations for SG162, SG182, and SG184 are included
alongside SG221. Together they cover four S-band contacts found in the
requested interval from `2025-11-01T21:59:51Z` through
`2025-11-05T21:59:51Z`.

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

The same restriction applies to the four November 2025 contacts below. ADX
contained no non-null `lr1_ranging_satRange` or `lr2_ranging_satRange`
measurements. The direct modem carrier-frequency fields were `0.0`. Nonzero
S-band converter references and Doppler/carrier offsets were present on sparse
rows sharing an exact timestamp, and the mode-4 diagnostic path assembled 507,
642, 571, and 685 complete frequency epochs for SG162, SG221, SG184, and SG182.
Those values are not an operational TRACK source until a site owner confirms
the absolute-frequency formulas, signs, coherent turnaround, and integration-
end semantics. The pedestal offset and applicable TLT calibration also remain
unavailable, so no TRACK file was written.

## Generated files

All windows are subsets of the same contact and stay inside its continuous
programmed-tracking interval.

| Requested window (UTC) | Observations | Checked-in output |
| --- | ---: | --- |
| `11:36:00`–`11:36:30` | 30 | [`output/30-second/ANGLE_SG221_2024-149CD_2026-08-26T10-23-05.tdm`](output/30-second/ANGLE_SG221_2024-149CD_2026-08-26T10-23-05.tdm) |
| `11:38:00`–`11:40:00` | 120 | [`output/2-minute/ANGLE_SG221_2024-149CD_2026-08-26T10-25-25.tdm`](output/2-minute/ANGLE_SG221_2024-149CD_2026-08-26T10-25-25.tdm) |
| `11:34:02`–`11:45:35` | 693 | [`output/full-program-track/ANGLE_SG221_2024-149CD_2026-08-26T10-25-28.tdm`](output/full-program-track/ANGLE_SG221_2024-149CD_2026-08-26T10-25-28.tdm) |

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

## November 2025 requested contacts

Each window starts after the contact's single initial `Idle` angle sample and
contains only paired azimuth/elevation samples reported as `Programtrack`.
S-band operation was confirmed from the active S-band converter frequencies.

| Antenna | Contact UUID | Program-tracking window (UTC) | Paired angles |
| --- | --- | --- | ---: |
| SG162 | `bcaf75af-00bf-4aa8-80fb-f830689001ae` | `2025-11-02T06:54:19Z`–`07:03:10Z` | 507 |
| SG221 | `d7c6b504-c929-4255-a6ce-8fb948eececc` | `2025-11-02T14:54:25Z`–`15:05:37Z` | 642 |
| SG184 | `ecf8c77f-2330-4f4d-81fb-96f051485874` | `2025-11-03T23:06:17Z`–`23:15:48Z` | 571 |
| SG182 | `050c7e87-83ca-47e5-bc4c-0d55bf51abd9` | `2025-11-04T13:40:02Z`–`13:51:27Z` | 685 |

Generate one file per contact after loading a valid `KOGS_API_KEY`:

```bash
uv run python scripts/write_tdm.py --config examples/ksat-tdm/awesat-1/sg162-angle.toml --contact-id bcaf75af-00bf-4aa8-80fb-f830689001ae --start 2025-11-02T06:54:19Z --stop 2025-11-02T07:03:10Z --output-dir examples/ksat-tdm/awesat-1/output/2025-11-requested/sg162
uv run python scripts/write_tdm.py --config examples/ksat-tdm/awesat-1/sg221-angle.toml --contact-id d7c6b504-c929-4255-a6ce-8fb948eececc --start 2025-11-02T14:54:25Z --stop 2025-11-02T15:05:37Z --output-dir examples/ksat-tdm/awesat-1/output/2025-11-requested/sg221
uv run python scripts/write_tdm.py --config examples/ksat-tdm/awesat-1/sg184-angle.toml --contact-id ecf8c77f-2330-4f4d-81fb-96f051485874 --start 2025-11-03T23:06:17Z --stop 2025-11-03T23:15:48Z --output-dir examples/ksat-tdm/awesat-1/output/2025-11-requested/sg184
uv run python scripts/write_tdm.py --config examples/ksat-tdm/awesat-1/sg182-angle.toml --contact-id 050c7e87-83ca-47e5-bc4c-0d55bf51abd9 --start 2025-11-04T13:40:02Z --stop 2025-11-04T13:51:27Z --output-dir examples/ksat-tdm/awesat-1/output/2025-11-requested/sg182
```

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
