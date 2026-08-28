# KSAT TRACK and ANGLE export

DART can write KSAT-profile CCSDS 503.0-B-2 `TRACK` mode-4 and `ANGLE` AZEL
files. Range modes 1 and 3 remain unavailable until an authoritative raw
round-trip delay source is selected; carrier-phase mode 2 is not defined by
the supplied KSAT profile. XEYN and XSYE angles remain unavailable until their
site-specific controller mappings are confirmed.

The implementation is intentionally narrow:

- `dart.tdm.ranging` owns the typed request, validation, KVN rendering, standard
  filename, and optional file write.
- `dart.tdm.angle` provides the equivalent narrow path for one AZEL contact.
- `dart.io.kogs` supplies the contact, antenna, spacecraft, ephemeris, and exact
  named link frequencies from `ctrl-config/v2/spacecrafts`.
- `dart.io.azure.fetch_contact_columns` performs the raw, one-contact query
  bounded to the KOGS reservation interval. It applies no solver filters.
- `dart.io.meos` is the fail-closed source for reviewed pedestal, TLT, and
  Doppler-correction constants. Missing calibration blocks TRACK, but ANGLE
  emits `UNKNOWN` calibration comments and structured warnings.

The receive frequency is the selected downlink center frequency plus the
explicitly configured ADX offset and sign. The transmit frequency is the
selected uplink frequency, optionally adjusted by its own mapped offset.
Neither mapping is inferred from column names. The integration-end column,
station column, offset columns, units, and signs are configurable for differing
antenna telemetry schemas.

## Python API

```python
from dart.tdm.ranging import FrequencySource, TrackRequest, write_track_tdm

request = TrackRequest(
    contact_id="CONTACT_UUID",
    band="S",
    integration_interval_s=1.0,
    turnaround_numerator=240,
    turnaround_denominator=221,
    transmit=FrequencySource("s_band_uplink_p1_1"),
    receive=FrequencySource(
        "s_band_downlink_p1_1",
        "lr1_receiver1_actualCarrierFrequencyOffset",
    ),
)
result = write_track_tdm(request, "delivery")
print(result.path)
```

ANGLE uses controller pointing readback without applying any model or
conversion:

```python
from dart.tdm.angle import AngleRequest, write_angle_tdm

result = write_angle_tdm(
    AngleRequest(
        contact_id="CONTACT_UUID",
        band="S",
        tracking_mode="PROGRAM",
    ),
    "delivery",
)
for warning in result.warnings:
    print(warning)
```

The default `antenna1_position_azimuth` and
`antenna1_position_elevation` mappings match current DART telemetry but have
not been confirmed as controller readback for every site. Every ANGLE result
therefore reports that confirmation warning. Tracking mode is a required
contact-wide input because no authoritative ADX mode mapping has been selected.

`KOGS_API_KEY` and the existing Azure ADX environment variables provide
credentials. Populate `dart.io.meos.TRACK_CALIBRATIONS` only with reviewed
station/band constants; an absent entry prevents TRACK export while ANGLE
reports warnings and writes `UNKNOWN` calibration comments.

## Command line

```bash
uv run python scripts/write_tdm.py \
  track \
  --contact-id CONTACT_UUID \
  --band S \
  --integration-interval 1 \
  --turnaround-numerator 240 \
  --turnaround-denominator 221 \
  --uplink-link s_band_uplink_p1_1 \
  --downlink-link s_band_downlink_p1_1 \
  --output-dir delivery
```

ANGLE export uses its own subcommand and does not accept TRACK parameters:

```bash
uv run python scripts/write_tdm.py \
  angle \
  --contact-id CONTACT_UUID \
  --band S \
  --tracking-mode PROGRAM \
  --output-dir delivery
```

Use the source-column, unit, and sign options when an antenna differs from the
default receiver-1 offset mapping. The configured timestamp must represent the
end of the integration interval. ANGLE additionally accepts
`--timestamp-column`, `--angle-1-column`, and `--angle-2-column`; overriding
them does not remove the requirement for site-owner confirmation.
