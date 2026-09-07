# TDM deployment profiles

Place reviewed `.yaml` profiles in this directory or point
`DART_TDM_PROFILE_DIR` at a protected deployment directory. Profiles contain
no credentials and are immutable by `(name, version)` once used by a job.

TRACK mode 4 requires an integration-end timestamp, exact control-config link
names, an explicit receive-frequency offset mapping and sign, a standard
turnaround ratio, and reviewed calibration:

```yaml
name: reviewed-site-track
version: 1
product: track
station: REPLACE_WITH_KSAT_STATION
band: S
integration_interval_s: 1.0
turnaround_numerator: 240
turnaround_denominator: 221
integration_end_column: CONFIRM_INTEGRATION_END_COLUMN
transmit:
  link_name: CONFIRM_UPLINK_LINK
receive:
  link_name: CONFIRM_DOWNLINK_LINK
  offset_column: CONFIRM_ABSOLUTE_OFFSET_COLUMN
  offset_unit: Hz
  offset_sign: 1
calibration:
  pedestal_offset_m: 0.0
  tlt_calibration_date: 2026-01-01
  correction_doppler_hz: 0.0
```

ANGLE profiles must record that their two mapped columns are confirmed
controller readback rather than commands:

```yaml
name: reviewed-site-angle
version: 1
product: angle
station: REPLACE_WITH_KSAT_STATION
band: S
tracking_mode: PROGRAM
angle_1_column: CONFIRM_AZIMUTH_READBACK_COLUMN
angle_2_column: CONFIRM_ELEVATION_READBACK_COLUMN
controller_readback_confirmed: true
```

The examples are schemas, not operational values. Do not rename and enable
them without site-owner review.
