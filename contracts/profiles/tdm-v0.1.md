# DART CCSDS TDM profile v0.1

DART emits CCSDS TDM 2.0 KVN with one metadata/data segment per
`(contact, station, spacecraft)` tuple.

Required header fields:

- `CCSDS_TDM_VERS = 2.0`
- `CREATION_DATE` in UTC
- `ORIGINATOR = DART`

Required segment metadata:

- `TIME_SYSTEM = UTC`
- `PARTICIPANT_1` is the receiving station
- `PARTICIPANT_2` is the spacecraft
- `MODE = SEQUENTIAL`
- `PATH = 2,1`
- `TRACK_ID` is the source contact/pass identifier
- `START_TIME` and `STOP_TIME` use UTC measurement epochs

The supported observation is `RECEIVE_FREQ`, expressed in MHz as required by
TDM. ADX carrier-frequency offset is converted to this absolute quantity only
by adding a positive, explicitly supplied nominal carrier frequency. A request
without a usable nominal frequency is rejected rather than emitted under an
incorrect CCSDS keyword.

Station ITRF coordinates, acquisition filters, ADX/KOGS provenance, raw and
presented counts, and per-sample operational metadata are carried in the DART
dataset envelope. Future phase observables require a separate reviewed profile;
they must not be represented as a superficially similar CCSDS observable.
