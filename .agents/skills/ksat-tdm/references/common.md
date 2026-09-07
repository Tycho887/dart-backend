# Common KSAT TDM profile

Source: `TDM-docs/Tracking Data Message Files _ Ranging.pdf`, pp. 1–3.

## Scope

KSAT uses CCSDS 503.0-B-2 TDM version 2.0 KVN. One track may produce up to four
files: `TRACK`, `ANGLE`, `SIGMET`, and `METEO`. A product is omitted when the
site cannot provide its data. The documented topology is one ground antenna,
one contact, one ground antenna, one spacecraft, and a two-way link; VLBI,
relay, bistatic, and other multi-site cases require a future extension.

## Filename

Use `<TYPE>_<GSID>_<SVID>_<DATE>.tdm`:

- `TYPE` is `TRACK`, `ANGLE`, `SIGMET`, or `METEO`.
- `GSID` is the KSAT station identifier.
- `SVID` is normally the COSPAR identifier.
- `DATE` equals `CREATION_DATE`, except colons in the time are replaced by
  dashes.
- Permit ASCII letters, digits, dash, and underscore; neither dash nor
  underscore may be the first character. Use a period only before `tdm`.
- Reject invalid identifier components instead of silently rewriting them;
  filename identifiers and `PARTICIPANT_*` values must remain consistent.

## Header and common metadata

- `CCSDS_TDM_VERS = 2.0` and `ORIGINATOR = KSAT`.
- `CREATION_DATE` is the UTC file-open time at one-second resolution and must
  match the filename timestamp.
- Header comments identify contents, KSAT site and location, WGS84 and ECEF
  antenna reference-point coordinates, pedestal offset `Lg`, latest TLT
  calibration date, and spacecraft common name/COSPAR/catalog identifiers.
  Write `UNKNOWN` for unavailable header-comment facts.
- DART derives COSPAR from the contact's KOGS TLE/OMM and cross-checks the KOGS
  catalog ID. Operational TRACK export requires both identifiers, pedestal
  offset, and applicable TLT calibration; ANGLE may retain `UNKNOWN`
  calibration comments with warnings.
- Every metadata segment contains `TIME_SYSTEM = UTC`.
- `PARTICIPANT_1` is the KSAT ground-station identifier.
- `PARTICIPANT_2` is normally the COSPAR ID; a 5- or 9-digit catalog ID is also
  allowed.

Use complete `META_START`/`META_STOP` and `DATA_START`/`DATA_STOP` blocks.
Observation lines have the form `KEYWORD = <UTC epoch> <value>`.
All emitted text must be ASCII.
