# TRACK product

Sources: `TDM-Ranging _ Ranging.pdf`, pp. 1–4; `Appendix-A1 Example Track Data
TDM Files _ Ranging.pdf`, pp. 1–3.

TRACK contains raw radiometric data. KSAT documents four numbered modes, but
integrated carrier phase is still in development and the supplied appendix has
no mode-2 example or carrier-phase data keyword. The DART writer therefore
supports modes 1, 3, and 4 only.

| Field | Mode 1 | Mode 3 | Mode 4 | Units/value |
| --- | --- | --- | --- | --- |
| `MODE` | yes | yes | yes | `SEQUENTIAL` |
| `TRANSMIT_BAND`, `RECEIVE_BAND` | yes | yes | yes | `S`, `X`, `Ka` |
| `TURNAROUND_NUMERATOR`, `TURNAROUND_DENOMINATOR` | no | yes | yes | supported ratio |
| `INTEGRATION_INTERVAL` | yes | yes | yes | seconds, 0.01–60 |
| `INTEGRATION_REF` | yes | yes | yes | `END` |
| `RANGE_UNITS` | yes | yes | no | `s` |
| `TRANSMIT_DELAY_1`, `RECEIVE_DELAY_1` | yes | yes | no | seconds |
| `CORRECTION_RANGE` | yes | yes | no | seconds |
| `CORRECTION_DOPPLER` | no | yes | yes | Hz |
| `CORRECTIONS_APPLIED` | yes | yes | yes | `NO` |
| `PATH` | yes | yes | yes | `1,2,1` |
| `RANGE` | yes | yes | no | round-trip delay, seconds |
| `TRANSMIT_FREQ_1`, `RECEIVE_FREQ_1` | no | yes | yes | Hz |

Do not emit `RANGE_MODULUS`; its need is marked TBD. Data timestamps have 1 us
resolution. Render range delay to 1 ps and frequencies to 1 microhertz. Emit
transmit frequency once unless uplink Doppler pre-steering changes it.

Confirm that the configured TRACK timestamp is the end of the integration
interval before emitting `INTEGRATION_REF = END`. Bands, interval, turnaround
ratio, calibration delays, and corrections are site/track metadata and must be
constant within a segment. Split the selection if any of them changes. Reject
an incomplete mapped TRACK row; do not silently discard it or infer a missing
frequency.

Supported same-band turnaround ratios from `Range Rate Model _ Ranging.pdf`,
p. 2: S `240/221`; X `880/749`; Ka `2720/2407`, `2760/2407`, or `2816/2407`.
KSAT does not currently support cross-band operations.

`RANGE` is raw round-trip modem delay, not distance. Corrections remain
unapplied; downstream processing owns the physical model.
