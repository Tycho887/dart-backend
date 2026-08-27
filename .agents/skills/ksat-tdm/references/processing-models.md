# Range and range-rate processing models

Sources: `Ranging Model _ Ranging.pdf`, pp. 1–2; `Delay Model _ Ranging.pdf`,
pp. 1–4; `Range Rate Model _ Ranging.pdf`, pp. 1–2.

The TDM generator preserves raw modem data and calibration terms. It does not
turn them into navigation range or range rate.

For range processing, KSAT supplies round-trip delay `Tm`, transmit/receive
delays `Ttx`/`Trx`, calibration correction `Tgc`, pedestal offset `Lg`, station
ECEF coordinates, and optionally local weather. The customer supplies the
troposphere, ionosphere, interplanetary-medium, spacecraft-turnaround, and
spacecraft-navigation-point models. The documents contain no complete signed
range equation; do not invent one.

Operational-path components must be removed except the actual free-space range
delay. Items shared by the calibration and operational paths are represented by
`CORRECTION_RANGE`; calibration-only items such as the TLT are removed before
that correction is written. Frequency-dependent manually estimated delays use
site lookup/interpolation before `TRANSMIT_DELAY_1` or `RECEIVE_DELAY_1` is
written.

Range-rate processing assumes a coherent spacecraft transponder. Inputs are
`TRANSMIT_FREQ_1` (`Fut`), `RECEIVE_FREQ_1` (`Fdr`), and the turnaround ratio
(`trn`/`trd`). The supplied document gives no conversion equation. It recommends
choosing an uplink frequency that is an integer multiple of the denominator to
reduce tuning-step error; this is advice, not a serialization requirement.
