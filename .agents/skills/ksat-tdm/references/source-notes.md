# Source authority and known errata

Treat the product-definition PDFs as authoritative over the appendices. Treat
CCSDS 503.0-B-2 as the baseline where the KSAT profile delegates syntax, but do
not use general CCSDS options to widen KSAT's stated support.

Known source issues:

- Appendix filenames use station `D32` while comments and `PARTICIPANT_1` use
  `H16`. A generated file must use one consistent configured station.
- Appendix chronology mixes a 2005 creation date, 2023 calibration, and 2024
  weather observations. Do not copy these values as a coherent example.
- The printed appendix code blocks are clipped and cannot be complete golden
  files.
- One mode-4 example shows malformed `INTEGRATION REF END`; the normative key
  is `INTEGRATION_REF = END`.
- Ranging prose sometimes drops `_1` from frequency/delay names and misspells
  `TURNAROUND`. Use the table/mapping forms `TRANSMIT_FREQ_1`,
  `RECEIVE_FREQ_1`, `TRANSMIT_DELAY_1`, `RECEIVE_DELAY_1`, and
  `TURNAROUND_NUMERATOR`.
- The overview mentions integrated carrier phase, but support is in development
  and no usable mode-2 carrier-phase definition is supplied.
- The ANGLE prose alternates between commanded angles and readback. Require an
  explicit site-specific ADX mapping or confirmation that DART's default
  position-angle columns have the required semantics.
- The normative `TDM-Meteorology` PDF is absent.

The generic `dart.tdm.legacy` output is a DART solver record, not a KSAT template:
it uses `ORIGINATOR = DART`, assigns participant roles differently, and writes
observations as separate `EPOCH` records. Use `dart.io.ksat_tdm` instead.
