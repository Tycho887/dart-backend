# ANGLE product

Sources: `TDM-Angle _ Ranging.pdf`, pp. 1–2; `Appendix-A2 Example Angle Data
TDM Files _ Ranging.pdf`, p. 1.

Generate ANGLE only when controller pointing-angle readback is available.
Additional metadata is:

- `RECEIVE_BAND`: `S`, `X`, or `Ka`.
- `ANGLE_TYPE`: `AZEL`, `XEYN`, or `XSYE`.
- `COMMENT TRACKING_MODE = AUTO|PROGRAM|SCAN`, a KSAT extension.

Every segment contains exactly one tracking mode. End the current data segment
and start a complete new metadata/data segment whenever that mode changes.

`ANGLE_1` and `ANGLE_2` are degrees at `1e-6` degree resolution:

- `AZEL`: azimuth, elevation.
- `XEYN` and `XSYE`: X-axis, Y-axis.

RADEC and dynamically controlled three-axis pedestals are unsupported.
Pre-positioned tilt-axis pedestals are provisional and require the pass-specific
antenna reference position in the header.

The source describes both commanded angles and controller readback; do not
silently choose between different ADX signals. DART's default mapping uses
`antenna1_position_azimuth` and `antenna1_position_elevation`; confirm that
these represent the required readback at the target site, or override them.
