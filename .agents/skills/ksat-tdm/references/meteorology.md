# METEO product — deferred

Source: `Appendix-A4 Example Meterology Data TDM File _ Ranging.pdf`, p. 1.

The supplied directory does not contain the linked normative
`TDM-Meteorology` document. The printed appendix clips its scrollable example,
so the exact standard temperature, pressure, and relative-humidity keywords,
units, resolutions, and complete data block cannot be recovered safely.

The available evidence establishes only:

- filename prefix `METEO`;
- common header and participant metadata;
- standard atmospheric temperature, pressure, and relative humidity are
  intended;
- the KSAT extension is encoded as
  `COMMENT WIND = <epoch> ddd@kk[Ggg]`, where `ddd` is direction, `kk` is wind
  speed, and optional `Ggg` is gust speed;
- alternate site formats may be used depending on metric availability.

Do not generate METEO from these incomplete definitions. The current exporter
must return a structured deferred reason until the normative page and selected
weather API establish the contract.
