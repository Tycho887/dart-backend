# ADX-backed export

Use `dart.io.ksat_adx` for live telemetry. Every query must select exactly one
contact and a bounded UTC interval. The selected ADX station must equal the
configured KSAT header site.

For TRACK mode 3, the minimum setup is:

```python
columns = KsatAdxColumnMap(
    track_timestamp="confirmed_integration_end_column",
    range_delay=AdxField("confirmed_round_trip_delay_column", "ps"),
    transmit_frequency=AdxField("confirmed_absolute_tx_column", "Hz"),
    receive_frequency=AdxField("confirmed_absolute_rx_column", "Hz"),
)
selection = KsatAdxQuery(
    contact_ids=(contact_id,),
    start_time=start_utc,
    stop_time=stop_utc,
    columns=columns,
)
result = export_ksat_tdm_bundle(
    selection,
    header,
    track=TrackMetadata(mode=3, ...),
    products=(KsatProduct.TRACK,),
    output_dir=output_directory,
)
```

Supply all mode-3 metadata explicitly: bands, integration interval, standard
turnaround ratio, transmit/receive delays, range correction, and Doppler
correction. Obtain contact, site, and spacecraft metadata from KOGS before
querying ADX. Use KOGS WGS-84 antenna coordinates and derive ITRF/ECEF
coordinates with satkit. Keep reviewed calibration facts that KOGS does not
expose in strict configuration; the ADX query does not derive header metadata.

Check both `result.generated` and `result.skipped`. A requested product can be
skipped for absent mappings, incomplete/non-finite data, station mismatch, or
unsupported configuration while another valid product is still returned.

The query returns raw telemetry. Do not use the solver query's elevation,
lock-state, Eb/N0, or Doppler filters for TDM delivery unless the delivery
requirement explicitly calls for those filters.

## Configuration-backed command

Use `scripts/write_tdm.py` for a one-contact command-line export. Its TOML is
loaded by `dart.io.ksat_export`, which strictly validates site, spacecraft,
header, product metadata, ADX columns, and source units before opening an ADX
client. `config/ksat-tdm.example.toml` documents every supported section.

Product sections determine the default selection: `[track]`, `[angle]`, and
`[sigmet]` enable TRACK, ANGLE, and SIGMET respectively. The `--product` option
may select a subset. Do not add METEO until both its normative definition and a
weather source exist.

Keep all credentials in the environment. The TOML must contain no secrets and
must declare the expected KOGS spacecraft, system, and station UUIDs. The
exporter reads `KOGS_API_KEY`, rejects identity mismatches before querying ADX,
and uses reverse geocoding only for the human-readable location. Geocoder or
missing-calibration failures produce `UNKNOWN` plus warnings; KOGS failures are
fatal. Both time bounds must be offset-aware ISO-8601 timestamps. The CLI
reports every written standard filename, warning, and skipped-product reason;
partial product success is a successful command.
