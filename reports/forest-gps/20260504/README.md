# FOREST reference OEMs

These recorded reference products remain on `main` because the benchmark
tools and reference-validation tests use them. Each spacecraft directory
contains its original OEM and `quality.json`, including the product checksum
and independent withheld-GPS validation results.

- FOREST-16, FOREST-17, and FOREST-18 have accepted `.oem` products.
- FOREST-19 retains its `.candidate.oem` designation; do not treat it as accepted.

The underlying GPS input exports remain in `gps-examples/`. See
[the GMAT workflow](../../../scripts/gmat/README.md) for generation and validation,
[reference inventories](../../../experiments/notes.md) for contact selections,
and [the experiment archive](../../../docs/experiment-archive.md) for study results.
