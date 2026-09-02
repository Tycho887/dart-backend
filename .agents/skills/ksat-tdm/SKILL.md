---
name: ksat-tdm
description: >-
  Create, validate, or review KSAT-profile CCSDS 503.0-B-2 Tracking Data
  Message files and the DART ADX-backed TRACK, ANGLE, and SIGMET export path.
  Use for KSAT TDM filenames, headers, metadata, observations, product
  selection, and ranging-model interpretation; do not use the legacy DART
  solver TDM as the KSAT authority.
---

# KSAT TDM

Generate KSAT delivery products from the supplied KSAT documentation and the
typed writers in `dart.io.ksat_tdm` and the bounded ADX query in `dart.io.azure`.

## Workflow

1. Read [common.md](references/common.md) for every product.
2. Read only the requested product definition:
   - [track.md](references/track.md) for range and range-rate data.
   - [angle.md](references/angle.md) for antenna pointing.
   - [signal-metrics.md](references/signal-metrics.md) for receive metrics.
   - [meteorology.md](references/meteorology.md) for the currently deferred METEO product.
3. Read [processing-models.md](references/processing-models.md) only when
   interpreting raw range/range-rate measurements or calibration terms.
4. Read [adx-export.md](references/adx-export.md) for database-backed export.
5. Read [source-notes.md](references/source-notes.md) before treating an
   appendix example as normative, resolving a discrepancy, or comparing the
   KSAT writer with the legacy DART writer.

Use `dart.io.ksat_tdm` for already typed product data. For ADX-backed export,
use `dart.io.azure.fetch_contact_columns` with a bounded contact/time query and
explicit source column plus unit mappings. The generic timestamp and antenna-angle defaults
match DART's current telemetry query but must be confirmed for the target site.
Do not treat a carrier-frequency offset as an absolute carrier frequency, or
Eb/N0 as PC/N0 or PR/N0.

## Required behavior

- Preserve raw KSAT measurements and declared correction terms; do not apply
  ranging, media, spacecraft, or Doppler models while serializing.
- Emit only products supported by the available site data. Report why a
  requested product was skipped.
- Reject undocumented modes, topologies, fields, ratios, or conversions rather
  than inferring them.
- Keep the legacy `dart.tdm.legacy` solver-record format separate. It reverses the
  KSAT participant roles and does not implement the KSAT observation grammar.

METEO serialization is deferred because the supplied directory omits the
normative `TDM-Meteorology` page and no weather API has been selected.
