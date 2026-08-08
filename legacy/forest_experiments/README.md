# Historical FOREST experiments

FOREST parquet and NovAtel BESTXYZ support is retained only to reproduce the
checked historical reports. Runtime services do not import these adapters.

The Python readers live in `dart.legacy`. Legacy CLI commands require explicit
satellite identifiers and never default to FOREST 16–19. Future reference
ephemerides enter the quality service as frame-explicit CCSDS OEM 2.0 KVN/XML.

Convert the historical reference files before rerunning a study:

```bash
uv run dart-legacy-bestxyz-to-oem \
  --raw-dir /path/to/raw-reference \
  --object-name OBJECT-NAME \
  --output /path/to/reference.oem
```
