# Historical FOREST experiments

FOREST parquet and NovAtel BESTXYZ support exists only to reproduce historical
reports. Production services do not import these adapters.

The loaders live in `dart_research.data`. Convert historical reference files
without installing a production CLI:

```bash
uv run --project research python -m dart_research.data.bestxyz_to_oem \
  --raw-dir /path/to/raw-reference \
  --object-name OBJECT-NAME \
  --output /path/to/reference.oem
```
