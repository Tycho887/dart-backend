# DART

DART is being reorganized into independent modules for passive-RF data
access, forward models, orbit determination, standards products, and antenna
control. The current stable Python surface is `dart.io`: a typed,
model-independent boundary around external data providers.

## IO quick start

```python
import asyncio

from dart.io import load_passes
from dart.io.adx import client_from_env
from dart.io.kogs import api_key_from_env


async def main() -> None:
    contacts, measurements = await load_passes(
        ["contact-uuid"],
        kogs_api_key=api_key_from_env(),
        adx_client=client_from_env(),
    )
    print(contacts[0])
    print(measurements)


asyncio.run(main())
```

`load_passes` resolves each KOGS contact, antenna, spacecraft, and ephemeris,
then retrieves the contact-bounded ADX measurements. It returns metadata in
the requested order and one timestamp-sorted Polars DataFrame. It does not
apply optimizer filters or select a high- or low-fidelity model.

The canonical measurement columns are:

```text
timestamp, contact_id, spacecraft_id, system_id, antenna_name,
tracking_epoch_offset_s, azimuth_deg, elevation_deg, carrier_lock,
ebn0, doppler_hz
```

For callers of the shared forward-model interface,
`dart.io.load_forward_context` converts this result into a
`ForwardModelContext` using explicit center-frequency and Doppler-variance
arguments.

## Provider modules

- `dart.io.kogs`: KOGS authentication, typed reads, contact metadata, and
  guarded scheduling mutations.
- `dart.io.adx`: ADX clients, bounded raw queries, and canonical measurements.
- `dart.io.orbital`: fail-closed absolute time-offset writes.
- `dart.io.ctrl_config`: read-only ctrl-config access.
- `dart.io.meos`: reviewed calibration data pending a live MEOS integration.
- `dart.io.parquet`: canonical replay of recorded ADX measurements.
- `dart.io.load`: asynchronous multi-provider workflows.

Credentials are loaded only through provider helpers and are passed explicitly
to orchestration functions. They are never stored in returned metadata.

## Development

```bash
uv sync
uv run pytest
uv run ruff check dart/io tests
uv run ty check dart/io
```

See [the suite architecture](docs/dart-suite-architecture.md) for the target
module boundaries. Optimizer, TDM, service, and control consumers are retained
as migration work and are not part of the current IO contract.
