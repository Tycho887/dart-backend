# Shadow scheduling safety

The current implementation is `dart/shadow_scheduler.py`; its CLI entry point
is `book_shadowpass.py`. The CLI requires both `--execute` and
`--confirm-booking` for mutation. `config/shadow-pass.example.yaml` contains no
operational identity.

Use injected fake clients and stores in ordinary tests. Never use production
credentials or network endpoints in the default test suite. Validate any
concrete KOGS booking, ephemeris-assignment, and cancellation endpoint against
the current API contract before first execution.
