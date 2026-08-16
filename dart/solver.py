"""Python facade over the Rust solver extension.

The extension (`dart_solver`, built by maturin) exposes a single function,
``solve(data: bytes) -> bytes``. This module handles the encode/decode around
it so callers work with the schema dataclasses directly.
"""

from __future__ import annotations

try:  # the extension is only importable once `uv sync`/`maturin develop` ran
    from dart_solver import solve as _solve
except ImportError as exc:  # pragma: no cover - exercised only on fresh checkouts
    raise ImportError(
        "the dart_solver extension is not built; run `uv sync` (or "
        "`maturin develop`) to build it into the venv"
    ) from exc

from dart.codec import SolverInput, decode_result, encode_input
from dart.schema import SolverResult


def solve(inp: SolverInput) -> SolverResult:
    """Pack ``inp``, hand it to the Rust solver, unpack the result.

    Raises the same schema/version errors as ``dart.codec``; solver-level
    failures (including "not implemented" results from scaffolding) are
    reported in the returned ``SolverResult``, not as exceptions.
    """
    payload = encode_input(inp)
    raw = _solve(payload)
    return decode_result(raw)
