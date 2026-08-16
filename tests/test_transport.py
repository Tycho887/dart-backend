"""End-to-end transport tests through the Rust extension.

Skipped when the extension is not built (run `uv sync` or
`maturin develop`). These prove the Python-encoded input survives the Rust
round trip and comes back as a well-formed SolverResult.
"""

import pytest

dart_solver = pytest.importorskip("dart_solver")

from dart.schema import (
    ForceModel,
    Observation,
    Rk89Input,
    SCHEMA_VERSION,
    Sgp4Input,
    SolverOptions,
    SolverResult,
    Station,
    Tle,
)
from dart.solver import solve

from test_codec import ISS_TLE, sample_rk89_input, sample_sgp4_input


def test_sgp4_transport_roundtrip():
    res = solve(sample_sgp4_input())
    assert isinstance(res, SolverResult)
    assert res.schema_version == SCHEMA_VERSION
    assert res.mode == "sgp4"
    # Scaffold contract: well-formed input reaches the sgp4 stub and comes
    # back as a not-implemented error result, not a transport failure.
    assert not res.success
    assert "scaffold" in res.message
    assert not res.converged
    assert res.residuals == ()


def test_rk89_transport_roundtrip():
    res = solve(sample_rk89_input())
    assert res.mode == "rk89"
    assert not res.success
    assert "scaffold" in res.message


def test_unknown_mode_comes_back_as_error_result():
    import msgpack

    from dart.codec import decode_result, encode_input

    raw = msgpack.unpackb(encode_input(sample_sgp4_input()))
    raw["mode"] = "jpl"
    res = decode_result(dart_solver.solve(msgpack.packb(raw)))
    assert not res.success
    assert "unknown mode" in res.message


def test_version_mismatch_comes_back_as_error_result():
    import msgpack

    from dart.codec import encode_input, decode_result

    raw = msgpack.unpackb(encode_input(sample_sgp4_input()))
    raw["schema_version"] = 999
    res = decode_result(dart_solver.solve(msgpack.packb(raw)))
    assert not res.success
    assert "schema_version mismatch" in res.message
