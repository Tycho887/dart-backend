"""MessagePack codec for the transport schema.

The codec is generic: it walks ``dataclasses.fields()`` in declaration order
(which fixes the msgpack key order and keeps the Rust side deterministic)
and rebuilds values from the declared type hints, so renames cannot silently
desync the two sides.

Version gate: any message whose ``schema_version`` does not match
``dart.schema.SCHEMA_VERSION`` is rejected with a clear error before any
further processing.
"""

from __future__ import annotations

import dataclasses
import types
import typing

import msgpack

from dart.schema import (
    Rk89Input,
    SCHEMA_VERSION,
    Sgp4Input,
    SolverResult,
)

SolverInput = Sgp4Input | Rk89Input

_UNION_ORIGINS = (typing.Union, types.UnionType)


class SchemaVersionError(ValueError):
    """Raised when a message's schema_version does not match SCHEMA_VERSION."""


class SchemaError(ValueError):
    """Raised when a message is malformed for the schema (bad mode, missing
    or mistyped fields)."""


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_input(inp: SolverInput) -> bytes:
    return msgpack.packb(_to_dict(inp), use_bin_type=False)


def encode_result(res: SolverResult) -> bytes:
    return msgpack.packb(_to_dict(res), use_bin_type=False)


def _to_dict(obj) -> dict:
    """dataclass -> plain dict, keyed by field name in declaration order."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def decode_input(data: bytes) -> SolverInput:
    raw = _unpack(data)
    _check_version(raw)
    mode = raw.get("mode")
    if mode == "sgp4":
        return _from_dict(Sgp4Input, raw)
    if mode == "rk89":
        return _from_dict(Rk89Input, raw)
    raise SchemaError(f"unknown mode {mode!r}: expected 'sgp4' or 'rk89'")


def decode_result(data: bytes) -> SolverResult:
    raw = _unpack(data)
    _check_version(raw)
    return _from_dict(SolverResult, raw)


def _unpack(data: bytes) -> dict:
    try:
        raw = msgpack.unpackb(data, raw=False)
    except Exception as exc:  # msgpack raises various exceptions on garbage
        raise SchemaError(f"failed to unpack message: {exc}") from exc
    if not isinstance(raw, dict):
        raise SchemaError("message is not a msgpack map")
    return raw


def _check_version(raw: dict) -> None:
    version = raw.get("schema_version")
    if version is None:
        raise SchemaError("missing required field 'schema_version'")
    if version != SCHEMA_VERSION:
        raise SchemaVersionError(
            f"schema_version mismatch: got {version}, expected {SCHEMA_VERSION}"
        )


def _from_dict(cls: type, data: dict):
    """Rebuild a frozen dataclass from a msgpack dict using its type hints."""
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            raise SchemaError(f"missing required field '{f.name}' for {cls.__name__}")
        kwargs[f.name] = _coerce(hints[f.name], data[f.name], f.name)
    return cls(**kwargs)


def _coerce(tp, value, name: str):
    origin = typing.get_origin(tp)

    if value is None:
        return None

    # Optional[X] / X | None: strip NoneType, coerce the rest
    if origin in _UNION_ORIGINS:
        for arg in typing.get_args(tp):
            if arg is type(None):
                continue
            return _coerce(arg, value, name)

    if origin is list:
        (arg,) = typing.get_args(tp)
        return [_coerce(arg, v, name) for v in value]

    if origin is tuple:
        args = typing.get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            arg = args[0]
            return tuple(_coerce(arg, v, name) for v in value)
        return tuple(_coerce(arg, v, name) for arg, v in zip(args, value))

    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise SchemaError(f"field '{name}' expected a map, got {type(value).__name__}")
        return _from_dict(tp, value)

    if tp is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise SchemaError(f"field '{name}' expected int, got {type(value).__name__}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SchemaError(f"field '{name}' expected float, got {type(value).__name__}")
        return float(value)
    if tp is bool:
        if not isinstance(value, bool):
            raise SchemaError(f"field '{name}' expected bool, got {type(value).__name__}")
        return value
    if tp is str:
        if not isinstance(value, str):
            raise SchemaError(f"field '{name}' expected str, got {type(value).__name__}")
        return value

    raise SchemaError(f"field '{name}': unsupported type {tp}")
