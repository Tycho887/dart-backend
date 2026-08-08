"""Dataset-agnostic command-line access to the service contracts."""

from __future__ import annotations

import argparse
from pathlib import Path

from .contracts import BatchRequest, QualityRequest
from .quality import assess_quality
from .services.optimizer import solve_batch


def _read(path: str, model):
    return model.model_validate_json(Path(path).read_text())


def _write(path: str | None, value) -> None:
    payload = value.model_dump_json(indent=2) + "\n"
    if path:
        Path(path).write_text(payload)
    else:
        print(payload, end="")


def _batch(args) -> int:
    _write(args.output, solve_batch(_read(args.input, BatchRequest)))
    return 0


def _quality(args) -> int:
    _write(args.output, assess_quality(_read(args.input, QualityRequest)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DART contract runner")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler, help_text in (
        ("batch", _batch, "run one batch optimization request"),
        ("quality", _quality, "compute residual and optional OEM metrics"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--input", required=True)
        command.add_argument("--output")
        command.set_defaults(function=handler)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
