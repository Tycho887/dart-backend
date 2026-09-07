"""Shared metadata helpers for compact KSAT delivery writers."""

from __future__ import annotations

import re

from dart.io.contact import ContactMetadata

BANDS = {"S", "X", "Ka"}
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
COLUMN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
COSPAR = re.compile(r"^\d{4}-\d{3}[A-Z]{1,3}$")


def validate_filename_identity(metadata: ContactMetadata) -> None:
    if not IDENTIFIER.fullmatch(metadata.antenna) or not COSPAR.fullmatch(
        metadata.cospar
    ):
        raise ValueError("KOGS identifiers are not valid KSAT filename components")
