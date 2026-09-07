"""Compatibility imports; OEM products now live in :mod:`dart.oem`."""

from dart.oem import Oem, read_oem, validate_oem

__all__ = ["Oem", "read_oem", "validate_oem"]
