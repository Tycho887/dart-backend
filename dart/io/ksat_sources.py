"""Source attribution shared by the KSAT metadata and export layers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class KsatAuthority(StrEnum):
    """Authorities from which a delivered KSAT value can originate."""

    KOGS = "KOGS"
    KOGS_EPHEMERIS = "KOGS_EPHEMERIS"
    ADX = "ADX"
    CONFIG = "CONFIG"
    DERIVED = "DERIVED"
    GEOCODER = "GEOCODER"
    MEOS = "MEOS"


@dataclass(frozen=True, slots=True)
class KsatProvenance:
    """One auditable value-to-authority association."""

    field: str
    authority: KsatAuthority
    detail: str | None = None


__all__ = ["KsatAuthority", "KsatProvenance"]
