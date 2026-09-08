"""FOREST18 LEOP contact inventory, extracted from doppler_parquet/forest18.parquet.

May 3–4, 2026 UTC. Supply the initial ephemeris ID explicitly when running; the committed GPS
OEM is the default reference. Contact ephemerides are never fit defaults.
"""

from pathlib import Path

SPACECRAFT_ID = "b221e85e-a0d5-44bd-abd2-930ff3e849b5"
CENTER_FREQUENCY_HZ = 2_216_300_000.0
CONTACT_IDS = (
    "53eafac8-7eae-4d9b-8adc-9066c668410c",
    "3ccef2f9-751b-4153-b810-267dd4c64a31",
    "ca8d1c55-2725-4217-bc02-13672ceec9ae",
    "a1bdfcbb-ff7d-47af-8ba1-7fb37ee15e65",
    "6d7785df-a7a7-4963-8fcc-bdebf3a6dc80",
    "f1550be3-6ca0-48b2-bd87-da21c1e885b6",
    "a5dc9c5c-0fb7-4bc9-9ba3-89a37ff6fac1",
    "b2015720-f0bb-48e7-b19a-131a592c9501",
    "49e74a6c-b21d-46c0-87bb-e4195b4a0dab",
    "6c22b929-f451-4e3f-bc5d-fc4e230c6c02",
    "5f62261f-f193-475b-ad2d-94a8b9fd0e75",
    "48de3125-1fd3-40df-be87-5fc55aa2485f",
    "2735b79f-1ae2-4be6-9b13-774d026efd72",
    "7654ef67-7aba-4f18-bf0a-65eacc1b6a22",
    "105f1f16-870e-4658-9a38-41c74ea5939c",
    "8db51bff-de04-4287-b3c8-4f949e7a8df1",
    "7f6f30f5-0820-469d-9ab6-71b9523e8939",
    "4d451626-a662-4148-8369-abbcd6001b6a",
)

REFERENCE_OBJECT_ID = "FOREST-18"
DEFAULT_REFERENCE_OEM = (
    Path(__file__).resolve().parents[2]
    / "reports/forest-gps/20260504/FOREST-18/FOREST-18.oem"
)
