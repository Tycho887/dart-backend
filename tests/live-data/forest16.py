"""FOREST16 LEOP contact inventory, extracted from doppler_parquet/forest16.parquet.

May 3–4, 2026 UTC. Supply the initial ephemeris ID explicitly when running; the committed GPS
OEM is the default reference. Contact ephemerides are never fit defaults.
"""

from pathlib import Path

SPACECRAFT_ID = "ebc4af2e-c5f5-4700-a103-d1fc1bf423bb"
CENTER_FREQUENCY_HZ = 2_216_300_000.0
CONTACT_IDS = (
    "a6252615-8c19-4f78-a6f0-227e3da97c2e",
    "b2bd28a8-6704-4a67-bd8b-2c84fb0473bc",
    "9adbc5c4-0b67-471e-a286-dda9a089a49f",
    "7815ef69-da31-40ef-853c-e552ebe841cc",
    "1c36ca34-a985-4809-9ca7-65e422e6c47d",
    "e5d79f0b-730b-4832-becf-f716d7322784",
    "bcec2df5-c0d0-4e89-91ae-f0568424de8a",
    "82723fb0-59f4-4de8-ac05-f9f3d9a7dcc7",
    "7cb89272-0ba6-4eaf-8445-917ec34e727e",
    "2a45c0d3-c3e9-45e8-a11f-5dbf6bda0693",
    "33061519-fa11-49e4-becd-6c770efd6e11",
    "85ec563d-ecb0-4c0b-9b2d-eeb9cec80587",
    "556b1cfa-0a8e-4d19-8f95-b123fd47cd60",
)

REFERENCE_OBJECT_ID = "FOREST-16"
DEFAULT_REFERENCE_OEM = (
    Path(__file__).resolve().parents[2]
    / "reports/forest-gps/20260504/FOREST-16/FOREST-16.oem"
)
