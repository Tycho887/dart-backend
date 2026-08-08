"""Adapter from FOREST parquet telemetry to canonical RF observations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import satkit as sk

from ..types import RFObservation, Station, TLEContext

DOPPLER_COLUMN = "lr1_receiver1_actualCarrierFrequencyOffset"


@dataclass(frozen=True, slots=True)
class ForestPass:
    satellite: str
    contact_id: str
    ground_station: str
    context: TLEContext
    observations: tuple[RFObservation, ...]

    @property
    def start_utc_s(self) -> float:
        return float(self.observations[0].epoch.as_unixtime())

    @property
    def end_utc_s(self) -> float:
        return float(self.observations[-1].epoch.as_unixtime())


def _required_columns() -> set[str]:
    return {
        "timestamp",
        "contact_id",
        "groundStation",
        "station_lat",
        "station_lon",
        "station_alt",
        "expected_frequency",
        "tle_line1",
        "tle_line2",
        "antenna1_position_azimuth",
        "antenna1_position_elevation",
        DOPPLER_COLUMN,
    }


def load_forest_passes(path: Path | str, satellite: str | None = None) -> list[ForestPass]:
    path = Path(path)
    frame = pl.read_parquet(path)
    missing = _required_columns() - set(frame.columns)
    if missing:
        raise ValueError(f"FOREST telemetry missing columns: {sorted(missing)}")
    frame = (
        frame.filter(
            pl.col(DOPPLER_COLUMN).is_not_null()
            & (pl.col(DOPPLER_COLUMN) != 0.0)
            & (pl.col(DOPPLER_COLUMN).abs() >= 0.1)
            & (pl.col("antenna1_position_elevation") > 1.0)
            & (pl.col("antenna1_position_elevation") < 89.0)
        )
        .sort("timestamp")
    )
    identifier = satellite or path.stem.replace("forest", "FOREST-").upper()
    result: list[ForestPass] = []
    for contact_id in frame["contact_id"].unique(maintain_order=True).to_list():
        contact = frame.filter(pl.col("contact_id") == contact_id)
        if contact.is_empty():
            continue
        station = Station(
            station_id=str(contact["groundStation"][0]),
            latitude_deg=float(contact["station_lat"][0]),
            longitude_deg=float(contact["station_lon"][0]),
            altitude_m=float(contact["station_alt"][0]),
        )
        tle = sk.TLE.from_lines([str(contact["tle_line1"][0]), str(contact["tle_line2"][0])])
        context = TLEContext(
            tle=tle,
            station=station,
            carrier_hz=float(contact["expected_frequency"][0]),
        )
        observations = []
        for index, row in enumerate(contact.iter_rows(named=True)):
            observations.append(
                RFObservation(
                    epoch=sk.time.from_datetime(row["timestamp"]),
                    station_id=station.station_id,
                    doppler_hz=float(row[DOPPLER_COLUMN]),
                    valid=True,
                    commanded_az_deg=float(row["antenna1_position_azimuth"]),
                    commanded_el_deg=float(row["antenna1_position_elevation"]),
                    sequence=index,
                    quality={"source": "FOREST", "contact_id": str(contact_id)},
                )
            )
        result.append(
            ForestPass(
                satellite=identifier,
                contact_id=str(contact_id),
                ground_station=station.station_id,
                context=context,
                observations=tuple(observations),
            )
        )
    return result
