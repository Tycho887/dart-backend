"""Build the deliberately narrow DART CCSDS TDM KVN profile."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from hashlib import sha256

from ..contracts import Measurement, TdmDocument


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def measurements_to_tdm(
    measurements: list[Measurement],
    nominal_carrier_frequency_hz: float,
) -> TdmDocument:
    """Serialize ADX carrier offsets as absolute CCSDS RECEIVE_FREQ values."""

    if nominal_carrier_frequency_hz <= 0.0:
        raise ValueError("nominal carrier frequency must be positive")
    if not measurements:
        raise ValueError("cannot create TDM without measurements")

    groups: dict[tuple[str, str, str], list[Measurement]] = defaultdict(list)
    for measurement in measurements:
        key = (measurement.pass_id, measurement.station_id, measurement.spacecraft_id)
        groups[key].append(measurement)

    lines = [
        "CCSDS_TDM_VERS = 2.0",
        f"CREATION_DATE = {_utc(datetime.now(UTC))}",
        "ORIGINATOR = DART",
    ]
    for (pass_id, station_id, spacecraft_id), group in groups.items():
        ordered = sorted(group, key=lambda item: item.time_tag)
        lines.extend(
            [
                "META_START",
                "TIME_SYSTEM = UTC",
                f"PARTICIPANT_1 = {station_id}",
                f"PARTICIPANT_2 = {spacecraft_id}",
                "MODE = SEQUENTIAL",
                "PATH = 2,1",
                f"TRACK_ID = {pass_id}",
                f"START_TIME = {_utc(ordered[0].time_tag)}",
                f"STOP_TIME = {_utc(ordered[-1].time_tag)}",
                "META_STOP",
                "DATA_START",
            ]
        )
        for measurement in ordered:
            if measurement.doppler_hz is None:
                continue
            frequency_mhz = (nominal_carrier_frequency_hz + measurement.doppler_hz) / 1_000_000.0
            lines.append(f"RECEIVE_FREQ = {_utc(measurement.time_tag)} {frequency_mhz:.12f}")
        lines.append("DATA_STOP")

    content = "\n".join(lines) + "\n"
    digest = sha256(content.encode()).hexdigest()
    return TdmDocument(content=content, sha256=digest)
