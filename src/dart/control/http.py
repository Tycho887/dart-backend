"""HTTP adapter for timestamped antenna-controller APIs."""

from __future__ import annotations

import datetime as dt
import json
import urllib.request

import satkit as sk

from ..types import RFObservation
from .interfaces import OffsetConvention


class HTTPAntennaBackend:
    def __init__(
        self,
        base_url: str,
        *,
        station_id: str,
        convention: OffsetConvention = OffsetConvention.PROPAGATION_ADVANCE,
        simulation_epoch=None,
        timeout_s: float = 5.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.station_id = station_id
        self.convention = convention
        self.simulation_epoch = simulation_epoch
        self.timeout_s = float(timeout_s)

    def _request(self, path: str, payload: dict | None = None) -> dict:
        request = urllib.request.Request(
            self.base_url + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"} if payload is not None else {},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.load(response)

    def apply_offset(self, offset_s: float) -> None:
        self._request("/offset", {"epoch_offset_seconds": float(offset_s)})

    def _epoch(self, payload: dict):
        if "measurement_utc_s" in payload:
            return sk.time.from_unixtime(float(payload["measurement_utc_s"]))
        if "measurement_time" in payload:
            value = dt.datetime.fromisoformat(str(payload["measurement_time"]).replace("Z", "+00:00"))
            return sk.time.from_datetime(value)
        if "sim_time_s" in payload and self.simulation_epoch is not None:
            return self.simulation_epoch + sk.duration(seconds=float(payload["sim_time_s"]))
        raise ValueError("reading lacks a usable measurement epoch")

    def read(self) -> RFObservation | None:
        payload = self._request("/readings")
        if payload.get("pass_over"):
            return None
        return RFObservation(
            epoch=self._epoch(payload),
            station_id=self.station_id,
            doppler_hz=float(payload["doppler_hz"] if "doppler_hz" in payload else payload["doppler_freq"]),
            phase_rad=(
                None if payload.get("phase_diff") is None else float(payload["phase_diff"])
            ),
            valid=bool(payload.get("valid", True)),
            commanded_az_deg=payload.get("az_cmd"),
            commanded_el_deg=payload.get("el_cmd"),
            applied_offset_s=payload.get("epoch_offset_seconds"),
            sequence=payload.get("sequence"),
            quality={"source": "http"},
        )

