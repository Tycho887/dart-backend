"""Versioned deployment-owned KSAT TDM profiles."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from dart.io.ksat_tdm import (
    AngleColumns,
    AngleRequest,
    FrequencySource,
    TrackColumns,
    TrackRequest,
)
from dart.io.meos import TrackCalibration

from .models import TdmJobRequest


class ProfileModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CalibrationProfile(ProfileModel):
    pedestal_offset_m: float = Field(ge=0.0)
    tlt_calibration_date: dt.date
    correction_doppler_hz: float = 0.0

    def runtime(self) -> TrackCalibration:
        return TrackCalibration(
            self.pedestal_offset_m,
            self.tlt_calibration_date,
            self.correction_doppler_hz,
        )


class FrequencyProfile(ProfileModel):
    link_name: str = Field(min_length=1)
    offset_column: str | None = None
    offset_unit: Literal["Hz", "kHz", "MHz"] = "Hz"
    offset_sign: Literal[-1, 1] = 1

    def runtime(self) -> FrequencySource:
        return FrequencySource(
            self.link_name,
            self.offset_column,
            self.offset_unit,
            self.offset_sign,
        )


class TrackProfile(ProfileModel):
    name: str = Field(min_length=1, max_length=100)
    version: int = Field(ge=1)
    product: Literal["track"]
    station: str = Field(min_length=1)
    band: Literal["S", "X", "Ka"]
    integration_interval_s: float = Field(ge=0.01, le=60.0)
    turnaround_numerator: int = Field(gt=0)
    turnaround_denominator: int = Field(gt=0)
    integration_end_column: str
    contact_column: str = "contact_id"
    station_column: str = "antenna_name"
    transmit: FrequencyProfile
    receive: FrequencyProfile
    calibration: CalibrationProfile

    def runtime(self, contact_id: str) -> TrackRequest:
        return TrackRequest(
            contact_id=contact_id,
            band=self.band,
            integration_interval_s=self.integration_interval_s,
            turnaround_numerator=self.turnaround_numerator,
            turnaround_denominator=self.turnaround_denominator,
            transmit=self.transmit.runtime(),
            receive=self.receive.runtime(),
            columns=TrackColumns(
                self.integration_end_column,
                self.contact_column,
                self.station_column,
            ),
            expected_station=self.station,
            calibration=self.calibration.runtime(),
        )


class AngleProfile(ProfileModel):
    name: str = Field(min_length=1, max_length=100)
    version: int = Field(ge=1)
    product: Literal["angle"]
    station: str = Field(min_length=1)
    band: Literal["S", "X", "Ka"]
    tracking_mode: Literal["AUTO", "PROGRAM", "SCAN"]
    timestamp_column: str = "timestamp"
    contact_column: str = "contact_id"
    station_column: str = "antenna_name"
    angle_1_column: str = "antenna1_position_azimuth"
    angle_2_column: str = "antenna1_position_elevation"
    controller_readback_confirmed: Literal[True]
    calibration: CalibrationProfile | None = None

    def runtime(self, contact_id: str) -> AngleRequest:
        return AngleRequest(
            contact_id=contact_id,
            band=self.band,
            tracking_mode=self.tracking_mode,
            columns=AngleColumns(
                self.timestamp_column,
                self.contact_column,
                self.station_column,
                self.angle_1_column,
                self.angle_2_column,
            ),
            expected_station=self.station,
            calibration=self.calibration.runtime() if self.calibration else None,
            controller_readback_confirmed=True,
        )


TdmProfile = Annotated[TrackProfile | AngleProfile, Field(discriminator="product")]
_PROFILE_ADAPTER = TypeAdapter(TdmProfile)


def load_tdm_profile_documents(directory: Path) -> list[dict]:
    """Load strict, non-secret deployment profiles in deterministic order."""
    if not directory.exists():
        return []
    documents = []
    for path in sorted(directory.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        items = raw if isinstance(raw, list) else [raw]
        for item in items:
            profile = _PROFILE_ADAPTER.validate_python(item)
            profile.runtime("00000000-0000-4000-8000-000000000000")
            documents.append(profile.model_dump(mode="json"))
    keys = [(item["name"], item["version"]) for item in documents]
    if len(keys) != len(set(keys)):
        raise ValueError("TDM profile names and versions must be unique")
    return documents


def validate_tdm_profile(request: TdmJobRequest, document: dict) -> TdmProfile:
    profile = _PROFILE_ADAPTER.validate_python(document)
    if (profile.name, profile.version) != (
        request.profile.name,
        request.profile.version,
    ):
        raise ValueError("TDM profile identity does not match the request")
    if profile.product != request.product:
        raise ValueError("TDM profile product does not match the request")
    profile.runtime(str(request.contact_id))
    return profile
