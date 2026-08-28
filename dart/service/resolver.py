"""Contact/config/telemetry adapters that produce deterministic solver inputs."""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dart.io.azure import fetch_contact_tracking_data
from dart.io.kogs import (
    get_contact,
    get_spacecraft,
    get_TLE,
    parse_ephemeris,
    parse_reservation,
    parse_satellite,
)
from dart.loaders.leo import build_sgp4_input, station_from_kogs, tle_from_inline
from dart.schema import FitParameter, Sgp4Input
from dart.time_solver import TimeSolverConfig

from .models import MeanElementsSolver, SolveJobRequest, TimeShiftSolver


class ResolutionError(Exception):
    def __init__(
        self,
        code: str,
        detail: str,
        *,
        retryable: bool = False,
        service: str | None = None,
    ):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.retryable = retryable
        self.service = service


@dataclass(frozen=True)
class ResolvedMetadata:
    contact_id: str
    spacecraft_id: str
    spacecraft_name: str
    system_id: str
    station_id: str | None
    ephemeris_id: str
    tle: Any
    nominal_center_frequency_hz: float
    frequency_provenance: dict
    kogs_provenance: dict


@dataclass(frozen=True)
class PreparedSolve:
    input: Sgp4Input
    time_config: TimeSolverConfig | None
    resolved_configuration: dict
    contact_record: dict


class ControlConfigV2:
    LINK_NAME = "s_band_downlink_p1_1"

    def __init__(self, root: Path):
        self.root = root.resolve()

    def observed_frequency(self, spacecraft_name: str) -> tuple[float, dict]:
        if Path(spacecraft_name).name != spacecraft_name or spacecraft_name in {
            ".",
            "..",
        }:
            raise ResolutionError(
                "invalid_spacecraft_config_name",
                f"Unsafe spacecraft configuration name {spacecraft_name!r}.",
            )
        path = (self.root / "spacecrafts" / f"{spacecraft_name}.yml").resolve()
        expected_parent = (self.root / "spacecrafts").resolve()
        if path.parent != expected_parent:
            raise ResolutionError(
                "invalid_spacecraft_config_name", "Configuration path escaped V2 root."
            )
        try:
            raw = path.read_bytes()
        except FileNotFoundError as exc:
            raise ResolutionError(
                "control_config_not_found",
                f"No V2 spacecraft configuration exists for {spacecraft_name!r}.",
            ) from exc
        try:
            document = yaml.safe_load(raw) or {}
            value = float(document["links"][self.LINK_NAME]["frequency"])
        except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
            raise ResolutionError(
                "control_config_frequency_invalid",
                f"V2 configuration for {spacecraft_name!r} has no valid "
                f"links.{self.LINK_NAME}.frequency.",
            ) from exc
        if not 1_000_000.0 <= value <= 100_000_000_000.0:
            raise ResolutionError(
                "control_config_frequency_invalid",
                f"Configured frequency {value} Hz is outside the service bounds.",
            )
        return value, {
            "source": "control_config_v2",
            "path": str(path),
            "link": self.LINK_NAME,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "value_hz": value,
        }


class InputResolver:
    def __init__(self, *, kogs_auth: str, control_config_dir: Path):
        self.kogs_auth = kogs_auth
        self.control_config = ControlConfigV2(control_config_dir)

    @staticmethod
    def _unwrap(payload: dict, key: str) -> dict:
        candidate = payload.get(key, payload)
        if not isinstance(candidate, dict):
            raise ResolutionError(
                "kogs_payload_invalid", f"KOGS {key} payload is not an object."
            )
        return candidate

    def resolve_metadata(self, request: SolveJobRequest) -> ResolvedMetadata:
        if not self.kogs_auth:
            raise ResolutionError(
                "kogs_credentials_missing", "KOGS_API_KEY is not configured."
            )
        contact_id = str(request.contact_ids[0])
        try:
            contact_payload = get_contact(self.kogs_auth, contact_id)
            reservation = parse_reservation(self._unwrap(contact_payload, "contact"))
        except ResolutionError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize KOGS client/parser failures
            raise self._external_error(
                "kogs_contact_failed", "KOGS contact lookup failed", exc
            )
        if not reservation.spacecraft_id:
            raise ResolutionError(
                "contact_spacecraft_missing", "Contact has no spacecraft_id."
            )
        if not reservation.system_id:
            raise ResolutionError("contact_system_missing", "Contact has no system_id.")
        if not reservation.ephemeris_id:
            raise ResolutionError(
                "contact_ephemeris_missing", "Contact has no ephemeris_id."
            )
        try:
            spacecraft_payload = get_spacecraft(
                self.kogs_auth, reservation.spacecraft_id
            )
            spacecraft = parse_satellite(self._unwrap(spacecraft_payload, "spacecraft"))
        except ResolutionError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize KOGS client/parser failures
            raise self._external_error(
                "kogs_spacecraft_failed", "KOGS spacecraft lookup failed", exc
            )
        if not spacecraft.name:
            raise ResolutionError(
                "spacecraft_name_missing", "KOGS spacecraft has no name."
            )
        try:
            ephemeris = parse_ephemeris(
                get_TLE(self.kogs_auth, reservation.ephemeris_id)
            )
            tle = tle_from_inline(ephemeris.inline_tle)
        except Exception as exc:  # noqa: BLE001 - normalize KOGS client/parser failures
            raise self._external_error(
                "kogs_ephemeris_failed", "KOGS ephemeris lookup failed", exc
            )

        if request.solver.nominal_center_frequency_hz is not None:
            frequency = request.solver.nominal_center_frequency_hz
            frequency_provenance = {"source": "request", "value_hz": frequency}
        else:
            frequency, frequency_provenance = self.control_config.observed_frequency(
                spacecraft.name
            )

        return ResolvedMetadata(
            contact_id=contact_id,
            spacecraft_id=reservation.spacecraft_id,
            spacecraft_name=spacecraft.name,
            system_id=reservation.system_id,
            station_id=reservation.station_id,
            ephemeris_id=reservation.ephemeris_id,
            tle=tle,
            nominal_center_frequency_hz=float(frequency),
            frequency_provenance=frequency_provenance,
            kogs_provenance={
                "contact_id": contact_id,
                "spacecraft_id": reservation.spacecraft_id,
                "system_id": reservation.system_id,
                "station_id": reservation.station_id,
                "ephemeris_id": reservation.ephemeris_id,
            },
        )

    def prepare_input(
        self,
        request: SolveJobRequest,
        profile: dict,
        effective: dict,
        metadata: ResolvedMetadata,
    ) -> PreparedSolve:
        filt = request.telemetry_filter
        try:
            telemetry = fetch_contact_tracking_data(
                metadata.contact_id,
                require_lock=filt.require_lock,
                min_elevation_deg=filt.min_elevation_deg,
                min_doppler_hz=filt.min_doppler_hz,
                max_doppler_hz=filt.max_doppler_hz,
            )
        except Exception as exc:  # noqa: BLE001 - normalize ADX SDK failures
            raise self._external_error(
                "adx_query_failed", "ADX telemetry query failed", exc, "adx"
            )
        if telemetry.is_empty():
            raise ResolutionError(
                "telemetry_empty", "No telemetry remained after filtering."
            )
        if len(telemetry) < filt.min_pass_measurements:
            raise ResolutionError(
                "insufficient_measurements",
                f"Contact has {len(telemetry)} filtered measurements; "
                f"{filt.min_pass_measurements} are required.",
            )
        system_ids = telemetry["system_id"].drop_nulls().unique().to_list()
        try:
            stations = {
                sid: station_from_kogs(self.kogs_auth, sid) for sid in system_ids
            }
        except Exception as exc:  # noqa: BLE001 - normalize KOGS client/parser failures
            raise self._external_error(
                "kogs_station_failed", "KOGS station lookup failed", exc
            )
        if not stations:
            raise ResolutionError(
                "station_metadata_missing", "Telemetry has no station system IDs."
            )

        internal_model = self._internal_parameterization(request.solver)
        inp = build_sgp4_input(
            telemetry,
            tle=metadata.tle,
            stations=stations,
            nominal_center_frequency_hz=metadata.nominal_center_frequency_hz,
            fit_model=internal_model,
            spacecraft_id=metadata.spacecraft_id,
        )
        physical = profile["physical_settings"]
        bias = FitParameter(**physical["pass_bias"])
        fit = dataclasses.replace(
            inp.fit,
            loss=effective["loss"],
            loss_scale=effective["loss_scale"],
            doppler_sigma_hz=effective["doppler_sigma_hz"],
            max_evaluations=effective["max_evaluations"],
            pass_biases=[bias],
        )
        time_config = None
        if isinstance(request.solver, MeanElementsSolver):
            fit = dataclasses.replace(
                fit,
                ftol_rel=effective["ftol_rel"],
                xtol_rel=effective["xtol_rel"],
                mean_anomaly=FitParameter(**physical["mean_anomaly"]),
                mean_motion=FitParameter(**physical["mean_motion"]),
                center_frequency=FitParameter(**physical["center_frequency"]),
            )
        else:
            fit = dataclasses.replace(
                fit,
                center_frequency=FitParameter(**physical["center_frequency"]),
            )
            time_config = TimeSolverConfig(
                time_shift_bounds_s=tuple(physical["time_shift_bounds_s"]),
                use_qmc=effective["use_qmc"],
                qmc_samples=effective["qmc_samples"],
                pointing_penalty_weight=physical["pointing_penalty_weight"],
                pointing_n_degrees=physical["pointing_n_degrees"],
                reg_weights=tuple(physical["reg_weights"]),
            )
        inp = dataclasses.replace(inp, fit=fit)
        resolved = {
            "profile": profile,
            "effective_optimizer": effective,
            "solver_kind": request.solver.kind,
            "parameterization": request.solver.parameterization,
            "internal_fit_model": internal_model,
            "nominal_center_frequency_hz": metadata.nominal_center_frequency_hz,
            "frequency_provenance": metadata.frequency_provenance,
            "telemetry_filter": filt.model_dump(mode="json"),
            "observation_count": len(inp.observations),
            "station_count": len(inp.stations),
        }
        contact = {
            "contact_id": metadata.contact_id,
            "spacecraft_id": metadata.spacecraft_id,
            "spacecraft_name": metadata.spacecraft_name,
            "system_id": metadata.system_id,
            "station_id": metadata.station_id,
            "ephemeris_id": metadata.ephemeris_id,
            "provenance": {
                "kogs": metadata.kogs_provenance,
                "frequency": metadata.frequency_provenance,
            },
        }
        return PreparedSolve(inp, time_config, resolved, contact)

    @staticmethod
    def _internal_parameterization(solver: MeanElementsSolver | TimeShiftSolver) -> str:
        if isinstance(solver, MeanElementsSolver):
            return solver.parameterization
        return {
            "time_shift": "mean_anomaly",
            "time_shift_bias": "mean_anomaly_mean_motion",
            "time_shift_bias_frequency": "mean_anomaly_mean_motion_frequency",
        }[solver.parameterization]

    @staticmethod
    def _external_error(
        code: str, detail: str, exc: Exception, service: str = "kogs"
    ) -> ResolutionError:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        retryable = status is None or status == 429 or status >= 500
        return ResolutionError(
            code, f"{detail}: {exc}", retryable=retryable, service=service
        )


def dataclass_document(value: Any) -> dict:
    return dataclasses.asdict(value)
