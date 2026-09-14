"""Acquire explicit priors and contact-bounded measurements through dart.io."""

import asyncio
import hashlib
import io
from dataclasses import dataclass

import polars as pl
import satkit as sk

from dart.io import (
    ContactMetadata,
    EphemerisMetadata,
    adx,
    ctrl_config,
    kogs,
    load_passes,
)
from dart.io.doppler import prepare_doppler
from dart.od import PriorStateData

from .config import ServiceSettings
from .models import ResolvedEstimateConfiguration


class InputValidationError(ValueError):
    """A safe, actionable input diagnostic without provider credentials."""


@dataclass(frozen=True)
class PreparedEstimate:
    prior: PriorStateData
    raw_measurements: bytes
    provenance: dict


def prepare_prior(
    configuration: ResolvedEstimateConfiguration,
    contacts: list[ContactMetadata],
    measurements: pl.DataFrame,
    ephemeris: EphemerisMetadata,
    frequency_hz: float,
) -> PriorStateData:
    request = configuration.request
    if [c.contact_id for c in contacts] != [str(cid) for cid in request.contact_ids]:
        raise InputValidationError("loaded contacts differ from requested order")
    if ephemeris.ephemeris_id != str(request.ephemeris_id):
        raise InputValidationError("selected prior ephemeris identity mismatch")
    if {c.spacecraft_id for c in contacts} != {ephemeris.spacecraft_id}:
        raise InputValidationError(
            "contacts and selected prior must belong to one spacecraft"
        )
    if not ephemeris.tle:
        raise InputValidationError("these profiles require a selected TLE prior")
    selection = request.measurement_selection
    selected = measurements.filter(
        pl.col("elevation_deg").is_finite()
        & (pl.col("elevation_deg") >= selection.min_elevation_deg)
        & pl.col("doppler_hz")
        .abs()
        .is_between(selection.min_abs_doppler_hz, selection.max_abs_doppler_hz)
    )
    if selection.min_ebn0_db is not None:
        selected = selected.filter(
            pl.col("ebn0").is_finite() & (pl.col("ebn0") >= selection.min_ebn0_db)
        )
    try:
        context, _ = prepare_doppler(
            contacts,
            selected,
            center_frequency_hz=frequency_hz,
            variance_hz2=selection.doppler_sigma_hz**2,
            min_samples=selection.min_samples_per_contact,
        )
    except ValueError as exc:
        raise InputValidationError(str(exc)) from exc
    epoch = min(o.time for o in context.observations) - sk.duration(seconds=1)
    return PriorStateData(context, ephemeris, epoch)


class InputResolver:
    def __init__(self, settings: ServiceSettings):
        self.settings = settings

    def _frequency(
        self,
        configuration: ResolvedEstimateConfiguration,
        contacts: list[ContactMetadata],
    ) -> tuple[float, dict]:
        override = configuration.request.nominal_center_frequency_hz
        if override is not None:
            return override, {"source": "request", "value_hz": override}
        name = contacts[0].spacecraft
        root = self.settings.control_config_v2_dir
        value = ctrl_config.get_observed_frequency(name, root=root)
        path = ctrl_config.spacecraft_config_path(root, name)
        return value, {
            "source": "ctrl-config-v2",
            "link": ctrl_config.OBSERVED_LINK_NAME,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "value_hz": value,
        }

    def prepare(self, configuration: ResolvedEstimateConfiguration) -> PreparedEstimate:
        key = self.settings.kogs_api_key
        if not key:
            raise InputValidationError("KOGS credentials are not configured")
        request = configuration.request
        ephemeris = kogs.get_ephemeris(
            key, str(request.ephemeris_id), timeout_seconds=30
        )
        with adx.client_from_env() as client:
            contacts, measurements = asyncio.run(
                load_passes(
                    [str(cid) for cid in request.contact_ids],
                    kogs_api_key=key,
                    adx_client=client,
                    timeout_seconds=30,
                )
            )
        frequency, frequency_provenance = self._frequency(configuration, contacts)
        prior = prepare_prior(
            configuration, contacts, measurements, ephemeris, frequency
        )
        buffer = io.BytesIO()
        measurements.write_parquet(buffer)
        return PreparedEstimate(
            prior,
            buffer.getvalue(),
            {
                "frequency": frequency_provenance,
                "raw_samples": len(measurements),
                "retained_samples": len(prior.observations.observations),
                "measurement_selection": request.measurement_selection.model_dump(
                    mode="json"
                ),
            },
        )
