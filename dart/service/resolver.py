"""Acquire selected or contact-linked priors and bounded inputs through dart.io."""

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
from dart.io.load import LoadError
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


def _latest_contact(contacts: list[ContactMetadata]) -> ContactMetadata:
    if not contacts:
        raise InputValidationError("no contacts available to select a prior ephemeris")
    return max(contacts, key=lambda contact: (contact.start, contact.contact_id))


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
    expected_id = (
        str(request.ephemeris_id)
        if request.ephemeris_id is not None
        else _latest_contact(contacts).ephemeris_id
    )
    if not ephemeris.ephemeris_id or ephemeris.ephemeris_id != expected_id:
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

    def _ephemeris(
        self,
        configuration: ResolvedEstimateConfiguration,
        contacts: list[ContactMetadata],
    ) -> tuple[EphemerisMetadata, dict]:
        requested_id = configuration.request.ephemeris_id
        if requested_id is not None:
            ephemeris = kogs.get_ephemeris(
                self.settings.kogs_api_key, str(requested_id), timeout_seconds=30
            )
            return ephemeris, {"source": "request", "ephemeris_id": str(requested_id)}
        contact = _latest_contact(contacts)
        return contact.ephemeris, {
            "source": "latest_contact",
            "contact_id": contact.contact_id,
            "contact_start": contact.start.isoformat(),
            "ephemeris_id": contact.ephemeris_id,
        }

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
        try:
            with adx.client_from_env() as client:
                contacts, measurements = asyncio.run(
                    load_passes(
                        [str(cid) for cid in request.contact_ids],
                        kogs_api_key=key,
                        adx_client=client,
                        timeout_seconds=30,
                    )
                )
        except LoadError as exc:
            if isinstance(exc.__cause__, kogs.KogsError):
                raise InputValidationError(
                    "Contact metadata or its associated ephemeris is unavailable or invalid; "
                    "check the selected contact and its ephemeris in KOGS."
                ) from exc
            raise
        ephemeris, prior_provenance = self._ephemeris(configuration, contacts)
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
                "prior_selection": prior_provenance,
                "frequency": frequency_provenance,
                "raw_samples": len(measurements),
                "retained_samples": len(prior.observations.observations),
                "measurement_selection": request.measurement_selection.model_dump(
                    mode="json"
                ),
            },
        )
