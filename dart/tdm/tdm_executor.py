"""Durable worker execution for one profiled KSAT TDM product."""

from __future__ import annotations

from collections.abc import Callable
from uuid import UUID

from pydantic import ValidationError

from dart.io.ksat_tdm import write_angle_tdm, write_track_tdm

from .config import ServiceSettings
from .database import ClaimedJob, Database
from .metrics import JOB_OUTCOMES, STAGE_LATENCY
from .models import TdmJobRequest
from .resolver import ResolutionError
from .tdm_profiles import AngleProfile, TrackProfile, validate_tdm_profile


class TdmJobExecutor:
    def __init__(
        self,
        database: Database,
        settings: ServiceSettings,
        cancel_if_requested: Callable[[ClaimedJob], bool],
    ):
        self.database = database
        self.settings = settings
        self.cancel_if_requested = cancel_if_requested

    def execute(self, job: ClaimedJob, run_id: UUID) -> None:
        request, profile = self._resolve(job)
        resolved = {
            "profile": profile.model_dump(mode="json"),
            "product": request.product,
        }
        parameterization = self._parameterization(profile)
        contact = {
            "contact_id": str(request.contact_id),
            "provenance": {"tdm_profile": resolved["profile"]},
        }
        self._store_resolution(job, run_id, resolved, contact, parameterization)
        if self.cancel_if_requested(job):
            return
        self.database.set_stage(
            job.id,
            self.settings.worker_id,
            "loading_telemetry",
            "loading_telemetry",
        )
        self.database.set_stage(
            job.id, self.settings.worker_id, "running", "running"
        )
        result, warnings = self._generate(request, profile)
        if result.metadata is not None:
            contact = self._resolved_contact(request, resolved, result.metadata)
            self._store_resolution(
                job, run_id, resolved, contact, parameterization
            )
        payload = result.text.encode("ascii")
        self.database.store_artifact(
            job_id=job.id,
            run_id=run_id,
            kind="tdm",
            content_type="text/plain; charset=us-ascii",
            data=payload,
            filename=result.filename,
            metadata={"product": request.product},
        )
        status = self.database.finish_success(
            job_id=job.id,
            worker_id=self.settings.worker_id,
            run_id=run_id,
            summary={
                "product": request.product,
                "filename": result.filename,
                "byte_count": len(payload),
            },
            warnings=warnings,
        )
        JOB_OUTCOMES.labels(outcome=status).inc()

    def _resolve(self, job: ClaimedJob):
        try:
            request = TdmJobRequest.model_validate(job.request_json)
            document = self.database.get_tdm_profile(
                request.profile.name, request.profile.version
            )
            return request, validate_tdm_profile(request, document)
        except (ValidationError, KeyError, ValueError) as exc:
            raise ResolutionError("stored_tdm_request_invalid", str(exc)) from exc

    def _generate(self, request, profile):
        with STAGE_LATENCY.labels(stage="running").time():
            try:
                runtime = profile.runtime(str(request.contact_id))
                if isinstance(profile, TrackProfile):
                    result = write_track_tdm(
                        runtime, kogs_api_key=self.settings.kogs_api_key
                    )
                    return result, []
                if isinstance(profile, AngleProfile):
                    result = write_angle_tdm(
                        runtime, kogs_api_key=self.settings.kogs_api_key
                    )
                    return result, list(result.warnings)
                raise TypeError("unknown TDM product")
            except (LookupError, NotImplementedError, TypeError, ValueError) as exc:
                raise ResolutionError("tdm_export_invalid", str(exc)) from exc
            except Exception as exc:
                raise ResolutionError(
                    "tdm_source_failed",
                    f"TDM source request failed: {exc}",
                    retryable=True,
                    service="tdm_source",
                ) from exc

    def _store_resolution(
        self, job, run_id, resolved, contact, parameterization
    ) -> None:
        self.database.store_resolution(
            job_id=job.id,
            worker_id=self.settings.worker_id,
            resolved_configuration=resolved,
            contact=contact,
            run_id=run_id,
            algorithm="ksat_tdm",
            parameterization=parameterization,
        )

    @staticmethod
    def _parameterization(profile) -> str:
        return "track_mode_4" if isinstance(profile, TrackProfile) else "angle_azel"

    @staticmethod
    def _resolved_contact(request, resolved, metadata) -> dict:
        return {
            "contact_id": str(request.contact_id),
            "spacecraft_id": metadata.spacecraft_id,
            "spacecraft_name": metadata.spacecraft,
            "system_id": metadata.system_id,
            "station_id": metadata.station_id,
            "ephemeris_id": metadata.ephemeris_id,
            "provenance": {
                "tdm_profile": resolved["profile"],
                "kogs": {
                    "spacecraft_id": metadata.spacecraft_id,
                    "system_id": metadata.system_id,
                    "station_id": metadata.station_id,
                    "ephemeris_id": metadata.ephemeris_id,
                },
            },
        }
