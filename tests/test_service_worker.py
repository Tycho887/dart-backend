from __future__ import annotations

from uuid import uuid4

from test_service_api import request_body

from dart.schema import Observation, Sgp4Input, SolverResult, Station, Tle
from dart.service.config import ServiceSettings
from dart.service.database import ClaimedJob
from dart.service.resolver import PreparedSolve, ResolvedMetadata
from dart.service.worker import Worker


class FakeResolver:
    def __init__(self):
        self.request = None

    def resolve_metadata(self, request):
        self.request = request
        return ResolvedMetadata(
            contact_id=str(request.contact_ids[0]),
            spacecraft_id="spacecraft-1",
            spacecraft_name="TESTSAT",
            system_id="system-1",
            station_id="station-1",
            ephemeris_id="ephemeris-1",
            tle=Tle("line1", "line2"),
            nominal_center_frequency_hz=2.2e9,
            frequency_provenance={"source": "request"},
            kogs_provenance={},
        )

    def prepare_input(self, request, profile, effective, metadata):
        inp = Sgp4Input(
            spacecraft_id="spacecraft-1",
            epoch_unix=1.0,
            tle=Tle("line1", "line2"),
            stations=[Station(id="system-1")],
            observations=[
                Observation(
                    epoch_unix=2.0,
                    doppler_hz=3.0,
                    azimuth_deg=4.0,
                    elevation_deg=5.0,
                    station_id="system-1",
                    contact_id=metadata.contact_id,
                )
            ],
        )
        return PreparedSolve(
            input=inp,
            time_config=None,
            resolved_configuration={"profile": profile, "effective": effective},
            contact_record={
                "contact_id": metadata.contact_id,
                "spacecraft_id": metadata.spacecraft_id,
                "spacecraft_name": metadata.spacecraft_name,
                "system_id": metadata.system_id,
                "station_id": metadata.station_id,
                "ephemeris_id": metadata.ephemeris_id,
                "provenance": {},
            },
        )


class FakeWorkerDatabase:
    def __init__(self, body):
        self.job = ClaimedJob(uuid4(), body, 1, 3)
        self.artifacts = []
        self.finished = None
        self.stages = []

    def recover_expired_leases(self):
        return 0

    def claim_job(self, worker_id, lease_seconds):
        job, self.job = self.job, None
        return job

    def heartbeat(self, *args):
        return True

    def cancellation_requested(self, job_id):
        return False

    def set_stage(self, job_id, worker_id, status, stage):
        self.stages.append(stage)

    def store_resolution(self, **kwargs):
        self.run_id = kwargs["run_id"]

    def store_artifact(self, **kwargs):
        self.artifacts.append((kwargs["kind"], kwargs["content_type"], kwargs["data"]))

    def finish_success(self, **kwargs):
        self.finished = ("succeeded", kwargs["summary"])
        return "succeeded"

    def finish_solver_failure(self, **kwargs):
        self.finished = ("failed", kwargs["summary"])

    def fail_or_retry(self, **kwargs):
        self.finished = ("failed", kwargs["error"])
        return "failed"

    def queue_metrics(self):
        return 0, 0.0


def test_worker_persists_snapshots_and_compact_result(monkeypatch):
    db = FakeWorkerDatabase(request_body())
    result = SolverResult(
        success=True,
        converged=True,
        message="ok",
        rms=12.5,
        parameter_names=("mean_anomaly_rad",),
        parameters=(0.01,),
        parameter_covariance=(0.0004,),
        covariance_rank=1,
        residuals=(1.0, -1.0),
    )
    monkeypatch.setattr("dart.service.worker.solve_mean_elements", lambda inp: result)
    settings = ServiceSettings(
        database_url="postgresql://unused/results",
        worker_id="worker-1",
        heartbeat_seconds=60,
        lease_seconds=300,
    )

    assert Worker(db, settings, resolver=FakeResolver()).run_once() is True

    assert db.stages == ["loading_telemetry", "running"]
    assert {(kind, content_type) for kind, content_type, _ in db.artifacts} == {
        ("normalized_solver_input", "application/json"),
        ("normalized_solver_input", "application/msgpack"),
        ("metadata_provenance", "application/json"),
        ("solver_output", "application/json"),
        ("solver_output", "application/msgpack"),
    }
    assert db.finished[0] == "succeeded"
    assert db.finished[1]["parameters"][0]["standard_uncertainty"] == 0.02
