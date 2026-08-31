from __future__ import annotations

from uuid import uuid4

from test_service_api import TDM_PROFILE, request_body, tdm_request_body

from dart.io.ksat_tdm import TrackResult
from dart.schema import Observation, Sgp4Input, SolverResult, Station, Tle
from dart.service.config import ServiceSettings
from dart.service.database import ClaimedJob
from dart.service.resolver import PreparedSolve, ResolvedMetadata
from dart.service.worker import Worker, result_summary


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
    def __init__(self, body, operation="solve"):
        self.job = ClaimedJob(uuid4(), body, 1, 3, operation)
        self.artifacts = []
        self.artifact_records = []
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
        self.artifact_records.append(kwargs)

    def get_tdm_profile(self, name, version):
        assert (name, version) == ("sg221-track", 1)
        return TDM_PROFILE

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


def test_worker_generates_durable_tdm_artifact(monkeypatch):
    db = FakeWorkerDatabase(tdm_request_body(), operation="tdm_export")
    monkeypatch.setattr(
        "dart.service.tdm_executor.write_track_tdm",
        lambda request, **kwargs: TrackResult(
            "TRACK_SG221_2024-149A_2026-08-28T12-34-56.tdm",
            "CCSDS_TDM_VERS = 2.0\nDATA_STOP\n",
        ),
    )
    settings = ServiceSettings(
        database_url="postgresql://unused/results",
        worker_id="worker-1",
        heartbeat_seconds=60,
        lease_seconds=300,
        kogs_api_key="secret",
    )

    assert Worker(db, settings, resolver=FakeResolver()).run_once() is True

    artifact = db.artifact_records[0]
    assert artifact["kind"] == "tdm"
    assert artifact["content_type"] == "text/plain; charset=us-ascii"
    assert artifact["filename"].startswith("TRACK_SG221_")
    assert artifact["metadata"] == {"product": "track"}
    assert db.finished == (
        "succeeded",
        {
            "product": "track",
            "filename": artifact["filename"],
            "byte_count": len(artifact["data"]),
        },
    )


def test_result_summary_assigns_rust_parameter_units():
    summary, _ = result_summary(
        SolverResult(
            parameter_names=(
                "delta_mean_anomaly_rad",
                "delta_mean_motion_rad_s",
                "doppler_bias_hz:pass-1",
            ),
            parameters=(1.0, 2.0, 3.0),
            parameter_covariance=(1.0,) * 9,
            covariance_rank=3,
        )
    )
    assert [item["unit"] for item in summary["parameters"]] == [
        "rad",
        "rad/s",
        "Hz",
    ]
