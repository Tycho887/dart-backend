from __future__ import annotations

from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from dart.service.api import create_app
from dart.service.config import ServiceSettings
from dart.service.database import IdempotencyConflict, JobNotFound, JobOwnershipConflict
from dart.service.profiles import profile_documents

CONTACT_ID = "11111111-1111-4111-8111-111111111111"
HEADERS = {
    "X-DART-Actor-ID": "operator-1",
    "X-DART-Actor-Type": "human",
    "Idempotency-Key": "submission-1",
}

TDM_PROFILE = {
    "name": "sg221-track",
    "version": 1,
    "product": "track",
    "station": "SG221",
    "band": "S",
    "integration_interval_s": 1.0,
    "turnaround_numerator": 240,
    "turnaround_denominator": 221,
    "integration_end_column": "timestamp",
    "transmit": {"link_name": "s_band_uplink_p1_1"},
    "receive": {
        "link_name": "s_band_downlink_p1_1",
        "offset_column": "receiver_offset",
    },
    "calibration": {
        "pedestal_offset_m": 4.0,
        "tlt_calibration_date": "2026-08-01",
        "correction_doppler_hz": -0.125,
    },
}


def request_body(*, kind="sgp4_mean_elements", strategy="single_contact"):
    if kind == "sgp4_mean_elements":
        solver = {
            "kind": kind,
            "parameterization": "mean_anomaly_mean_motion",
            "nominal_center_frequency_hz": 2.2e9,
            "optimizer": {
                "profile": "sgp4-production",
                "version": 1,
                "overrides": {"loss": "soft_l1", "max_evaluations": 300},
            },
        }
    else:
        solver = {
            "kind": kind,
            "parameterization": "time_shift_bias",
            "optimizer": {
                "profile": "time-shift-production",
                "version": 1,
                "overrides": {"use_qmc": True, "qmc_samples": 20},
            },
        }
    return {
        "contact_ids": [CONTACT_ID],
        "strategy": strategy,
        "ephemeris_id": None,
        "solver": solver,
        "telemetry_filter": {
            "require_lock": False,
            "min_elevation_deg": 1.0,
            "min_doppler_hz": 1.0,
            "max_doppler_hz": 100000.0,
            "min_pass_measurements": 50,
        },
        "client_context": {"label": "operator test", "tags": ["leop"]},
    }


def tdm_request_body():
    return {
        "contact_id": CONTACT_ID,
        "product": "track",
        "profile": {"name": "sg221-track", "version": 1},
        "client_context": {"label": "TDM export", "tags": ["ksat"]},
    }


class FakeDatabase:
    def __init__(self):
        self.jobs = {}

    def migrate(self):
        pass

    def healthcheck(self):
        pass

    def list_profiles(self):
        return profile_documents()

    def list_tdm_profiles(self):
        return [TDM_PROFILE]

    def get_tdm_profile(self, name, version):
        if (name, version) != ("sg221-track", 1):
            raise KeyError((name, version))
        return TDM_PROFILE

    def submit_job(
        self,
        *,
        request_json,
        actor_id,
        actor_type,
        idempotency_key,
        max_attempts,
        operation="solve",
    ):
        key = (actor_id, idempotency_key)
        existing = self.jobs.get(key)
        if existing:
            if existing[1] != request_json or existing[-1] != operation:
                raise IdempotencyConflict(idempotency_key)
            return existing[0], True
        job_id = uuid4()
        self.jobs[key] = (
            job_id,
            request_json,
            actor_type,
            max_attempts,
            "queued",
            operation,
        )
        return job_id, False

    def cancel_job(self, job_id, actor_id):
        for (owner, _), value in self.jobs.items():
            if value[0] == job_id:
                if owner != actor_id:
                    raise JobOwnershipConflict(str(job_id))
                return {
                    "job_id": job_id,
                    "status": "canceled",
                    "cancel_requested": True,
                }
        raise JobNotFound(str(job_id))


def client():
    app = create_app(
        settings=ServiceSettings(database_url="postgresql://unused/results"),
        database=FakeDatabase(),
        migrate_on_start=False,
    )
    return TestClient(app)


def test_submit_and_idempotent_replay():
    with client() as http:
        first = http.post("/v1/solve-jobs", json=request_body(), headers=HEADERS)
        second = http.post("/v1/solve-jobs", json=request_body(), headers=HEADERS)
    assert first.status_code == 202
    assert UUID(first.json()["job_id"])
    assert first.json()["idempotent_replay"] is False
    assert second.json() == {**first.json(), "idempotent_replay": True}
    assert "location" not in {key.lower() for key in first.headers}


def test_idempotency_conflict_is_problem_json():
    with client() as http:
        assert (
            http.post(
                "/v1/solve-jobs", json=request_body(), headers=HEADERS
            ).status_code
            == 202
        )
        changed = request_body()
        changed["telemetry_filter"]["min_elevation_deg"] = 2.0
        response = http.post("/v1/solve-jobs", json=changed, headers=HEADERS)
    assert response.status_code == 409
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "idempotency_key_reused"


def test_joint_is_advertised_but_rejected():
    with client() as http:
        capabilities = http.get("/v1/capabilities").json()
        response = http.post(
            "/v1/solve-jobs/validate",
            json=request_body(strategy="joint"),
            headers={k: v for k, v in HEADERS.items() if k != "Idempotency-Key"},
        )
    assert (
        next(s for s in capabilities["strategies"] if s["name"] == "joint")["available"]
        is False
    )
    assert response.status_code == 409
    assert response.json()["code"] == "capability_unavailable"


def test_strict_tagged_union_and_required_identity_headers():
    body = request_body()
    body["solver"]["unknown"] = True
    with client() as http:
        strict = http.post("/v1/solve-jobs", json=body, headers=HEADERS)
        missing_actor = http.post(
            "/v1/solve-jobs", json=request_body(), headers={"Idempotency-Key": "x"}
        )
    assert strict.status_code == 422
    assert strict.json()["code"] == "request_validation_failed"
    assert missing_actor.status_code == 422


def test_validation_resolves_profile_without_external_calls():
    headers = {k: v for k, v in HEADERS.items() if k != "Idempotency-Key"}
    with client() as http:
        explicit = http.post(
            "/v1/solve-jobs/validate", json=request_body(), headers=headers
        )
        deferred = http.post(
            "/v1/solve-jobs/validate",
            json=request_body(kind="sgp4_time_shift"),
            headers=headers,
        )
    assert explicit.json()["frequency_source"] == "request"
    assert explicit.json()["effective_settings"]["max_evaluations"] == 300
    assert deferred.json()["frequency_source"] == "control_config_deferred"


def test_openapi_has_dispatch_surface_without_result_reads():
    with client() as http:
        schema = http.get("/v1/openapi.json").json()
    paths = schema["paths"]
    assert "/v1/solve-jobs" in paths
    assert "/v1/tdm-jobs" in paths
    assert "/v1/jobs/{job_id}/cancel" in paths
    assert "/v1/jobs/{job_id}" not in paths
    assert "/v1/jobs/{job_id}/result" not in paths


def test_tdm_validation_submission_and_discovery():
    validation_headers = {
        key: value for key, value in HEADERS.items() if key != "Idempotency-Key"
    }
    with client() as http:
        profiles = http.get("/v1/tdm-profiles").json()
        validation = http.post(
            "/v1/tdm-jobs/validate",
            json=tdm_request_body(),
            headers=validation_headers,
        )
        submission = http.post(
            "/v1/tdm-jobs", json=tdm_request_body(), headers=HEADERS
        )
    assert profiles == [TDM_PROFILE]
    assert validation.status_code == 200
    assert validation.json()["resolved_profile"]["product"] == "track"
    assert submission.status_code == 202


def test_tdm_profile_product_mismatch_is_rejected():
    body = tdm_request_body()
    body["product"] = "angle"
    with client() as http:
        response = http.post("/v1/tdm-jobs", json=body, headers=HEADERS)
    assert response.status_code == 422
    assert response.json()["code"] == "tdm_profile_not_found"


def test_browser_preflight_allows_grafana_headers():
    with client() as http:
        response = http.options(
            "/v1/solve-jobs",
            headers={
                "Origin": "https://grafana.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": (
                    "content-type,idempotency-key,x-dart-actor-id,x-dart-actor-type"
                ),
            },
        )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
