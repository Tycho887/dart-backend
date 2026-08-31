"""Docker-backed PostgreSQL/Timescale integration tests.

Set ``DART_RUN_DATABASE_TESTS=1`` to enable these tests. The fixture starts an
isolated TimescaleDB container and removes it when the session finishes.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import psycopg
import pytest
from test_service_api import request_body, tdm_request_body

from dart.service.config import ServiceSettings
from dart.service.database import Database, IdempotencyConflict, JobOwnershipConflict

pytestmark = pytest.mark.skipif(
    os.getenv("DART_RUN_DATABASE_TESTS") != "1",
    reason="set DART_RUN_DATABASE_TESTS=1 to start the Docker database fixture",
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def database():
    port = _free_port()
    name = f"dart-service-test-{uuid4().hex[:10]}"
    subprocess.run(
        [
            "docker",
            "run",
            "--detach",
            "--name",
            name,
            "--env",
            "POSTGRES_PASSWORD=postgres",
            "--env",
            "POSTGRES_DB=results",
            "--publish",
            f"127.0.0.1:{port}:5432",
            "timescale/timescaledb:latest-pg16",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    dsn = f"postgresql://postgres:postgres@127.0.0.1:{port}/results"
    deadline = time.monotonic() + 60
    while True:
        try:
            with psycopg.connect(dsn):
                break
        except psycopg.OperationalError:
            if time.monotonic() >= deadline:
                subprocess.run(["docker", "logs", name], check=False)
                raise
            time.sleep(0.25)
    db = Database(ServiceSettings(database_url=dsn))
    db.migrate()
    try:
        yield db
    finally:
        db.close()
        subprocess.run(
            ["docker", "rm", "--force", name], check=False, capture_output=True
        )


def test_idempotency_claim_artifacts_and_grafana_views(database):
    body = request_body()
    job_id, replay = database.submit_job(
        request_json=body,
        actor_id="operator-1",
        actor_type="human",
        idempotency_key="database-test-1",
        max_attempts=3,
    )
    replay_id, replay = database.submit_job(
        request_json=body,
        actor_id="operator-1",
        actor_type="human",
        idempotency_key="database-test-1",
        max_attempts=3,
    )
    assert replay is True and replay_id == job_id
    changed = request_body()
    changed["telemetry_filter"]["min_elevation_deg"] = 5.0
    with pytest.raises(IdempotencyConflict):
        database.submit_job(
            request_json=changed,
            actor_id="operator-1",
            actor_type="human",
            idempotency_key="database-test-1",
            max_attempts=3,
        )

    claimed = database.claim_job("worker-1", 300)
    assert claimed.id == job_id
    run_id = uuid4()
    database.store_resolution(
        job_id=job_id,
        worker_id="worker-1",
        resolved_configuration={"frequency_hz": 2.2e9},
        contact={
            "contact_id": body["contact_ids"][0],
            "spacecraft_id": "spacecraft-1",
            "spacecraft_name": "TESTSAT",
            "system_id": "system-1",
            "station_id": "station-1",
            "ephemeris_id": "ephemeris-1",
            "provenance": {"source": "test"},
        },
        run_id=run_id,
        algorithm="sgp4_mean_elements",
        parameterization="mean_anomaly_mean_motion",
    )
    database.store_artifact(
        job_id=job_id,
        run_id=run_id,
        kind="normalized_solver_input",
        content_type="application/json",
        data={"schema_version": 2},
    )
    assert (
        database.finish_success(
            job_id=job_id,
            worker_id="worker-1",
            run_id=run_id,
            summary={"converged": True, "residual_rms_hz": 1.5},
            warnings=[],
        )
        == "succeeded"
    )

    with database.pool.connection() as conn:
        status = conn.execute(
            "SELECT * FROM dart.job_status_v1 WHERE job_id = %s", (job_id,)
        ).fetchone()
        result = conn.execute(
            "SELECT * FROM dart.job_results_v1 WHERE job_id = %s", (job_id,)
        ).fetchone()
        artifact = conn.execute(
            "SELECT sha256 FROM dart.job_artifacts WHERE job_id = %s", (job_id,)
        ).fetchone()
    assert status["status"] == "succeeded"
    assert status["operation"] == "solve"
    assert status["contact_id"] == uuid4().__class__(body["contact_ids"][0])
    assert result["result_summary"]["converged"] is True
    assert len(artifact["sha256"]) == 64


def test_tdm_artifact_grafana_view(database):
    body = tdm_request_body()
    job_id, _ = database.submit_job(
        request_json=body,
        actor_id="operator-1",
        actor_type="human",
        idempotency_key="tdm-database-test",
        max_attempts=3,
        operation="tdm_export",
    )
    claimed = database.claim_job("worker-tdm", 300)
    assert claimed.id == job_id
    assert claimed.operation == "tdm_export"
    run_id = uuid4()
    database.store_resolution(
        job_id=job_id,
        worker_id="worker-tdm",
        resolved_configuration={"profile": body["profile"]},
        contact={"contact_id": body["contact_id"], "provenance": {}},
        run_id=run_id,
        algorithm="ksat_tdm",
        parameterization="track_mode_4",
    )
    filename = "TRACK_SG221_2024-149A_2026-08-28T12-34-56.tdm"
    text = "CCSDS_TDM_VERS = 2.0\nDATA_STOP\n"
    database.store_artifact(
        job_id=job_id,
        run_id=run_id,
        kind="tdm",
        content_type="text/plain; charset=us-ascii",
        data=text.encode("ascii"),
        filename=filename,
        metadata={"product": "track"},
    )
    database.finish_success(
        job_id=job_id,
        worker_id="worker-tdm",
        run_id=run_id,
        summary={"product": "track", "filename": filename},
        warnings=[],
    )

    with database.pool.connection() as conn:
        artifact = conn.execute(
            "SELECT * FROM dart.tdm_artifacts_v1 WHERE job_id = %s", (job_id,)
        ).fetchone()
    assert artifact["product"] == "track"
    assert artifact["filename"] == filename
    assert artifact["tdm_text"] == text
    assert artifact["byte_count"] == len(text)


def test_concurrent_claims_are_distinct(database):
    ids = []
    for index in range(2):
        job_id, _ = database.submit_job(
            request_json=request_body(),
            actor_id="worker-test",
            actor_type="service",
            idempotency_key=f"concurrent-{index}",
            max_attempts=3,
        )
        ids.append(job_id)
    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(
            executor.map(
                lambda worker: database.claim_job(worker, 300),
                ["worker-a", "worker-b"],
            )
        )
    assert {claim.id for claim in claims} == set(ids)


def test_cancel_is_actor_owned(database):
    job_id, _ = database.submit_job(
        request_json=request_body(),
        actor_id="owner",
        actor_type="human",
        idempotency_key="cancel-me",
        max_attempts=3,
    )
    with pytest.raises(JobOwnershipConflict):
        database.cancel_job(job_id, "somebody-else")
    canceled = database.cancel_job(job_id, "owner")
    assert canceled["status"] == "canceled"
    assert canceled["cancel_requested"] is True


def test_expired_leases_recover_then_exhaust(database):
    job_id, _ = database.submit_job(
        request_json=request_body(),
        actor_id="lease-test",
        actor_type="service",
        idempotency_key="lease-recovery",
        max_attempts=2,
    )
    first = database.claim_job("worker-lease-a", 300)
    assert first.id == job_id and first.attempt_count == 1
    with database.pool.connection() as conn, conn.transaction():
        conn.execute(
            "UPDATE dart.jobs SET lease_expires_at = now() - interval '1 second' WHERE id = %s",
            (job_id,),
        )
    assert database.recover_expired_leases() == 1
    second = database.claim_job("worker-lease-b", 300)
    assert second.id == job_id and second.attempt_count == 2
    with database.pool.connection() as conn, conn.transaction():
        conn.execute(
            "UPDATE dart.jobs SET lease_expires_at = now() - interval '1 second' WHERE id = %s",
            (job_id,),
        )
    assert database.recover_expired_leases() == 1
    with database.pool.connection() as conn:
        row = conn.execute(
            "SELECT status, terminal_error FROM dart.jobs WHERE id = %s", (job_id,)
        ).fetchone()
    assert row["status"] == "failed"
    assert row["terminal_error"]["code"] == "worker_lease_exhausted"


def test_transient_failure_is_retried_then_terminal(database):
    job_id, _ = database.submit_job(
        request_json=request_body(),
        actor_id="retry-test",
        actor_type="service",
        idempotency_key="retry-policy",
        max_attempts=2,
    )
    first = database.claim_job("worker-retry", 300)
    error = {"code": "adx_query_failed", "retryable": True}
    assert (
        database.fail_or_retry(
            job=first,
            worker_id="worker-retry",
            error=error,
            retryable=True,
        )
        == "queued"
    )
    with database.pool.connection() as conn, conn.transaction():
        conn.execute(
            "UPDATE dart.jobs SET available_at = now() WHERE id = %s", (job_id,)
        )
    second = database.claim_job("worker-retry", 300)
    assert second.attempt_count == 2
    assert (
        database.fail_or_retry(
            job=second,
            worker_id="worker-retry",
            error=error,
            retryable=True,
        )
        == "failed"
    )
