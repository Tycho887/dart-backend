"""Opt-in tests against an isolated PostgreSQL/TimescaleDB database."""

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import numpy as np
import pytest
from psycopg import sql
from service_fixtures import configuration, prior_fixture

from dart.service.config import ServiceSettings
from dart.service.database import (
    Database,
    IdempotencyConflict,
    JobOwnershipConflict,
    LeaseLost,
)
from dart.service.resolver import PreparedEstimate
from dart.service.serialization import (
    document,
    prior_document,
    result_rows,
    software_versions,
)
from dart.service.worker import Worker, run_estimate


@pytest.fixture
def database():
    url = os.getenv("DART_TEST_DATABASE_URL")
    if not url:
        pytest.skip("set DART_TEST_DATABASE_URL to an isolated test database")
    settings = ServiceSettings(
        url,
        database_schema=f"test_{uuid4().hex}",
        gateway_token="test",
        heartbeat_seconds=0.2,
    )
    db = Database(settings)
    try:
        db.migrate()
        db.migrate()
        yield db
    finally:
        with db.pool.connection() as conn:
            conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(settings.database_schema)
                )
            )
        db.close()


def submit(db, config=None, key=None):
    return db.submit_estimate(
        config or configuration(), "engineer", "human", key or str(uuid4())
    )


def test_concurrent_idempotency_claim_and_cancellation(database):
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda _: submit(database, key="one"), range(4)))
    assert len({r.estimate_uuid for r in responses}) == 1
    assert sum(not r.idempotent_replay for r in responses) == 1
    with pytest.raises(IdempotencyConflict):
        submit(database, configuration("hifi"), "one")
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = list(pool.map(lambda _: database.claim_job("worker", 30), range(2)))
    job = next(j for j in jobs if j is not None)
    assert sum(j is not None for j in jobs) == 1
    with pytest.raises(JobOwnershipConflict):
        database.cancel_job(job.id, "another-user")
    database.cancel_job(job.id, "engineer")
    assert database.set_stage(job, "worker", "running") is False
    replay = submit(database, key="one")
    assert replay.status == "canceled" and replay.idempotent_replay


def test_lease_recovery_fences_old_attempt_even_with_same_worker_id(database):
    accepted = submit(database)
    first = database.claim_job("worker", 30)
    with database.pool.connection() as conn:
        conn.execute(
            database.query(
                "UPDATE {s}.jobs SET lease_expires_at=now()-interval '1 second' WHERE id=%s"
            ),
            (accepted.job_id,),
        )
    assert database.recover_expired_leases() == 1
    second = database.claim_job("worker", 30)
    assert second.estimate_uuid == first.estimate_uuid and second.run_id != first.run_id
    for action in (
        lambda: database.heartbeat(first, "worker", 30),
        lambda: database.set_stage(first, "worker", "running"),
        lambda: database.fail_or_retry(first, "worker", {"code": "stale"}, False),
    ):
        with pytest.raises(LeaseLost):
            action()
    assert database.set_stage(second, "worker", "running")


@pytest.mark.parametrize(
    "model", ["lofi-time", "lofi-time-frequency", "lofi-elements", "hifi"]
)
def test_worker_publishes_frozen_fit_and_artifacts(database, model):
    config = configuration(model, multipass=model != "lofi-time")
    prior = prior_fixture(config)

    class FrozenResolver:
        def prepare(self, configuration):
            return PreparedEstimate(prior, b"raw fixture", {"source": "test"})

    accepted = submit(database, config)
    worker = Worker(database, database.settings, resolver=FrozenResolver())
    assert worker.run_once()
    expected, _, _ = run_estimate(prior, config)
    with database.pool.connection() as conn:
        summary = conn.execute(
            database.query("SELECT * FROM {s}.estimates_v1 WHERE estimate_uuid=%s"),
            (accepted.estimate_uuid,),
        ).fetchone()
        parameters = conn.execute(
            database.query(
                "SELECT * FROM {s}.estimate_parameters_v1 WHERE estimate_uuid=%s ORDER BY ordinal"
            ),
            (accepted.estimate_uuid,),
        ).fetchall()
        artifacts = conn.execute(
            database.query(
                "SELECT * FROM {s}.estimate_artifacts WHERE estimate_uuid=%s"
            ),
            (accepted.estimate_uuid,),
        ).fetchall()
    assert summary["status"] == ("succeeded" if expected.success else "failed")
    assert [p["parameter_name"] for p in parameters] == list(expected.parameter_names)
    np.testing.assert_allclose([p["value"] for p in parameters], expected.parameters)
    assert all(p["standard_uncertainty"] is None for p in parameters)
    assert len(artifacts) == 4
    assert all(
        hashlib.sha256(bytes(a["payload"])).hexdigest() == a["sha256"]
        for a in artifacts
    )
    assert all(a["byte_count"] == len(a["payload"]) for a in artifacts)
    assert database.load_prior(accepted.estimate_uuid) == prior_document(prior)


def test_atomic_publication_and_cancel_during_fit(database):
    config = configuration()
    submit(database, config)
    job = database.claim_job("worker", 30)
    prior = prior_fixture(config)
    database.store_prior(
        job, "worker", prior_document(prior), b"raw", {}, software_versions()
    )
    output, optimizer, scan = run_estimate(prior, config)
    parameters, diagnostics = result_rows(prior, config, optimizer, output)
    invalid = [*parameters, parameters[0]]
    with pytest.raises(Exception, match="duplicate key"):
        database.persist_estimate_result(
            job,
            "worker",
            invalid,
            diagnostics,
            document(output),
            document(optimizer),
            scan,
        )
    with database.pool.connection() as conn:
        assert (
            conn.execute(
                database.query("SELECT count(*) AS n FROM {s}.estimate_parameters")
            ).fetchone()["n"]
            == 0
        )
    database.cancel_job(job.id, "engineer")
    assert (
        database.persist_estimate_result(
            job,
            "worker",
            parameters,
            diagnostics,
            document(output),
            document(optimizer),
            scan,
        )
        == "canceled"
    )
    with database.pool.connection() as conn:
        assert (
            conn.execute(
                database.query("SELECT count(*) AS n FROM {s}.estimate_parameters")
            ).fetchone()["n"]
            == 0
        )


def test_transient_retry_keeps_inputs_and_terminal_replay(database):
    accepted = submit(database, key="retry")
    job = database.claim_job("worker", 30)
    prior = prior_fixture(job.configuration)
    database.store_prior(
        job, "worker", prior_document(prior), b"raw", {}, software_versions()
    )
    assert (
        database.fail_or_retry(job, "worker", {"code": "temporary"}, True) == "queued"
    )
    with database.pool.connection() as conn:
        conn.execute(
            database.query("UPDATE {s}.jobs SET available_at=now() WHERE id=%s"),
            (job.id,),
        )
    new = database.claim_job("worker", 30)
    assert new.run_id != job.run_id
    assert database.load_prior(new.estimate_uuid) == prior_document(prior)
    database.fail_or_retry(new, "worker", {"code": "permanent"}, False)
    assert submit(database, key="retry").status == "failed"
    assert accepted.estimate_uuid == new.estimate_uuid
