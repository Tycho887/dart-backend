"""Worker failure diagnostics identify code paths without logging provider data."""

import json
import logging
import threading
from dataclasses import replace
from unittest.mock import Mock
from uuid import uuid4

import psycopg
import pytest
from service_fixtures import configuration

from dart.io.load import LoadError
from dart.service import cli
from dart.service import worker as worker_module
from dart.service.config import ServiceSettings
from dart.service.database import ClaimedJob, Database, LeaseLost
from dart.service.diagnostics import exception_chain, exception_summary, log_failure
from dart.service.resolver import InputValidationError
from dart.service.worker import Worker, _failure


@pytest.fixture
def job():
    return ClaimedJob(uuid4(), uuid4(), uuid4(), 1, 3, configuration())


@pytest.fixture
def worker():
    db = Mock(spec=Database)
    db.set_stage.return_value = True
    db.fail_or_retry.return_value = "failed"
    return Worker(db, ServiceSettings("unused"), resolver=Mock())


def missing_dependency():
    # The literal, exception note, and local variable must never enter a log.
    provider_secret = "secret-from-provider"
    error = ModuleNotFoundError(provider_secret, name="example.optional")
    error.add_note("secret-in-note")
    raise error


def failure_records(caplog):
    return [json.loads(r.message) for r in caplog.records if hasattr(r, "event")]


@pytest.mark.parametrize(
    "target,phase",
    [
        ("_prior", "preparation"),
        ("run_estimate", "fitting"),
        ("result_rows", "result_serialization"),
        ("document", "result_serialization"),
        ("persist_estimate_result", "persistence"),
    ],
)
def test_job_failure_identifies_module_stack_phase_and_run(
    worker, job, monkeypatch, caplog, target, phase
):
    monkeypatch.setattr(worker, "_prior", Mock(return_value=object()))
    monkeypatch.setattr(worker_module, "run_estimate", Mock(return_value=(1, 2, [])))
    monkeypatch.setattr(worker_module, "result_rows", Mock(return_value=([], [])))
    monkeypatch.setattr(worker_module, "document", Mock(return_value={}))
    owners = {"_prior": worker, "persist_estimate_result": worker.database}
    monkeypatch.setattr(
        owners.get(target, worker_module),
        target,
        lambda *args: missing_dependency(),
    )
    worker._process(job)
    error, retryable = worker.database.fail_or_retry.call_args.args[2:]
    assert error == {
        "code": "worker_execution_failed",
        "detail": f"Estimate failed during {phase}: missing Python module 'example.optional'.",
        "exception_type": "ModuleNotFoundError",
        "missing_module": "example.optional",
        "phase": phase,
        "run_id": str(job.run_id),
        "retryable": False,
    }
    assert retryable is False
    (record,) = failure_records(caplog)
    assert record["estimate_uuid"] == str(job.estimate_uuid)
    assert record["job_id"] == str(job.id)
    assert record["run_id"] == error["run_id"]
    assert record["attempt"] == 1 and record["worker_id"] == worker.settings.worker_id
    assert record["phase"] == phase
    frame = record["traceback"][0]["frames"][-1]
    assert frame["file"] == __file__ and frame["line"] > 0
    assert frame["function"] == "missing_dependency"
    assert "secret-from-provider" not in caplog.text + str(error)
    assert "secret-in-note" not in caplog.text + str(error)
    assert all(r.exc_info is None for r in caplog.records)


def test_wrapped_provider_failure_is_safe_and_keeps_retry_policy(caplog):
    try:
        try:
            raise psycopg.OperationalError("postgresql://user:secret-password@host/db")
        except psycopg.OperationalError as cause:
            raise LoadError(
                "ADX", "contact", "Authorization: Bearer secret-token"
            ) from cause
    except LoadError as exc:
        log_failure(
            logging.getLogger(__name__), exc, phase="preparation", worker_id="w"
        )
        error, retryable = _failure(exc, "preparation")
    assert retryable is True and error["retryable"] is True
    assert error["exception_type"] == "OperationalError"
    (record,) = failure_records(caplog)
    assert [e["exception_type"] for e in record["traceback"]] == [
        "LoadError",
        "OperationalError",
    ]
    assert all(e["frames"] for e in record["traceback"])
    assert "secret-password" not in caplog.text + str(error)
    assert "secret-token" not in caplog.text + str(error)


@pytest.mark.parametrize(
    "name", [None, "bad\nmodule", "https://secret@host", "auth=secret"]
)
def test_missing_module_does_not_fall_back_to_exception_message(name):
    error, retryable = _failure(ModuleNotFoundError("secret-message", name=name))
    assert "missing_module" not in error
    assert "secret" not in str(error)
    assert retryable is False


def test_exception_chain_handles_context_suppression_and_cycles():
    cause = ModuleNotFoundError("hidden", name="example")
    wrapper = RuntimeError("hidden")
    wrapper.__context__ = cause
    assert exception_summary(wrapper)["missing_module"] == "example"
    wrapper.__suppress_context__ = True
    assert exception_chain(wrapper) == [wrapper]
    wrapper.__cause__ = cause
    cause.__cause__ = wrapper
    assert exception_chain(wrapper) == [wrapper, cause]


def test_safe_validation_message_is_preserved():
    error, retryable = _failure(
        InputValidationError("selected prior ephemeris identity mismatch")
    )
    assert error["code"] == "input_invalid"
    assert error["detail"] == "selected prior ephemeris identity mismatch"
    assert retryable is False


def test_failure_recording_error_keeps_job_context(worker, job, caplog, monkeypatch):
    monkeypatch.setattr(worker, "_prior", lambda *_: missing_dependency())
    worker.database.fail_or_retry.side_effect = psycopg.OperationalError("secret-dsn")
    with pytest.raises(psycopg.OperationalError):
        worker._process(job)
    initial, recording = failure_records(caplog)
    assert initial["phase"] == "preparation"
    assert recording["phase"] == "failure_recording"
    assert recording["exception_type"] == "OperationalError"
    assert recording["run_id"] == str(job.run_id)
    assert "secret-dsn" not in caplog.text


def test_heartbeat_failure_keeps_job_context(worker, job, caplog):
    worker.settings = replace(worker.settings, heartbeat_seconds=0.001)
    attempted = threading.Event()

    def fail_heartbeat(*args):
        attempted.set()
        raise psycopg.OperationalError("secret-heartbeat-dsn")

    worker.database.heartbeat.side_effect = fail_heartbeat
    with worker._heartbeat(job):
        assert attempted.wait(timeout=2)
    (record,) = failure_records(caplog)
    assert record["phase"] == "heartbeat" and record["run_id"] == str(job.run_id)
    assert "secret-heartbeat-dsn" not in caplog.text


def test_loop_failure_is_logged_and_worker_continues(worker, monkeypatch, caplog):
    worker.settings = replace(worker.settings, poll_seconds=0.001)
    run = Mock(
        side_effect=[psycopg.OperationalError("secret-loop-dsn"), KeyboardInterrupt]
    )
    monkeypatch.setattr(worker, "run_once", run)
    worker.run_forever()
    assert run.call_count == 2
    (record,) = failure_records(caplog)
    assert record["phase"] == "worker_iteration"
    assert record["worker_id"] == worker.settings.worker_id
    assert "job_id" not in record
    assert "secret-loop-dsn" not in caplog.text


@pytest.mark.parametrize("lease_lost", [False, True])
def test_failure_does_not_override_lease_fencing(worker, job, monkeypatch, lease_lost):
    if lease_lost:
        monkeypatch.setattr(worker, "_prior", Mock(side_effect=LeaseLost))
    else:
        worker.database.set_stage.return_value = False
    worker._process(job)
    worker.database.fail_or_retry.assert_not_called()
    worker.database.persist_estimate_result.assert_not_called()


def test_startup_logs_build_identity(monkeypatch, caplog):
    monkeypatch.setattr(cli, "Database", Mock())
    monkeypatch.setattr(cli, "Worker", Mock())
    monkeypatch.setattr(cli, "software_versions", lambda: {"build_sha256": "a" * 64})
    with caplog.at_level(logging.INFO):
        cli.worker()
    assert "build_sha256=" + "a" * 64 in caplog.text


def test_startup_failure_is_sanitized(monkeypatch, caplog):
    db = Mock()
    db.healthcheck.side_effect = psycopg.OperationalError("secret-startup-dsn")
    monkeypatch.setattr(cli, "Database", Mock(return_value=db))
    monkeypatch.setattr(cli, "software_versions", lambda: {})
    with pytest.raises(SystemExit) as result:
        cli.worker()
    assert result.value.code == 1
    assert failure_records(caplog)[0]["phase"] == "startup"
    assert "secret-startup-dsn" not in caplog.text
    db.close.assert_called_once()
