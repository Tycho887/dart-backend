from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from dart.contracts import (
    BatchRequest,
    BatchResult,
    DatasetPacket,
    DatasetQuery,
    OptimizerConfiguration,
    OptimizerModel,
    PipelineStage,
    RunRequest,
    RunStatus,
    StageError,
    TimeOffsetMetaparameters,
    TLEData,
)
from dart.gateway import jobs, worker


class FakeCursor:
    def __init__(self, row: Any = None, rows: list[Any] | None = None) -> None:
        self.row = row
        self.rows = [] if rows is None else rows
        self.executed: list[tuple[str, Any]] = []
        self.rowcount = 1

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, statement: str, parameters: Any = None) -> None:
        self.executed.append((statement, parameters))

    def fetchone(self) -> Any:
        return self.row

    def fetchall(self) -> list[Any]:
        return self.rows


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self.cursor_value = cursor

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def cursor(self) -> FakeCursor:
        return self.cursor_value


class UnusedProvider:
    def fetch(
        self,
        query: DatasetQuery,
        nominal_carrier_frequency_hz: float,
    ) -> DatasetPacket:
        del query, nominal_carrier_frequency_hz
        raise AssertionError("provider should not run after an artifact conflict")

    def fetch_with_heartbeat(
        self,
        query: DatasetQuery,
        nominal_carrier_frequency_hz: float,
        heartbeat: Callable[[], None],
    ) -> DatasetPacket:
        del query, nominal_carrier_frequency_hz, heartbeat
        raise AssertionError("provider should not run after an artifact conflict")


class UnusedServices:
    def solve(self, request: BatchRequest) -> BatchResult:
        del request
        raise AssertionError("services should not run after an artifact conflict")

    def postprocess(self, request: Any) -> Any:
        del request
        raise AssertionError("services should not run after an artifact conflict")


def test_load_dataset_wraps_invalid_stored_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor = FakeCursor(row=("{not-json",))
    monkeypatch.setattr(jobs, "_connect", lambda: FakeConnection(cursor))

    with pytest.raises(jobs.ArtifactConflictError, match="stored dataset artifact is invalid"):
        jobs.load_dataset("run-1")


def test_list_candidates_wraps_invalid_stored_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor = FakeCursor(rows=[("candidate-1", 0, "{not-json", "pending", None, None, None)])
    monkeypatch.setattr(jobs, "_connect", lambda: FakeConnection(cursor))

    with pytest.raises(jobs.ArtifactConflictError, match="stored candidate artifact is invalid"):
        jobs.list_candidates("run-1")


def test_process_next_retries_invalid_stored_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    claimed = jobs.ClaimedRun("run-1", _request(), "lease-1", 1)
    errors = []
    monkeypatch.setattr(worker.jobs, "claim", lambda: claimed)
    monkeypatch.setattr(
        worker.jobs,
        "load_dataset",
        lambda _run_id: (_ for _ in ()).throw(jobs.ArtifactConflictError("invalid dataset")),
    )
    monkeypatch.setattr(worker.jobs, "retry_or_fail", lambda _claimed, error: errors.append(error))

    assert worker.process_next(worker.Worker(UnusedProvider(), UnusedServices()))

    assert len(errors) == 1
    assert errors[0].stage is PipelineStage.PERSISTENCE


def test_stale_attempt_finalizer_marks_exhausted_runs_and_emits_events() -> None:
    cursor = FakeCursor(rows=[("run-1", jobs.MAX_ATTEMPTS)])

    assert jobs._finalize_stale_exhausted_runs(cursor) == 1

    assert "attempts >= %s" in cursor.executed[0][0]
    assert cursor.executed[0][1][1] == jobs.MAX_ATTEMPTS
    assert len(cursor.executed) == 2
    assert "pipeline_stage_events" in cursor.executed[1][0]


def test_retry_backoff_is_bounded_and_exponential() -> None:
    assert [jobs.retry_backoff_seconds(attempt) for attempt in range(1, 4)] == [5, 10, 20]
    assert jobs.retry_backoff_seconds(7) == jobs.MAX_RETRY_BACKOFF_SECONDS
    assert jobs.retry_backoff_seconds(100) == jobs.MAX_RETRY_BACKOFF_SECONDS


def test_claim_only_considers_retries_eligible_at_the_database_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursor = FakeCursor()
    monkeypatch.setattr(jobs, "_connect", lambda: FakeConnection(cursor))

    assert jobs.claim() is None

    statement, parameters = cursor.executed[1]
    assert "next_attempt_at <= NOW()" in statement
    assert "ORDER BY next_attempt_at, created_at, run_id" in statement
    assert parameters == (jobs.MAX_ATTEMPTS,)


def test_retry_persists_database_clock_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor = FakeCursor(row=(RunStatus.QUEUED.value, 1))
    monkeypatch.setattr(jobs, "_connect", lambda: FakeConnection(cursor))
    claimed = jobs.ClaimedRun("run-1", _request(), "lease-1", attempt=1)
    error = StageError(
        stage=PipelineStage.ACQUISITION,
        error_type="AdxTransportError",
        message="dataset acquisition failed",
    )

    assert jobs.retry_or_fail(claimed, error) is RunStatus.QUEUED

    statement, parameters = cursor.executed[0]
    assert "next_attempt_at=CASE" in statement
    assert "NOW()+(%s * INTERVAL '1 second')" in statement
    assert parameters[:3] == (
        jobs.MAX_ATTEMPTS,
        jobs.MAX_ATTEMPTS,
        jobs.retry_backoff_seconds(claimed.attempt),
    )


def test_schema_persists_retry_eligibility_and_queue_index() -> None:
    schema = jobs._schema_sql()

    assert "next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT NOW()" in schema
    assert "ON pipeline_runs (status, next_attempt_at, created_at)" in schema


def _request() -> RunRequest:
    return RunRequest(
        query=DatasetQuery(contact_ids=["contact-1"]),
        optimizer_configuration=OptimizerConfiguration(
            reference_tle=TLEData(
                name="TEST",
                line1="1 57912U 23146X   24099.49439401  .00006757  00000+0  51475-3 0  9997",
                line2="2 57912  43.0018 157.5807 0001420 272.5369  87.5310 15.02537576 31746",
            ),
            spacecraft_id="spacecraft-1",
            nominal_carrier_frequency_hz=2.2e9,
        ),
        candidate_metaparameters=[TimeOffsetMetaparameters(model=OptimizerModel.TIME_OFFSET)],
    )
