"""PostgreSQL-backed durable orchestration and immutable scientific artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
from math import isclose, log, pi
from pathlib import Path
from sys import float_info
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from ...contracts import (
    BatchResult,
    CandidateRecord,
    CandidateStatus,
    DatasetPacket,
    InformationCriterion,
    ObservableChannel,
    PipelineStage,
    QualityResult,
    RunRecord,
    RunRequest,
    RunStatus,
    SelectionScore,
    StageError,
)

MAX_ATTEMPTS = 3
DEFAULT_LEASE_SECONDS = 1_800
RETRY_BACKOFF_BASE_SECONDS = 5
MAX_RETRY_BACKOFF_SECONDS = 300
TERMINAL_STATUSES = {RunStatus.SUCCEEDED, RunStatus.PARTIAL, RunStatus.FAILED}
CANDIDATE_TERMINAL_STATUSES = {
    CandidateStatus.SUCCEEDED,
    CandidateStatus.UNHEALTHY,
    CandidateStatus.INELIGIBLE,
    CandidateStatus.FAILED,
}


class IdempotencyConflictError(ValueError):
    """An idempotency key was reused for a different canonical request."""


class ArtifactConflictError(RuntimeError):
    """A supposedly immutable artifact conflicts with its stored provenance."""


class LeaseLostError(RuntimeError):
    """A worker attempted a write after its run lease expired or changed owner."""


class ResidualIdentityError(ValueError):
    """A solver result does not cover exactly the acquired measurement identities."""


@dataclass(frozen=True)
class ClaimedRun:
    run_id: str
    request: RunRequest
    lease_token: str
    attempt: int


def _connect() -> Any:
    import psycopg

    return psycopg.connect(os.environ["DART_DATABASE_URL"])


def lease_seconds() -> int:
    """Return the bounded durable lease duration shared with external-call clients."""

    raw = os.getenv("DART_RUN_LEASE_SECONDS", str(DEFAULT_LEASE_SECONDS))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError("DART_RUN_LEASE_SECONDS must be an integer") from exc
    if value <= 0:
        raise RuntimeError("DART_RUN_LEASE_SECONDS must be positive")
    return value


def retry_backoff_seconds(attempt: int) -> int:
    """Return the bounded delay before retrying a completed failed attempt."""

    delay = RETRY_BACKOFF_BASE_SECONDS
    remaining_doublings = max(attempt - 1, 0)
    while remaining_doublings and delay < MAX_RETRY_BACKOFF_SECONDS:
        delay = min(delay * 2, MAX_RETRY_BACKOFF_SECONDS)
        remaining_doublings -= 1
    return delay


def _schema_sql() -> str:
    source_schema = (
        Path(__file__).resolve().parents[4] / "deploy" / "database" / "001_processing_runs.sql"
    )
    if source_schema.is_file():
        return source_schema.read_text()
    return files("dart.services.orchestrator").joinpath("schema.sql").read_text()


def initialize() -> None:
    """Apply the disposable baseline under one PostgreSQL transaction advisory lock."""

    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("dart-processing-runs-v0",))
        cursor.execute(_schema_sql())


def canonical_json(value: BaseModel) -> str:
    """Serialize a contract payload identically across idempotent writes."""

    return json.dumps(
        _normalize_signed_zero(value.model_dump(mode="json")),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def artifact_digest(value: BaseModel) -> str:
    return sha256(canonical_json(value).encode()).hexdigest()


def create(request: RunRequest, idempotency_key: str) -> str:
    """Atomically create one durable run or return its idempotent predecessor."""

    run_id = str(uuid4())
    payload = canonical_json(request)
    digest = sha256(payload.encode()).hexdigest()
    source_orbit = canonical_json(request.optimizer_configuration.reference_tle)
    solver_config = canonical_json(request.optimizer_configuration)
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO pipeline_runs
                (run_id, pipeline_name, idempotency_key, status, request_sha256,
                 request_json, selection_criterion)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
            ON CONFLICT (idempotency_key) DO NOTHING
            RETURNING run_id
            """,
            (
                run_id,
                request.pipeline,
                idempotency_key,
                RunStatus.QUEUED.value,
                digest,
                payload,
                request.selection_criterion.value,
            ),
        )
        inserted = cursor.fetchone()
        if inserted is None:
            cursor.execute(
                "SELECT run_id, request_sha256 FROM pipeline_runs WHERE idempotency_key=%s",
                (idempotency_key,),
            )
            existing = cursor.fetchone()
            if existing is None:
                raise RuntimeError("idempotent run insertion did not return a stored run")
            if existing[1] != digest:
                raise IdempotencyConflictError(
                    "idempotency key was already used for a different request"
                )
            return str(existing[0])
        cursor.execute(
            """
            INSERT INTO run_inputs
                (run_id, contract_version, source_orbit_json, solver_config_json)
            VALUES (%s, %s, %s::jsonb, %s::jsonb)
            """,
            (run_id, request.schema_version, source_orbit, solver_config),
        )
        candidate_rows = [
            (
                str(uuid4()),
                run_id,
                index,
                metaparameters.model.value,
                canonical_json(metaparameters),
                CandidateStatus.PENDING.value,
            )
            for index, metaparameters in enumerate(request.candidate_metaparameters)
        ]
        cursor.executemany(
            """
            INSERT INTO run_candidates
                (candidate_id, run_id, candidate_index, model, metaparameters_json, status)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s)
            """,
            candidate_rows,
        )
        _event(cursor, run_id, 0, "orchestrator", RunStatus.QUEUED.value, {}, None)
    return run_id


def finalize_stale_exhausted_runs() -> int:
    """Terminally fail stale runs that exhausted all leases without a worker cleanup."""

    with _connect() as connection, connection.cursor() as cursor:
        return _finalize_stale_exhausted_runs(cursor)


def claim() -> ClaimedRun | None:
    """Claim one resumable non-terminal run and attach a new lease owner token."""

    lease_token = str(uuid4())
    duration = lease_seconds()
    with _connect() as connection, connection.cursor() as cursor:
        _finalize_stale_exhausted_runs(cursor)
        cursor.execute(
            """
            SELECT run_id, request_json
            FROM pipeline_runs
            WHERE status NOT IN ('succeeded', 'partial', 'failed')
              AND (status = 'queued' OR lease_until IS NULL OR lease_until < NOW())
              AND next_attempt_at <= NOW()
              AND attempts < %s
            ORDER BY next_attempt_at, created_at, run_id
            FOR UPDATE SKIP LOCKED
            LIMIT 1
            """,
            (MAX_ATTEMPTS,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        cursor.execute(
            """
            UPDATE pipeline_runs
            SET status=%s, attempts=attempts+1, lease_token=%s,
                lease_until=NOW()+(%s * INTERVAL '1 second'), updated_at=NOW()
            WHERE run_id=%s
            RETURNING attempts
            """,
            (RunStatus.ACQUIRING.value, lease_token, duration, row[0]),
        )
        attempt = cursor.fetchone()[0]
        _event(cursor, str(row[0]), attempt, "acquisition", RunStatus.ACQUIRING.value, {}, None)
        return ClaimedRun(
            run_id=str(row[0]),
            request=RunRequest.model_validate(row[1]),
            lease_token=lease_token,
            attempt=attempt,
        )


def renew_lease(claimed: ClaimedRun) -> None:
    """Renew ownership immediately before a potentially long external operation."""

    duration = lease_seconds()
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE pipeline_runs
            SET lease_until=NOW()+(%s * INTERVAL '1 second'), updated_at=NOW()
            WHERE run_id=%s AND lease_token=%s AND lease_until > NOW()
            """,
            (duration, claimed.run_id, claimed.lease_token),
        )
        if cursor.rowcount != 1:
            raise LeaseLostError("run lease was lost before external operation")


def load_dataset(run_id: str) -> DatasetPacket | None:
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT dataset_json FROM run_inputs WHERE run_id=%s", (run_id,))
        row = cursor.fetchone()
    if row is None:
        return None
    try:
        payload = row[0]
    except (IndexError, TypeError) as exc:
        raise ArtifactConflictError("stored dataset artifact has an invalid row shape") from exc
    if payload is None:
        return None
    return _decode_dataset(payload)


def list_candidates(run_id: str) -> list[CandidateRecord]:
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT candidates.candidate_id, candidates.candidate_index,
                   candidates.metaparameters_json, candidates.status,
                   candidates.error_json, results.result_json, metrics.result_json
            FROM run_candidates AS candidates
            LEFT JOIN solver_results AS results
                ON (results.run_id, results.candidate_id) =
                   (candidates.run_id, candidates.candidate_id)
            LEFT JOIN metric_sets AS metrics
                ON (metrics.metric_set_id, metrics.candidate_id) =
                   (candidates.selected_metric_set_id, candidates.candidate_id)
            WHERE candidates.run_id=%s
            ORDER BY candidates.candidate_index
            """,
            (run_id,),
        )
        rows = cursor.fetchall()
    try:
        return [_candidate_record(row) for row in rows]
    except (IndexError, TypeError, ValueError) as exc:
        raise ArtifactConflictError("stored candidate artifact is invalid") from exc


def mark_candidate_optimizing(claimed: ClaimedRun, candidate_id: str) -> None:
    with _connect() as connection, connection.cursor() as cursor:
        _require_lease(cursor, claimed)
        _update_candidate_status(
            cursor,
            claimed,
            candidate_id,
            CandidateStatus.OPTIMIZING,
            allowed=(CandidateStatus.PENDING, CandidateStatus.OPTIMIZING),
        )
        _set_run_status(cursor, claimed, RunStatus.OPTIMIZING, "optimization", candidate_id)


def mark_candidate_postprocessing(claimed: ClaimedRun, candidate_id: str) -> None:
    with _connect() as connection, connection.cursor() as cursor:
        _require_lease(cursor, claimed)
        _update_candidate_status(
            cursor,
            claimed,
            candidate_id,
            CandidateStatus.POSTPROCESSING,
            allowed=(CandidateStatus.OPTIMIZING, CandidateStatus.POSTPROCESSING),
        )
        _set_run_status(cursor, claimed, RunStatus.POSTPROCESSING, "postprocessing", candidate_id)


def store_acquisition(claimed: ClaimedRun, packet: DatasetPacket) -> None:
    """Store the acquired artifact once, accepting only an identical retry."""

    payload = canonical_json(packet)
    dataset_digest = sha256(payload.encode()).hexdigest()
    with _connect() as connection, connection.cursor() as cursor:
        _require_lease(cursor, claimed)
        cursor.execute(
            "SELECT dataset_sha256 FROM run_inputs WHERE run_id=%s FOR UPDATE", (claimed.run_id,)
        )
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError("run input row is missing")
        if row[0] is not None:
            if row[0] != dataset_digest:
                raise ArtifactConflictError("stored acquisition differs from retry acquisition")
        else:
            cursor.execute(
                """
                UPDATE run_inputs
                SET tdm_sha256=%s, tdm_kvn=%s, dataset_sha256=%s, dataset_json=%s::jsonb,
                    provenance_json=%s::jsonb
                WHERE run_id=%s AND dataset_sha256 IS NULL
                """,
                (
                    packet.tdm.sha256,
                    packet.tdm.content,
                    dataset_digest,
                    payload,
                    json.dumps(packet.provenance, sort_keys=True),
                    claimed.run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ArtifactConflictError("acquisition write was not applied exactly once")
        _set_run_status(cursor, claimed, RunStatus.OPTIMIZING, "acquisition", None)


def store_candidate_result(
    claimed: ClaimedRun,
    candidate_id: str,
    result: BatchResult,
) -> None:
    """Store one candidate result and its exact channel-normalized residual artifacts."""

    payload = canonical_json(result)
    result_digest = sha256(payload.encode()).hexdigest()
    solver_version = _solver_version()
    with _connect() as connection, connection.cursor() as cursor:
        _require_lease(cursor, claimed)
        _require_candidate_model(cursor, claimed.run_id, candidate_id, result.model.value)
        packet = _load_dataset_cursor(cursor, claimed.run_id)
        validate_residual_measurement_ids(result, packet)
        cursor.execute(
            """
            INSERT INTO solver_results
                (result_id, run_id, candidate_id, solver_version, contract_version,
                 result_sha256, result_json, parameters_json, covariance_json, diagnostics_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb)
            ON CONFLICT (candidate_id) DO NOTHING
            RETURNING result_id
            """,
            (
                str(uuid4()),
                claimed.run_id,
                candidate_id,
                solver_version,
                result.schema_version,
                result_digest,
                payload,
                canonical_json(result.parameters),
                canonical_json(result.covariance),
                canonical_json(result.diagnostics),
            ),
        )
        inserted = cursor.fetchone()
        if inserted is None:
            cursor.execute(
                """
                SELECT result_sha256, solver_version, contract_version
                FROM solver_results
                WHERE run_id=%s AND candidate_id=%s
                """,
                (claimed.run_id, candidate_id),
            )
            existing = cursor.fetchone()
            if existing is None:
                raise ArtifactConflictError("stored solver result belongs to another run")
            if existing != (result_digest, solver_version, result.schema_version):
                raise ArtifactConflictError("stored solver result differs from retry provenance")
        else:
            _store_residuals(cursor, claimed.run_id, candidate_id, result, packet)
        _update_candidate_status(
            cursor,
            claimed,
            candidate_id,
            CandidateStatus.POSTPROCESSING,
            allowed=(CandidateStatus.OPTIMIZING, CandidateStatus.POSTPROCESSING),
        )
        _set_run_status(cursor, claimed, RunStatus.POSTPROCESSING, "optimization", candidate_id)


def store_candidate_quality(
    claimed: ClaimedRun,
    candidate_id: str,
    quality: QualityResult,
) -> None:
    """Store a versioned quality artifact and point the candidate at its selected version."""

    payload = canonical_json(quality)
    quality_digest = sha256(payload.encode()).hexdigest()
    postprocessor_version = _postprocessor_version()
    with _connect() as connection, connection.cursor() as cursor:
        _require_lease(cursor, claimed)
        cursor.execute(
            """
            SELECT results.result_json, results.result_sha256, runs.selection_criterion
            FROM solver_results AS results
            JOIN pipeline_runs AS runs ON runs.run_id = results.run_id
            WHERE results.run_id=%s AND results.candidate_id=%s
            FOR UPDATE
            """,
            (claimed.run_id, candidate_id),
        )
        row = cursor.fetchone()
        if row is None:
            raise ArtifactConflictError("cannot postprocess a missing candidate result")
        try:
            result = _decode_batch_result(row[0])
            solver_result_digest = row[1]
            criterion = InformationCriterion(row[2])
        except (IndexError, TypeError, ValueError) as exc:
            raise ArtifactConflictError("stored solver result artifact is invalid") from exc
        validate_selection_score(result, quality, criterion)
        status = _candidate_terminal_status(result.diagnostics.healthy, quality)
        cursor.execute(
            """
            SELECT metric_set_id, quality_sha256, solver_result_sha256, contract_version
            FROM metric_sets
            WHERE run_id=%s AND candidate_id=%s AND postprocessor_version=%s
            FOR UPDATE
            """,
            (claimed.run_id, candidate_id, postprocessor_version),
        )
        existing = cursor.fetchone()
        if existing is None:
            metric_set_id = str(uuid4())
            cursor.execute(
                """
                INSERT INTO metric_sets
                    (metric_set_id, run_id, candidate_id, postprocessor_version, contract_version,
                     solver_result_sha256, quality_sha256, result_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    metric_set_id,
                    claimed.run_id,
                    candidate_id,
                    postprocessor_version,
                    quality.schema_version,
                    solver_result_digest,
                    quality_digest,
                    payload,
                ),
            )
            _store_metric_values(cursor, metric_set_id, quality)
        else:
            metric_set_id = str(existing[0])
            if existing[1:] != (quality_digest, solver_result_digest, quality.schema_version):
                raise ArtifactConflictError(
                    "stored postprocessor result differs from retry provenance"
                )
        cursor.execute(
            """
            UPDATE run_candidates
            SET status=%s, selected_metric_set_id=%s, updated_at=NOW()
            WHERE run_id=%s AND candidate_id=%s AND status = ANY(%s)
            """,
            (
                status.value,
                metric_set_id,
                claimed.run_id,
                candidate_id,
                [CandidateStatus.POSTPROCESSING.value, status.value],
            ),
        )
        if cursor.rowcount != 1:
            raise ArtifactConflictError("candidate status cannot transition to stored quality")
        _event(
            cursor,
            claimed.run_id,
            claimed.attempt,
            "postprocessing",
            status.value,
            {"postprocessor_version": postprocessor_version},
            candidate_id,
        )


def store_candidate_error(
    claimed: ClaimedRun,
    candidate_id: str,
    error: StageError,
) -> None:
    """Persist one typed candidate error from a non-terminal candidate state."""

    payload = canonical_json(error)
    with _connect() as connection, connection.cursor() as cursor:
        _require_lease(cursor, claimed)
        cursor.execute(
            """
            UPDATE run_candidates
            SET status=%s, error_json=COALESCE(error_json, %s::jsonb), updated_at=NOW()
            WHERE run_id=%s AND candidate_id=%s AND status = ANY(%s)
            RETURNING error_json
            """,
            (
                CandidateStatus.FAILED.value,
                payload,
                claimed.run_id,
                candidate_id,
                [
                    CandidateStatus.PENDING.value,
                    CandidateStatus.OPTIMIZING.value,
                    CandidateStatus.POSTPROCESSING.value,
                ],
            ),
        )
        row = cursor.fetchone()
        if row is None:
            raise ArtifactConflictError(
                "candidate is terminal or does not belong to the active run"
            )
        stored = _json_payload(row[0])
        if _canonical_mapping(stored) != payload:
            raise ArtifactConflictError("stored candidate error differs from retry error")
        _event(
            cursor,
            claimed.run_id,
            claimed.attempt,
            error.stage.value,
            CandidateStatus.FAILED.value,
            error.model_dump(mode="json"),
            candidate_id,
        )


def complete_run(
    claimed: ClaimedRun,
    status: RunStatus,
    selected_candidate_id: str | None,
) -> None:
    """Finish a run only when candidate states and selected result agree exactly."""

    if status not in TERMINAL_STATUSES:
        raise ValueError("complete_run requires a terminal run status")
    with _connect() as connection, connection.cursor() as cursor:
        _require_lease(cursor, claimed)
        cursor.execute(
            """
            SELECT candidate_id, status, selected_metric_set_id
            FROM run_candidates
            WHERE run_id=%s
            ORDER BY candidate_index
            FOR UPDATE
            """,
            (claimed.run_id,),
        )
        candidates = cursor.fetchall()
        if not candidates:
            raise ArtifactConflictError("run has no candidates")
        candidate_statuses = {str(row[0]): CandidateStatus(row[1]) for row in candidates}
        if any(value not in CANDIDATE_TERMINAL_STATUSES for value in candidate_statuses.values()):
            raise ValueError("all candidates must be terminal before completing a run")
        succeeded = [
            str(row[0])
            for row in candidates
            if CandidateStatus(row[1]) is CandidateStatus.SUCCEEDED and row[2] is not None
        ]
        expected_status = _expected_run_status(candidate_statuses, succeeded)
        if status is not expected_status:
            raise ValueError("run terminal status does not match candidate states")
        if expected_status is RunStatus.FAILED:
            if selected_candidate_id is not None:
                raise ValueError("a failed run cannot select a candidate")
        else:
            if selected_candidate_id not in succeeded:
                raise ValueError(
                    "selected candidate must be a scored successful candidate in this run"
                )
        cursor.execute(
            """
            UPDATE pipeline_runs
            SET status=%s, selected_candidate_id=%s, lease_token=NULL, lease_until=NULL,
                updated_at=NOW()
            WHERE run_id=%s AND lease_token=%s AND lease_until > NOW()
            """,
            (status.value, selected_candidate_id, claimed.run_id, claimed.lease_token),
        )
        if cursor.rowcount != 1:
            raise LeaseLostError("run lease was lost before completion")
        _event(
            cursor,
            claimed.run_id,
            claimed.attempt,
            "selection",
            status.value,
            {},
            selected_candidate_id,
        )


def retry_or_fail(claimed: ClaimedRun, error: StageError) -> RunStatus:
    """Release a failed global stage for retry, or fail after the final attempt."""

    with _connect() as connection, connection.cursor() as cursor:
        delay = retry_backoff_seconds(claimed.attempt)
        cursor.execute(
            """
            UPDATE pipeline_runs
            SET status=CASE WHEN attempts < %s THEN 'queued' ELSE 'failed' END,
                next_attempt_at=CASE
                    WHEN attempts < %s THEN NOW()+(%s * INTERVAL '1 second')
                    ELSE NOW()
                END,
                errors_json=errors_json || %s::jsonb,
                lease_token=NULL, lease_until=NULL, updated_at=NOW()
            WHERE run_id=%s AND lease_token=%s AND lease_until > NOW()
            RETURNING status, attempts
            """,
            (
                MAX_ATTEMPTS,
                MAX_ATTEMPTS,
                delay,
                json.dumps([error.model_dump(mode="json")]),
                claimed.run_id,
                claimed.lease_token,
            ),
        )
        row = cursor.fetchone()
        if row is None:
            raise LeaseLostError("run lease was lost before retry transition")
        status = RunStatus(row[0])
        _event(
            cursor,
            claimed.run_id,
            row[1],
            error.stage.value,
            status.value,
            error.model_dump(mode="json"),
            None,
        )
    return status


def get(run_id: str) -> RunRecord | None:
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT status, created_at, updated_at, attempts, request_sha256, errors_json,
                   selected_candidate_id
            FROM pipeline_runs
            WHERE run_id=%s
            """,
            (run_id,),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    return RunRecord(
        run_id=run_id,
        status=RunStatus(row[0]),
        created_at=row[1],
        updated_at=row[2],
        attempts=row[3],
        request_sha256=row[4],
        errors=[StageError.model_validate(value) for value in row[5] or []],
        candidates=list_candidates(run_id),
        selected_candidate_id=None if row[6] is None else str(row[6]),
    )


def validate_residual_measurement_ids(result: BatchResult, packet: DatasetPacket) -> None:
    """Ensure persisted residual rows and the immutable dataset identify the same samples."""

    expected = {measurement.measurement_id for measurement in packet.measurements}
    actual = {record.measurement_id for record in result.residuals}
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    detail = []
    if missing:
        detail.append(f"missing={','.join(missing)}")
    if unknown:
        detail.append(f"unknown={','.join(unknown)}")
    raise ResidualIdentityError(
        "solver residual measurement IDs must match acquisition: " + "; ".join(detail)
    )


def _candidate_record(row: Any) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=str(row[0]),
        candidate_index=row[1],
        metaparameters=_json_payload(row[2]),
        status=CandidateStatus(row[3]),
        error=None if row[4] is None else StageError.model_validate(_json_payload(row[4])),
        result=None if row[5] is None else BatchResult.model_validate(_json_payload(row[5])),
        quality=None if row[6] is None else QualityResult.model_validate(_json_payload(row[6])),
    )


def _finalize_stale_exhausted_runs(cursor: Any) -> int:
    error = StageError(
        stage=PipelineStage.PERSISTENCE,
        error_type="AttemptLimitExceeded",
        message="run exhausted its lease attempts without a terminal transition",
    )
    cursor.execute(
        """
        UPDATE pipeline_runs
        SET status='failed', errors_json=errors_json || %s::jsonb,
            lease_token=NULL, lease_until=NULL, updated_at=NOW()
        WHERE status NOT IN ('succeeded', 'partial', 'failed')
          AND attempts >= %s
          AND (lease_until IS NULL OR lease_until < NOW())
        RETURNING run_id, attempts
        """,
        (json.dumps([error.model_dump(mode="json")]), MAX_ATTEMPTS),
    )
    rows = cursor.fetchall()
    for run_id, attempt in rows:
        _event(
            cursor,
            str(run_id),
            attempt,
            error.stage.value,
            RunStatus.FAILED.value,
            error.model_dump(mode="json"),
            None,
        )
    return len(rows)


def _expected_run_status(
    candidate_statuses: dict[str, CandidateStatus],
    succeeded: list[str],
) -> RunStatus:
    if not succeeded:
        return RunStatus.FAILED
    if all(status is CandidateStatus.SUCCEEDED for status in candidate_statuses.values()):
        return RunStatus.SUCCEEDED
    return RunStatus.PARTIAL


def _json_payload(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _normalize_signed_zero(value: Any) -> Any:
    """Canonicalize recursively serialized numeric values without mutating contracts."""

    if isinstance(value, float) and value == 0.0:
        return 0.0
    if isinstance(value, dict):
        return {key: _normalize_signed_zero(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_signed_zero(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_signed_zero(item) for item in value)
    return value


def _decode_dataset(payload: Any) -> DatasetPacket:
    try:
        return DatasetPacket.model_validate(_json_payload(payload))
    except (TypeError, ValueError) as exc:
        raise ArtifactConflictError("stored dataset artifact is invalid") from exc


def _decode_batch_result(payload: Any) -> BatchResult:
    try:
        return BatchResult.model_validate(_json_payload(payload))
    except (TypeError, ValueError) as exc:
        raise ArtifactConflictError("stored solver result artifact is invalid") from exc


def _canonical_mapping(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _require_lease(cursor: Any, claimed: ClaimedRun) -> None:
    cursor.execute(
        """
        SELECT 1 FROM pipeline_runs
        WHERE run_id=%s AND lease_token=%s AND lease_until > NOW()
        FOR UPDATE
        """,
        (claimed.run_id, claimed.lease_token),
    )
    if cursor.fetchone() is None:
        raise LeaseLostError("run lease is no longer held by this worker")


def _set_run_status(
    cursor: Any,
    claimed: ClaimedRun,
    status: RunStatus,
    stage: str,
    candidate_id: str | None,
) -> None:
    cursor.execute(
        """
        UPDATE pipeline_runs
        SET status=%s, lease_until=NOW()+(%s * INTERVAL '1 second'), updated_at=NOW()
        WHERE run_id=%s AND lease_token=%s AND lease_until > NOW()
        """,
        (status.value, lease_seconds(), claimed.run_id, claimed.lease_token),
    )
    if cursor.rowcount != 1:
        raise LeaseLostError("run lease was lost during stage transition")
    _event(cursor, claimed.run_id, claimed.attempt, stage, status.value, {}, candidate_id)


def _update_candidate_status(
    cursor: Any,
    claimed: ClaimedRun,
    candidate_id: str,
    status: CandidateStatus,
    *,
    allowed: tuple[CandidateStatus, ...],
) -> None:
    cursor.execute(
        """
        UPDATE run_candidates
        SET status=%s, updated_at=NOW()
        WHERE run_id=%s AND candidate_id=%s AND status = ANY(%s)
        """,
        (status.value, claimed.run_id, candidate_id, [item.value for item in allowed]),
    )
    if cursor.rowcount != 1:
        raise ArtifactConflictError("candidate status cannot transition from its stored state")


def _require_candidate_model(
    cursor: Any,
    run_id: str,
    candidate_id: str,
    result_model: str,
) -> None:
    cursor.execute(
        "SELECT model FROM run_candidates WHERE run_id=%s AND candidate_id=%s",
        (run_id, candidate_id),
    )
    row = cursor.fetchone()
    if row is None:
        raise ArtifactConflictError("solver result references an unknown candidate")
    if row[0] != result_model:
        raise ArtifactConflictError("solver result model differs from candidate model")


def _load_dataset_cursor(cursor: Any, run_id: str) -> DatasetPacket:
    cursor.execute("SELECT dataset_json FROM run_inputs WHERE run_id=%s", (run_id,))
    row = cursor.fetchone()
    if row is None:
        raise ArtifactConflictError("cannot store result before acquisition")
    try:
        payload = row[0]
    except (IndexError, TypeError) as exc:
        raise ArtifactConflictError("stored dataset artifact has an invalid row shape") from exc
    if payload is None:
        raise ArtifactConflictError("cannot store result before acquisition")
    return _decode_dataset(payload)


def _store_residuals(
    cursor: Any,
    run_id: str,
    candidate_id: str,
    result: BatchResult,
    packet: DatasetPacket,
) -> None:
    validate_residual_measurement_ids(result, packet)
    epochs = {
        measurement.measurement_id: measurement.time_tag for measurement in packet.measurements
    }
    rows = [
        (
            epochs[record.measurement_id],
            run_id,
            candidate_id,
            record.measurement_id,
            channel.channel.value,
            channel.consumed,
            channel.predicted,
            channel.residual,
            channel.robust_weight,
        )
        for record in result.residuals
        for channel in record.channels
    ]
    cursor.executemany(
        """
        INSERT INTO solver_residuals
            (timestamp, run_id, candidate_id, measurement_id, channel, consumed,
             predicted, residual, robust_weight)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (candidate_id, measurement_id, channel, timestamp) DO NOTHING
        """,
        rows,
    )


def _store_metric_values(cursor: Any, metric_set_id: str, quality: QualityResult) -> None:
    groups = {
        "convergence": quality.convergence.metrics,
        "residuals": quality.residuals.metrics,
        "selection": quality.selection.model_dump(mode="json"),
        "truth": {} if quality.truth is None else quality.truth.metrics,
    }
    rows = [
        (metric_set_id, group, name, json.dumps(value))
        for group, metrics in groups.items()
        for name, value in metrics.items()
    ]
    cursor.executemany(
        """
        INSERT INTO metric_values
            (metric_set_id, metric_group, metric_name, metric_value)
        VALUES (%s, %s, %s, %s::jsonb)
        """,
        rows,
    )


def _candidate_terminal_status(result_is_healthy: bool, quality: QualityResult) -> CandidateStatus:
    if not result_is_healthy:
        return CandidateStatus.UNHEALTHY
    if not quality.selection.eligible:
        return CandidateStatus.INELIGIBLE
    return CandidateStatus.SUCCEEDED


def _solver_version() -> str:
    return os.getenv("DART_SOLVER_VERSION", "0.2.0")


def _postprocessor_version() -> str:
    return os.getenv("DART_POSTPROCESSOR_VERSION", "0.2.0")


def _event(
    cursor: Any,
    run_id: str,
    attempt: int,
    stage: str,
    status: str,
    detail: dict[str, Any],
    candidate_id: str | None,
) -> None:
    cursor.execute(
        """
        INSERT INTO pipeline_stage_events
            (timestamp, run_id, candidate_id, attempt, stage, status, detail_json)
        VALUES (NOW(), %s, %s, %s, %s, %s, %s::jsonb)
        """,
        (run_id, candidate_id, attempt, stage, status, json.dumps(detail, sort_keys=True)),
    )


class SelectionValidationError(ValueError):
    """A postprocessor score does not describe the supplied solver result."""


def expected_selection_score(
    result: BatchResult,
    criterion: InformationCriterion,
) -> SelectionScore:
    """Recompute the agreed Gaussian information-criterion fields from a fit."""

    residuals = [
        channel.residual
        for record in result.residuals
        for channel in record.channels
        if (
            channel.channel is ObservableChannel.DOPPLER
            and channel.consumed
            and channel.residual is not None
        )
    ]
    observations = len(residuals)
    parameter_count = len(result.covariance.parameter_order)
    residual_sum_squares = sum(value * value for value in residuals)
    if observations == 0:
        return SelectionScore(
            criterion=criterion,
            eligible=False,
            observations=0,
            fitted_parameter_count=parameter_count,
            residual_sum_squares_hz2=0.0,
        )
    variance = max(residual_sum_squares / observations, float_info.min)
    deviance = observations * (log(2.0 * pi) + 1.0 + log(variance))
    aic = deviance + 2.0 * parameter_count
    if criterion is InformationCriterion.AIC:
        score = aic
    elif criterion is InformationCriterion.BIC:
        score = deviance + parameter_count * log(observations)
    else:
        denominator = observations - parameter_count - 1
        if denominator <= 0:
            return SelectionScore(
                criterion=criterion,
                eligible=False,
                observations=observations,
                fitted_parameter_count=parameter_count,
                residual_sum_squares_hz2=residual_sum_squares,
            )
        score = aic + (2.0 * parameter_count * (parameter_count + 1) / denominator)
    return SelectionScore(
        criterion=criterion,
        eligible=True,
        score=score,
        observations=observations,
        fitted_parameter_count=parameter_count,
        residual_sum_squares_hz2=residual_sum_squares,
    )


def validate_selection_score(
    result: BatchResult,
    quality: QualityResult,
    criterion: InformationCriterion,
) -> None:
    """Reject remote quality metadata that cannot be recomputed from the fit."""

    expected = expected_selection_score(result, criterion)
    actual = quality.selection
    if actual.criterion is not expected.criterion:
        raise SelectionValidationError(
            "postprocessor selection criterion differs from the run request"
        )
    if actual.eligible is not expected.eligible:
        raise SelectionValidationError("postprocessor selection eligibility differs from the fit")
    if actual.observations != expected.observations:
        raise SelectionValidationError(
            "postprocessor selection observation count differs from the fit"
        )
    if actual.fitted_parameter_count != expected.fitted_parameter_count:
        raise SelectionValidationError(
            "postprocessor selection parameter count differs from the fit"
        )
    if not isclose(
        actual.residual_sum_squares_hz2,
        expected.residual_sum_squares_hz2,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise SelectionValidationError("postprocessor selection residual sum of squares differs")
    if expected.score is None:
        if actual.score is not None:
            raise SelectionValidationError("an ineligible selection cannot carry a score")
        return
    if actual.score is None or not isclose(
        actual.score, expected.score, rel_tol=1e-12, abs_tol=1e-9
    ):
        raise SelectionValidationError("postprocessor selection score differs from the fit")
