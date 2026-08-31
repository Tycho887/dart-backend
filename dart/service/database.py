"""Direct psycopg persistence for submission, leases, events, and artifacts."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from .config import ServiceSettings
from .profiles import profile_documents
from .tdm_profiles import load_tdm_profile_documents


class IdempotencyConflict(Exception):
    pass


class JobNotFound(Exception):
    pass


class JobOwnershipConflict(Exception):
    pass


@dataclass(frozen=True)
class ClaimedJob:
    id: UUID
    request_json: dict
    attempt_count: int
    max_attempts: int
    operation: str = "solve"


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


class Database:
    def __init__(self, settings: ServiceSettings, *, open_pool: bool = True):
        settings.validate()
        self.settings = settings
        self.schema = settings.database_schema
        self.pool = ConnectionPool(
            conninfo=settings.database_url,
            min_size=1,
            max_size=10,
            open=open_pool,
            kwargs={"row_factory": dict_row},
        )

    def open(self) -> None:
        self.pool.open(wait=True)

    def close(self) -> None:
        self.pool.close()

    def _table(self, name: str) -> sql.Composed:
        return sql.SQL("{}.{}").format(
            sql.Identifier(self.schema), sql.Identifier(name)
        )

    def migrate(self) -> None:
        migration_dir = importlib.resources.files("dart.service.migrations")
        files = sorted(
            item for item in migration_dir.iterdir() if item.name.endswith(".sql")
        )
        with self.pool.connection() as conn, conn.transaction():
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext('dart-service-migrations'))"
            )
            conn.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                    sql.Identifier(self.schema)
                )
            )
            conn.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {} "
                    "(version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
                ).format(self._table("schema_migrations"))
            )
            applied = {
                row["version"]
                for row in conn.execute(
                    sql.SQL("SELECT version FROM {}").format(
                        self._table("schema_migrations")
                    )
                ).fetchall()
            }
            for item in files:
                if item.name in applied:
                    continue
                body = item.read_text(encoding="utf-8").replace(
                    "{{schema}}", f'"{self.schema}"'
                )
                conn.execute(body)
                conn.execute(
                    sql.SQL(
                        "INSERT INTO {} (version) VALUES (%s) ON CONFLICT DO NOTHING"
                    ).format(self._table("schema_migrations")),
                    (item.name,),
                )
            for profile in profile_documents():
                conn.execute(
                    sql.SQL(
                        "INSERT INTO {} (name, version, solver_kind, definition) "
                        "VALUES (%s, %s, %s, %s) ON CONFLICT (name, version) DO NOTHING"
                    ).format(self._table("optimizer_profiles")),
                    (
                        profile["name"],
                        profile["version"],
                        profile["solver_kind"],
                        Jsonb(profile),
                    ),
                )
            for profile in load_tdm_profile_documents(
                self.settings.tdm_profile_dir
            ):
                conn.execute(
                    sql.SQL(
                        "INSERT INTO {} (name, version, product, definition) "
                        "VALUES (%s, %s, %s, %s) "
                        "ON CONFLICT (name, version) DO NOTHING"
                    ).format(self._table("tdm_profiles")),
                    (
                        profile["name"],
                        profile["version"],
                        profile["product"],
                        Jsonb(profile),
                    ),
                )

    def healthcheck(self) -> None:
        with self.pool.connection() as conn:
            conn.execute("SELECT 1").fetchone()

    def queue_metrics(self) -> tuple[int, float]:
        with self.pool.connection() as conn:
            row = conn.execute(
                sql.SQL(
                    "SELECT count(*)::integer AS depth, "
                    "COALESCE(extract(epoch FROM now() - min(created_at)), 0)::float8 AS age "
                    "FROM {} WHERE status = 'queued'"
                ).format(self._table("jobs"))
            ).fetchone()
        return row["depth"], row["age"]

    def list_profiles(self) -> list[dict]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                sql.SQL("SELECT definition FROM {} ORDER BY name, version").format(
                    self._table("optimizer_profiles")
                )
            ).fetchall()
        return [row["definition"] for row in rows]

    def list_tdm_profiles(self) -> list[dict]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                sql.SQL("SELECT definition FROM {} ORDER BY name, version").format(
                    self._table("tdm_profiles")
                )
            ).fetchall()
        return [row["definition"] for row in rows]

    def get_tdm_profile(self, name: str, version: int) -> dict:
        with self.pool.connection() as conn:
            row = conn.execute(
                sql.SQL(
                    "SELECT definition FROM {} WHERE name = %s AND version = %s"
                ).format(self._table("tdm_profiles")),
                (name, version),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown TDM profile {name!r} version {version}")
        return row["definition"]

    def submit_job(
        self,
        *,
        request_json: dict,
        actor_id: str,
        actor_type: str,
        idempotency_key: str,
        max_attempts: int,
        operation: str = "solve",
    ) -> tuple[UUID, bool]:
        request_hash = sha256_json(
            {"operation": operation, "request": request_json}
        )
        job_id = uuid4()
        context = request_json.get("client_context", {})
        with self.pool.connection() as conn, conn.transaction():
            inserted = conn.execute(
                sql.SQL(
                    "INSERT INTO {} "
                    "(id, operation, status, stage, request_json, request_hash, actor_id, actor_type, "
                    " idempotency_key, label, tags, max_attempts) "
                    "VALUES (%s, %s, 'queued', 'queued', %s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (actor_id, idempotency_key) DO NOTHING RETURNING id"
                ).format(self._table("jobs")),
                (
                    job_id,
                    operation,
                    Jsonb(request_json),
                    request_hash,
                    actor_id,
                    actor_type,
                    idempotency_key,
                    context.get("label"),
                    Jsonb(context.get("tags", [])),
                    max_attempts,
                ),
            ).fetchone()
            replay = inserted is None
            if replay:
                existing = conn.execute(
                    sql.SQL(
                        "SELECT id, request_hash FROM {} WHERE actor_id = %s AND idempotency_key = %s"
                    ).format(self._table("jobs")),
                    (actor_id, idempotency_key),
                ).fetchone()
                if existing["request_hash"] != request_hash:
                    raise IdempotencyConflict(idempotency_key)
                job_id = existing["id"]
            else:
                conn.execute(
                    sql.SQL(
                        "INSERT INTO {} (job_id, event_type, to_status, stage) "
                        "VALUES (%s, 'submitted', 'queued', 'queued')"
                    ).format(self._table("job_events")),
                    (job_id,),
                )
                conn.execute("SELECT pg_notify('dart_jobs', %s)", (str(job_id),))
        return job_id, replay

    def cancel_job(self, job_id: UUID, actor_id: str) -> dict:
        with self.pool.connection() as conn, conn.transaction():
            row = conn.execute(
                sql.SQL("SELECT * FROM {} WHERE id = %s FOR UPDATE").format(
                    self._table("jobs")
                ),
                (job_id,),
            ).fetchone()
            if row is None:
                raise JobNotFound(str(job_id))
            if row["actor_id"] != actor_id:
                raise JobOwnershipConflict(str(job_id))
            terminal = row["status"] in {"succeeded", "failed", "canceled"}
            new_status = row["status"]
            cancel_requested = row["cancel_requested"]
            if not terminal:
                cancel_requested = True
                if row["status"] == "queued":
                    new_status = "canceled"
                    conn.execute(
                        sql.SQL(
                            "UPDATE {} SET status = 'canceled', stage = 'canceled', "
                            "cancel_requested = true, updated_at = now(), finished_at = now() WHERE id = %s"
                        ).format(self._table("jobs")),
                        (job_id,),
                    )
                else:
                    conn.execute(
                        sql.SQL(
                            "UPDATE {} SET cancel_requested = true, updated_at = now() WHERE id = %s"
                        ).format(self._table("jobs")),
                        (job_id,),
                    )
                conn.execute(
                    sql.SQL(
                        "INSERT INTO {} (job_id, event_type, from_status, to_status, stage) "
                        "VALUES (%s, 'cancel_requested', %s, %s, %s)"
                    ).format(self._table("job_events")),
                    (
                        job_id,
                        row["status"],
                        new_status,
                        "canceled" if new_status == "canceled" else row["stage"],
                    ),
                )
            return {
                "job_id": job_id,
                "status": new_status,
                "cancel_requested": cancel_requested,
            }

    def recover_expired_leases(self) -> int:
        with self.pool.connection() as conn, conn.transaction():
            exhausted = conn.execute(
                sql.SQL(
                    "UPDATE {} SET status = 'failed', stage = 'failed', finished_at = now(), "
                    "terminal_error = %s, lease_owner = NULL, lease_expires_at = NULL, "
                    "heartbeat_at = NULL, updated_at = now() "
                    "WHERE status IN ('resolving_inputs', 'loading_telemetry', 'running') "
                    "AND lease_expires_at < now() AND attempt_count >= max_attempts RETURNING id"
                ).format(self._table("jobs")),
                (
                    Jsonb(
                        {
                            "type": "urn:dart:problem:worker_lease_exhausted",
                            "title": "Worker Lease Exhausted",
                            "status": 503,
                            "detail": "The final worker lease expired.",
                            "code": "worker_lease_exhausted",
                            "retryable": False,
                        }
                    ),
                ),
            ).fetchall()
            rows = conn.execute(
                sql.SQL(
                    "UPDATE {} SET status = 'queued', stage = 'queued', available_at = now(), "
                    "lease_owner = NULL, lease_expires_at = NULL, heartbeat_at = NULL, updated_at = now() "
                    "WHERE status IN ('resolving_inputs', 'loading_telemetry', 'running') "
                    "AND lease_expires_at < now() AND attempt_count < max_attempts RETURNING id"
                ).format(self._table("jobs"))
            ).fetchall()
            for row in rows:
                conn.execute(
                    sql.SQL(
                        "INSERT INTO {} (job_id, event_type, to_status, stage, diagnostic) "
                        "VALUES (%s, 'lease_recovered', 'queued', 'queued', %s)"
                    ).format(self._table("job_events")),
                    (row["id"], Jsonb({"reason": "expired_lease"})),
                )
            for row in exhausted:
                conn.execute(
                    sql.SQL(
                        "INSERT INTO {} (job_id, event_type, to_status, stage, diagnostic) "
                        "VALUES (%s, 'lease_exhausted', 'failed', 'failed', %s)"
                    ).format(self._table("job_events")),
                    (row["id"], Jsonb({"reason": "expired_final_lease"})),
                )
            return len(rows) + len(exhausted)

    def claim_job(self, worker_id: str, lease_seconds: int) -> ClaimedJob | None:
        with self.pool.connection() as conn, conn.transaction():
            row = conn.execute(
                sql.SQL(
                    "WITH candidate AS ("
                    " SELECT id FROM {} WHERE status = 'queued' AND available_at <= now() "
                    " AND attempt_count < max_attempts "
                    " AND cancel_requested = false ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1"
                    ") UPDATE {} j SET status = 'resolving_inputs', stage = 'resolving_inputs', "
                    "attempt_count = attempt_count + 1, lease_owner = %s, "
                    "lease_expires_at = now() + (%s * interval '1 second'), heartbeat_at = now(), "
                    "started_at = COALESCE(started_at, now()), updated_at = now() "
                    "FROM candidate WHERE j.id = candidate.id "
                    "RETURNING j.id, j.request_json, j.attempt_count, j.max_attempts, j.operation"
                ).format(self._table("jobs"), self._table("jobs")),
                (worker_id, lease_seconds),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                sql.SQL(
                    "INSERT INTO {} (job_id, event_type, from_status, to_status, stage, diagnostic) "
                    "VALUES (%s, 'claimed', 'queued', 'resolving_inputs', 'resolving_inputs', %s)"
                ).format(self._table("job_events")),
                (
                    row["id"],
                    Jsonb({"worker_id": worker_id, "attempt": row["attempt_count"]}),
                ),
            )
            return ClaimedJob(**row)

    def heartbeat(self, job_id: UUID, worker_id: str, lease_seconds: int) -> bool:
        with self.pool.connection() as conn, conn.transaction():
            row = conn.execute(
                sql.SQL(
                    "UPDATE {} SET heartbeat_at = now(), "
                    "lease_expires_at = now() + (%s * interval '1 second'), updated_at = now() "
                    "WHERE id = %s AND lease_owner = %s AND status NOT IN ('succeeded','failed','canceled') "
                    "RETURNING id"
                ).format(self._table("jobs")),
                (lease_seconds, job_id, worker_id),
            ).fetchone()
            return row is not None

    def cancellation_requested(self, job_id: UUID) -> bool:
        with self.pool.connection() as conn:
            row = conn.execute(
                sql.SQL("SELECT cancel_requested FROM {} WHERE id = %s").format(
                    self._table("jobs")
                ),
                (job_id,),
            ).fetchone()
            return bool(row and row["cancel_requested"])

    def set_stage(self, job_id: UUID, worker_id: str, status: str, stage: str) -> None:
        with self.pool.connection() as conn, conn.transaction():
            old = conn.execute(
                sql.SQL("SELECT status FROM {} WHERE id = %s FOR UPDATE").format(
                    self._table("jobs")
                ),
                (job_id,),
            ).fetchone()
            if old is None:
                raise JobNotFound(str(job_id))
            updated = conn.execute(
                sql.SQL(
                    "UPDATE {} SET status = %s, stage = %s, updated_at = now() "
                    "WHERE id = %s AND lease_owner = %s RETURNING id"
                ).format(self._table("jobs")),
                (status, stage, job_id, worker_id),
            ).fetchone()
            if updated is None:
                raise RuntimeError(f"worker {worker_id} lost lease for {job_id}")
            conn.execute(
                sql.SQL(
                    "INSERT INTO {} (job_id, event_type, from_status, to_status, stage) "
                    "VALUES (%s, 'stage_changed', %s, %s, %s)"
                ).format(self._table("job_events")),
                (job_id, old["status"], status, stage),
            )

    def store_resolution(
        self,
        *,
        job_id: UUID,
        worker_id: str,
        resolved_configuration: dict,
        contact: dict,
        run_id: UUID,
        algorithm: str,
        parameterization: str,
    ) -> None:
        with self.pool.connection() as conn, conn.transaction():
            updated = conn.execute(
                sql.SQL(
                    "UPDATE {} SET resolved_configuration = %s, updated_at = now() "
                    "WHERE id = %s AND lease_owner = %s RETURNING id"
                ).format(self._table("jobs")),
                (Jsonb(resolved_configuration), job_id, worker_id),
            ).fetchone()
            if updated is None:
                raise RuntimeError(f"worker {worker_id} lost lease for {job_id}")
            conn.execute(
                sql.SQL(
                    "INSERT INTO {} (job_id, ordinal, contact_id, spacecraft_id, spacecraft_name, "
                    "system_id, station_id, ephemeris_id, provenance) "
                    "VALUES (%s, 0, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (job_id, ordinal) DO UPDATE SET "
                    "spacecraft_id=EXCLUDED.spacecraft_id, spacecraft_name=EXCLUDED.spacecraft_name, "
                    "system_id=EXCLUDED.system_id, station_id=EXCLUDED.station_id, "
                    "ephemeris_id=EXCLUDED.ephemeris_id, provenance=EXCLUDED.provenance"
                ).format(self._table("job_contacts")),
                (
                    job_id,
                    contact["contact_id"],
                    contact.get("spacecraft_id"),
                    contact.get("spacecraft_name"),
                    contact.get("system_id"),
                    contact.get("station_id"),
                    contact.get("ephemeris_id"),
                    Jsonb(contact.get("provenance", {})),
                ),
            )
            conn.execute(
                sql.SQL(
                    "INSERT INTO {} (id, job_id, ordinal, algorithm, parameterization, resolved_settings, status, started_at) "
                    "VALUES (%s, %s, 0, %s, %s, %s, 'running', now()) "
                    "ON CONFLICT (job_id, ordinal) DO UPDATE SET resolved_settings=EXCLUDED.resolved_settings, "
                    "status='running', started_at=COALESCE({}.started_at, now()), terminal_error=NULL"
                ).format(self._table("job_runs"), self._table("job_runs")),
                (
                    run_id,
                    job_id,
                    algorithm,
                    parameterization,
                    Jsonb(resolved_configuration),
                ),
            )

    def store_artifact(
        self,
        *,
        job_id: UUID,
        run_id: UUID,
        kind: str,
        content_type: str,
        data: bytes | dict,
        filename: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        raw = data if isinstance(data, bytes) else canonical_json(data)
        digest = hashlib.sha256(raw).hexdigest()
        with self.pool.connection() as conn, conn.transaction():
            conn.execute(
                sql.SQL(
                    "INSERT INTO {} (id, job_id, run_id, kind, content_type, sha256, payload, json_payload, filename, metadata) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (job_id, kind, version, content_type) DO UPDATE SET "
                    "run_id=EXCLUDED.run_id, sha256=EXCLUDED.sha256, payload=EXCLUDED.payload, "
                    "json_payload=EXCLUDED.json_payload, filename=EXCLUDED.filename, "
                    "metadata=EXCLUDED.metadata, created_at=now()"
                ).format(self._table("job_artifacts")),
                (
                    uuid4(),
                    job_id,
                    run_id,
                    kind,
                    content_type,
                    digest,
                    data if isinstance(data, bytes) else None,
                    None if isinstance(data, bytes) else Jsonb(data),
                    filename,
                    Jsonb(metadata or {}),
                ),
            )

    def finish_success(
        self,
        *,
        job_id: UUID,
        worker_id: str,
        run_id: UUID,
        summary: dict,
        warnings: list[str],
    ) -> str:
        with self.pool.connection() as conn, conn.transaction():
            job = conn.execute(
                sql.SQL(
                    "SELECT status, cancel_requested FROM {} WHERE id = %s FOR UPDATE"
                ).format(self._table("jobs")),
                (job_id,),
            ).fetchone()
            status = "canceled" if job["cancel_requested"] else "succeeded"
            run_status = "canceled" if status == "canceled" else "succeeded"
            updated = conn.execute(
                sql.SQL(
                    "UPDATE {} SET status=%s, stage=%s, warnings=%s, finished_at=now(), updated_at=now(), "
                    "lease_owner=NULL, lease_expires_at=NULL, heartbeat_at=NULL WHERE id=%s AND lease_owner=%s"
                ).format(self._table("jobs")),
                (status, status, Jsonb(warnings), job_id, worker_id),
            ).rowcount
            if updated != 1:
                raise RuntimeError(f"worker {worker_id} lost lease for {job_id}")
            conn.execute(
                sql.SQL(
                    "UPDATE {} SET status=%s, result_summary=%s, finished_at=now() WHERE id=%s"
                ).format(self._table("job_runs")),
                (run_status, Jsonb(summary), run_id),
            )
            conn.execute(
                sql.SQL(
                    "INSERT INTO {} (job_id, event_type, from_status, to_status, stage) "
                    "VALUES (%s, %s, %s, %s, %s)"
                ).format(self._table("job_events")),
                (job_id, status, job["status"], status, status),
            )
            return status

    def finish_canceled(self, job_id: UUID, worker_id: str) -> None:
        with self.pool.connection() as conn, conn.transaction():
            row = conn.execute(
                sql.SQL(
                    "UPDATE {} SET status='canceled', stage='canceled', finished_at=now(), updated_at=now(), "
                    "lease_owner=NULL, lease_expires_at=NULL, heartbeat_at=NULL "
                    "WHERE id=%s AND lease_owner=%s RETURNING status"
                ).format(self._table("jobs")),
                (job_id, worker_id),
            ).fetchone()
            if row:
                conn.execute(
                    sql.SQL(
                        "UPDATE {} SET status='canceled', finished_at=now() "
                        "WHERE job_id=%s AND status='running'"
                    ).format(self._table("job_runs")),
                    (job_id,),
                )
                conn.execute(
                    sql.SQL(
                        "INSERT INTO {} (job_id, event_type, to_status, stage) "
                        "VALUES (%s, 'canceled', 'canceled', 'canceled')"
                    ).format(self._table("job_events")),
                    (job_id,),
                )

    def finish_solver_failure(
        self,
        *,
        job_id: UUID,
        worker_id: str,
        run_id: UUID,
        summary: dict,
        error: dict,
        warnings: list[str],
    ) -> None:
        with self.pool.connection() as conn, conn.transaction():
            updated = conn.execute(
                sql.SQL(
                    "UPDATE {} SET status='failed', stage='failed', terminal_error=%s, warnings=%s, "
                    "finished_at=now(), updated_at=now(), lease_owner=NULL, lease_expires_at=NULL, heartbeat_at=NULL "
                    "WHERE id=%s AND lease_owner=%s"
                ).format(self._table("jobs")),
                (Jsonb(error), Jsonb(warnings), job_id, worker_id),
            ).rowcount
            if updated != 1:
                raise RuntimeError(f"worker {worker_id} lost lease for {job_id}")
            conn.execute(
                sql.SQL(
                    "UPDATE {} SET status='failed', result_summary=%s, terminal_error=%s, finished_at=now() "
                    "WHERE id=%s"
                ).format(self._table("job_runs")),
                (Jsonb(summary), Jsonb(error), run_id),
            )
            conn.execute(
                sql.SQL(
                    "INSERT INTO {} (job_id, event_type, to_status, stage, diagnostic) "
                    "VALUES (%s, 'solver_failed', 'failed', 'failed', %s)"
                ).format(self._table("job_events")),
                (job_id, Jsonb(error)),
            )

    def fail_or_retry(
        self,
        *,
        job: ClaimedJob,
        worker_id: str,
        error: dict,
        retryable: bool,
    ) -> str:
        will_retry = retryable and job.attempt_count < job.max_attempts
        status = "queued" if will_retry else "failed"
        delay_seconds = min(300, 5 * (2 ** max(0, job.attempt_count - 1)))
        with self.pool.connection() as conn, conn.transaction():
            if will_retry:
                conn.execute(
                    sql.SQL(
                        "UPDATE {} SET status='queued', stage='queued', available_at=now() + (%s * interval '1 second'), "
                        "terminal_error=%s, lease_owner=NULL, lease_expires_at=NULL, heartbeat_at=NULL, updated_at=now() "
                        "WHERE id=%s AND lease_owner=%s"
                    ).format(self._table("jobs")),
                    (delay_seconds, Jsonb(error), job.id, worker_id),
                )
            else:
                conn.execute(
                    sql.SQL(
                        "UPDATE {} SET status='failed', stage='failed', terminal_error=%s, finished_at=now(), "
                        "lease_owner=NULL, lease_expires_at=NULL, heartbeat_at=NULL, updated_at=now() "
                        "WHERE id=%s AND lease_owner=%s"
                    ).format(self._table("jobs")),
                    (Jsonb(error), job.id, worker_id),
                )
                conn.execute(
                    sql.SQL(
                        "UPDATE {} SET status='failed', terminal_error=%s, finished_at=now() "
                        "WHERE job_id=%s AND status='running'"
                    ).format(self._table("job_runs")),
                    (Jsonb(error), job.id),
                )
            conn.execute(
                sql.SQL(
                    "INSERT INTO {} (job_id, event_type, to_status, stage, diagnostic) "
                    "VALUES (%s, %s, %s, %s, %s)"
                ).format(self._table("job_events")),
                (
                    job.id,
                    "retry_scheduled" if will_retry else "failed",
                    status,
                    status,
                    Jsonb(error),
                ),
            )
            if will_retry:
                conn.execute("SELECT pg_notify('dart_jobs', %s)", (str(job.id),))
        return status
