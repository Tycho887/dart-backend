"""Transactional estimates and a lease-fenced PostgreSQL work queue."""

import hashlib
import importlib.resources
from dataclasses import dataclass
from typing import LiteralString
from uuid import UUID, uuid4

from psycopg import Connection, sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from .config import ServiceSettings
from .models import (
    EstimateAccepted,
    ForwardModelProfile,
    OptimizerProfile,
    ResolvedEstimateConfiguration,
)
from .profiles import forward_model_profiles, optimizer_profiles
from .serialization import json_bytes

TERMINAL = {"succeeded", "failed", "canceled"}


class IdempotencyConflict(Exception):
    pass


class JobNotFound(Exception):
    pass


class JobOwnershipConflict(Exception):
    pass


class LeaseLost(Exception):
    pass


@dataclass(frozen=True)
class ClaimedJob:
    id: UUID
    estimate_uuid: UUID
    run_id: UUID
    attempt_count: int
    max_attempts: int
    configuration: ResolvedEstimateConfiguration


class Database:
    def __init__(self, settings: ServiceSettings, *, open_pool: bool = True):
        settings.validate()
        self.settings = settings
        self.pool: ConnectionPool[Connection[dict]] = ConnectionPool(
            settings.database_url,
            min_size=1,
            max_size=10,
            open=open_pool,
            kwargs={"row_factory": dict_row},
        )

    def open(self) -> None:
        self.pool.open(wait=True)

    def close(self) -> None:
        self.pool.close()

    def query(self, text: LiteralString) -> sql.Composed:
        return sql.SQL(text).format(s=sql.Identifier(self.settings.database_schema))

    def migrate(self) -> None:
        directory = importlib.resources.files("dart.service.migrations")
        with self.pool.connection() as conn, conn.transaction():
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext('dart-service-migrations'))"
            )
            conn.execute(self.query("CREATE SCHEMA IF NOT EXISTS {s}"))
            conn.execute(
                self.query(
                    "CREATE TABLE IF NOT EXISTS {s}.schema_migrations (version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
                )
            )
            applied = {
                row["version"]
                for row in conn.execute(
                    self.query("SELECT version FROM {s}.schema_migrations")
                )
            }
            for item in sorted(directory.iterdir(), key=lambda p: p.name):
                if item.name.endswith(".sql") and item.name not in applied:
                    conn.execute(
                        item.read_text()
                        .replace(
                            "{{schema}}",
                            sql.Identifier(self.settings.database_schema).as_string(
                                conn
                            ),
                        )
                        .encode()
                    )
                    conn.execute(
                        self.query(
                            "INSERT INTO {s}.schema_migrations(version) VALUES (%s)"
                        ),
                        (item.name,),
                    )
            self._seed_profiles(conn)

    def _seed_profiles(self, conn: Connection[dict]) -> None:
        for table, profiles in (
            ("forward_model_profiles", forward_model_profiles()),
            ("optimizer_profiles", optimizer_profiles()),
        ):
            target = sql.Identifier(self.settings.database_schema, table)
            for profile in profiles:
                data = profile.model_dump(mode="json")
                conn.execute(
                    sql.SQL(
                        "INSERT INTO {} (name,version,definition) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING"
                    ).format(target),
                    (profile.name, profile.version, Jsonb(data)),
                )
                stored = conn.execute(
                    sql.SQL(
                        "SELECT definition FROM {} WHERE name=%s AND version=%s"
                    ).format(target),
                    (profile.name, profile.version),
                ).fetchone()
                assert stored is not None
                if stored["definition"] != data:
                    raise ValueError(
                        f"immutable profile changed: {profile.name} v{profile.version}; increment its version"
                    )

    def healthcheck(self) -> None:
        with self.pool.connection() as conn:
            conn.execute(self.query("SELECT estimate_uuid FROM {s}.estimates LIMIT 0"))

    def list_profiles(self, family: str) -> list[dict]:
        tables = {
            "forward_model": "forward_model_profiles",
            "optimizer": "optimizer_profiles",
        }
        with self.pool.connection() as conn:
            rows = conn.execute(
                sql.SQL(
                    "SELECT definition FROM {} WHERE definition ? 'label' ORDER BY name,version"
                ).format(sql.Identifier(self.settings.database_schema, tables[family]))
            )
            return [r["definition"] for r in rows]

    def resolve_profiles(
        self,
        model_name: str,
        model_version: int,
        optimizer_name: str,
        optimizer_version: int,
    ) -> tuple[ForwardModelProfile, OptimizerProfile]:
        models = {
            (p["name"], p["version"]): p for p in self.list_profiles("forward_model")
        }
        optimizers = {
            (p["name"], p["version"]): p for p in self.list_profiles("optimizer")
        }
        return (
            ForwardModelProfile.model_validate(models[model_name, model_version]),
            OptimizerProfile.model_validate(
                optimizers[optimizer_name, optimizer_version]
            ),
        )

    def submit_estimate(
        self,
        configuration: ResolvedEstimateConfiguration,
        actor_id: str,
        actor_type: str,
        idempotency_key: str,
    ) -> EstimateAccepted:
        request = configuration.request.model_dump(mode="json")
        digest = hashlib.sha256(json_bytes(request)).hexdigest()
        job_id, estimate_uuid = uuid4(), uuid4()
        with self.pool.connection() as conn, conn.transaction():
            inserted = conn.execute(
                self.query("""INSERT INTO {s}.jobs
                (id,operation,status,stage,request_json,request_hash,actor_id,actor_type,idempotency_key,label,max_attempts)
                VALUES (%s,'estimate','queued','queued',%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT(actor_id,idempotency_key) DO NOTHING RETURNING id"""),
                (
                    job_id,
                    Jsonb(request),
                    digest,
                    actor_id,
                    actor_type,
                    idempotency_key,
                    request["label"],
                    self.settings.max_attempts,
                ),
            ).fetchone()
            if inserted is None:
                row = conn.execute(
                    self.query("""SELECT j.id AS job_id,j.status,j.request_hash,j.actor_type,e.estimate_uuid
                    FROM {s}.jobs j LEFT JOIN {s}.estimates e ON e.job_id=j.id
                    WHERE actor_id=%s AND idempotency_key=%s"""),
                    (actor_id, idempotency_key),
                ).fetchone()
                assert row is not None
                if (
                    row["request_hash"] != digest
                    or row["estimate_uuid"] is None
                    or row["actor_type"] != actor_type
                ):
                    raise IdempotencyConflict(idempotency_key)
                return EstimateAccepted(
                    job_id=row["job_id"],
                    estimate_uuid=row["estimate_uuid"],
                    status=row["status"],
                    idempotent_replay=True,
                )
            conn.execute(
                self.query("""INSERT INTO {s}.estimates
                (estimate_uuid,job_id,forward_model_name,forward_model_version,optimizer_name,optimizer_version,configuration,prior_ephemeris_id)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"""),
                (
                    estimate_uuid,
                    job_id,
                    configuration.forward_model.name,
                    configuration.forward_model.version,
                    configuration.optimizer.name,
                    configuration.optimizer.version,
                    Jsonb(configuration.model_dump(mode="json")),
                    request["ephemeris_id"],
                ),
            )
            for ordinal, contact_id in enumerate(request["contact_ids"]):
                conn.execute(
                    self.query(
                        "INSERT INTO {s}.estimate_contacts(estimate_uuid,contact_id,ordinal) VALUES (%s,%s,%s)"
                    ),
                    (estimate_uuid, contact_id, ordinal),
                )
            self._event(conn, job_id, "submitted", "queued")
            conn.execute("SELECT pg_notify('dart_jobs',%s)", (str(job_id),))
        return EstimateAccepted(
            job_id=job_id,
            estimate_uuid=estimate_uuid,
            status="queued",
            idempotent_replay=False,
        )

    def _event(
        self,
        conn: Connection[dict],
        job_id: UUID,
        event: str,
        status: str,
        diagnostic: dict | None = None,
    ) -> None:
        conn.execute(
            self.query(
                "INSERT INTO {s}.job_events(job_id,event_type,to_status,stage,diagnostic) VALUES (%s,%s,%s,%s,%s)"
            ),
            (job_id, event, status, status, Jsonb(diagnostic or {})),
        )

    def claim_job(self, worker_id: str, lease_seconds: int) -> ClaimedJob | None:
        with self.pool.connection() as conn, conn.transaction():
            row = conn.execute(
                self.query("""SELECT j.id,j.attempt_count,j.max_attempts,e.estimate_uuid,e.configuration
                FROM {s}.jobs j JOIN {s}.estimates e ON e.job_id=j.id
                WHERE j.operation='estimate' AND j.status='queued' AND j.available_at<=now()
                AND NOT j.cancel_requested AND j.attempt_count<j.max_attempts
                ORDER BY j.created_at FOR UPDATE OF j SKIP LOCKED LIMIT 1""")
            ).fetchone()
            if row is None:
                return None
            attempt = row["attempt_count"] + 1
            run_id = uuid4()
            config = ResolvedEstimateConfiguration.model_validate(row["configuration"])
            conn.execute(
                self.query("""UPDATE {s}.jobs SET status='resolving_inputs',stage='resolving_inputs',attempt_count=%s,
                lease_owner=%s,lease_expires_at=clock_timestamp()+(%s*interval '1 second'),heartbeat_at=clock_timestamp(),
                started_at=COALESCE(started_at,now()),updated_at=now(),terminal_error=NULL WHERE id=%s"""),
                (attempt, worker_id, lease_seconds, row["id"]),
            )
            conn.execute(
                self.query("""INSERT INTO {s}.job_runs(id,job_id,ordinal,algorithm,parameterization,resolved_settings,status,started_at)
                VALUES (%s,%s,%s,%s,%s,%s,'running',now())"""),
                (
                    run_id,
                    row["id"],
                    attempt,
                    config.forward_model.model,
                    config.forward_model.name,
                    Jsonb(row["configuration"]),
                ),
            )
            self._event(
                conn,
                row["id"],
                "claimed",
                "resolving_inputs",
                {"attempt": attempt, "run_id": str(run_id)},
            )
            return ClaimedJob(
                row["id"],
                row["estimate_uuid"],
                run_id,
                attempt,
                row["max_attempts"],
                config,
            )

    def _owned_job(
        self, conn: Connection[dict], job: ClaimedJob, worker_id: str
    ) -> dict:
        row = conn.execute(
            self.query("""SELECT j.* FROM {s}.jobs j JOIN {s}.job_runs r ON r.job_id=j.id
            WHERE j.id=%s AND r.id=%s AND r.ordinal=j.attempt_count AND j.attempt_count=%s
            AND j.lease_owner=%s AND j.lease_expires_at>clock_timestamp()
            AND j.status NOT IN ('succeeded','failed','canceled') FOR UPDATE OF j"""),
            (job.id, job.run_id, job.attempt_count, worker_id),
        ).fetchone()
        if row is None:
            raise LeaseLost(str(job.id))
        return row

    def heartbeat(self, job: ClaimedJob, worker_id: str, lease_seconds: int) -> None:
        with self.pool.connection() as conn, conn.transaction():
            self._owned_job(conn, job, worker_id)
            conn.execute(
                self.query(
                    "UPDATE {s}.jobs SET heartbeat_at=clock_timestamp(),lease_expires_at=clock_timestamp()+(%s*interval '1 second') WHERE id=%s"
                ),
                (lease_seconds, job.id),
            )

    def set_stage(self, job: ClaimedJob, worker_id: str, stage: str) -> bool:
        if stage not in {"resolving_inputs", "loading_telemetry", "running"}:
            raise ValueError("invalid worker stage")
        with self.pool.connection() as conn, conn.transaction():
            row = self._owned_job(conn, job, worker_id)
            if row["cancel_requested"]:
                self._terminal(conn, job, "canceled")
                return False
            conn.execute(
                self.query(
                    "UPDATE {s}.jobs SET status=%s,stage=%s,updated_at=now() WHERE id=%s"
                ),
                (stage, stage, job.id),
            )
            self._event(conn, job.id, "stage_changed", stage)
            return True

    def cancel_job(self, job_id: UUID, actor_id: str) -> dict:
        with self.pool.connection() as conn, conn.transaction():
            row = conn.execute(
                self.query(
                    "SELECT * FROM {s}.jobs WHERE id=%s AND operation='estimate' FOR UPDATE"
                ),
                (job_id,),
            ).fetchone()
            if row is None:
                raise JobNotFound(str(job_id))
            if row["actor_id"] != actor_id:
                raise JobOwnershipConflict(str(job_id))
            if row["status"] not in TERMINAL:
                conn.execute(
                    self.query("""UPDATE {s}.jobs SET cancel_requested=true, updated_at=now(),
                    status=CASE WHEN status='queued' THEN 'canceled' ELSE status END,
                    stage=CASE WHEN status='queued' THEN 'canceled' ELSE stage END,
                    finished_at=CASE WHEN status='queued' THEN now() ELSE finished_at END
                    WHERE id=%s"""),
                    (job_id,),
                )
                row["cancel_requested"] = True
                row["status"] = (
                    "canceled" if row["status"] == "queued" else row["status"]
                )
                self._event(conn, job_id, "cancel_requested", row["status"])
            return {
                "job_id": job_id,
                "status": row["status"],
                "cancel_requested": row["cancel_requested"],
            }

    def recover_expired_leases(self) -> int:
        with self.pool.connection() as conn, conn.transaction():
            rows = conn.execute(
                self.query("""SELECT id,attempt_count,max_attempts,cancel_requested FROM {s}.jobs
                WHERE operation='estimate' AND lease_expires_at<=clock_timestamp()
                AND status NOT IN ('succeeded','failed','canceled') FOR UPDATE SKIP LOCKED""")
            ).fetchall()
            for row in rows:
                self._recover(conn, row)
            return len(rows)

    def _recover(self, conn: Connection[dict], row: dict) -> None:
        status = "queued" if row["attempt_count"] < row["max_attempts"] else "failed"
        if row["cancel_requested"]:
            status = "canceled"
        error = {
            "code": "lease_expired",
            "detail": "Worker lease expired.",
            "retryable": status == "queued",
        }
        conn.execute(
            self.query(
                "UPDATE {s}.job_runs SET status='failed',terminal_error=%s,finished_at=now() WHERE job_id=%s AND ordinal=%s"
            ),
            (Jsonb(error), row["id"], row["attempt_count"]),
        )
        conn.execute(
            self.query("""UPDATE {s}.jobs SET status=%s,stage=%s,terminal_error=%s,lease_owner=NULL,
            lease_expires_at=NULL,heartbeat_at=NULL,updated_at=now(),available_at=now(),
            finished_at=CASE WHEN %s='queued' THEN NULL ELSE now() END WHERE id=%s"""),
            (status, status, Jsonb(error), status, row["id"]),
        )
        self._event(conn, row["id"], "lease_expired", status, error)

    def _terminal(
        self,
        conn: Connection[dict],
        job: ClaimedJob,
        status: str,
        error: dict | None = None,
    ) -> None:
        conn.execute(
            self.query("""UPDATE {s}.jobs SET status=%s,stage=%s,terminal_error=%s,finished_at=now(),updated_at=now(),
            lease_owner=NULL,lease_expires_at=NULL,heartbeat_at=NULL WHERE id=%s"""),
            (status, status, Jsonb(error) if error else None, job.id),
        )
        conn.execute(
            self.query(
                "UPDATE {s}.job_runs SET status=%s,terminal_error=%s,finished_at=now() WHERE id=%s"
            ),
            (status, Jsonb(error) if error else None, job.run_id),
        )
        self._event(conn, job.id, status, status, error)

    def fail_or_retry(
        self, job: ClaimedJob, worker_id: str, error: dict, retryable: bool
    ) -> str:
        with self.pool.connection() as conn, conn.transaction():
            row = self._owned_job(conn, job, worker_id)
            if row["cancel_requested"]:
                self._terminal(conn, job, "canceled")
                return "canceled"
            if not retryable or job.attempt_count >= job.max_attempts:
                self._terminal(conn, job, "failed", error)
                return "failed"
            conn.execute(
                self.query(
                    "UPDATE {s}.job_runs SET status='failed',terminal_error=%s,finished_at=now() WHERE id=%s"
                ),
                (Jsonb(error), job.run_id),
            )
            conn.execute(
                self.query("""UPDATE {s}.jobs SET status='queued',stage='queued',terminal_error=%s,updated_at=now(),
                available_at=now()+(%s*interval '1 second'),lease_owner=NULL,lease_expires_at=NULL,heartbeat_at=NULL WHERE id=%s"""),
                (Jsonb(error), min(300, 5 * 2 ** (job.attempt_count - 1)), job.id),
            )
            self._event(conn, job.id, "retry_scheduled", "queued", error)
            conn.execute("SELECT pg_notify('dart_jobs',%s)", (str(job.id),))
            return "queued"

    def store_artifact(
        self,
        conn: Connection[dict],
        job: ClaimedJob,
        kind: str,
        filename: str,
        content_type: str,
        payload: bytes,
        provenance: dict | None = None,
    ) -> UUID:
        artifact_uuid = uuid4()
        conn.execute(
            self.query("""INSERT INTO {s}.estimate_artifacts
            (artifact_uuid,estimate_uuid,run_id,kind,filename,content_type,payload,sha256,provenance)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)"""),
            (
                artifact_uuid,
                job.estimate_uuid,
                job.run_id,
                kind,
                filename,
                content_type,
                payload,
                hashlib.sha256(payload).hexdigest(),
                Jsonb(provenance or {}),
            ),
        )
        return artifact_uuid

    def load_prior(self, estimate_uuid: UUID) -> dict | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                self.query("SELECT prior FROM {s}.estimates WHERE estimate_uuid=%s"),
                (estimate_uuid,),
            ).fetchone()
        if row is None:
            raise JobNotFound(str(estimate_uuid))
        return row["prior"]

    def store_prior(
        self,
        job: ClaimedJob,
        worker_id: str,
        prior: dict,
        raw_measurements: bytes,
        provenance: dict,
        versions: dict,
    ) -> None:
        with self.pool.connection() as conn, conn.transaction():
            self._owned_job(conn, job, worker_id)
            updated = conn.execute(
                self.query("""UPDATE {s}.estimates SET prior=%s,epoch=to_timestamp(%s),
                spacecraft_id=%s,spacecraft_name=%s,software_version=%s WHERE estimate_uuid=%s AND prior IS NULL RETURNING estimate_uuid"""),
                (
                    Jsonb(prior),
                    prior["epoch_unix_s"],
                    prior["ephemeris"]["spacecraft_id"],
                    prior["contacts"][0]["spacecraft"],
                    Jsonb(versions),
                    job.estimate_uuid,
                ),
            ).fetchone()
            if updated is None:
                raise ValueError("estimate inputs are already frozen")
            for contact in prior["contacts"]:
                conn.execute(
                    self.query(
                        "UPDATE {s}.estimate_contacts SET provenance=%s WHERE estimate_uuid=%s AND contact_id=%s"
                    ),
                    (Jsonb(contact), job.estimate_uuid, contact["contact_id"]),
                )
            self.store_artifact(
                conn,
                job,
                "normalized_prior",
                "prior.json",
                "application/json",
                json_bytes(prior),
                provenance,
            )
            self.store_artifact(
                conn,
                job,
                "raw_measurements",
                "measurements.parquet",
                "application/vnd.apache.parquet",
                raw_measurements,
                provenance,
            )

    def persist_estimate_result(
        self,
        job: ClaimedJob,
        worker_id: str,
        parameters: list[dict],
        diagnostics: dict,
        output: dict,
        initialized_optimizer: dict,
        initialization_scan: list,
    ) -> str:
        with self.pool.connection() as conn, conn.transaction():
            row = self._owned_job(conn, job, worker_id)
            if row["cancel_requested"]:
                self._terminal(conn, job, "canceled")
                return "canceled"
            for parameter in parameters:
                conn.execute(
                    self.query("""INSERT INTO {s}.estimate_parameters
                    (estimate_uuid,parameter_name,ordinal,role,value,unit,initial_value,lower_bound,upper_bound,scale,standard_uncertainty,contact_id)
                    VALUES (%(estimate_uuid)s,%(parameter_name)s,%(ordinal)s,%(role)s,%(value)s,%(unit)s,%(initial_value)s,%(lower_bound)s,%(upper_bound)s,%(scale)s,%(standard_uncertainty)s,%(contact_id)s)"""),
                    {"estimate_uuid": job.estimate_uuid, **parameter},
                )
            values = {**diagnostics, "estimate_uuid": job.estimate_uuid}
            for key in ("covariance", "parameter_order", "warnings"):
                values[key] = None if values[key] is None else Jsonb(values[key])
            conn.execute(
                self.query("""INSERT INTO {s}.estimate_diagnostics
                (estimate_uuid,success,optimizer_status,message,objective,optimality,function_evaluations,jacobian_evaluations,
                 observation_count,whitened_residual_rms,residual_rms_hz,covariance,covariance_rank,parameter_order,warnings)
                VALUES (%(estimate_uuid)s,%(success)s,%(optimizer_status)s,%(message)s,%(objective)s,%(optimality)s,%(function_evaluations)s,
                %(jacobian_evaluations)s,%(observation_count)s,%(whitened_residual_rms)s,%(residual_rms_hz)s,%(covariance)s,%(covariance_rank)s,%(parameter_order)s,%(warnings)s)"""),
                values,
            )
            self.store_artifact(
                conn,
                job,
                "optimizer_output",
                "output.json",
                "application/json",
                json_bytes(output),
            )
            self.store_artifact(
                conn,
                job,
                "initialized_optimizer",
                "optimizer.json",
                "application/json",
                json_bytes(initialized_optimizer),
                {"scan": initialization_scan},
            )
            conn.execute(
                self.query(
                    "UPDATE {s}.estimates SET published_run_id=%s WHERE estimate_uuid=%s"
                ),
                (job.run_id, job.estimate_uuid),
            )
            conn.execute(
                self.query("UPDATE {s}.job_runs SET result_summary=%s WHERE id=%s"),
                (Jsonb(diagnostics), job.run_id),
            )
            status = "succeeded" if diagnostics["success"] else "failed"
            error = (
                None
                if diagnostics["success"]
                else {
                    "code": "optimizer_failed",
                    "detail": diagnostics["message"],
                    "retryable": False,
                }
            )
            self._terminal(conn, job, status, error)
            return status

    def queue_metrics(self) -> tuple[int, float]:
        with self.pool.connection() as conn:
            row = conn.execute(
                self.query("""SELECT count(*)::integer AS depth,
                COALESCE(extract(epoch FROM now()-min(created_at)),0)::float8 AS age
                FROM {s}.jobs WHERE status='queued' AND operation='estimate'""")
            ).fetchone()
            assert row is not None
            return row["depth"], row["age"]
