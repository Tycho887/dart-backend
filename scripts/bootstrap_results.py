"""Prepare private deployment credentials, migrate results, and grant scoped roles."""

import argparse
import os
import secrets
from pathlib import Path

from dotenv import dotenv_values
from psycopg import sql
from psycopg.conninfo import make_conninfo

from dart.service.config import ServiceSettings
from dart.service.database import Database


def prepare(source: Path, target: Path) -> dict[str, str]:
    values = {k: v for k, v in dotenv_values(source).items() if v is not None}
    existing = dotenv_values(target) if target.exists() else {}
    for name in (
        "DART_RESULTS_APP_PASSWORD",
        "DART_RESULTS_GRAFANA_PASSWORD",
        "DART_GATEWAY_TOKEN",
    ):
        values[name] = (
            existing.get(name) or values.get(name) or secrets.token_urlsafe(32)
        )
    if not values.get("POSTGRES_PASSWORD") or not values.get(
        "GF_SECURITY_ADMIN_PASSWORD"
    ):
        raise ValueError(
            "source env file must configure database and Grafana administrator passwords"
        )
    values["DART_RESULTS_APP_URL"] = make_conninfo(
        host="timescaledb",
        dbname="results",
        user="dart_app",
        password=values["DART_RESULTS_APP_PASSWORD"],
    )
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.chmod(target, 0o600)
    with os.fdopen(fd, "w") as stream:
        for key, value in values.items():
            escaped = (
                value.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
            )
            stream.write(f"{key}='{escaped}'\n")
    return values


def bootstrap(values: dict[str, str], host: str, port: int) -> None:
    url = make_conninfo(
        host=host,
        port=port,
        dbname="results",
        user="postgres",
        password=values["POSTGRES_PASSWORD"],
    )
    db = Database(ServiceSettings(url))
    try:
        db.migrate()
        with db.pool.connection() as conn, conn.transaction():
            conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
            for role, password in (
                ("dart_app", values["DART_RESULTS_APP_PASSWORD"]),
                ("dart_grafana", values["DART_RESULTS_GRAFANA_PASSWORD"]),
            ):
                if not conn.execute(
                    "SELECT 1 FROM pg_roles WHERE rolname=%s", (role,)
                ).fetchone():
                    conn.execute(
                        sql.SQL("CREATE ROLE {} LOGIN").format(sql.Identifier(role))
                    )
                conn.execute(
                    sql.SQL("ALTER ROLE {} PASSWORD {}").format(
                        sql.Identifier(role), sql.Literal(password)
                    )
                )
                conn.execute(
                    sql.SQL("GRANT CONNECT ON DATABASE results TO {}").format(
                        sql.Identifier(role)
                    )
                )
                conn.execute(
                    sql.SQL("GRANT USAGE ON SCHEMA dart TO {}").format(
                        sql.Identifier(role)
                    )
                )
            tables = (
                "jobs",
                "job_runs",
                "job_events",
                "estimates",
                "estimate_contacts",
                "estimate_parameters",
                "estimate_diagnostics",
                "estimate_artifacts",
            )
            for table in tables:
                conn.execute(
                    sql.SQL("GRANT SELECT,INSERT,UPDATE ON {} TO dart_app").format(
                        sql.Identifier("dart", table)
                    )
                )
            conn.execute(
                "GRANT USAGE,SELECT ON ALL SEQUENCES IN SCHEMA dart TO dart_app"
            )
            conn.execute(
                "GRANT SELECT ON dart.forward_model_profiles,dart.optimizer_profiles TO dart_app"
            )
            views = (
                "estimates_v1",
                "estimate_contacts_v1",
                "estimate_parameters_v1",
                "estimate_diagnostics_v1",
                "estimate_events_v1",
                "estimate_artifacts_v1",
            )
            for view in views:
                conn.execute(
                    sql.SQL("GRANT SELECT ON {} TO dart_grafana").format(
                        sql.Identifier("dart", view)
                    )
                )
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--runtime-env", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5434)
    args = parser.parse_args()
    values = prepare(args.env_file, args.runtime_env)
    if not args.prepare_only:
        bootstrap(values, args.host, args.port)
    print(
        "Private runtime configuration prepared."
        if args.prepare_only
        else "Results migrations and scoped roles configured."
    )


if __name__ == "__main__":
    main()
