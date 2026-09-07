CREATE SCHEMA IF NOT EXISTS {{schema}};

CREATE TABLE IF NOT EXISTS {{schema}}.schema_migrations (
    version text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {{schema}}.optimizer_profiles (
    name text NOT NULL,
    version integer NOT NULL CHECK (version > 0),
    solver_kind text NOT NULL,
    definition jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (name, version)
);

CREATE TABLE IF NOT EXISTS {{schema}}.jobs (
    id uuid PRIMARY KEY,
    operation text NOT NULL DEFAULT 'solve',
    status text NOT NULL CHECK (
        status IN ('queued', 'resolving_inputs', 'loading_telemetry', 'running',
                   'succeeded', 'failed', 'canceled')
    ),
    stage text NOT NULL,
    request_json jsonb NOT NULL,
    request_hash text NOT NULL,
    resolved_configuration jsonb,
    actor_id text NOT NULL,
    actor_type text NOT NULL CHECK (actor_type IN ('human', 'service')),
    idempotency_key text NOT NULL,
    label text,
    tags jsonb NOT NULL DEFAULT '[]'::jsonb,
    cancel_requested boolean NOT NULL DEFAULT false,
    attempt_count integer NOT NULL DEFAULT 0,
    max_attempts integer NOT NULL DEFAULT 3 CHECK (max_attempts > 0),
    available_at timestamptz NOT NULL DEFAULT now(),
    lease_owner text,
    lease_expires_at timestamptz,
    heartbeat_at timestamptz,
    terminal_error jsonb,
    warnings jsonb NOT NULL DEFAULT '[]'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    finished_at timestamptz,
    UNIQUE (actor_id, idempotency_key)
);

CREATE INDEX IF NOT EXISTS jobs_claim_idx
    ON {{schema}}.jobs (status, available_at, created_at)
    WHERE status = 'queued';
CREATE INDEX IF NOT EXISTS jobs_lease_idx
    ON {{schema}}.jobs (lease_expires_at)
    WHERE lease_expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS jobs_tags_gin_idx ON {{schema}}.jobs USING gin (tags);

CREATE TABLE IF NOT EXISTS {{schema}}.job_contacts (
    job_id uuid NOT NULL REFERENCES {{schema}}.jobs(id) ON DELETE CASCADE,
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    contact_id uuid NOT NULL,
    spacecraft_id text,
    spacecraft_name text,
    system_id text,
    station_id text,
    ephemeris_id text,
    provenance jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (job_id, ordinal)
);
CREATE INDEX IF NOT EXISTS job_contacts_contact_idx
    ON {{schema}}.job_contacts (contact_id);

CREATE TABLE IF NOT EXISTS {{schema}}.job_runs (
    id uuid PRIMARY KEY,
    job_id uuid NOT NULL REFERENCES {{schema}}.jobs(id) ON DELETE CASCADE,
    ordinal integer NOT NULL DEFAULT 0,
    algorithm text NOT NULL,
    parameterization text NOT NULL,
    resolved_settings jsonb NOT NULL,
    status text NOT NULL,
    result_summary jsonb,
    terminal_error jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    finished_at timestamptz,
    UNIQUE (job_id, ordinal)
);

CREATE TABLE IF NOT EXISTS {{schema}}.job_artifacts (
    id uuid PRIMARY KEY,
    job_id uuid NOT NULL REFERENCES {{schema}}.jobs(id) ON DELETE CASCADE,
    run_id uuid REFERENCES {{schema}}.job_runs(id) ON DELETE CASCADE,
    kind text NOT NULL,
    version integer NOT NULL DEFAULT 1 CHECK (version > 0),
    content_type text NOT NULL,
    sha256 text NOT NULL,
    payload bytea,
    json_payload jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK ((payload IS NULL) <> (json_payload IS NULL)),
    UNIQUE (job_id, kind, version, content_type)
);

CREATE TABLE IF NOT EXISTS {{schema}}.job_events (
    id bigserial PRIMARY KEY,
    job_id uuid NOT NULL REFERENCES {{schema}}.jobs(id) ON DELETE CASCADE,
    event_type text NOT NULL,
    from_status text,
    to_status text,
    stage text,
    diagnostic jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS job_events_job_time_idx
    ON {{schema}}.job_events (job_id, created_at, id);

CREATE OR REPLACE VIEW {{schema}}.job_status_v1 AS
SELECT
    j.id AS job_id,
    j.status,
    j.stage,
    j.actor_id,
    j.actor_type,
    j.label,
    j.tags,
    j.cancel_requested,
    j.attempt_count,
    j.max_attempts,
    j.created_at,
    j.updated_at,
    j.started_at,
    j.finished_at,
    j.warnings,
    j.terminal_error,
    c.contact_id,
    c.spacecraft_id,
    c.spacecraft_name,
    c.system_id,
    c.station_id,
    c.ephemeris_id
FROM {{schema}}.jobs j
LEFT JOIN {{schema}}.job_contacts c ON c.job_id = j.id AND c.ordinal = 0;

CREATE OR REPLACE VIEW {{schema}}.job_results_v1 AS
SELECT
    j.id AS job_id,
    j.status,
    j.finished_at,
    r.algorithm,
    r.parameterization,
    r.status AS run_status,
    r.result_summary,
    r.terminal_error,
    c.contact_id,
    c.spacecraft_id,
    c.spacecraft_name
FROM {{schema}}.jobs j
JOIN {{schema}}.job_runs r ON r.job_id = j.id AND r.ordinal = 0
LEFT JOIN {{schema}}.job_contacts c ON c.job_id = j.id AND c.ordinal = 0;

CREATE OR REPLACE VIEW {{schema}}.job_events_v1 AS
SELECT id, job_id, event_type, from_status, to_status, stage, diagnostic, created_at
FROM {{schema}}.job_events;
