CREATE TABLE IF NOT EXISTS {{schema}}.tdm_profiles (
    name text NOT NULL,
    version integer NOT NULL CHECK (version > 0),
    product text NOT NULL CHECK (product IN ('track', 'angle')),
    definition jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (name, version)
);

ALTER TABLE {{schema}}.jobs
    DROP CONSTRAINT IF EXISTS jobs_operation_check;
ALTER TABLE {{schema}}.jobs
    ADD CONSTRAINT jobs_operation_check
    CHECK (operation IN ('solve', 'tdm_export'));

ALTER TABLE {{schema}}.job_artifacts
    ADD COLUMN IF NOT EXISTS filename text,
    ADD COLUMN IF NOT EXISTS metadata jsonb NOT NULL DEFAULT '{}'::jsonb;

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
    c.ephemeris_id,
    j.operation
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
    c.spacecraft_name,
    j.operation
FROM {{schema}}.jobs j
JOIN {{schema}}.job_runs r ON r.job_id = j.id AND r.ordinal = 0
LEFT JOIN {{schema}}.job_contacts c ON c.job_id = j.id AND c.ordinal = 0;

CREATE OR REPLACE VIEW {{schema}}.tdm_artifacts_v1 AS
SELECT
    j.id AS job_id,
    a.id AS artifact_id,
    c.contact_id,
    a.metadata ->> 'product' AS product,
    a.filename,
    a.content_type,
    a.sha256,
    octet_length(a.payload) AS byte_count,
    j.warnings,
    a.created_at,
    convert_from(a.payload, 'UTF8') AS tdm_text
FROM {{schema}}.jobs j
JOIN {{schema}}.job_artifacts a ON a.job_id = j.id AND a.kind = 'tdm'
LEFT JOIN {{schema}}.job_contacts c ON c.job_id = j.id AND c.ordinal = 0
WHERE j.operation = 'tdm_export' AND a.payload IS NOT NULL;
