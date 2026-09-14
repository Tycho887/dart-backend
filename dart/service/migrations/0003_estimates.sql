-- Additive migration: historical solve/TDM records remain untouched.
ALTER TABLE {{schema}}.jobs DROP CONSTRAINT jobs_operation_check;
ALTER TABLE {{schema}}.jobs ADD CONSTRAINT jobs_operation_check
    CHECK (operation IN ('solve', 'tdm_export', 'estimate'));
ALTER TABLE {{schema}}.optimizer_profiles ALTER COLUMN solver_kind DROP NOT NULL;

CREATE TABLE {{schema}}.forward_model_profiles (
    name text NOT NULL, version integer NOT NULL CHECK (version > 0),
    definition jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (name, version)
);
CREATE TABLE {{schema}}.estimates (
    estimate_uuid uuid PRIMARY KEY,
    job_id uuid NOT NULL UNIQUE REFERENCES {{schema}}.jobs(id),
    forward_model_name text NOT NULL, forward_model_version integer NOT NULL,
    optimizer_name text NOT NULL, optimizer_version integer NOT NULL,
    configuration jsonb NOT NULL,
    spacecraft_id text,
    spacecraft_name text,
    prior_ephemeris_id uuid NOT NULL,
    prior jsonb,
    epoch timestamptz,
    software_version jsonb,
    published_run_id uuid REFERENCES {{schema}}.job_runs(id),
    created_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (forward_model_name, forward_model_version)
        REFERENCES {{schema}}.forward_model_profiles(name, version),
    FOREIGN KEY (optimizer_name, optimizer_version)
        REFERENCES {{schema}}.optimizer_profiles(name, version)
);
CREATE INDEX estimates_spacecraft_time_idx ON {{schema}}.estimates(spacecraft_id, created_at DESC);
CREATE TABLE {{schema}}.estimate_contacts (
    estimate_uuid uuid NOT NULL REFERENCES {{schema}}.estimates,
    contact_id uuid NOT NULL, ordinal integer NOT NULL CHECK (ordinal >= 0),
    provenance jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (estimate_uuid, contact_id), UNIQUE (estimate_uuid, ordinal)
);
CREATE INDEX estimate_contacts_contact_idx ON {{schema}}.estimate_contacts(contact_id);
CREATE TABLE {{schema}}.estimate_parameters (
    estimate_uuid uuid NOT NULL REFERENCES {{schema}}.estimates,
    parameter_name text NOT NULL, ordinal integer NOT NULL CHECK (ordinal >= 0),
    role text NOT NULL CHECK (role IN ('estimate','consider','fixed')),
    value double precision NOT NULL, unit text NOT NULL,
    initial_value double precision NOT NULL, lower_bound double precision NOT NULL,
    upper_bound double precision NOT NULL, scale double precision NOT NULL CHECK (scale > 0),
    standard_uncertainty double precision CHECK (standard_uncertainty >= 0),
    contact_id uuid,
    PRIMARY KEY (estimate_uuid, parameter_name), UNIQUE (estimate_uuid, ordinal),
    CHECK (lower_bound < upper_bound),
    FOREIGN KEY (estimate_uuid, contact_id) REFERENCES {{schema}}.estimate_contacts
);
CREATE TABLE {{schema}}.estimate_diagnostics (
    estimate_uuid uuid PRIMARY KEY REFERENCES {{schema}}.estimates,
    success boolean NOT NULL, optimizer_status integer NOT NULL, message text NOT NULL,
    objective double precision NOT NULL, optimality double precision NOT NULL,
    function_evaluations integer NOT NULL, jacobian_evaluations integer,
    observation_count integer NOT NULL,
    whitened_residual_rms double precision NOT NULL, residual_rms_hz double precision NOT NULL,
    covariance jsonb, covariance_rank integer,
    parameter_order jsonb NOT NULL, warnings jsonb NOT NULL DEFAULT '[]'::jsonb
);
CREATE TABLE {{schema}}.estimate_artifacts (
    artifact_uuid uuid PRIMARY KEY,
    estimate_uuid uuid NOT NULL REFERENCES {{schema}}.estimates,
    run_id uuid NOT NULL REFERENCES {{schema}}.job_runs(id),
    kind text NOT NULL, version integer NOT NULL DEFAULT 1 CHECK (version > 0),
    filename text NOT NULL, content_type text NOT NULL,
    payload bytea NOT NULL, sha256 text NOT NULL CHECK (sha256 ~ '^[a-f0-9]{64}$'),
    byte_count bigint GENERATED ALWAYS AS (octet_length(payload)) STORED,
    provenance jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (estimate_uuid, run_id, kind, version, filename)
);
CREATE INDEX estimate_artifacts_estimate_idx ON {{schema}}.estimate_artifacts(estimate_uuid);

CREATE VIEW {{schema}}.estimates_v1 AS
SELECT e.estimate_uuid, e.job_id, j.status, j.stage, j.cancel_requested,
       j.actor_id, j.label, j.attempt_count, j.created_at, j.updated_at, j.finished_at,
       e.spacecraft_id, e.spacecraft_name, e.prior_ephemeris_id, e.epoch,
       e.forward_model_name, e.forward_model_version, e.optimizer_name, e.optimizer_version,
       e.configuration, e.prior, e.software_version, e.published_run_id,
       (SELECT jsonb_agg(c.contact_id ORDER BY c.ordinal) FROM {{schema}}.estimate_contacts c
        WHERE c.estimate_uuid=e.estimate_uuid) AS contact_ids,
       j.terminal_error
FROM {{schema}}.estimates e JOIN {{schema}}.jobs j ON j.id=e.job_id;
CREATE VIEW {{schema}}.estimate_parameters_v1 AS SELECT * FROM {{schema}}.estimate_parameters;
CREATE VIEW {{schema}}.estimate_diagnostics_v1 AS SELECT * FROM {{schema}}.estimate_diagnostics;
CREATE VIEW {{schema}}.estimate_contacts_v1 AS SELECT * FROM {{schema}}.estimate_contacts;
CREATE VIEW {{schema}}.estimate_events_v1 AS
SELECT e.estimate_uuid, v.* FROM {{schema}}.estimates e
JOIN {{schema}}.job_events v ON v.job_id=e.job_id;
CREATE VIEW {{schema}}.estimate_artifacts_v1 AS
SELECT artifact_uuid, estimate_uuid, run_id, kind, version, filename, content_type,
       sha256, byte_count, provenance, created_at FROM {{schema}}.estimate_artifacts;
