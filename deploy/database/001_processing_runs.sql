-- Canonical pre-v1 durable-run baseline. Recreate disposable databases from
-- this file; deployed wheels receive this same resource as dart.services.orchestrator/schema.sql.

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id UUID PRIMARY KEY,
    pipeline_name TEXT NOT NULL CHECK (pipeline_name = 'batch_od'),
    idempotency_key TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status IN (
        'queued', 'acquiring', 'optimizing', 'postprocessing',
        'succeeded', 'partial', 'failed'
    )),
    request_sha256 TEXT NOT NULL,
    request_json JSONB NOT NULL,
    selection_criterion TEXT NOT NULL CHECK (selection_criterion IN ('bic', 'aic', 'aicc')),
    errors_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    lease_token UUID,
    lease_until TIMESTAMPTZ,
    selected_candidate_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS pipeline_runs_queue_idx
    ON pipeline_runs (status, next_attempt_at, created_at);

CREATE TABLE IF NOT EXISTS run_inputs (
    run_id UUID PRIMARY KEY REFERENCES pipeline_runs(run_id),
    contract_version TEXT NOT NULL,
    source_orbit_json JSONB NOT NULL,
    solver_config_json JSONB NOT NULL,
    tdm_sha256 TEXT,
    tdm_kvn TEXT,
    dataset_sha256 TEXT,
    dataset_json JSONB,
    provenance_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK ((dataset_sha256 IS NULL) = (dataset_json IS NULL))
);

CREATE TABLE IF NOT EXISTS run_candidates (
    candidate_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES pipeline_runs(run_id),
    candidate_index INTEGER NOT NULL CHECK (candidate_index >= 0),
    model TEXT NOT NULL,
    metaparameters_json JSONB NOT NULL,
    status TEXT NOT NULL CHECK (status IN (
        'pending', 'optimizing', 'postprocessing', 'succeeded',
        'unhealthy', 'ineligible', 'failed'
    )),
    error_json JSONB,
    selected_metric_set_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, candidate_index),
    UNIQUE (run_id, candidate_id)
);
CREATE INDEX IF NOT EXISTS run_candidates_run_idx
    ON run_candidates (run_id, candidate_index);

CREATE TABLE IF NOT EXISTS solver_results (
    result_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES pipeline_runs(run_id),
    candidate_id UUID NOT NULL UNIQUE,
    solver_version TEXT NOT NULL,
    contract_version TEXT NOT NULL,
    result_sha256 TEXT NOT NULL,
    result_json JSONB NOT NULL,
    parameters_json JSONB NOT NULL,
    covariance_json JSONB NOT NULL,
    diagnostics_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, candidate_id),
    FOREIGN KEY (run_id, candidate_id)
        REFERENCES run_candidates (run_id, candidate_id)
);
CREATE INDEX IF NOT EXISTS solver_results_run_idx ON solver_results (run_id);

CREATE TABLE IF NOT EXISTS solver_residuals (
    timestamp TIMESTAMPTZ NOT NULL,
    run_id UUID NOT NULL REFERENCES pipeline_runs(run_id),
    candidate_id UUID NOT NULL,
    measurement_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    consumed BOOLEAN NOT NULL,
    predicted DOUBLE PRECISION,
    residual DOUBLE PRECISION,
    robust_weight DOUBLE PRECISION,
    PRIMARY KEY (candidate_id, measurement_id, channel, timestamp),
    FOREIGN KEY (run_id, candidate_id)
        REFERENCES run_candidates (run_id, candidate_id)
);
SELECT create_hypertable('solver_residuals', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS solver_residuals_run_idx
    ON solver_residuals (run_id, candidate_id, timestamp);

CREATE TABLE IF NOT EXISTS metric_sets (
    metric_set_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES pipeline_runs(run_id),
    candidate_id UUID NOT NULL,
    postprocessor_version TEXT NOT NULL,
    contract_version TEXT NOT NULL,
    solver_result_sha256 TEXT NOT NULL,
    quality_sha256 TEXT NOT NULL,
    result_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (candidate_id, postprocessor_version),
    UNIQUE (metric_set_id, candidate_id),
    FOREIGN KEY (run_id, candidate_id)
        REFERENCES run_candidates (run_id, candidate_id)
);
CREATE INDEX IF NOT EXISTS metric_sets_run_idx ON metric_sets (run_id, candidate_id);

CREATE TABLE IF NOT EXISTS metric_values (
    metric_set_id UUID NOT NULL REFERENCES metric_sets(metric_set_id),
    metric_group TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value JSONB NOT NULL,
    unit TEXT,
    PRIMARY KEY (metric_set_id, metric_group, metric_name)
);

CREATE TABLE IF NOT EXISTS pipeline_stage_events (
    timestamp TIMESTAMPTZ NOT NULL,
    run_id UUID NOT NULL REFERENCES pipeline_runs(run_id),
    candidate_id UUID,
    attempt INTEGER NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    detail_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    FOREIGN KEY (run_id, candidate_id)
        REFERENCES run_candidates (run_id, candidate_id)
);
SELECT create_hypertable('pipeline_stage_events', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS pipeline_stage_events_run_idx
    ON pipeline_stage_events (run_id, timestamp DESC);

DO $$
BEGIN
    ALTER TABLE pipeline_runs
        ADD CONSTRAINT pipeline_runs_selected_candidate_fk
        FOREIGN KEY (run_id, selected_candidate_id)
        REFERENCES run_candidates (run_id, candidate_id);
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

DO $$
BEGIN
    ALTER TABLE run_candidates
        ADD CONSTRAINT run_candidates_selected_metric_fk
        FOREIGN KEY (selected_metric_set_id, candidate_id)
        REFERENCES metric_sets (metric_set_id, candidate_id);
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

CREATE OR REPLACE VIEW dart_run_results AS
SELECT runs.run_id,
       runs.pipeline_name,
       runs.status,
       runs.selected_candidate_id,
       runs.selection_criterion,
       runs.created_at,
       runs.updated_at,
       runs.attempts,
       selected_result.result_json AS selected_result_json,
       selected_metrics.result_json AS selected_metrics_json,
       runs.errors_json
FROM pipeline_runs AS runs
LEFT JOIN run_candidates AS selected_candidate
    ON (selected_candidate.run_id, selected_candidate.candidate_id) =
       (runs.run_id, runs.selected_candidate_id)
LEFT JOIN solver_results AS selected_result
    ON (selected_result.run_id, selected_result.candidate_id) =
       (runs.run_id, runs.selected_candidate_id)
LEFT JOIN metric_sets AS selected_metrics
    ON (selected_metrics.metric_set_id, selected_metrics.candidate_id) =
       (selected_candidate.selected_metric_set_id, selected_candidate.candidate_id);
