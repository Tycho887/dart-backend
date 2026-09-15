-- Automatic priors are resolved by the worker before inputs are frozen.
ALTER TABLE {{schema}}.estimates ALTER COLUMN prior_ephemeris_id DROP NOT NULL;
ALTER TABLE {{schema}}.estimates ADD CONSTRAINT estimates_resolved_prior_id_check
    CHECK (prior IS NULL OR prior_ephemeris_id IS NOT NULL);
