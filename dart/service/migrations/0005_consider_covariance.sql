ALTER TABLE {{schema}}.estimate_diagnostics
    ADD COLUMN covariance_method text;

CREATE OR REPLACE VIEW {{schema}}.estimate_diagnostics_v1 AS
SELECT * FROM {{schema}}.estimate_diagnostics;
