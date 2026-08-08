# Pre-v1 roadmap

## Completed structural work

- Three production services and four runtime roles are isolated under
  `dart.services`.
- Language-neutral v0.1 contracts generate one shared Python projection.
- Production propagation is centralized; research code and dependencies are
  quarantined in a separate project.
- Legacy runtimes and the local general-purpose CLI are removed after archival.
- Deployment assets, migrations, Grafana snapshots, and maintenance tools are
  grouped under `deploy/` and `tools/`.

## Remaining release gates

- Run the real ADX/KOGS tests with all required credentials.
- Run persistence and interruption scenarios against an isolated TimescaleDB.
- Verify the configured ingress and a bounded end-to-end Grafana workflow.
- Re-run Ruff, ty, contracts, offline tests, wheel inspection, research tests,
  and both production image boundary probes for every release candidate.

Future tracker, scheduler, UKF, phase, or control capabilities require their
own language-neutral contracts and production evidence before promotion from
research.
