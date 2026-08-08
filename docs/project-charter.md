# DART project charter

## Purpose

DART provides reproducible post-pass passive-RF orbit estimation. Its first
production workflow acquires Doppler observations and metadata from ADX/KOGS,
normalizes and filters them, performs a batch fit, computes independent quality
metrics, stores the complete evidence chain in TimescaleDB, and presents the
result in Grafana.

The product is pre-v1. Correctness, explicit scientific meaning, and a small
maintainable implementation take priority over compatibility with the merged
prototype repositories.

## Initial production capability

The named `batch_od` pipeline is:

1. acquire ADX telemetry and KOGS station/orbit metadata;
2. apply recorded filters and produce a CCSDS TDM input artifact;
3. run the stateless batch solver;
4. compute residual, convergence, and optional reference-OEM metrics;
5. persist immutable inputs and solver outputs plus versioned derived metrics;
6. expose run state and results to the existing Grafana instance.

The source observation epochs, reference orbit, filters, hyperparameters,
software versions, and hashes must be sufficient to reproduce every result.

## Boundaries

The initial production scope is Doppler batch processing. The following remain
research or future modules until each receives its own contract, validation
evidence, and operational design:

- realtime UKF processing;
- antenna acquisition and closed-loop control;
- contact scheduling;
- interferometric phase-difference and SDR processing.

FOREST is a validation dataset, not a production API concept. Its accepted
reports and compact regression fixtures remain evidence for the batch method.

## Design principles

- Services may be implemented in different languages.
- Language-neutral wire contracts, not shared Python classes, define service
  boundaries.
- The orchestrator owns workflow state and persistence. Scientific services are
  stateless.
- Named workflows are implemented explicitly. DART does not provide a generic
  user-configurable DAG engine before one is demonstrably needed.
- CCSDS formats are used only where their semantics match the data. DART
  envelopes cover job control, provenance, diagnostics, and genuinely custom
  observables.
