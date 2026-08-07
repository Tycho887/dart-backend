# Reports and analysis guide

## Evidence-first catalog

The report tree is intentionally divided by evidence and readiness status.
“Production” means the operationally relevant real-data experiment; it does
not mean the code is flight-qualified or that the retrospective cohort is a
blinded confirmation.

| Report family | Data and estimator | Permitted conclusion |
| --- | --- | --- |
| [`../reports/production/`](../reports/production/) | Recorded FOREST Doppler + raw BESTXYZ; full-pass robust batch LS. | Current Doppler-only post-pass accuracy evidence. |
| [`../reports/experimental/ukf/`](../reports/experimental/ukf/) | Recorded FOREST Doppler + raw BESTXYZ; static UKF. | Experimental filter behavior only. |
| [`../reports/experimental/henault_phase/`](../reports/experimental/henault_phase/) | Synthetic phase-difference channel, with recorded geometry/residuals in some trials. | Henault-style model and algorithm research only. |
| [`../reports/reference/`](../reports/reference/) | Manifests, contact inventory, and mixed audit data. | Traceability and reproduction. |
| [`../reports/archive/`](../reports/archive/) | Earlier mixed artifacts. | Provenance only. |

## Recommended reading order

For the actual Doppler-only post-pass result:

1. [`../reports/production/protocol.md`](../reports/production/protocol.md)
2. [`../reports/production/doppler_batch_ls.md`](../reports/production/doppler_batch_ls.md)

For experimental work, read the relevant branch independently:

1. [`../reports/experimental/ukf/ukf_replay.md`](../reports/experimental/ukf/ukf_replay.md)
   for the real-Doppler but experimental sequential filter.
2. [`../reports/experimental/henault_phase/README.md`](../reports/experimental/henault_phase/README.md),
   then its protocol and experiment-specific report, for phase-difference or
   synthetic work.

## Canonical artifacts

JSON is the canonical machine-readable output and CSV is a flat per-pass view.
Markdown is a derived review view. The scoped JSON bundles intentionally omit
the other estimator family, while
[`forest_replay_audit.*`](../reports/reference/forest_replay_audit.json)
retains the combined replay for traceability.

| Path | Contents |
| --- | --- |
| `reports/production/doppler_batch_ls.*` | Real Doppler-only post-pass batch-LS scope; no UKF or phase fields. |
| `reports/experimental/ukf/ukf_replay.*` | Experimental UKF-only scope; no batch comparison fields. |
| `reports/experimental/henault_phase/window_simulation.*` | Closure and independent-dynamics paired phase simulations. |
| `reports/experimental/henault_phase/empirical_window_simulation.*` | Residual-block experiment with synthetic phase. |
| `reports/experimental/henault_phase/model_ablation.*` | Fully synthetic held-out state-model study. |
| `reports/reference/observation_inventory.*` | Contact filtering, eligibility, and GPS-coverage inventory. |

## Interpretation rules

- Real data is not synonymous with operational status: the UKF replay is
  experimental even though its Doppler and GPS inputs are real.
- “Post-pass batch LS” is not synonymous with real-time tracking. It uses the
  whole contact and its same-pass score is a backcast.
- No real phase performance is measured. All complete-phase values use a
  synthetic, perfectly available phase channel with a hypothetical exact
  baseline.
- Closure simulation is an implementation check, not independent validation.
- Historical replay cannot validate counterfactual acquisition or beam
  retention.
- Pass-level summaries are primary; high-rate samples and overlapping GPS
  forecast epochs are not independent trials.

## Reproduction map

```bash
# Contact inventory / input traceability
uv run dart inventory-forest \
  --data-dir deprecated/dart-v1/data \
  --raw-gps-dir deprecated/dart-v1/data/Ororatech-HFS-GNSS-data-raw \
  --output reports/reference/observation_inventory.json

# One replay, written as an audit bundle plus two evidence-scoped reports
uv run dart replay-forest \
  --data-dir deprecated/dart-v1/data \
  --raw-gps-dir deprecated/dart-v1/data/Ororatech-HFS-GNSS-data-raw \
  --satellites 16 17 18 19 \
  --output reports/reference/forest_replay_audit.json \
  --batch-report-output reports/production/doppler_batch_ls.json \
  --ukf-report-output reports/experimental/ukf/ukf_replay.json

# Henault-style paired Doppler / synthetic-phase trials
uv run dart simulate-windows \
  --data-dir deprecated/dart-v1/data \
  --satellites 16 17 18 19 \
  --tiers closure independent_dynamics \
  --seeds 0 \
  --output reports/experimental/henault_phase/window_simulation.json

# Residual-block stress test: real Doppler residuals, synthetic phase
uv run dart simulate-windows \
  --data-dir deprecated/dart-v1/data \
  --satellites 18 19 \
  --calibration-satellites 16 17 \
  --tiers empirical_residual \
  --seeds 0 1 \
  --output reports/experimental/henault_phase/empirical_window_simulation.json

# Fully synthetic multi-pass state-model experiment
uv run dart model-ablation \
  --seeds 0 1 2 \
  --output reports/experimental/henault_phase/model_ablation.json
```

Verify external inputs before reproduction:

```bash
sha256sum -c reports/reference/data_manifest.sha256
```
