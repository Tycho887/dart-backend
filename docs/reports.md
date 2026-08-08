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

## Historical reproduction status

The former research CLI is deliberately not installed or exposed by the
production wheel. The committed historical reports remain provenance, but
re-running them requires a separately reviewed research workspace; it is not a
supported production workflow.

Verify external inputs before reproduction:

```bash
sha256sum -c reports/reference/data_manifest.sha256
```
