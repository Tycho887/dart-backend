# Reference and traceability artifacts

| Artifact | Purpose |
| --- | --- |
| `data_manifest.sha256` | Checksums for the external FOREST parquet and raw BESTXYZ inputs. |
| `observation_inventory.{json,csv}` | All raw contacts, filtering counts, eligibility, and GPS coverage. |
| `forest_replay_audit.{json,csv,md}` | Combined batch-plus-UKF replay export retained so scoped reports can be traced to the same pass set. |

The audit export is intentionally mixed and is not a headline-performance
report. Use `../production/doppler_batch_ls.*` or
`../experimental/ukf/ukf_replay.*` instead.
