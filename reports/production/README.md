# Production evidence: real Doppler-only post-pass batch LS

This area contains the one report family that may be used for the current
operationally relevant accuracy statement: recorded FOREST passive Doppler,
fit after the pass with the robust batch least-squares estimator, then scored
against independent raw BESTXYZ positions at their receiver measurement
epochs.

It deliberately excludes the UKF, interferometric phase difference,
closed-loop control, and synthetic truth. Those belong under
[`../experimental/`](../experimental/).

Read [protocol.md](protocol.md) before citing
[doppler_batch_ls.md](doppler_batch_ls.md). The JSON and CSV beside that
Markdown report are the scoped machine-readable batch-LS bundle.

For a technical-review narrative covering the complete May 2026 LEOP contact
inventory, exact TLE priors, per-pass measurement counts, error computation,
and leakage controls, read [forest_leop_may_2026.md](forest_leop_may_2026.md).
Its companion [61-contact CSV](forest_leop_may_2026.csv) preserves the complete
raw-to-result audit trail.

An illustrated [Word edition](forest_leop_may_2026.docx) embeds the original
eligible-pass timeline, an [all-contact UTC status timeline](contact_timeline_filter_status.png),
and a report-local [burst-radio residual diagnostic](burst_radio_residual_clusters.png)
from the archived GMM/DBSCAN study. The Word edition keeps one compact
four-column fitted-pass table; the complete contact, fit, and audit tables
remain in the Markdown and CSV. In the UTC timeline, green segments passed the
`>=301` presented-Doppler rule and red segments did not. Regenerate the Word
edition and generated figure assets with:

```bash
uv run --extra plot python scripts/generate_forest_leop_docx.py
```
