# Report map

This directory is organized by the evidence behind a result, rather than by
the implementation that produced it. A result can use real Doppler and still
be experimental; conversely, a familiar estimator name does not make a
simulation operational evidence.

| Area | What belongs there | What it can support |
| --- | --- | --- |
| [`production/`](production/) | Recorded FOREST Doppler evaluated against raw receiver-epoch BESTXYZ with the post-pass robust batch LS fit. | The current Doppler-only post-pass accuracy evidence. |
| [`../research/reports/experimental/`](../research/reports/experimental/) | Static UKF replay, phase-difference work, closed-loop trials, and synthetic studies. | Algorithm and measurement-model research only. |
| [`reference/`](reference/) | Input manifest, contact inventory, and a combined audit export. | Traceability and reproduction, not headline performance. |
| [`../research/reports/archive/`](../research/reports/archive/) | Superseded research artifacts retained for provenance. | Historical comparison only. |

“Production” here means the operationally relevant real-data experiment, not
that the software or estimator is flight-qualified. All current FOREST work is
retrospective and exploratory; it is not a blinded confirmation.

## Where to start

1. For the real Doppler-only batch-LS result, read the
   [production protocol](production/protocol.md), then the generated
   [batch-LS evaluation](production/doppler_batch_ls.md).
2. For the experimental UKF on those same recorded Doppler passes, read
   [experimental UKF](../research/reports/experimental/ukf/).
3. For Henault-style interferometric phase difference, read
   [phase research](../research/reports/experimental/henault_phase/). No result in
   that branch is real dual-antenna phase validation.

Do not combine tables across these areas without restating their evidence
class, cohort, and causal/post-pass status.
