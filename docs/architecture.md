# DART architecture and scientific boundaries

## Production data flow

```text
Grafana ──server-side proxy──> Orchestrator API
                                   │
                         durable batch_od run
                                   │
                 ┌─────────────────┼─────────────────┐
                 ▼                 ▼                 ▼
           ADX/KOGS proxy     Batch solver      Postprocessor
                 │                 │                 │
                 └──── TDM ────────┴── result ───────┘
                                   │
                                   ▼
                 TimescaleDB inputs/results/metrics
```

The orchestrator is the only database writer. The API proxy, solver, and
postprocessor are stateless HTTP services and can be replaced by
contract-compatible implementations in other languages.

CCSDS TDM is the reproducible observation artifact, OEM is the optional
reference or propagated state artifact, and OMM is used for fitted mean
elements when lossless. DART JSON envelopes cover run control, hashes,
station coordinates, hyperparameters, covariance, residuals, diagnostics, and
derived metrics.

## Scientific invariants

- Observation time is the UTC measurement epoch, never packet arrival time.
- Positions and velocities identify their frame and SI units.
- Doppler input is a carrier-frequency offset in hertz. It becomes TDM
  `RECEIVE_FREQ` only after adding a known nominal carrier and converting to
  the CCSDS unit.
- The source TLE or OMM and acquired observations are immutable.
- Parameter and covariance order are explicit.
- Reference OEM truth is available only to postprocessing and cannot influence
  a solver result.
- Station geometry is resolved and stored with the run so later KOGS changes
  cannot silently alter reproduction.

## Validation meaning

The production report family covers real FOREST Doppler batch fits scored
against receiver-epoch BESTXYZ positions. It demonstrates retrospective
post-pass behavior for the documented cohort, not guaranteed accuracy or live
beam retention.

UKF replay, closed-loop acquisition, interferometric phase, and mean-element
ablation reports remain research evidence. The explicit Doppler-only
two-element mean-anomaly/mean-motion solver is production code, but these
ablation reports are not its operational validation. The research modules
cannot be cited as production features and are not imported by production
service entry points.

## Operational risks

- Time offset and frequency bias may be weakly observable on short or one-sided
  arcs.
- A scalar offset cannot repair altitude, plane, drag, or cross-track errors.
- Correlated transmitter and orbit-model errors invalidate white-noise
  covariance assumptions.
- ADX/KOGS authentication or semantic changes must fail the live integration
  gate; tests may not silently use recorded substitutes.
- Pipeline retries must not duplicate solver outputs or metric sets.
- Dashboard changes must be inspected and applied to the live Grafana instance
  through its API, with a saved export used only as a synchronized snapshot.
