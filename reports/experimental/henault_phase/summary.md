# Henault-style phase-difference experiments: preliminary summary

**Status:** experimental and predominantly synthetic. This summary contains no
production Doppler-only batch-LS conclusion and no real phase-performance
claim.

## Geometry-matched simulation

With one paired seed on each of 15 recorded-window geometries, white-noise
Doppler closure gives median trial errors of 1.051 km for batch and 1.233 km
for the static UKF, with no failures. Independent numerical dynamics over the
same short windows gives nearly identical values. That agreement confirms only
that the two synthetic truth tiers remain close on these short arcs; it does
not reproduce a real error distribution.

Under complete, perfectly calibrated synthetic phase, median closure error
falls to 0.008 km for staged batch and 0.010 km for the static UKF. Three of
15 phase-enabled batch trials still degrade because phase ambiguity is not
fully eliminated; the worst phase-enabled batch trial is 5.459 km. These are
idealized upper-bound-style information results, not field expectations.

## Empirical-residual experiment

F16/F17 Doppler residual blocks were injected into F18/F19 window geometry
using two paired seeds per window. The rough residual model produces the
following synthetic trial outcomes:

| Channels | Estimator | Trials | Median error (km) | Worst (km) | Health failures |
| --- | --- | ---: | ---: | ---: | ---: |
| Doppler | Batch | 18 | 3.918 | 21.516 | 0 |
| Doppler | Static UKF | 18 | 4.934 | 285.652 | 7 |
| Doppler + complete phase | Batch | 18 | 5.130 | 19.973 | 0 |
| Doppler + complete phase | Static UKF | 18 | 12.980 | 285.653 | 8 |

Complete phase does not repair a filter that fails to localize enough to use
it, and the phase-enabled batch can enter a wrapped alias when its Doppler
initializer is poor. The residual bootstrap is an exploratory stress test; it
does not establish a calibrated simulator or phase benefit.

## Synthetic state-model ablation

| Truth family | Channels | Time-offset model median (km) | Mean-anomaly/motion model median (km) |
| --- | --- | ---: | ---: |
| Pure time offset | Doppler | 0.128 | 0.165 |
| Pure time offset | Doppler + phase | 0.004 | 0.231 |
| Mean anomaly and mean motion error | Doppler | 20.298 | 0.109 |
| Mean anomaly and mean motion error | Doppler + phase | 21.194 | 0.176 |

Both models fit two synthetic passes and are scored on a third. Complete phase
helps the correctly specified scalar-offset model, but does not rescue a
misspecified scalar model and slightly degrades the mean-element result in
this small trial. The result demonstrates model matching in simulation, not
real-data superiority.

## Next evidence needed

The justified next step is calibrated, independently held-out dual-antenna
phase data with cycle-slip and availability characterization. Expanding the
idealized phase or mean-element study before that would add simulation detail
without adding operational validation.
