# Migration provenance

The unified project is assembled in the workspace-root repository from two
legacy working trees.  They remain read-only and ignored until numerical and
behavioral parity has been established.

| Source | Recorded revision | Role retained |
| --- | --- | --- |
| `deprecated/dart-v1` | `572c172` plus local working changes | Doppler model, batch phase-shift fit, FOREST loader, BESTXYZ scoring |
| `deprecated/autofinder` | `9dee99a` plus local working changes | Henault world model, dither acquisition, live control behavior |

No nested `.git`, `.venv`, cache, generated plot, or bundled research PDF is
copied into the unified repository.  Full FOREST telemetry and GPS exports
remain external inputs.  The original trees should only be removed after the
root package passes both legacy-regression and new end-to-end validation.

## Intentional behavior changes

- Every estimator uses the same full Satkit frame/state transformation.
- The public offset sign follows the GPS-validated Dart convention
  (`SGP4(t + offset_s)`).  Backends with a lag-positive convention must set
  their sign conversion explicitly.
- Measurement-noise configuration is a standard deviation; covariance uses
  its square.
- Phase is optional and its innovation is circular.
- Antenna encoder commands are visibility/control metadata, not independent
  orbit observations.
