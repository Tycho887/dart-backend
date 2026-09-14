# Six-parameter multipass solver tuning

`experiments.solver_tuning.run_solver_tuning(study, reference, *,
pass_counts=(5, 6, 8), trials=100, seed=42)` tunes SciPy settings on all six
archived matched FOREST anchors. It is also exported from
`experiments.trajectory_tuning`; that module's original spacecraft-holdout
workflow and CLI remain unchanged.

Restore the existing publication into an ignored, new workspace, then run:

```bash
uv sync --extra tuning
uv run python -m experiments.study_artifacts restore \
  reports/forest-studies/trajectories experiments/results/solver-tuning
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m experiments.solver_tuning \
  --study experiments/results/solver-tuning \
  --reference experiments/results/solver-tuning/forecast-reference
```

The runner reproduces the five/eight-pass baselines within 1 mm of the archived
scores and establishes the six-pass baseline before searching. Each study has
100 total trials, including the enqueued Soft-L1 / 200 Hz profile. Trials run
sequentially, with up to four concurrent anchor fits. Each fit estimates six
SGP4 orbit corrections and a constant Doppler bias for each contact. B*, time
offset, and center-frequency corrections stay fixed.

Contact groups, quality gates, unit observation variance, bounds, scales and
priors are frozen. Search dimensions are loss (`linear`, `soft_l1`, `huber`,
`cauchy`, `arctan`), logarithmic `f_scale` (10–10,000 Hz, nonlinear losses only),
independent logarithmic `ftol`/`xtol`/`gtol` (1e-12–1e-4), and `x_scale`
(`profile` or `jac`). `OptimizerContext.loss_scale` is SciPy's `f_scale`;
`max_evaluations` is `max_nfev`. TRF, the dense exact trust-region solver,
1,000 evaluations and the Rust Jacobian stay fixed. The existing deterministic
phase scan selects its seed using each trial's Doppler loss. Neither fitting
nor phase initialization accepts GPS data.

The objective is the arithmetic mean of six per-anchor 3D position RMS values,
each measured against actual reference samples over the full reservation at
zero time offset. Every required fit must converge and every local window must
score completely; otherwise Optuna receives infinity and JSON records the
failure with a null objective. Forecast scores cover 48 hours from reservation
stop. Forecast failures and regressions remain explicit and do not change local
ranking. Equal objective values select the earliest trial number.

The winning profile is shared by every anchor for a given pass count. Each
winner is refitted in a fresh cache directory. Confirmation checks parameter
vectors, local scores and exact orbit-descriptor bytes. Winners are also
applied to the original 38-anchor cohort, retaining insufficient-history and
screening failures in denominators. This is additional dataset evaluation, not
a holdout or a generalization claim. Candidate reference annotations are
preserved, including unverified accuracy in raw GPS gaps.

## Resuming and artifacts

Re-run the same command to resume **toward** 100 total trials, not add 100 more.
`--trials` can increase the total target. SQLite stores trials and prepared
settings; numeric TPE and random-sampler RNG states preserve the seeded sequence
across process restarts. The Optuna version is pinned in the immutable manifest
because its RNG accessors are private. An interrupted prepared trial resumes its
original settings; interruption before preparation is committed is recorded as
a failed trial and still counts toward the target. A process lock prevents two
runners from writing the same study concurrently.

The runner rejects changed input bytes, references, groups, search configuration,
source hashes, package versions, or numerical runtime. Output lives under
`STUDY/solver-tuning/`. Each count has a SQLite database and per-trial JSON, a
baseline, winner confirmation, and cohort result. JSON preserves parameters,
termination reasons, evaluation counts, runtime and failures. Fit directories
retain exact orbit descriptors, profiles and seeds. Full residual/Jacobian
outputs for baselines and winners are copied to `diagnostics/`; independent
refits retain theirs separately. `README.md` and `per-anchor.csv` compare results.

Publish a completed workspace with `experiments.study_artifacts export`, sharing
exact original inputs with the existing sibling publications. Exclude repeated
trial residual/Jacobian outputs and measurement copies only after preserving
baseline/winner diagnostics. Retain SQLite, trial JSON, profiles, fitted vectors,
orbit descriptors, failures, source snapshots and manifests. Verify checksums
and restore into a new directory before treating the publication as complete.
