# GPS-measured accuracy of a multipass run

The next study should test whether a selected set of Doppler contacts produces
one accurate orbit at the evaluation epoch and throughout a 48-hour forecast.
Report measured errors against an explicit GPS reference. A covariance trace,
FIM condition number, low Doppler residual, or successful optimizer termination
does not establish an error bound. Statistical calibration is deferred.

## Evidence and priorities

On the six archived anchors with eight usable historical contacts, the
six-parameter model produced these zero-offset results:

| Selection | Local median RMS, km | 48-hour median RMS, km | Local <5 km |
|---|---:|---:|---:|
| Single | 237.908 | 2308.277 | 0/6 |
| Latest 3 | 106.172 | 1720.258 | 0/6 |
| Latest 5 | 4.120 | 22.930 | 3/6 |
| Latest 8 | 3.680 | 27.908 | 5/6 |
| Individual condition ranking | 6.363 | 25.405 | 2/6 |
| Individual trace ranking | 5.243 | 28.630 | 3/6 |

The single-pass medians exclude one failed fit; success denominators include it.
These six anchors overlap in history and are not six independent replications.
Across the original 38 anchors, the best fixed configuration succeeded on 8/38,
below the 19/38 target. Larger histories improve local accuracy, but additional
passes did not monotonically improve the forecast. Selection by isolated-pass
information also changed the number of retained contacts, confounding that
comparison. These observations motivate the ordered experiments below.

### 1. Complementary information at equal contact counts

Use a common pinned prior and one Doppler-only L pilot fitted to the candidate
inventory. At that same pilot orbit and parameterization, evaluate all six orbit
sensitivities for each contact. Apply the configured residual weights and project
out that contact's constant frequency-bias column. Sum the resulting orbit
information matrices for each proposed set. Individual rank-deficient contacts
may complement each other and must not be rejected merely for low individual rank.

Compare latest-contact selection with greedy joint log-determinant selection at
exactly 3, 5, and 8 contacts. Both use the same quality-screened candidate inventory,
common pilot, observation gates, final six-parameter profile, and anchor. Require
the anchor in each selected set. Freeze the pilot during selection to isolate the
effect of set composition; fit the final shared orbit after selection.

For a concrete first experiment, take the latest 16 usable contacts ending by the
anchor, limited to 72 hours. Use existing orbit parameter scales and maximize
`logdet(I_selected + lambda * identity)`, with
`lambda = 1e-6 * largest_eigenvalue(I_all_candidates)`. This ridge makes selection
well-defined before full rank; it is a numerical selection regularizer, not a
physical prior covariance. Fail if all candidate information is zero or invalid.
Break equal gains by recency, then contact UUID. Use maintained linear algebra in
the Rust numerical core. Keep the existing individual trace/condition selectors
and single-pass fits as controls, identifying their actual contact counts.

### 2. Contact count versus observation span

Repeat equal-count comparisons with maximum ages of 12, 24, 48, and 72 hours.
Report actual span, contact count, antenna diversity, and ascending/descending
geometry. Freeze matched anchors before fitting so short histories cannot improve
the score by disappearing. Also report operational coverage over the wider cohort.
This separates complementary geometry from stale-data/model-mismatch effects.

### 3. Measurement weighting and residual structure

Plot signed residuals within each contact and by antenna. Compare the existing
sample-weighted Soft-L1 loss with equal aggregate weight per contact, and tune its
transition using development data only. Check whether long or densely sampled
contacts dominate, and whether multi-kHz residual excursions correspond to known
telemetry fields or tracking transitions. Retain raw observations and explicit
selection reasons. Do not identify outliers using GPS orbit errors.

### 4. Forecast dynamics

Inspect radial/along-track/cross-track growth at 6, 12, 24, and 48 hours, alongside
mean-motion sensitivity and fitted parameter bounds. After the preceding controls,
compare fixed B* with a bounded B* fit only on spans that constrain it. A longer
span may expose SGP4 or prior-model errors rather than provide useful information.
Compare against the uncorrected prior; reserve force-model/full-state changes for
a separate explicit experiment. Subsecond post-fit timing alignment remains a
diagnostic: the completed sweep did not explain most of the cross-track error.

## Run and evaluation contract

A run has one spacecraft UUID, explicit candidate and selected contact UUIDs,
one pinned prior/ephemeris ID, a resolved optimizer profile, and a fixed anchor.
The fit estimates one shared orbit and one constant Doppler bias per selected
contact. Save fit status and reason even if it cannot produce a usable orbit.
Selection and fitting receive no GPS reference or GPS-derived scores.

The comparison epoch is the anchor reservation end, shared by all strategies.
Local accuracy uses actual reference samples over that anchor's full reservation;
forecast accuracy uses actual reference samples over the following 48 hours.
Record requested interval, actual first/last sample epochs, sample count, frame,
units, and reference checksum/quality status. Do not silently interpolate an exact
epoch state from absent samples; report that point score as unavailable when there
is no exact sample. A nearby sample's score must carry its actual epoch.

Report position RMS in metres, velocity RMS in metres/second, RTN position RMS,
and sampled maximum position error for each window. The maximum describes sampled
epochs, not a continuous-time bound. Keep local and forecast scores separate.
Do not add velocity or maximum-error acceptance thresholds before an operational
requirement defines them. Existing position success remains strictly `<5000 m`.

Missing coverage, failed quality screening, insufficient history, selection
failure, fit nonconvergence, and propagation failure are explicit outcomes.
Never substitute a shorter forecast. Preserve the historical 38-anchor comparison
and publish a new, separately frozen cohort for an expanded inventory. Report
failure/coverage counts alongside medians of scored runs. Never choose the winning
model separately for each run using GPS.

GPS here is a reference estimate, not exact truth. FOREST-19's local reference is a
candidate; extended FOREST-16/18/19 references also remain candidates. Preserve
their assessments and the unverified accuracy inside raw GPS gaps in every report.

## Proposed function skeleton (future implementation)

Build on the existing types and keep acquisition outside the solver:

```python
select_contact_set(
    contacts: Sequence[ContactMetadata],
    data: PriorStateData,
    anchor: ContactMetadata,
    pilot: Sgp4Orbit,
    optimizer: OptimizerContext,
    *, count: int, max_age: timedelta,
    method: Literal["latest", "joint_logdet"],
) -> ContactGroup

# Existing APIs after selecting/subsetting normalized observations:
fit(data: PriorStateData, optimizer: OptimizerContext) -> OptimizerOutput
resolve_solution(data: PriorStateData, output: OptimizerOutput) -> OrbitSolution
evaluate_trajectory(orbit, reference_segments, start, stop, offsets_s=(0.0,))
```

The selector returns IDs, status, and reasons using `ContactGroup`; it does not
create an external-data client or a second optimizer. The evaluation extension
adds sampled maximum error and requested/sample epoch metadata to an evaluation
record, separate from `OptimizerOutput`. No service or MessagePack schema change
is part of this cleanup. Future service integration must reuse durable job/run/
artifact primitives and require an explicit reference for GPS evaluation.

## Larger inventory, tuning, and acceptance

Acquire a larger causal contact inventory through existing clients, with reference
coverage extending 48 hours beyond each anchor. Use chronological development and
held-out blocks per spacecraft. Leave at least 120 hours between development and
held-out anchors for the maximum 72-hour lookback plus 48-hour forecast, and verify
that neither contacts nor evaluation intervals overlap across the split.

Make contact count, maximum age, selector, contact weighting, and robust-loss scale
explicit configuration values. Reuse the existing Optuna runner and independent
local/48-hour objectives. Freeze inventories, split, denominator, and failure policy
before tuning; stricter filters cannot drop difficult anchors from objectives.
Select configurations on development data and evaluate held-out blocks once.
Report the archived pilot separately from this new study. Do not claim calibrated
probabilities or uncertainty bounds from the small, overlapping archived cohort.

Before implementing the new selector, test complementary individually deficient
passes, nuisance-bias invariance, common-epoch/scaling consistency, deterministic
ties, causal selection, exact retained counts, and zero-information failures.
Acceptance requires an equal-count comparison with unchanged gates/profiles,
explicit failure denominators, and a measured local/forecast tradeoff on held-out
data. Improvements in FIM alone do not satisfy the accuracy objective.
