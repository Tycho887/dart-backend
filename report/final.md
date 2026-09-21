# FOREST results: next-hour position RMS after fitting

**The 12-hour and 24-hour limits select training data. Every error below is
scored over the hour immediately after the selected fit/prefix completes.**
These are not errors over the training arc, nor forecasts deliberately aged
to the end of the 12-hour or 24-hour collection limit.

This is the authoritative metric and comparison for
[results.md](../raw_results/results.md). Earlier deadline-aged comparisons and
KPT revisions based on them are withdrawn from this report.

## Median and observed worst-case errors

Each row summarizes four spacecraft-level scores with equal spacecraft weight.
A full-set row uses all eligible passes available at the selected prefix.
A worst-omission row first takes the largest RMS among all N fits that each
omit one of those N passes, then summarizes those four spacecraft maxima.

| Starting condition | Fit population | Median: first 12 h of data | Maximum: first 12 h of data | Median: first 24 h of data | Maximum: first 24 h of data |
| --- | --- | ---: | ---: | ---: | ---: |
| Informed prior, L+n | Full set of available passes | **3.39 km** | **10.57 km** | **1.16 km** | **5.16 km** |
| Informed prior, L+n | Worst N−1 omission per spacecraft | **7.12 km** | **11.59 km** | **3.03 km** | **6.10 km** |
| Poor prior, Cartesian no drag | Full set of available passes | **8.26 km** | **17.15 km** | **4.25 km** | **7.85 km** |
| Poor prior, Cartesian no drag | Worst N−1 omission per spacecraft | Not available for all four | Not available for all four | **13.52 km** | **32.63 km** |

The maxima are observed errors, not guaranteed bounds or population p95
estimates. No fold or spacecraft is removed because its error is large.
The poor-prior 12-hour full fits come from v5; v6 supplies the all-pass 24-hour
comparison and its omission fits. There is no complete poor-prior 12-hour
omission experiment for all four spacecraft, so no maximum is inferred for it.

## Exact selection and scoring rule

1. Start the collection clock at completion of the first eligible pass under
   the experiment's telemetry selection. It is not time since separation.
2. For collection limit $T=12$ or $24$ hours, select the **latest cumulative
   prefix completed by $t_{0,i}+T$**, independently of its error. Use every
   eligible pass in that prefix; do not select the best earlier result.
3. Let $c_i(T)$ be that prefix's completion time. Score the full fit and every
   omission fit on the **same** reference samples in $(c_i(T),c_i(T)+1\,\mathrm{h}]$.
   Keep that window fixed when the omitted pass is the first or last one.

The three-dimensional position RMS error is

$$
E_i(T)=\sqrt{\frac{1}{M_i}\sum_{j=1}^{M_i}
\left\|\hat{\mathbf r}_i(t_j)-\mathbf r_{i,\mathrm{ref}}(t_j)\right\|_2^2},
\qquad t_j\in(c_i(T),\;c_i(T)+1\,\mathrm{h}].
$$

The collection limit $t_{0,i}+T$ does **not** replace $c_i(T)$ in this expression.
For example, FOREST-16's informed-prior 12-hour selection completes at 11.08 h
and is scored over approximately 11.08–12.08 h. Its 24-hour selection completes
at 16.06 h and is scored over approximately 16.06–17.06 h.

Fit availability is approximated by prefix completion, assuming zero ingestion
and computation latency. The archive ends before 24 h for every spacecraft;
“first 24 h of data” means all eligible observations recorded within that limit,
not 24 hours of continuous observations. All selected scores have complete OEM
coverage with 60 reference samples per fit.

## Informed prior: v7 cumulative L+n

| Spacecraft | Passes: 12 h / 24 h | Fit completed at: 12 h / 24 h selection | Full-set RMS: 12 h | Full-set RMS: 24 h | Worst omission RMS: 12 h | Worst omission RMS: 24 h |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| FOREST-16 | 6 / 10 | 11.08 / 16.06 h | 10.57 km | 0.39 km | 11.16 km | 0.77 km |
| FOREST-17 | 7 / 10 | 11.45 / 16.23 h | 2.48 km | 5.16 km | 3.08 km | 6.10 km |
| FOREST-18 | 9 / 10 | 11.89 / 14.10 h | 1.25 km | 1.69 km | 2.49 km | 3.28 km |
| FOREST-19 | 7 / 11 | 10.76 / 18.73 h | 4.30 km | 0.63 km | 11.59 km | 2.79 km |

All full fits and all omissions in these selected prefixes converged and passed
v7's recorded quality screen. The 12-hour selection has 29 omission fits; the
24-hour selection has 41. Every omission is included.

The cohort median and maximum improve with the later selection. Individual
results are not monotonic: FOREST-17 and FOREST-18 worsen, while FOREST-16 and
FOREST-19 improve. Earlier intermediate spikes also remain in the original
[time-accuracy report](../raw_results/forest-experiment-v7/README.md). This table
compares the two selected prefixes, not every update issued before each limit.

V7 uses updated priors, finite Doppler, carrier lock, and finite Eb/N0 strictly
above 5 dB. No WN-versus-burst-radio distinction is inferred: those earlier
studies concern different spacecraft and are not pooled into these results.

## Poor prior: v5/v6 Cartesian full state, no drag

| Spacecraft | Passes: 12 h / 24 h | Fit completed at: 12 h / 24 h selection | Full-set RMS: 12 h | Full-set RMS: 24 h | Worst omission RMS: 24 h |
| --- | ---: | --- | ---: | ---: | ---: |
| FOREST-16 | 5 / 8 | 11.08 / 16.06 h | 6.51 km | 3.42 km | 8.23 km |
| FOREST-17 | 6 / 9 | 11.45 / 16.23 h | 10.01 km | 7.85 km | 13.24 km |
| FOREST-18 | 8 / 8 | 10.89 / 10.89 h | 5.09 km | 5.09 km | 13.80 km |
| FOREST-19 | 7 / 10 | 10.76 / 18.73 h | 17.15 km | 3.24 km | 32.63 km |

The v5 bundle supplies the shorter cumulative full fits. Its final all-pass
contact sets and next-hour scores agree with the v6 no-drag baseline for all
four spacecraft. FOREST-18 has no additional eligible passes between the two
limits under this selection, so its full-fit result is unchanged.

The 35 v6 omission trajectories were rescored over the hour after the full
prefix, so their errors use the same metric as the full fits and v7 CV. This
is different from v6's original **during-omitted-pass** position scores. All
35 omissions have complete coverage in the common next-hour window. No shorter
12-hour omission results were invented or substituted from another window.

All these full fits and omissions converged. V6 did not apply v7's operational
quality screen. The poor prior is an inaccurate pre-launch TLE, not prior-free
IOD, and the experiment does not establish performance across TLE-age bands.
The two experiments use different telemetry selections and first-pass clocks;
FOREST-18's v7 clock starts about 3.21 h earlier than its v5/v6 clock.

## Statistical interpretation

Student's t intervals below use one **full-set next-hour RMS** per spacecraft,
with $n=4$, sample standard deviation $s$, and three degrees of freedom:

$$
\mathrm{CI}_{0.95}(\mu)=\bar E\pm t_{0.975,3}\frac{s}{\sqrt{4}},
\qquad t_{0.975,3}=3.18245.
$$

| Starting condition | Collection limit | Mean full-set RMS | Two-sided 95% interval for the mean |
| --- | ---: | ---: | ---: |
| Informed prior | 12 h | 4.65 km | −1.94–11.23 km |
| Informed prior | 24 h | 1.97 km | −1.53–5.47 km |
| Poor prior | 12 h | 9.69 km | 1.12–18.26 km |
| Poor prior | 24 h | 4.90 km | 1.50–8.30 km |

These are provisional estimates assuming independent representative cases and
an approximately normal distribution of spacecraft-level RMS. Negative lower
limits are the unmodified normal-case approximation; physical error is
nonnegative. The four related spacecraft, their repeated collection limits,
and overlapping omission fits do not provide independent population trials.
The intervals concern the **mean**, not the median, maximum, or p95.

FOREST-19 retains a candidate GPS reference, and reference uncertainty is not
separately propagated. These limitations prevent a validated population p95
claim. KPT thresholds must distinguish full-fit performance from single-pass
omission resilience; the observed omission maximum is not a p95 estimate.
No target widening is justified here by the superseded deadline-aged scores.

## Reproducibility and superseded comparisons

- [Collection-window scores](collection-window-fits.csv): **121 scores**,
  comprising 16 full-set scores and 105 omission scores, with training counts,
  collection limits, exact fit/scoring epochs, source identities and hashes.
- [Provenance](collection-window-provenance.json): selection rules, scorer
  hashes, validation, and the missing poor-prior 12-hour CV cohort.
- Sources: [v5 fits](../raw_results/forest-experiment-v5/fits.csv),
  [v6 results](../raw_results/forest-experiment-v6/README.md), and
  [v7 results](../raw_results/forest-experiment-v7/README.md).

The existing next-hour scorer was applied to saved trajectories; no fits or
propagation were rerun. Every published RMS was cross-checked directly from
the saved vector residuals. Archived v5/v6/v7 full-fit scores and v7 omission
scores were reproduced; v5/v6 all-pass baseline parity was also verified.

The older `kpt-deadline-fits.csv` is retained only as a superseded analysis of
ageing products at the collection cutoff. It is not a source for these tables.
In particular, the previous informed-prior medians 4.14/5.06 km and maxima
10.98/11.27 km measured a different forecast window. The original 1.16/5.16 km
final-prefix summary is valid for the **24-hour collection selection** under
this report's next-hour-after-fit metric; it is not the 12-hour result.
