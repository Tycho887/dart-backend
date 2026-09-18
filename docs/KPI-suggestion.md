# Suggested KPIs — CAS500-2 passive-Doppler orbit determination

These are **provisional design targets aiming for ≥90% reliability**. The
campaign supports their selection, but four related spacecraft provide too
little independent evidence to establish that reliability.

## Targets and margins

Errors are **3D position-vector RMSE over the following hour**. IOD uses
cumulative full-state estimation; informed-prior refinement uses cumulative L+n.

| KPI | Proposed target | Observed evidence | Engineering headroom |
| --- | --- | --- | --- |
| First informed-prior solution | **≤5 km by 16 h after separation**, passing the quality screen | **4/4 spacecraft**; first approved errors 1.31–3.52 km, delivered by 12.6 h | 1.5 km and 3.4 h |
| Subsequent informed-prior updates | **≤12 km**, with ≥4 eligible passes and quality acceptance | **22/22 fits**; maximum 10.40 km | 1.6 km |
| IOD-like recovery | **≤25 km by 20 h after separation**, with ≥6 eligible passes | **15/15 forecasts** with ≥6 passes; maximum 20.97 km. All spacecraft reached six passes by 17.6 h | 4 km and 2.4 h |
| Product at 24 h after the first usable pass | **≤20 km**, using the latest available solution | **4/4 spacecraft per prior category**; maxima 13.3 km for IOD and 15.4 km for informed priors | At least 4.6 km |

Headroom is the allowance above observed errors or delivery times, not a
statistical confidence bound. Full-state results were unscreened; an operational
acceptance rule remains part of qualifying the IOD target.

## Clocks and interpretation

- **Separation to delivery:** includes waiting for usable contacts. Separation
  is approximately 08:00 UTC on 3 May 2026.
- **First usable pass to delivery:** start at completion of the first pass
  meeting the orbit-fit observation gates. A companion target is **≤5 km within
  12 h**; all four first approved solutions arrived within 3.0–8.7 h, using
  2–5 passes. The six-hour target was met by only **2/4** after quality screening.
- **Product age and forecast horizon:** record both. The 24-hour row uses ageing
  saved solutions; the archive contains less than 24 hours of eligible pass
  collection after each spacecraft’s first usable pass. At 24 h after
  separation instead, maximum errors were 11.0 km and 12.9 km.

Initial acquisition and continued accuracy are separate metrics: **17/25**
quality-approved informed-prior updates were below 5 km, versus **24/25** below
10 km. This motivates the wider allowance for subsequent updates.

## Qualification

Stratify results by eligible pass count, prior quality, and **original TLE epoch
age**: <24 h, 1–3 days, 3–7 days, and >7 days. The two experimental prior
categories have nearly identical epochs, so their accuracy difference establishes
a prior-quality comparison rather than an age-performance curve.

Include missed deliveries, rejected solutions, and errors above 50 km in
reliability assessment; report missing validation coverage separately. The
50 km plotting cutoff must not filter KPI failures.

The observed fits share spacecraft and observations, and FOREST-19 uses a
candidate GPS reference. As a qualification benchmark, **29 independent
successes with zero failures** give a one-sided 95% lower reliability bound
above 90%; each claimed operating cohort needs suitable evidence.

Sources: [methods and results](concise-math.md) and
[frozen fit results](../raw_results/forest-experiment-v5/fits.csv).
