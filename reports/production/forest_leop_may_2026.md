# FOREST-16/17/18/19 May 2026 passive RF orbit evaluation

## Executive result

This report evaluates passive radio-frequency orbit determination during May 2026 launch operations. It covers 61 contacts from 2026-05-03 09:40:38.191475 UTC to 2026-05-04 05:26:49.694147 UTC.

Fifteen contacts had at least 301 accepted Doppler measurements. The estimator fitted these contacts. Eleven fitted contacts had at least five GPS position records during the same contact. These 11 contacts form the primary result set.

This is a retrospective full-contact evaluation. The estimator used all accepted measurements after each contact ended. The results do not show real-time performance. This evaluation does not qualify the system for flight.

| Method | Best contact median error (km) | Median contact error (km) | Worst contact median error (km) | Primary contacts improved |
| --- | ---: | ---: | ---: | ---: |
| Initial orbit estimate | 2.437 | 9.310 | 21.391 | — |
| Robust full-contact fit | 0.494 | 3.857 | 24.856 | 7/11 |

All 15 fits passed the acceptance checks. Fourteen fitted contacts had GPS data. The fit improved 10 of these contacts and made 4 worse.

The best observed contact had a median error of 0.494 km. This is a best-case result from the recorded data. It is not a guaranteed accuracy.

The results show the potential performance for future launch operations. This potential depends on sufficient Doppler coverage, enough Doppler change, and stable receiver data.

## Terms and definitions

- **Launch and early orbit operations (LEOP):** The work done immediately after launch to find and control a spacecraft.
- **Passive radio-frequency orbit determination (passive RF OD):** An orbit estimate that uses received radio frequency changes. It does not transmit a ranging signal.
- **Contact:** A recorded period when one ground station observed one spacecraft.
- **Accepted Doppler measurement:** A valid frequency-offset value that passed the data-quality filters.
- **Global Positioning System (GPS) reference:** A GPS position record used to measure error after the Doppler fit. The estimator does not use it during the fit.
- **Residual:** The measured Doppler value minus the Doppler value predicted by the model.
- **Initial orbit estimate:** The spacecraft orbit information available before the Doppler fit.
- **Primary contact:** A fitted contact with at least five GPS records during that contact.
- **Limited-GPS contact:** A fitted contact with one to four GPS records. It is not part of the primary result set.
- **GPS-free contact:** A fitted contact with no GPS record during that contact. It has no same-contact position-error result.
- **Median error:** The middle position-error value after the values are put in order.
- **Standard deviation:** A value that shows how widely a set of values spreads around its average.
- **Hertz (Hz):** The unit of frequency. One hertz is one cycle each second.
- **Coordinated Universal Time (UTC):** The common time standard used for all contact times in this report.
- **Gauss-Newton method:** An iterative fit method that reduces the sum of squared residuals.
- **Gaussian mixture model (GMM):** A statistical method that represents residuals as one or more bell-shaped groups. Each group is a component.
- **Density-based spatial clustering of applications with noise (DBSCAN):** A method that groups nearby items and leaves isolated items ungrouped.
- **Robust loss:** A fit rule that gives less influence to very large residuals.
- **Soft-L1 loss:** The robust loss used for the final fit. It limits the influence of large residuals without deleting them.
- **Retrospective full-contact evaluation:** An evaluation that uses a complete recorded contact after that contact ended.

## Data and contact selection

The evaluation used recorded carrier-frequency offsets and GPS positions for all four spacecraft.

An accepted Doppler measurement met all of these conditions:

- The frequency offset was present and not zero.
- The absolute frequency offset was at least 0.1 Hz.
- The recorded antenna elevation was more than 1 degree and less than 89 degrees.

A contact required at least 301 accepted Doppler measurements before fitting. A fitted contact required at least five GPS records to enter the primary result set.

The team selected the 301-measurement rule after it reviewed the early contacts. Those contacts had too few measurements or too much radio interference. The rule is specific to this evaluation. It is not a universal limit for passive RF OD.

| Selection stage | Measurements or contacts retained |
| --- | ---: |
| Raw telemetry measurements | 176,474 |
| Measurements with a frequency offset | 67,882 |
| Nonzero frequency offsets | 9,727 |
| Absolute frequency offset of at least 0.1 Hz | 9,727 |
| Accepted Doppler measurements | 9,568 |
| Recorded contacts | 61 |
| Fitted contacts | 15 |
| Primary contacts | 11 |

| Spacecraft | Recorded contacts | Fitted contacts | Primary contacts |
| --- | ---: | ---: | ---: |
| FOREST-16 | 13 | 3 | 2 |
| FOREST-17 | 15 | 3 | 3 |
| FOREST-18 | 18 | 6 | 4 |
| FOREST-19 | 15 | 3 | 2 |

The estimator did not fit 46 contacts. Eighteen had no accepted Doppler measurement. The other 28 had between 1 and 300 accepted Doppler measurements.

## Burst-radio errors and estimator change

### Cause of the earlier fit problems

The earlier Gauss-Newton method used a standard squared-error cost. This method works well when most residuals are small and follow one bell-shaped group.

The FOREST burst radios sometimes produced errors of tens of kilohertz. The squared-error cost gave these large errors too much influence. A short burst could outweigh hundreds of representative measurements.

The estimator changed both time and frequency. During a short contact, one correction can partly imitate the other. Large radio errors, small Doppler changes, and a poor starting value could therefore move the fit to a wrong solution.

### Residual analysis

The residual study examined each contact separately. It removed duplicate times and contacts with fewer than 100 residuals. It then used a GMM with one to five groups. A statistical rule balanced the fit quality against the number of groups.

The study removed a group if it represented less than 25 percent of its contact. It then used DBSCAN to group the retained GMM components by their average and standard deviation.

The larger residual archive contained 817 retained components from 527 contacts. DBSCAN put 685 components in a narrow group. This group had an average offset of -180.55 Hz and an average standard deviation of 351.24 Hz.

DBSCAN put 78 components in a broad group. This group had an average offset of +654.08 Hz and an average standard deviation of 29,770.65 Hz. DBSCAN left 54 components ungrouped.

The residual archive is larger than the 61-contact performance set. It describes the radio environment, not the reported orbit accuracy. The group values are not universal hardware limits.

The analysis found a narrow error group and a separate burst-error group. A single bell-shaped error model did not describe both groups.

### Estimator selection

Development tests compared standard loss and robust loss. Separate stress tests added known time errors and measurement noise.

These tests supported four changes. The final estimator uses soft-L1 loss, correction limits, several starting values, and a minimum data rule. The development tests used recorded data and were retrospective. Only the fixed final settings produced the accuracy results in this report.

## Initial orbit estimates

Each fitted contact started from one spacecraft-specific initial orbit estimate. The estimator did not change this initial estimate. It calculated a time correction and a frequency correction for each contact.

- **FOREST-16:** Initial estimate time 2026-05-03 09:21:26 UTC; used for 3 fitted contacts.
- **FOREST-17:** Initial estimate time 2026-05-03 09:19:03 UTC; used for 3 fitted contacts.
- **FOREST-18:** Initial estimate time 2026-05-03 09:20:18 UTC; used for 6 fitted contacts.
- **FOREST-19:** Initial estimate time 2026-05-03 09:21:02 UTC; used for 3 fitted contacts.

## Analyzed contacts and results

The times below show the first and last accepted Doppler measurements. The position errors use GPS records from the same contact. The median error is the primary measure.

| Spacecraft | Contact ID | Station | Accepted Doppler interval (UTC) | Doppler measurements | GPS records | Time correction (s) | Frequency correction (Hz) | Initial median error (km) | Corrected median error (km) | GPS status |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FOREST-16 | `e5d79f0b-730b-4832-becf-f716d7322784` | PUNTA_ARENAS | 2026-05-03 15:42:15.405285 UTC – 2026-05-03 15:49:59.655731 UTC | 340 | 6 | 3.770 | 975.138 | 9.310 | 19.154 | Primary contact |
| FOREST-16 | `33061519-fa11-49e4-becd-6c770efd6e11` | SVALSAT | 2026-05-04 00:26:50.187208 UTC – 2026-05-04 00:36:32.536875 UTC | 475 | 1 | 2.282 | 648.128 | 18.166 | 1.014 | Limited-GPS contact |
| FOREST-16 | `556b1cfa-0a8e-4d19-8f95-b123fd47cd60` | PUNTA_ARENAS | 2026-05-04 02:39:19.219744 UTC – 2026-05-04 02:49:56.523857 UTC | 401 | 11 | 3.198 | 729.813 | 21.140 | 3.013 | Primary contact |
| FOREST-17 | `2610bd5f-3a3f-4756-be38-d31392d4324c` | HARTEBEESTHOEK | 2026-05-03 20:04:03.162277 UTC – 2026-05-03 20:13:59.537732 UTC | 349 | 7 | -4.136 | -929.899 | 6.395 | 24.856 | Primary contact |
| FOREST-17 | `36c92afa-8853-4630-a788-ad2e63912d51` | TROLL | 2026-05-03 21:54:34.185557 UTC – 2026-05-03 22:04:24.590709 UTC | 370 | 10 | -0.441 | 71.973 | 7.186 | 3.857 | Primary contact |
| FOREST-17 | `d623db96-56fd-4783-9f1e-bf3933f4dbd8` | PUNTA_ARENAS | 2026-05-04 01:06:44.176799 UTC – 2026-05-04 01:13:48.439478 UTC | 344 | 5 | -0.509 | 755.427 | 8.337 | 4.500 | Primary contact |
| FOREST-18 | `49e74a6c-b21d-46c0-87bb-e4195b4a0dab` | PUNTA_ARENAS | 2026-05-03 17:18:55.440628 UTC – 2026-05-03 17:28:01.741531 UTC | 471 | 9 | 0.112 | 944.281 | 9.976 | 10.826 | Primary contact |
| FOREST-18 | `48de3125-1fd3-40df-be87-5fc55aa2485f` | TROLL | 2026-05-03 20:18:25.198752 UTC – 2026-05-03 20:28:19.546636 UTC | 341 | 10 | -2.207 | 128.810 | 15.350 | 1.377 | Primary contact |
| FOREST-18 | `2735b79f-1ae2-4be6-9b13-774d026efd72` | SVALSAT | 2026-05-03 21:13:16.051067 UTC – 2026-05-03 21:23:40.437676 UTC | 336 | 4 | -2.541 | 165.122 | 14.463 | 4.769 | Limited-GPS contact |
| FOREST-18 | `7654ef67-7aba-4f18-bf0a-65eacc1b6a22` | SVALSAT | 2026-05-03 22:49:19.221401 UTC – 2026-05-03 23:00:58.667437 UTC | 339 | 3 | -2.233 | -45.201 | 16.744 | 0.307 | Limited-GPS contact |
| FOREST-18 | `105f1f16-870e-4658-9a38-41c74ea5939c` | TROLL | 2026-05-03 23:31:14.18241 UTC – 2026-05-03 23:39:09.458704 UTC | 322 | 7 | -2.770 | -63.312 | 19.555 | 1.390 | Primary contact |
| FOREST-18 | `8db51bff-de04-4287-b3c8-4f949e7a8df1` | AWARUA | 2026-05-04 01:23:19.261813 UTC – 2026-05-04 01:32:09.500393 UTC | 337 | 8 | -3.042 | 763.899 | 21.391 | 1.585 | Primary contact |
| FOREST-19 | `d2af3474-5932-431d-ae42-c7ce901c93ba` | TROLL | 2026-05-03 18:43:07.219705 UTC – 2026-05-03 18:52:27.545014 UTC | 358 | 8 | -0.377 | -85.525 | 2.437 | 5.249 | Primary contact |
| FOREST-19 | `df2618ab-46b7-49bb-a148-ea9f89466869` | AWARUA | 2026-05-03 23:47:15.296679 UTC – 2026-05-03 23:58:01.646697 UTC | 521 | 11 | 0.303 | 648.238 | 2.746 | 0.494 | Primary contact |
| FOREST-19 | `e629dfe8-8210-4af9-a61d-f02b5377a4fd` | SVALSAT | 2026-05-04 05:11:51.150377 UTC – 2026-05-04 05:19:49.540117 UTC | 357 | 0 | 0.168 | 866.285 | — | — | GPS-free contact |

Four fitted contacts are not primary contacts. Three are limited-GPS contacts. One is a GPS-free contact.

The 0.307 km FOREST-18 result uses only three GPS records. It is a limited-GPS result. It does not replace the 0.494 km best primary result.

## Doppler model and final estimator

The model predicts Doppler from the relative motion of the spacecraft and the ground station. It also includes a constant frequency correction for each contact.

The estimator changes two values. The time correction moves the spacecraft along its initial orbit. The frequency correction accounts for a constant radio-frequency offset.

The final fit uses soft-L1 loss with a 700 Hz transition value. Small residuals retain their normal influence. Large residuals receive less influence.

The time correction is limited to -120 through +120 seconds. The frequency correction is limited to -100,000 through +100,000 Hz. The fit starts from several time values and keeps the best soft-L1 result.

A fit is accepted only if it completes, finds both corrections, and does not stop at a limit. All 15 fits met these conditions.

This method does not estimate a complete orbit state. It cannot correct all types of orbit error.

## Position-error calculation

The evaluation compares the estimated three-dimensional position with the GPS position at the same time. The distance between these positions is the position error in kilometres.

The initial error uses no time correction. The corrected error uses the fitted time correction. The contact result is the median of its GPS position errors.

The report gives each contact equal weight. A long contact or a contact with more GPS records cannot dominate the summary.

## Later GPS comparison

For this comparison, the fitted time correction remains constant after the contact ends. The evaluation then compares the orbit estimate with later GPS positions.

Each time interval requires at least five GPS records. Each value is the median of the contact median errors.

| Time after contact | Contacts | Initial median error (km) | Corrected median error (km) | Contacts improved |
| --- | ---: | ---: | ---: | ---: |
| 0–1 h | 15 | 9.676 | 2.617 | 11/15 |
| 1–3 h | 15 | 12.740 | 3.509 | 11/15 |
| 3–6 h | 15 | 16.735 | 5.860 | 11/15 |
| 6–12 h | 15 | 22.378 | 10.519 | 12/15 |
| 12–24 h | 15 | 31.735 | 19.798 | 11/15 |

## Controls and limitations

The evaluation used these controls:

- The contact accounting includes all 61 recorded contacts.
- The report includes contacts that became worse after the fit.
- GPS data does not enter a production fit. It measures error after the fit.
- The same data-quality rules apply to all contacts.
- The primary result set uses the same five-record GPS rule for all contacts.
- GPS position and orbit position are compared at the same measurement time.
- The summary gives each contact equal weight.

The study is retrospective. The team reviewed the contacts before it completed this evaluation. The estimator settings were also developed with recorded data.

A future confirmation must use fixed rules on new contacts. This report shows potential performance under the recorded conditions. It does not guarantee operational accuracy.

## Verification summary

The review counted all 61 contacts and applied the same selection rules to each contact. It confirmed 15 fitted contacts and 11 primary contacts.

The review also checked the contact times, measurement counts, GPS counts, initial estimates, and fit results. The result tables include all 15 fitted contacts.

## Appendix A — all 61 recorded contacts

Measurement counts are shown in this order: raw / frequency offset present / nonzero / at least 0.1 Hz / accepted.

| Spacecraft | Contact ID | Station | Recorded interval (UTC) | Accepted Doppler interval (UTC) | Measurement counts | GPS records | Status |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| FOREST-16 | `a6252615-8c19-4f78-a6f0-227e3da97c2e` | SVALSAT | 2026-05-03 09:40:40.060749 UTC – 2026-05-03 10:09:27.049946 UTC | 2026-05-03 10:01:16.884186 UTC – 2026-05-03 10:01:24.893618 UTC | 4134 / 1703 / 10 / 10 / 9 | 0 | Not fitted: fewer than 301 accepted Doppler measurements |
| FOREST-16 | `b2bd28a8-6704-4a67-bd8b-2c84fb0473bc` | AWARUA | 2026-05-03 10:20:36.15872 UTC – 2026-05-03 10:49:55.926637 UTC | 2026-05-03 10:31:18.426844 UTC – 2026-05-03 10:39:37.754525 UTC | 4204 / 1721 / 297 / 297 / 297 | 0 | Not fitted: fewer than 301 accepted Doppler measurements |
| FOREST-16 | `9adbc5c4-0b67-471e-a286-dda9a089a49f` | SVALSAT | 2026-05-03 11:27:39 UTC – 2026-05-03 11:38:36.577378 UTC | — – — | 1543 / 443 / 0 / 0 / 0 | 0 | Not fitted: no accepted Doppler measurements |
| FOREST-16 | `7815ef69-da31-40ef-853c-e552ebe841cc` | TROLL | 2026-05-03 12:12:32.14899 UTC – 2026-05-03 12:42:04.105706 UTC | — – — | 3751 / 1488 / 0 / 0 / 0 | 0 | Not fitted: no accepted Doppler measurements |
| FOREST-16 | `1c36ca34-a985-4809-9ca7-65e422e6c47d` | SVALSAT | 2026-05-03 12:56:20.05508 UTC – 2026-05-03 13:22:21.964978 UTC | 2026-05-03 13:07:11.460412 UTC – 2026-05-03 13:14:46.719821 UTC | 3568 / 1503 / 187 / 187 / 187 | 7 | Not fitted: fewer than 301 accepted Doppler measurements |
| FOREST-16 | `e5d79f0b-730b-4832-becf-f716d7322784` | PUNTA_ARENAS | 2026-05-03 15:31:25.050098 UTC – 2026-05-03 16:00:51.081325 UTC | 2026-05-03 15:42:15.405285 UTC – 2026-05-03 15:49:59.655731 UTC | 4267 / 1749 / 340 / 340 / 340 | 6 | Primary contact |
| FOREST-16 | `bcec2df5-c0d0-4e89-91ae-f0568424de8a` | SVALSAT | 2026-05-03 17:50:03.055974 UTC – 2026-05-03 18:15:58.955542 UTC | 2026-05-03 18:00:40.520348 UTC – 2026-05-03 18:02:07.617611 UTC | 3623 / 1533 / 44 / 44 / 44 | 2 | Not fitted: fewer than 301 accepted Doppler measurements |
| FOREST-16 | `82723fb0-59f4-4de8-ac05-f9f3d9a7dcc7` | SVALSAT | 2026-05-03 19:35:01.058298 UTC – 2026-05-03 19:50:52.619807 UTC | 2026-05-03 19:39:48.218286 UTC – 2026-05-03 19:44:49.403602 UTC | 2529 / 938 / 252 / 252 / 252 | 1 | Not fitted: fewer than 301 accepted Doppler measurements |
| FOREST-16 | `7cb89272-0ba6-4eaf-8445-917ec34e727e` | HARTEBEESTHOEK | 2026-05-03 21:39:48.182416 UTC – 2026-05-03 21:54:30.671411 UTC | 2026-05-03 21:40:25.189144 UTC – 2026-05-03 21:51:03.539207 UTC | 2349 / 883 / 278 / 278 / 192 | 11 | Not fitted: fewer than 301 accepted Doppler measurements |
| FOREST-16 | `2a45c0d3-c3e9-45e8-a11f-5dbf6bda0693` | SVALSAT | 2026-05-03 22:47:21.05696 UTC – 2026-05-03 23:05:01.746432 UTC | 2026-05-03 22:49:21.113395 UTC – 2026-05-03 22:51:49.208673 UTC | 2841 / 1040 / 110 / 110 / 110 | 2 | Not fitted: fewer than 301 accepted Doppler measurements |
| FOREST-16 | `33061519-fa11-49e4-becd-6c770efd6e11` | SVALSAT | 2026-05-04 00:23:14.048872 UTC – 2026-05-04 00:40:56.6895 UTC | 2026-05-04 00:26:50.187208 UTC – 2026-05-04 00:36:32.536875 UTC | 2863 / 1048 / 475 / 475 / 475 | 1 | Limited-GPS contact: 1 record |
| FOREST-16 | `85ec563d-ecb0-4c0b-9b2d-eeb9cec80587` | SVALSAT | 2026-05-04 01:58:43.060348 UTC – 2026-05-04 02:16:21.696483 UTC | 2026-05-04 02:01:32.182395 UTC – 2026-05-04 02:01:37.17855 UTC | 2843 / 1042 / 6 / 6 / 6 | 0 | Not fitted: fewer than 301 accepted Doppler measurements |
| FOREST-16 | `556b1cfa-0a8e-4d19-8f95-b123fd47cd60` | PUNTA_ARENAS | 2026-05-04 02:35:53.050254 UTC – 2026-05-04 02:53:42.66359 UTC | 2026-05-04 02:39:19.219744 UTC – 2026-05-04 02:49:56.523857 UTC | 2912 / 1070 / 405 / 405 / 401 | 11 | Primary contact |
| FOREST-17 | `c12b9960-e925-4286-9cd1-6a887d4e8289` | SVALSAT | 2026-05-03 09:40:38.191475 UTC – 2026-05-03 10:09:32.948508 UTC | — – — | 4150 / 1709 / 0 / 0 / 0 | 0 | Not fitted: no accepted Doppler measurements |
| FOREST-17 | `75d4144d-d3fc-4e18-8f14-b17da65092cb` | AWARUA | 2026-05-03 10:30:06.10634 UTC – 2026-05-03 10:42:51 UTC | — – — | 2295 / 765 / 0 / 0 / 0 | 0 | Not fitted: no accepted Doppler measurements |
| FOREST-17 | `118e1a7d-4ae2-4982-a40b-7d6eb0ead2e4` | TROLL | 2026-05-03 10:36:52.106507 UTC – 2026-05-03 11:06:10.080052 UTC | — – — | 3689 / 1464 / 0 / 0 / 0 | 0 | Not fitted: no accepted Doppler measurements |
| FOREST-17 | `dc06b459-269d-446a-97ef-9b6febcd8679` | SVALSAT | 2026-05-03 11:27:41.154902 UTC – 2026-05-03 11:38:44 UTC | 2026-05-03 11:28:51.196429 UTC – 2026-05-03 11:38:13.465414 UTC | 1979 / 658 / 247 / 247 / 247 | 0 | Not fitted: fewer than 301 accepted Doppler measurements |
| FOREST-17 | `0a2e4679-2976-4bed-bd7a-ffd4f45e0a60` | TROLL | 2026-05-03 12:22:12 UTC – 2026-05-03 12:35:06.543016 UTC | — – — | 2085 / 655 / 0 / 0 / 0 | 0 | Not fitted: no accepted Doppler measurements |
| FOREST-17 | `6569e3a1-7aa1-4ff3-a2c0-cd1cd88fd85a` | SVALSAT | 2026-05-03 12:56:30.150282 UTC – 2026-05-03 13:22:38.992042 UTC | 2026-05-03 13:07:21.493384 UTC – 2026-05-03 13:14:53.798579 UTC | 3617 / 1525 / 213 / 213 / 213 | 8 | Not fitted: fewer than 301 accepted Doppler measurements |
| FOREST-17 | `4ef3b909-6d7f-460d-bf48-41aed65abf0a` | TROLL | 2026-05-03 13:48:19.093456 UTC – 2026-05-03 14:17:19.125152 UTC | — – — | 3659 / 1458 / 0 / 0 / 0 | 0 | Not fitted: no accepted Doppler measurements |
| FOREST-17 | `5233c319-41fd-4896-847f-112ddfe9c56d` | TROLL | 2026-05-03 16:58:11.141154 UTC – 2026-05-03 17:26:34.092114 UTC | 2026-05-03 17:10:33.509256 UTC – 2026-05-03 17:18:29.828082 UTC | 3550 / 1422 / 264 / 264 / 264 | 0 | Not fitted: fewer than 301 accepted Doppler measurements |
| FOREST-17 | `74df0163-cf66-41f9-b960-409fca756c93` | SVALSAT | 2026-05-03 17:50:30.174645 UTC – 2026-05-03 18:16:35.006705 UTC | 2026-05-03 18:01:08.495501 UTC – 2026-05-03 18:02:21.53164 UTC | 3680 / 1558 / 51 / 51 / 51 | 1 | Not fitted: fewer than 301 accepted Doppler measurements |
| FOREST-17 | `2610bd5f-3a3f-4756-be38-d31392d4324c` | HARTEBEESTHOEK | 2026-05-03 20:03:32.133648 UTC – 2026-05-03 20:20:55.764887 UTC | 2026-05-03 20:04:03.162277 UTC – 2026-05-03 20:13:59.537732 UTC | 2834 / 1044 / 368 / 368 / 349 | 7 | Primary contact |
| FOREST-17 | `36c92afa-8853-4630-a788-ad2e63912d51` | TROLL | 2026-05-03 21:51:59.119639 UTC – 2026-05-03 22:09:43.700544 UTC | 2026-05-03 21:54:34.185557 UTC – 2026-05-03 22:04:24.590709 UTC | 2544 / 889 / 370 / 370 / 370 | 10 | Primary contact |
| FOREST-17 | `d61bfc77-1d1c-4e82-91fd-6e7489e73cb3` | SVALSAT | 2026-05-03 22:48:07.142387 UTC – 2026-05-03 23:05:54.694529 UTC | 2026-05-03 22:50:05.198813 UTC – 2026-05-03 22:53:09.312098 UTC | 2859 / 1046 / 103 / 103 / 103 | 2 | Not fitted: fewer than 301 accepted Doppler measurements |
| FOREST-17 | `d623db96-56fd-4783-9f1e-bf3933f4dbd8` | PUNTA_ARENAS | 2026-05-04 01:02:54.061707 UTC – 2026-05-04 01:18:41.547445 UTC | 2026-05-04 01:06:44.176799 UTC – 2026-05-04 01:13:48.439478 UTC | 2545 / 947 / 344 / 344 / 344 | 5 | Primary contact |
| FOREST-17 | `01448cbd-3659-4184-89c2-7883e4a3c6c7` | SVALSAT | 2026-05-04 01:59:41.097891 UTC – 2026-05-04 02:17:24.702523 UTC | 2026-05-04 02:02:09.187411 UTC – 2026-05-04 02:03:18.224597 UTC | 2843 / 1040 / 62 / 62 / 62 | 1 | Not fitted: fewer than 301 accepted Doppler measurements |
| FOREST-17 | `9303555b-beda-4b39-ac33-b6df9a7e9952` | SVALSAT | 2026-05-04 03:35:06.058524 UTC – 2026-05-04 03:52:44.79613 UTC | 2026-05-04 03:37:43.144481 UTC – 2026-05-04 03:48:32.569257 UTC | 2836 / 1039 / 240 / 240 / 240 | 6 | Not fitted: fewer than 301 accepted Doppler measurements |
| FOREST-18 | `53eafac8-7eae-4d9b-8adc-9066c668410c` | SVALSAT | 2026-05-03 09:40:41.120205 UTC – 2026-05-03 10:09:27.977468 UTC | — – — | 4111 / 1692 / 0 / 0 / 0 | 0 | Not fitted: no accepted Doppler measurements |
| FOREST-18 | `3ccef2f9-751b-4153-b810-267dd4c64a31` | AWARUA | 2026-05-03 10:30:02.157114 UTC – 2026-05-03 10:32:25 UTC | — – — | 429 / 143 / 0 / 0 / 0 | 0 | Not fitted: no accepted Doppler measurements |
| FOREST-18 | `ca8d1c55-2725-4217-bc02-13672ceec9ae` | SVALSAT | 2026-05-03 11:18:07.087341 UTC – 2026-05-03 11:45:43.99511 UTC | — – — | 3930 / 1637 / 0 / 0 / 0 | 0 | Not fitted: no accepted Doppler measurements |
| FOREST-18 | `a1bdfcbb-ff7d-47af-8ba1-7fb37ee15e65` | SVALSAT | 2026-05-03 11:31:17 UTC – 2026-05-03 11:38:32 UTC | — – — | 1228 / 396 / 0 / 0 / 0 | 0 | Not fitted: no accepted Doppler measurements |
| FOREST-18 | `6d7785df-a7a7-4963-8fcc-bdebf3a6dc80` | TROLL | 2026-05-03 12:22:00.107874 UTC – 2026-05-03 12:34:55 UTC | 2026-05-03 12:24:13.202667 UTC – 2026-05-03 12:34:40.516597 UTC | 2323 / 774 / 72 / 72 / 71 | 10 | Not fitted: fewer than 301 accepted Doppler measurements |
| FOREST-18 | `f1550be3-6ca0-48b2-bd87-da21c1e885b6` | SVALSAT | 2026-05-03 12:56:19.140207 UTC – 2026-05-03 13:22:20.084922 UTC | — – — | 3636 / 1538 / 0 / 0 / 0 | 0 | Not fitted: no accepted Doppler measurements |
| FOREST-18 | `a5dc9c5c-0fb7-4bc9-9ba3-89a37ff6fac1` | SVALSAT | 2026-05-03 13:06:52 UTC – 2026-05-03 13:17:49.469197 UTC | — – — | 1756 / 628 / 0 / 0 / 0 | 0 | Not fitted: no accepted Doppler measurements |
| FOREST-18 | `b2015720-f0bb-48e7-b19a-131a592c9501` | PUNTA_ARENAS | 2026-05-03 13:56:45.053388 UTC – 2026-05-03 14:22:37.918427 UTC | 2026-05-03 14:08:47.471582 UTC – 2026-05-03 14:10:39.528736 UTC | 3608 / 1525 / 29 / 29 / 29 | 2 | Not fitted: fewer than 301 accepted Doppler measurements |
| FOREST-18 | `49e74a6c-b21d-46c0-87bb-e4195b4a0dab` | PUNTA_ARENAS | 2026-05-03 17:07:14.061741 UTC – 2026-05-03 17:35:22.002084 UTC | 2026-05-03 17:18:55.440628 UTC – 2026-05-03 17:28:01.741531 UTC | 4017 / 1663 / 487 / 487 / 471 | 9 | Primary contact |
| FOREST-18 | `6c22b929-f451-4e3f-bc5d-fc4e230c6c02` | SVALSAT | 2026-05-03 17:50:01.097961 UTC – 2026-05-03 18:15:55.085095 UTC | 2026-05-03 18:00:37.493457 UTC – 2026-05-03 18:01:43.556265 UTC | 3624 / 1535 / 32 / 32 / 32 | 1 | Not fitted: fewer than 301 accepted Doppler measurements |
| FOREST-18 | `5f62261f-f193-475b-ad2d-94a8b9fd0e75` | SVALSAT | 2026-05-03 18:00:19 UTC – 2026-05-03 18:10:18.481309 UTC | — – — | 1676 / 581 / 0 / 0 / 0 | 0 | Not fitted: no accepted Doppler measurements |
| FOREST-18 | `48de3125-1fd3-40df-be87-5fc55aa2485f` | TROLL | 2026-05-03 20:15:20.081551 UTC – 2026-05-03 20:33:08.697798 UTC | 2026-05-03 20:18:25.198752 UTC – 2026-05-03 20:28:19.546636 UTC | 2557 / 892 / 341 / 341 / 341 | 10 | Primary contact |
| FOREST-18 | `2735b79f-1ae2-4be6-9b13-774d026efd72` | SVALSAT | 2026-05-03 21:12:35.105666 UTC – 2026-05-03 21:24:38 UTC | 2026-05-03 21:13:16.051067 UTC – 2026-05-03 21:23:40.437676 UTC | 2117 / 697 / 339 / 339 / 336 | 4 | Limited-GPS contact: 4 records |
| FOREST-18 | `7654ef67-7aba-4f18-bf0a-65eacc1b6a22` | SVALSAT | 2026-05-03 22:47:18.129477 UTC – 2026-05-03 23:04:57.723599 UTC | 2026-05-03 22:49:19.221401 UTC – 2026-05-03 23:00:58.667437 UTC | 2857 / 1049 / 339 / 339 / 339 | 3 | Limited-GPS contact: 3 records |
| FOREST-18 | `105f1f16-870e-4658-9a38-41c74ea5939c` | TROLL | 2026-05-03 23:28:25.108077 UTC – 2026-05-03 23:44:45.634323 UTC | 2026-05-03 23:31:14.18241 UTC – 2026-05-03 23:39:09.458704 UTC | 2309 / 812 / 322 / 322 / 322 | 7 | Primary contact |
| FOREST-18 | `8db51bff-de04-4287-b3c8-4f949e7a8df1` | AWARUA | 2026-05-04 01:20:16.264068 UTC – 2026-05-04 01:35:43.599906 UTC | 2026-05-04 01:23:19.261813 UTC – 2026-05-04 01:32:09.500393 UTC | 2485 / 928 / 342 / 342 / 337 | 8 | Primary contact |
| FOREST-18 | `7f6f30f5-0820-469d-9ab6-71b9523e8939` | SVALSAT | 2026-05-04 01:58:39.310713 UTC – 2026-05-04 02:16:15.689935 UTC | 2026-05-04 02:01:18.301457 UTC – 2026-05-04 02:08:06.448033 UTC | 2805 / 1024 / 170 / 170 / 170 | 0 | Not fitted: fewer than 301 accepted Doppler measurements |
| FOREST-18 | `4d451626-a662-4148-8369-abbcd6001b6a` | PUNTA_ARENAS | 2026-05-04 04:13:06.0568 UTC – 2026-05-04 04:28:39.548511 UTC | 2026-05-04 04:16:11.165606 UTC – 2026-05-04 04:24:33.539715 UTC | 2504 / 934 / 212 / 212 / 212 | 9 | Not fitted: fewer than 301 accepted Doppler measurements |
| FOREST-19 | `4a452e93-b302-4029-b583-7e0e061cbc64` | SVALSAT | 2026-05-03 09:40:41.142928 UTC – 2026-05-03 10:09:28.976325 UTC | — – — | 4097 / 1685 / 0 / 0 / 0 | 0 | Not fitted: no accepted Doppler measurements |
| FOREST-19 | `827d2f56-abc5-46cf-97b9-bc8883c9acca` | SVALSAT | 2026-05-03 09:46:29.065422 UTC – 2026-05-03 10:04:28.670885 UTC | — – — | 2855 / 1064 / 0 / 0 / 0 | 0 | Not fitted: no accepted Doppler measurements |
| FOREST-19 | `dc8c6112-48a4-4713-8b24-daa04ac09048` | SVALSAT | 2026-05-03 09:53:00 UTC – 2026-05-03 10:01:00 UTC | — – — | 1431 / 475 / 0 / 0 / 0 | 0 | Not fitted: no accepted Doppler measurements |
| FOREST-19 | `28e194fe-197d-4ec1-b4e4-c206bc5ebb54` | AWARUA | 2026-05-03 10:40:55 UTC – 2026-05-03 10:42:45 UTC | 2026-05-03 10:40:56.219612 UTC – 2026-05-03 10:42:29.338365 UTC | 331 / 110 / 84 / 84 / 70 | 0 | Not fitted: fewer than 301 accepted Doppler measurements |
| FOREST-19 | `7825d181-49f4-46e5-9bfe-0e9a99500190` | AWARUA | 2026-05-03 11:57:22.188431 UTC – 2026-05-03 12:20:59.983401 UTC | — – — | 3466 / 1405 / 0 / 0 / 0 | 0 | Not fitted: no accepted Doppler measurements |
| FOREST-19 | `a22fbafe-9e67-4eb3-b02b-7f108aa1b80b` | TROLL | 2026-05-03 12:22:00.108505 UTC – 2026-05-03 12:34:54.651116 UTC | 2026-05-03 12:24:37.30326 UTC – 2026-05-03 12:32:52.578205 UTC | 2322 / 774 / 189 / 189 / 189 | 3 | Not fitted: fewer than 301 accepted Doppler measurements |
| FOREST-19 | `f80c6227-6cc9-441f-ae4f-13c37b361b00` | SVALSAT | 2026-05-03 12:56:20.117565 UTC – 2026-05-03 13:22:20.019669 UTC | 2026-05-03 13:07:07.540632 UTC – 2026-05-03 13:13:17.65616 UTC | 3622 / 1531 / 148 / 148 / 148 | 4 | Not fitted: fewer than 301 accepted Doppler measurements |
| FOREST-19 | `2e914640-332d-4846-8126-0c7d58e2a568` | TROLL | 2026-05-03 15:23:04.07893 UTC – 2026-05-03 15:51:28.079573 UTC | 2026-05-03 15:35:44.538784 UTC – 2026-05-03 15:40:23.709613 UTC | 3542 / 1418 / 156 / 156 / 156 | 5 | Not fitted: fewer than 301 accepted Doppler measurements |
| FOREST-19 | `399cf06a-3730-4818-8f75-270388e8fd0a` | SVALSAT | 2026-05-03 17:50:01.153676 UTC – 2026-05-03 18:15:56.076481 UTC | 2026-05-03 18:00:34.47949 UTC – 2026-05-03 18:08:12.724352 UTC | 3653 / 1549 / 275 / 275 / 275 | 7 | Not fitted: fewer than 301 accepted Doppler measurements |
| FOREST-19 | `d2af3474-5932-431d-ae42-c7ce901c93ba` | TROLL | 2026-05-03 18:40:28.112289 UTC – 2026-05-03 18:57:41.703199 UTC | 2026-05-03 18:43:07.219705 UTC – 2026-05-03 18:52:27.545014 UTC | 2467 / 865 / 358 / 358 / 358 | 8 | Primary contact |
| FOREST-19 | `5d035d45-52f9-4d77-86fa-7e4917f23992` | SVALSAT | 2026-05-03 21:11:15.059507 UTC – 2026-05-03 21:28:19.760669 UTC | 2026-05-03 21:17:47.314688 UTC – 2026-05-03 21:21:59.45518 UTC | 2736 / 1006 / 159 / 159 / 159 | 0 | Not fitted: fewer than 301 accepted Doppler measurements |
| FOREST-19 | `8c5807c9-daa1-4957-a3a5-2711acfae882` | SVALSAT | 2026-05-03 22:47:18.096097 UTC – 2026-05-03 23:04:57.652053 UTC | 2026-05-03 22:49:13.185074 UTC – 2026-05-03 22:52:04.293769 UTC | 2842 / 1041 / 89 / 89 / 89 | 3 | Not fitted: fewer than 301 accepted Doppler measurements |
| FOREST-19 | `df2618ab-46b7-49bb-a148-ea9f89466869` | AWARUA | 2026-05-03 23:44:06.241471 UTC – 2026-05-04 00:01:54.810689 UTC | 2026-05-03 23:47:15.296679 UTC – 2026-05-03 23:58:01.646697 UTC | 2878 / 1054 / 531 / 531 / 521 | 11 | Primary contact |
| FOREST-19 | `c862d412-bfa6-4ff1-bd12-935e690521fb` | SVALSAT | 2026-05-04 01:58:39.12675 UTC – 2026-05-04 02:16:16.644855 UTC | 2026-05-04 02:00:37.156173 UTC – 2026-05-04 02:01:46.222667 UTC | 2837 / 1040 / 30 / 30 / 30 | 0 | Not fitted: fewer than 301 accepted Doppler measurements |
| FOREST-19 | `e629dfe8-8210-4af9-a61d-f02b5377a4fd` | SVALSAT | 2026-05-04 05:09:13.048181 UTC – 2026-05-04 05:26:49.694147 UTC | 2026-05-04 05:11:51.150377 UTC – 2026-05-04 05:19:49.540117 UTC | 2831 / 1037 / 357 / 357 / 357 | 0 | GPS-free contact |
