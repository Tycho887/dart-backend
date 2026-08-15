# Full simplified pass results — 25-measurement experiment

This sensitivity experiment fits all 41 passes with at least 25 accepted Doppler measurements. It uses the same robust full-pass estimator as the 301-measurement production evaluation; only the minimum measurement count changed. 37 fits are healthy and 4 are unhealthy.

Duration is the full recorded contact interval. Measurements is the accepted Doppler count supplied to the estimator. Offset is the estimated TLE time offset, and bias is the constant frequency correction estimated for the pass.

Offset variance is the local covariance estimate for the fitted time offset. It is model-based and is not calibrated for correlated Doppler errors or burst interference. Doppler RMSE is the conventional unweighted post-fit root-mean-square measured-minus-predicted residual, including burst outliers.

Prior and new TLE accuracy are median three-dimensional position errors against GPS. Where GPS exists during the full recorded contact, all those fixes are used. Otherwise, the first five GPS fixes after the contact are used and the future time range is shown explicitly. The new TLE accuracy applies the fitted time offset to the prior orbit estimate. Frequency bias is not encoded in the TLE.

Of the 41 passes, 38 are scored with GPS from the recorded contact and 3 use future GPS. 3 contact-GPS results contain fewer than five fixes.

## Gated result summary

A pass is valid when it has at least 250 accepted Doppler measurements, offset variance no greater than 15 s², and a healthy fit. 15 of 41 passes are valid.

| Orbit estimate | Best accuracy (km) | Median accuracy (km) | Worst accuracy (km) |
| --- | ---: | ---: | ---: |
| Prior TLE | 2.382 | 9.310 | 21.147 |
| New TLE | 0.288 | 3.008 | 24.862 |

The summary gives each valid pass equal weight and uses the GPS reference identified in its table row.

### FOREST-19 RIC degradation example

The following plot shows prior and corrected radial, in-track, and cross-track position residuals for the FOREST-19 AWARUA contact with 17 contact-time GPS fixes. Residuals are predicted minus GPS reference position. The dashed vertical line is the recorded contact end; subsequent GPS points compare the predictions within a 24-hour forecast window.

![FOREST-19 AWARUA RIC position residuals](deliverables/forest19_awarua_ric_residuals.png)

The machine-readable deliverables contain the exact prior TLE, fitted variables, full covariance matrix, contact UUID, and GPS fixes used for the accuracy calculation. The complete file index is in [the deliverable manifest](deliverables/manifest.json).

### All valid contacts

| Contact UUID / JSON | Satellite | Pass start (UTC) | Station | Measurements | Offset variance (s²) | GPS reference | Prior accuracy (km) | New TLE accuracy (km) |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| [b2bd28a8-6704-4a67-bd8b-2c84fb0473bc](deliverables/contacts/b2bd28a8-6704-4a67-bd8b-2c84fb0473bc.json) | FOREST-16 | 2026-05-03 10:20:36 | AWARUA | 297 | 2.098 | Future GPS (+116–120 min) | 4.923‡ | 0.541‡ |
| [e5d79f0b-730b-4832-becf-f716d7322784](deliverables/contacts/e5d79f0b-730b-4832-becf-f716d7322784.json) | FOREST-16 | 2026-05-03 15:31:25 | PUNTA_ARENAS | 340 | 2.540 | Contact GPS (28) | 9.310 | 19.154 |
| [33061519-fa11-49e4-becd-6c770efd6e11](deliverables/contacts/33061519-fa11-49e4-becd-6c770efd6e11.json) | FOREST-16 | 2026-05-04 00:23:14 | SVALSAT | 475 | 0.556 | Contact GPS (6) | 18.412 | 1.232 |
| [556b1cfa-0a8e-4d19-8f95-b123fd47cd60](deliverables/contacts/556b1cfa-0a8e-4d19-8f95-b123fd47cd60.json) | FOREST-16 | 2026-05-04 02:35:53 | PUNTA_ARENAS | 401 | 0.882 | Contact GPS (18) | 21.147 | 3.008 |
| [2610bd5f-3a3f-4756-be38-d31392d4324c](deliverables/contacts/2610bd5f-3a3f-4756-be38-d31392d4324c.json) | FOREST-17 | 2026-05-03 20:03:32 | HARTEBEESTHOEK | 349 | 4.453 | Contact GPS (13) | 6.384 | 24.862 |
| [36c92afa-8853-4630-a788-ad2e63912d51](deliverables/contacts/36c92afa-8853-4630-a788-ad2e63912d51.json) | FOREST-17 | 2026-05-03 21:51:59 | TROLL | 370 | 1.246 | Contact GPS (17) | 7.167 | 3.838 |
| [d623db96-56fd-4783-9f1e-bf3933f4dbd8](deliverables/contacts/d623db96-56fd-4783-9f1e-bf3933f4dbd8.json) | FOREST-17 | 2026-05-04 01:02:54 | PUNTA_ARENAS | 344 | 11.569 | Contact GPS (13) | 8.305 | 4.468 |
| [49e74a6c-b21d-46c0-87bb-e4195b4a0dab](deliverables/contacts/49e74a6c-b21d-46c0-87bb-e4195b4a0dab.json) | FOREST-18 | 2026-05-03 17:07:14 | PUNTA_ARENAS | 471 | 8.299 | Contact GPS (28) | 9.917 | 10.767 |
| [48de3125-1fd3-40df-be87-5fc55aa2485f](deliverables/contacts/48de3125-1fd3-40df-be87-5fc55aa2485f.json) | FOREST-18 | 2026-05-03 20:15:20 | TROLL | 341 | 0.517 | Contact GPS (17) | 15.370 | 1.351 |
| [2735b79f-1ae2-4be6-9b13-774d026efd72](deliverables/contacts/2735b79f-1ae2-4be6-9b13-774d026efd72.json) | FOREST-18 | 2026-05-03 21:12:35 | SVALSAT | 336 | 5.297 | Contact GPS (5) | 14.452 | 4.779 |
| [7654ef67-7aba-4f18-bf0a-65eacc1b6a22](deliverables/contacts/7654ef67-7aba-4f18-bf0a-65eacc1b6a22.json) | FOREST-18 | 2026-05-03 22:47:18 | SVALSAT | 339 | 1.240 | Contact GPS (5) | 16.722 | 0.288 |
| [105f1f16-870e-4658-9a38-41c74ea5939c](deliverables/contacts/105f1f16-870e-4658-9a38-41c74ea5939c.json) | FOREST-18 | 2026-05-03 23:28:25 | TROLL | 322 | 14.807 | Contact GPS (14) | 19.570 | 1.372 |
| [d2af3474-5932-431d-ae42-c7ce901c93ba](deliverables/contacts/d2af3474-5932-431d-ae42-c7ce901c93ba.json) | FOREST-19 | 2026-05-03 18:40:28 | TROLL | 358 | 2.354 | Contact GPS (16) | 2.382 | 5.196 |
| [df2618ab-46b7-49bb-a148-ea9f89466869](deliverables/contacts/df2618ab-46b7-49bb-a148-ea9f89466869.json) | FOREST-19 | 2026-05-03 23:44:06 | AWARUA | 521 | 0.769 | Contact GPS (17) | 2.746 | 0.494 |
| [e629dfe8-8210-4af9-a61d-f02b5377a4fd](deliverables/contacts/e629dfe8-8210-4af9-a61d-f02b5377a4fd.json) | FOREST-19 | 2026-05-04 05:09:13 | SVALSAT | 357 | 0.823 | Contact GPS (6) | 3.097 | 1.841 |

## FOREST-16

Prior TLE:

```text
1 90916U 00000AAA 26123.38989170  .00000000  00000-0  15851-2 0  9996
2 90916  97.7445  21.6711 0001088 184.3194 124.1089 14.91831094    03
```

| Pass start (UTC) | Station | Duration | Measurements | Gate result | Fit | Offset (s) | Offset variance (s²) | Bias (Hz) | Doppler RMSE (Hz) | GPS reference | Prior accuracy (km) | New TLE accuracy (km) |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 2026-05-03 10:20:36 | AWARUA | 29:20 | 297 | Valid | Healthy | +0.719 | 2.098 | +693.479 | 17111.493 | Future GPS (+116–120 min) | 4.923‡ | 0.541‡ |
| 2026-05-03 12:56:20 | SVALSAT | 26:02 | 187 | Invalid: <250 measurements | Healthy | +14.123 | 381.081 | +1842.089 | 33754.940 | Contact GPS (23) | 5.727 | 101.115 |
| 2026-05-03 15:31:25 | PUNTA_ARENAS | 29:26 | 340 | Valid | Healthy | +3.770 | 2.540 | +975.138 | 23177.265 | Contact GPS (28) | 9.310 | 19.154 |
| 2026-05-03 17:50:03 | SVALSAT | 25:56 | 44 | Invalid: <250 measurements | At bound | -120.000 | 32874.227 | -10988.044 | 36752.176 | Contact GPS (13) | 11.099 | 918.009 |
| 2026-05-03 19:35:01 | SVALSAT | 15:52 | 252 | Invalid: variance >15 s² | Healthy | +2.464 | 38.887 | +974.388 | 20280.090 | Contact GPS (6) | 12.799 | 5.879 |
| 2026-05-03 21:39:48 | HARTEBEESTHOEK | 14:42 | 192 | Invalid: <250 measurements | Healthy | +12.149 | 222.750 | +1445.473 | 35462.819 | Contact GPS (15) | 15.158 | 76.666 |
| 2026-05-03 22:47:21 | SVALSAT | 17:41 | 110 | Invalid: <250 measurements | Healthy | -0.326 | 404.743 | +781.893 | 14230.258 | Contact GPS (5) | 16.206 | 18.673 |
| 2026-05-04 00:23:14 | SVALSAT | 17:43 | 475 | Valid | Healthy | +2.282 | 0.556 | +648.128 | 17676.768 | Contact GPS (6) | 18.412 | 1.232 |
| 2026-05-04 02:35:53 | PUNTA_ARENAS | 17:50 | 401 | Valid | Healthy | +3.198 | 0.882 | +729.813 | 28473.304 | Contact GPS (18) | 21.147 | 3.008 |

## FOREST-17

Prior TLE:

```text
1 90917U 00000AAA 26123.38823777  .00000000  00000-0  15851-2 0  9997
2 90917  97.7396  21.6764 0005261 314.5846 344.9937 14.90540572    06
```

| Pass start (UTC) | Station | Duration | Measurements | Gate result | Fit | Offset (s) | Offset variance (s²) | Bias (Hz) | Doppler RMSE (Hz) | GPS reference | Prior accuracy (km) | New TLE accuracy (km) |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 2026-05-03 11:27:41 | SVALSAT | 11:03 | 247 | Invalid: <250 measurements | Healthy | -2.419 | 37.255 | +5253.238 | 19730.504 | Future GPS (+22–26 min) | 1.000‡ | 17.335‡ |
| 2026-05-03 12:56:30 | SVALSAT | 26:09 | 213 | Invalid: <250 measurements | Healthy | +21.537 | 646.837 | +4048.593 | 34778.435 | Contact GPS (26) | 1.790 | 164.579 |
| 2026-05-03 16:58:11 | TROLL | 28:23 | 264 | Invalid: variance >15 s² | Healthy | -1.155 | 16.284 | -275.606 | 23381.285 | Contact GPS (8) | 3.436 | 5.323 |
| 2026-05-03 17:50:30 | SVALSAT | 26:05 | 51 | Invalid: <250 measurements | At bound | +120.000 | 45207.386 | +12918.439 | 36966.680 | Contact GPS (18) | 4.551 | 910.988 |
| 2026-05-03 20:03:32 | HARTEBEESTHOEK | 17:24 | 349 | Valid | Healthy | -4.136 | 4.453 | -929.899 | 33314.162 | Contact GPS (13) | 6.384 | 24.862 |
| 2026-05-03 21:51:59 | TROLL | 17:45 | 370 | Valid | Healthy | -0.441 | 1.246 | +71.973 | 17670.177 | Contact GPS (17) | 7.167 | 3.838 |
| 2026-05-03 22:48:07 | SVALSAT | 17:48 | 103 | Invalid: <250 measurements | Healthy | +0.160 | 818.467 | +130.618 | 28877.364 | Contact GPS (4) | 7.759† | 8.970† |
| 2026-05-04 01:02:54 | PUNTA_ARENAS | 15:47 | 344 | Valid | Healthy | -0.509 | 11.569 | +755.427 | 14586.761 | Contact GPS (13) | 8.305 | 4.468 |
| 2026-05-04 01:59:41 | SVALSAT | 17:44 | 62 | Invalid: <250 measurements | Healthy | +49.106 | 13087.969 | +1016.384 | 22072.714 | Contact GPS (5) | 9.783 | 380.889 |
| 2026-05-04 03:35:06 | SVALSAT | 17:39 | 240 | Invalid: <250 measurements | Healthy | -1.501 | 9.567 | +1050.683 | 17622.420 | Contact GPS (10) | 11.312 | 0.414 |

## FOREST-18

Prior TLE:

```text
1 90918U 00000AAA 26123.38910698  .00000000  00000-0  15851-2 0  9997
2 90918  97.7376  21.6797 0001897  93.2348 211.0047 14.92051430    08
```

| Pass start (UTC) | Station | Duration | Measurements | Gate result | Fit | Offset (s) | Offset variance (s²) | Bias (Hz) | Doppler RMSE (Hz) | GPS reference | Prior accuracy (km) | New TLE accuracy (km) |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 2026-05-03 12:22:00 | TROLL | 12:55 | 71 | Invalid: <250 measurements | At bound | +120.000 | 2173.769 | +78413.723 | 80281.781 | Contact GPS (13) | 2.903 | 907.964 |
| 2026-05-03 13:56:45 | PUNTA_ARENAS | 25:53 | 29 | Invalid: <250 measurements | Healthy | -4.217 | 2354.150 | +346.894 | 13701.901 | Contact GPS (26) | 4.506 | 27.333 |
| 2026-05-03 17:07:14 | PUNTA_ARENAS | 28:08 | 471 | Valid | Healthy | +0.112 | 8.299 | +944.281 | 23644.362 | Contact GPS (28) | 9.917 | 10.767 |
| 2026-05-03 17:50:01 | SVALSAT | 25:54 | 32 | Invalid: <250 measurements | At bound | -120.000 | 135242.081 | -13269.105 | 39024.931 | Contact GPS (23) | 9.511 | 897.739 |
| 2026-05-03 20:15:20 | TROLL | 17:49 | 341 | Valid | Healthy | -2.207 | 0.517 | +128.810 | 16343.500 | Contact GPS (17) | 15.370 | 1.351 |
| 2026-05-03 21:12:35 | SVALSAT | 12:03 | 336 | Valid | Healthy | -2.541 | 5.297 | +165.122 | 18503.206 | Contact GPS (5) | 14.452 | 4.779 |
| 2026-05-03 22:47:18 | SVALSAT | 17:40 | 339 | Valid | Healthy | -2.233 | 1.240 | -45.201 | 17022.154 | Contact GPS (5) | 16.722 | 0.288 |
| 2026-05-03 23:28:25 | TROLL | 16:21 | 322 | Valid | Healthy | -2.770 | 14.807 | -63.312 | 20128.507 | Contact GPS (14) | 19.570 | 1.372 |
| 2026-05-04 01:20:16 | AWARUA | 15:27 | 337 | Invalid: variance >15 s² | Healthy | -3.042 | 16.442 | +763.899 | 20334.488 | Contact GPS (15) | 21.386 | 1.592 |
| 2026-05-04 01:58:39 | SVALSAT | 17:36 | 170 | Invalid: <250 measurements | Healthy | -2.644 | 8.329 | -20.806 | 30956.947 | Contact GPS (9) | 21.445 | 1.498 |
| 2026-05-04 04:13:06 | PUNTA_ARENAS | 15:33 | 212 | Invalid: <250 measurements | Healthy | -4.118 | 54.415 | +728.369 | 22494.568 | Contact GPS (16) | 26.367 | 4.760 |

## FOREST-19

Prior TLE:

```text
1 90919U 00000AAA 26123.38960698  .00000000  00000-0  15851-2 0  9993
2 90919  97.7424  21.6734 0001942 141.5194 165.3859 14.92112300    07
```

| Pass start (UTC) | Station | Duration | Measurements | Gate result | Fit | Offset (s) | Offset variance (s²) | Bias (Hz) | Doppler RMSE (Hz) | GPS reference | Prior accuracy (km) | New TLE accuracy (km) |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 2026-05-03 10:40:55 | AWARUA | 01:50 | 70 | Invalid: <250 measurements | Healthy | +102.074 | 56912.076 | +242.161 | 11663.816 | Future GPS (+101–110 min) | 2.116‡ | 768.083‡ |
| 2026-05-03 12:22:00 | TROLL | 12:55 | 189 | Invalid: <250 measurements | Healthy | -0.312 | 3.991 | +1343.454 | 18934.551 | Contact GPS (6) | 2.107 | 4.457 |
| 2026-05-03 12:56:20 | SVALSAT | 26:00 | 148 | Invalid: <250 measurements | Healthy | -20.843 | 644.380 | -3779.380 | 37782.668 | Contact GPS (23) | 2.845 | 160.482 |
| 2026-05-03 15:23:04 | TROLL | 28:24 | 156 | Invalid: <250 measurements | Healthy | -68.017 | 92.126 | -18224.599 | 34063.029 | Contact GPS (28) | 3.298 | 516.609 |
| 2026-05-03 17:50:01 | SVALSAT | 25:55 | 275 | Invalid: variance >15 s² | Healthy | -0.238 | 57.941 | -432.216 | 18898.122 | Contact GPS (18) | 3.725 | 5.514 |
| 2026-05-03 18:40:28 | TROLL | 17:14 | 358 | Valid | Healthy | -0.377 | 2.354 | -85.525 | 17160.987 | Contact GPS (16) | 2.382 | 5.196 |
| 2026-05-03 21:11:15 | SVALSAT | 17:05 | 159 | Invalid: <250 measurements | Healthy | -1.537 | 24.329 | +95.805 | 17863.464 | Contact GPS (4) | 3.808† | 15.420† |
| 2026-05-03 22:47:18 | SVALSAT | 17:40 | 89 | Invalid: <250 measurements | Healthy | -16.917 | 4285.237 | -615.721 | 21251.291 | Contact GPS (8) | 3.869 | 131.827 |
| 2026-05-03 23:44:06 | AWARUA | 17:49 | 521 | Valid | Healthy | +0.303 | 0.769 | +648.238 | 17663.529 | Contact GPS (17) | 2.746 | 0.494 |
| 2026-05-04 01:58:39 | SVALSAT | 17:38 | 30 | Invalid: <250 measurements | Healthy | +12.562 | 95357.357 | -810.299 | 25498.948 | Contact GPS (2) | 4.305† | 90.735† |
| 2026-05-04 05:09:13 | SVALSAT | 17:37 | 357 | Valid | Healthy | +0.168 | 0.823 | +866.285 | 23159.140 | Contact GPS (6) | 3.097 | 1.841 |

† Accuracy is based on fewer than five GPS fixes from the recorded contact and is a limited-GPS result.

‡ No GPS was available during the recorded contact. Accuracy uses the first five later GPS fixes over the time range shown, so it is a forecast comparison rather than during-pass accuracy.

Fits marked **At bound** reached the ±120 s offset limit and failed the numerical health check. Their diagnostic values are retained but should not be treated as valid corrected TLE results.
