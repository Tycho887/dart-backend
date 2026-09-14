# Previous benchmark findings

These results belong to the retired FOREST studies, not benchmark defaults.
Historical source is available at Git revision `e5e851d`; publication manifests
also identify their own source snapshots. Reports and bundles remain under
[reports/forest-studies](../reports/forest-studies/README.md).

## Tuned optimizer settings

All settings below used six SGP4 orbit corrections plus one Doppler bias per
contact, unit observation variance, the existing orbit bounds/scales,
`x_scale="profile"`, and `max_evaluations=1000`. The historical runs also used
quality screening and a Doppler-only phase scan. Passing the optimizer settings
alone to the new benchmark does not reproduce that preprocessing/initialization.
Each search used all six matched anchors, 100 trials, seed 42; these are
**dataset-specific results, not holdout validation**.

| Passes | loss | loss_scale | ftol | xtol | gtol |
| --- | --- | --- | --- | --- | --- |
| 5 | cauchy | 962.0911382672281 | 8.024043405411047e-06 | 2.7504149061003033e-12 | 1.8066064221763204e-07 |
| 6 | soft_l1 | 340.2722778658969 | 9.488348606577822e-05 | 1.446437141665793e-12 | 9.751969812940085e-09 |
| 8 | soft_l1 | 443.84115662271665 | 2.1903756539591055e-05 | 2.1352005373497463e-12 | 4.012528472626066e-09 |

Mean local position RMS improved from 5.553/4.121/4.161 km to
4.664/3.709/3.798 km for five/six/eight passes, with independently reproduced
refits. Eight-pass mean forecast RMS worsened from 33.830 to 38.380 km and its
local count below 5 km fell from 5/6 to 4/6. Individual forecasts could regress
also for five and six passes. Additional cohort evaluations retained all 38
anchors in success denominators; they were not a holdout.

Source: [published tuning results](../reports/forest-studies/solver-tuning/README.md).
The earlier fixed-configuration study reached local RMS below 5 km on only
8/38 anchors; no configuration reached full zero-offset 48-hour RMS below 5 km.
See [trajectory findings](../reports/forest-studies/trajectories/FINDINGS.md).

## Reference products and contact inventories

Nominal frequency: **2,216,300,000 Hz**. These original inventories have 13, 15,
18, and 15 contacts; some lack usable Doppler or reference coverage. The benchmark
requires usable observations for every supplied contact, so choose an explicit
subset when needed.

The original May 3 noon–May 5 noon UTC reference OEMs have 2,881 one-minute
samples. FOREST-16/17/18 were accepted (withheld GPS RMS 82.9/35.0/56.1 m);
FOREST-19 remains a candidate (166.7 m). This validation metric does not bound
absolute accuracy throughout the OEM, particularly inside raw GPS gaps.
The separately extended forecast references also retained candidate assessments
for FOREST-16/18/19. Preserve the quality reports alongside reference products;
the new benchmark records identity and hashes, not an inferred accuracy rating.

The OEMs use FOREST names as OBJECT_ID. Pass the listed spacecraft UUID as
`reference_spacecraft_id` to bind them explicitly. Priors are selected separately
from the contact-associated ephemerides; the reference never initializes a fit.

### FOREST-16

- Spacecraft: `ebc4af2e-c5f5-4700-a103-d1fc1bf423bb`
- Initial ephemeris: `23f53a9c-7307-43d0-844d-29934384ef0c`
- OEM: [FOREST-16.oem](../reports/forest-gps/20260504/FOREST-16/FOREST-16.oem)

Contact IDs, chronological order:

```text
a6252615-8c19-4f78-a6f0-227e3da97c2e
b2bd28a8-6704-4a67-bd8b-2c84fb0473bc
9adbc5c4-0b67-471e-a286-dda9a089a49f
7815ef69-da31-40ef-853c-e552ebe841cc
1c36ca34-a985-4809-9ca7-65e422e6c47d
e5d79f0b-730b-4832-becf-f716d7322784
bcec2df5-c0d0-4e89-91ae-f0568424de8a
82723fb0-59f4-4de8-ac05-f9f3d9a7dcc7
7cb89272-0ba6-4eaf-8445-917ec34e727e
2a45c0d3-c3e9-45e8-a11f-5dbf6bda0693
33061519-fa11-49e4-becd-6c770efd6e11
85ec563d-ecb0-4c0b-9b2d-eeb9cec80587
556b1cfa-0a8e-4d19-8f95-b123fd47cd60
```

### FOREST-17

- Spacecraft: `81515c13-627d-4a0a-bc94-50ec0e9231cd`
- Initial ephemeris: `a79ad36d-0ea1-4b6a-9309-3cb603d0ae6e`
- OEM: [FOREST-17.oem](../reports/forest-gps/20260504/FOREST-17/FOREST-17.oem)

Contact IDs, chronological order:

```text
c12b9960-e925-4286-9cd1-6a887d4e8289
75d4144d-d3fc-4e18-8f14-b17da65092cb
118e1a7d-4ae2-4982-a40b-7d6eb0ead2e4
dc06b459-269d-446a-97ef-9b6febcd8679
0a2e4679-2976-4bed-bd7a-ffd4f45e0a60
6569e3a1-7aa1-4ff3-a2c0-cd1cd88fd85a
4ef3b909-6d7f-460d-bf48-41aed65abf0a
5233c319-41fd-4896-847f-112ddfe9c56d
74df0163-cf66-41f9-b960-409fca756c93
2610bd5f-3a3f-4756-be38-d31392d4324c
36c92afa-8853-4630-a788-ad2e63912d51
d61bfc77-1d1c-4e82-91fd-6e7489e73cb3
d623db96-56fd-4783-9f1e-bf3933f4dbd8
01448cbd-3659-4184-89c2-7883e4a3c6c7
9303555b-beda-4b39-ac33-b6df9a7e9952
```

### FOREST-18

- Spacecraft: `b221e85e-a0d5-44bd-abd2-930ff3e849b5`
- Initial ephemeris: `52e670bc-e434-46df-b206-bebe4068bbd0`
- OEM: [FOREST-18.oem](../reports/forest-gps/20260504/FOREST-18/FOREST-18.oem)

Contact IDs, chronological order:

```text
53eafac8-7eae-4d9b-8adc-9066c668410c
3ccef2f9-751b-4153-b810-267dd4c64a31
ca8d1c55-2725-4217-bc02-13672ceec9ae
a1bdfcbb-ff7d-47af-8ba1-7fb37ee15e65
6d7785df-a7a7-4963-8fcc-bdebf3a6dc80
f1550be3-6ca0-48b2-bd87-da21c1e885b6
a5dc9c5c-0fb7-4bc9-9ba3-89a37ff6fac1
b2015720-f0bb-48e7-b19a-131a592c9501
49e74a6c-b21d-46c0-87bb-e4195b4a0dab
6c22b929-f451-4e3f-bc5d-fc4e230c6c02
5f62261f-f193-475b-ad2d-94a8b9fd0e75
48de3125-1fd3-40df-be87-5fc55aa2485f
2735b79f-1ae2-4be6-9b13-774d026efd72
7654ef67-7aba-4f18-bf0a-65eacc1b6a22
105f1f16-870e-4658-9a38-41c74ea5939c
8db51bff-de04-4287-b3c8-4f949e7a8df1
7f6f30f5-0820-469d-9ab6-71b9523e8939
4d451626-a662-4148-8369-abbcd6001b6a
```

### FOREST-19

- Spacecraft: `e72db085-5680-457c-ab1b-5b2f8e2d739c`
- Initial ephemeris: `378baf66-330b-4c28-a127-84659699e638`
- OEM: [FOREST-19.candidate.oem](../reports/forest-gps/20260504/FOREST-19/FOREST-19.candidate.oem)

Contact IDs, chronological order:

```text
4a452e93-b302-4029-b583-7e0e061cbc64
827d2f56-abc5-46cf-97b9-bc8883c9acca
dc8c6112-48a4-4713-8b24-daa04ac09048
28e194fe-197d-4ec1-b4e4-c206bc5ebb54
7825d181-49f4-46e5-9bfe-0e9a99500190
a22fbafe-9e67-4eb3-b02b-7f108aa1b80b
f80c6227-6cc9-441f-ae4f-13c37b361b00
2e914640-332d-4846-8126-0c7d58e2a568
399cf06a-3730-4818-8f75-270388e8fd0a
d2af3474-5932-431d-ae42-c7ce901c93ba
5d035d45-52f9-4d77-86fa-7e4917f23992
8c5807c9-daa1-4957-a3a5-2711acfae882
df2618ab-46b7-49bb-a148-ea9f89466869
c862d412-bfa6-4ff1-bd12-935e690521fb
e629dfe8-8210-4af9-a61d-f02b5377a4fd
```
