# Synthetic state-model ablation

**Evidence/status:** synthetic model-identifiability experiment. Both models fit two synthetic passes and are scored on a third. Doppler and idealized complete Henault-style phase-difference trials share the same truth; this is not a real-data accuracy or operational-superiority claim.

| Truth family | Channels | Fitted model | Trials | Median held-out error (km) | Worst trial median (km) |
| --- | --- | --- | ---: | ---: | ---: |
| offset | doppler | time_offset | 3 | 0.128 | 0.234 |
| offset | doppler | mean_anomaly_motion | 3 | 0.165 | 0.265 |
| offset | doppler_phase | time_offset | 3 | 0.004 | 0.007 |
| offset | doppler_phase | mean_anomaly_motion | 3 | 0.231 | 0.272 |
| mean_elements | doppler | time_offset | 3 | 20.298 | 20.444 |
| mean_elements | doppler | mean_anomaly_motion | 3 | 0.109 | 0.307 |
| mean_elements | doppler_phase | time_offset | 3 | 21.194 | 21.293 |
| mean_elements | doppler_phase | mean_anomaly_motion | 3 | 0.176 | 0.177 |
