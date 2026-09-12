# What explains the high error?

The strongest evidence points to weak single-pass orbit constraints, with large
remaining measurement outliers. Post-fit subsecond alignment alone does not remove
most of the error. This diagnostic does not rule out a timestamp issue inside
Doppler acquisition or fitting. Forecast drift remains a separate limitation after
local accuracy improves.

## Timing

The complete −1…+1 s sweep reduced robust L-only median local RMS from **16.892
to 15.291 km**, and L+n from **20.333 to 18.812 km**. Neither configuration gained
a pass below 5 km. Fixed positive shifts of 0.350 and 0.707 s worsened both medians.

Individual optimal offsets span both sweep boundaries. The median offsets are
0.094 s for L and 0.206 s for L+n; ten of 31 fits in each configuration prefer a
boundary. These GPS-assisted alignments cannot identify a universal clock bias.

The median cross-track error component is **12.851 km for L** and **12.804 km for
L+n**. Moving along the fitted trajectory cannot substantially remove that error.
All archived tracking epoch offsets are zero. GPS references already use receiver
GPS time; packet latency must not be applied as an additional correction.

## Multipass improvement and remaining forecast error

The six-parameter model improves substantially with more contacts. On the six
fixed anchors with eight usable historical contacts:

| Strategy | Local median RMS, km | 48-hour median RMS, km | Local <5 km |
|---|---:|---:|---:|
| Single pass | 237.908 | 2308.277 | 0/6 |
| Latest 3 | 106.172 | 1720.258 | 0/6 |
| Latest 5 | 4.120 | 22.930 | 3/6 |
| Latest 8 | 3.680 | 27.908 | 5/6 |
| Condition selection | 6.363 | 25.405 | 2/6 |
| Trace selection | 5.243 | 28.630 | 3/6 |

Single-pass medians use five converged fits; its sixth fit failed. Other rows
have six scored fits. Failures remain in all success denominators.

Across the original 38 anchors, the best fixed success count is **8/38**, from
six-parameter five-pass fitting. That configuration had only 18 usable windows.
Eight-pass fitting scored **5/38**, with only six usable windows. The original
19/38 target was therefore **not achieved**.

Individual-pass information ranking did not beat the latest five/eight contacts
in this pilot. Ranking isolated passes does not explicitly reward complementary
geometry, and selection retains fewer contacts. The current results do not
separate these two effects.

The forecast improvement is real, but no fixed configuration has a complete
48-hour RMS below 5 km for any anchor. Applying each local optimal time offset
to its forecast has limited benefit: for eight-pass six-parameter fits the median
forecast RMS changes from 27.908 to 29.016 km. Unrestricted interpretation as a
clock calibration would be misleading.

Unweighted Doppler residual RMS remains around 8 kHz even in useful fits. Soft-L1
reduces outlier influence; it does not remove those observations or prove the
remaining measurement model is correct. The full orbit information is severely
conditioned in short passes despite numerical full rank.

## Next tests supported by the reusable implementation

1. Replay a larger historical inventory so five/eight-pass configurations have
   enough anchors to meaningfully test the original 19/38 goal.
2. Compare joint-information or geometry-aware selection with the current
   individual-pass rankings, controlling for retained contact count.
3. Diagnose residual mean-motion/drag and other model errors over the forecast,
   together with telemetry outliers. B* and other corrections were fixed in this
   study; the observed drift does not by itself identify which correction is needed.

Local references retain the original assessments, including the FOREST-19
candidate. Extended forecast references for FOREST-16/18/19 remain candidates
with withheld RMS 130.3/125.1/135.0 m and failed convergence checks; FOREST-17
passed at 62.8 m. Accuracy in raw GPS gaps is unverified. These limitations remain
in every reference artifact and should accompany forecast comparisons.
