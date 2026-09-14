# FOREST study publications

These are historical publications. The study, reporting, and restoration scripts
were retired during benchmark consolidation. Run the commands below from the
historical Git revision `e5e851d` (for example, in a separate worktree); each
publication also records its original source/runtime requirements. The bundles
and their individual files remain unchanged. For new work, use
[the contact-list benchmark](../../docs/benchmark.md); selected findings and exact
tuned settings are retained in [notes](../../experiments/notes.md).

These compact publications preserve the completed FOREST studies in Git. The
original generated directories contained thousands of files, mostly reproducible
trajectory arrays and repeated inputs. Each publication provides one essential
bundle plus readable summaries and figures. No fitting algorithm changed.

| Publication | Original files | Original bytes | Essential bundle bytes |
|---|---:|---:|---:|
| [Acquisition](acquisition/README.md) | 108 | 30,513,695 | 4,255,912 |
| [Independent passes](pass-accuracy/README.md) | 2,987 | 34,664,835 | 6,206,324 |
| [Forecast reference](forecast-reference/publication.json) | 105 | 159,152,444 | 3,134,458 |
| [Trajectories](trajectories/README.md) | 11,114 | 703,265,934 | 22,873,578 |

A subsequent [solver-settings study](solver-tuning/README.md) ran 100 trials each
for five, six, and eight passes. On the same six anchors, mean local RMS improved
from 5.553/4.121/4.161 km to 4.664/3.709/3.798 km, with exact independent refits.
These are dataset-specific results. Eight-pass mean forecast RMS worsened, and
its local count below 5 km fell from 5/6 to 4/6. The publication includes all
trial records, per-anchor comparisons, and the additional 38-anchor evaluation.

The completed studies total 14,314 original files. Including the two interrupted
trajectory attempts listed in [pruning.json](pruning.json), the cleanup covers
15,425 files and 943,927,110 bytes. The 26 publication files occupy approximately
38.5 MB, a 96% reduction in the active artifact footprint. The completed deletion
record is [cleanup.json](cleanup.json). Git history retains the original compact
publications; discarded caches and interrupted fit outputs are not archived.

## What the results establish

The original target was not reached: the best fixed configuration achieved local
position RMS below 5 km on **8/38** anchors. On the six matched anchors, six-parameter
five/eight-pass fits achieved median local RMS **4.120/3.680 km** and 48-hour RMS
**22.930/27.908 km**. No configuration achieved a full zero-offset 48-hour RMS below
5 km. Post-fit time shifts did not explain most single-pass error.

See [the findings](trajectories/FINDINGS.md), [comparison](trajectories/comparison.csv),
and [the next multipass study](../../docs/multipass-run-study.md). The next study
will compare complementary information at equal pass counts, observation span,
measurement weighting, and forecast dynamics. It will measure errors against GPS;
calibrated production uncertainty is deferred.

## Preservation and replay

Bundles retain exact input bytes, contact metadata, priors, orbit descriptors,
optimizer outputs/profiles, selection/failure records, score tables, timing curves,
tuning results, and reference observations/OEM/quality assessments. Repeated bytes
are stored once by SHA-256. Shared inputs resolve through sibling publications;
tracked source and local GPS bytes resolve through Git revision
`26bc425f13a5f39ae3da0e8fa7384803ecdd39c3`.

Historical source versions that differ from that revision remain in the bundles.
Native-runtime hashes and package versions are preserved; native binaries are
rebuilt from source rather than archived. The recorded runtime is still available
in this workspace and was used for publication verification. Git history and all
sibling publication directories are required for a fresh restoration.

```bash
uv run python -m experiments.study_artifacts verify reports/forest-studies/trajectories
uv run python -m experiments.study_artifacts restore \
  reports/forest-studies/trajectories /tmp/forest-restored
uv run python -m experiments.verify_trajectory_publication /tmp/forest-restored
```

Restoration requires a new destination. It verifies bundle, dependency, Git-source,
and readable-summary checksums before writing. It restores original relative paths
and materializes shared inputs automatically. It needs no acquisition providers or
original result directories. Reference relocation preserves checksum enforcement.

To generate fresh trajectories, timing sweeps, and CSVs without fitting again:

```bash
uv run python -m experiments.trajectory_report --study /tmp/forest-restored \
  --reference /tmp/forest-restored/forecast-reference
uv run python -m experiments.trajectory_diagnostics --study /tmp/forest-restored
```

The restored study omits completed evaluation-cache markers along with the state
arrays, so replay regenerates those arrays. Raw JSON score records remain available
before replay. Historical README links describe the original expanded layout;
generated CSVs and arrays become available after replay.

For a fresh fit comparison, restore `acquisition` and `pass-accuracy` similarly,
then supply their paths to `experiments.forest_trajectories --archive ... --previous
...`. Use a new output directory. Each `publication.json` records the exact export
exclusion patterns. `python -m experiments.study_artifacts export --help` documents
publication options for future studies.

## Verification and interpretation

[Replay verification](replay-verification.json) covers all 912 comparison rows,
1,173 scored local/forecast windows, unchanged outcome counts, and four timing
sweeps. Nominal position differences were exactly zero under the recorded runtime;
the allowed tolerance was 1 mm. Recorded unavailable windows were also reproduced.

The full test and checksum results are recorded in [validation.json](validation.json).
FOREST-19's local reference and FOREST-16/18/19's forecast references retain their
candidate assessments. Raw GPS gap accuracy remains unverified. Neither FIM
conditioning nor GPS-assisted timing alignment is a calibrated uncertainty bound.
