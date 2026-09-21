# Experiment archive

`main` contains DART, the service and dashboard, deployment files, all current
study and benchmark tools, and their tests. Recorded GPS, Doppler, reference
OEMs, and quality metadata remain available where the tools expect them.
Generated experiment products and historical reports live separately.

## Results branch

The local branch `archive/forest-v5-v7` preserves source and reports at the
cleanup boundary. Its archive commit is
`49cb3c6ec9762135254e702417a028865a0d1306`; the shared source snapshot is
`95a6c6420b957405a4c6f140f4364effd62316a3`.

Read its index without switching branches:

```bash
git show archive/forest-v5-v7:ARCHIVE.md
git show archive/forest-v5-v7:archive-validation.md
git show archive/forest-v5-v7:report/final.md
```

For browsing the results alongside main, create a separate checkout:

```bash
git worktree add ../dart-study-results archive/forest-v5-v7
```

The archive branch has not been pushed. Its compact V5–V7 tables, plots,
metadata, final collection-window report, and selection provenance retain
their original paths under `raw_results/` and `report/`. The earlier
deadline-based report is explicitly superseded. Archived measurements and
KPI examples from `docs/math.md`, older study summaries, campaign papers, and
the retired solver demo remain on that branch too.

## External artifacts

The granular artifacts are in `/home/michaeljohansen/DART/data/forest-v5-v7`,
or `../data/forest-v5-v7` from this checkout. This is local storage outside Git.
It contains complete V5, V6, and V7 result directories plus the small frozen
input cache, preserving their original relative paths:

```text
raw_results/forest-experiment-v5/
raw_results/forest-experiment-v6/
raw_results/forest-experiment-v7/
experiments/results/forest-inputs/
```

The external `archive-manifest.json` records the source revision, sizes, and
SHA-256 checksums for all 5,957 files (2,091,449,780 bytes). The same manifest
and checksum list are committed on the archive branch. Verify the external
copy before use:

```bash
(cd ../data/forest-v5-v7 && sha256sum --quiet -c SHA256SUMS)
```

Restore only the artifacts needed by a command, into paths that do not already
contain local results. For example, the V6/V7 runners accept the external V5
bundle directly as their positional `bundle` argument. To use the original
default paths, copy the saved directories back into the ignored locations:

```bash
mkdir -p raw_results experiments/results
cp -an ../data/forest-v5-v7/raw_results/. raw_results/
cp -an ../data/forest-v5-v7/experiments/results/forest-inputs experiments/results/
```

The optional frozen-inventory test skips when `forest-inputs` has not been
restored. Ordinary runtime and synthetic benchmark tests need no external
archive. New experiments continue writing to the existing ignored output
paths; no runner or public interface was changed by cleanup.

## Historical reproduction

Use each artifact's frozen `source/` and recorded dependencies for historical
reproduction; V5's source is inside its ZIP. V6/V7 also include `v5-source/`
for their earlier input provenance. The recorded Git revision alone may omit
working-tree edits included in those frozen snapshots. Keep the originals
unchanged and rebuild into a separate directory.

The archival checks rebuilt all V5–V7 reports without refitting. Their numerical
tables matched within absolute/relative tolerance `1e-12`. V6/V7 used a native
binary matching their recorded hash. V5's native hash differs from the installed
binary, so its successful report rebuild is a compatibility check rather than
an exact reconstruction of its original numerical environment. See the archive
validation record for details and retained provenance limits.

Older ignored raw results, V6 initial checkpoints, and redundant run caches
were intentionally discarded after verifying V5–V7. Old tracked material
remains recoverable from the archive branch and Git history. Some studies
already removed before this cleanup are documented at revision `e5e851d`;
their former local result links are not evidence that those artifacts are
included in the V5–V7 archive.
