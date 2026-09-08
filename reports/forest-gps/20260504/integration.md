# Integration with remote main

The `forest-experiment` branch preserves the original GPS implementation in
`c2e9244` and merges the 29 incoming commits through `origin/main` at `ba5e584`.
The README and ignore-rule conflicts retain both main's new architecture and
the GPS workflow. BESTXYZ provider access moved to `dart.io.gps`, and OEM
product validation moved to `dart.oem`; compatibility imports preserve the
previous GPS import paths. Main's Rust implementation is `crates/forward-models`.

## Verification

- `uv sync --locked` rebuilt main's Python/Rust extension.
- The merged Python suite passed: 177 passed, one optional GMAT test skipped.
- The five snapshot checks separately verify all payload checksums, source
  hashes, screened observations, complete holdout residuals, OEM grids, and
  accepted/candidate status without requiring GMAT.
- `cargo test --manifest-path crates/forward-models/Cargo.toml`: 25 passed.
- The opt-in real GMAT synthetic recovery and export-resume test passed.
- Ruff passed for `dart/io`, all tests, and the GPS adapter, CLI, compatibility
  imports, and OEM module. `uv run ty check dart/io` passed.
- All four saved observation arrays and rejection lists match a fresh load
  exactly. Accepted products revalidated against the native propagation
  reports; FOREST-19 reproduced the same failing quality report.
- OEM bytes remain unchanged. No acceptance criteria or numerical settings
  were relaxed during integration.

## Complexity review

The repository-required cyclomatic-complexity skill was applied to the GPS
changes. The table uses a consistent manual AST decision count (base one,
branches/loops/exception handlers/conditional expressions, Boolean alternatives,
and comprehension loops/filters), not a Radon score. Focused tests passed before
and after the helper extraction. Public call signatures were preserved.

| Function | Before | After |
|---|---:|---:|
| `load_bestxyz` | 51 | 12 |
| `runtime_manifest` | 18 | 6 |
| `assess_stage` | 19 | 10 |
| `read_oem` | 20 | 11 |
| `validate_oem` | 17 | 1 |
| CLI `main` | 13 | 9 |

Extracted helpers separate receiver-time resolution, uncertainty screening,
duplicate selection, runtime coverage, metadata checks, and native state
report parsing. Each new helper scores at most 10. `load_bestxyz`, `read_oem`,
and the unchanged `write_script` (11) remain worth watching if expanded;
their remaining branches express input assembly or file syntax.
