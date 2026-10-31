# Contributing

Bugs, improvements and documentation gaps are tracked as
[GitHub Issues](https://github.com/aidan-epperly/ncalg/issues).  File them
with the issue forms on the website (*Bug report*, *Improvement*,
*Documentation*), or from the command line with `scripts/issues.py`, which
writes **one issue file** to [`issues/`](issues/README.md) and mirrors it to
GitHub.  The files work offline and travel with the repository.

Downstream projects depend on ncalg as a path or git dependency.  A problem
found there is reported here, with a reproduction that does not need the
downstream project.

The current list of local issue files: [`issues/README.md`](issues/README.md),
or `python scripts/issues.py list`.

## Filing

From the ncalg directory (any Python 3, no packages needed):

```bash
python scripts/issues.py new bug "Relaxation::try_build rejects ..." --area elimination --severity 2 --reporter "Your Name"
python scripts/issues.py new improvement "Warm start for the native IPM" --area solver
python scripts/issues.py new docs "README does not say ..." --area docs
```

The command prints the new file's path; fill in the template sections.  It
records the current commit.  Add `--needs-repro` for something seen once and
not reproduced.  Pass a finished body with `--body-file report.md`.

A good report has:

- **the smallest reproduction**, self-contained: a few lines of Rust, a
  command, or a small input file attached to the issue;
- **the value a trusted method gives** next to the wrong one (exact
  diagonalization, SCS/CSDP on the moment form, a published number);
- the suspected location (`src/ncsos.rs:1427`) when known.

One problem per issue.  A follow-up observation on an open issue goes into
that file under a dated `## Addendum (YYYY-MM-DD)` heading; a new problem gets
a new issue that links the old one (`#N`).

### Severity (bugs)

| | meaning | example |
|---|---|---|
| **S1** | wrong result without an error | an "upper bound" below the true value |
| **S2** | crash, panic or error on valid input | `try_build` declares a feasible system inconsistent |
| **S3** | performance or usability | symmetry reduction silently skipped, 400 s instead of seconds |

S1 comes first: a silent wrong number in a paper is worse than a panic.

### Areas

`groebner`, `reduction`, `relaxation`, `elimination`, `symmetry`, `solver`,
`export` (SDPA / moment-form files), `gns`, `scripts` (Python/Julia solver
scripts), `examples`, `docs`, `build`.  Comma-separate several.

### Status

`open` → `fixed` / `wontfix` / `duplicate`; `needs-repro` for unconfirmed reports.

## Fixing

1. Add a regression test that fails before the fix (`tests/bug_report_tests.rs`
   or the area's test file).
2. Reference the issue in the commit message; a GitHub closing keyword
   ("Fixes" and the issue number) closes it when the commit reaches `main`.
3. Record the fix locally:

   ```bash
   python scripts/issues.py close 12 --commit abc1234 --note "One sentence on what the fix does."
   ```

   This appends a `## Resolution` section, sets `fixed_in`, regenerates the
   index, and closes the GitHub issue too when `gh` is available.  If the issue
   was closed on GitHub first, `python scripts/issues.py pull` copies that back.

## Performance

Correctness is guarded by tests.  Speed is guarded by a few cheap habits and
one script, so that a slowdown is caught when it lands.  In one such case, an
accuracy fix to the presolve made the swap-algebra (Quantum Max Cut)
relaxations 5–10× slower: the presolve took 57 s of a 63 s solve.  No other
problem in the test set slowed down.

1. **Cost note.**  An issue or commit that changes a step every solve or
   build runs (presolve, elimination, per-iteration kernels, relaxation
   build passes) states the old and new cost in the problem's parameters:
   `n` (block size), `m` (variables), `nnz`, number of blocks, stacked rows.
   It also names the problem family that is worst for the new cost.  One
   line, e.g. "QR of the stacked F rows: O(rows · n²), rows ≈ m · n when the
   F_j share positions (QMC)".
2. **Canary.**  After changing `src/sdp/`, `src/linalg/`, the relaxation
   build (`src/ncsos.rs`, `src/symmetry/`, `src/reduce.rs`, `src/groebner.rs`)
   or the Schur and Newton code, run

   ```bash
   python scripts/canary.py
   ```

   It builds and runs `examples/canary.rs` (release), compares every
   instance's counters and phase times with this machine's baseline
   (`canary/baseline-<host>.json`, a local file that is not committed, since
   timings only compare on one machine), and exits non-zero on a regression.
   Counters (iterations, operator products, factorization retries, kernel
   searches) are deterministic, so any change in them is reported.  Times are
   flagged above 1.5× the baseline and 0.2 s.  A certified bound that differs
   from its solve's value is a regression, and two instances
   (`rescaled_blocks`, `tiny_localizer`) are rescaled copies whose values must
   match the unscaled problems.  `cargo run --release --example canary --
   --rescale` re-solves every instance under random block and variable scales
   (`RESCALE_MODE=blocks|vars`, `RESCALE_ONLY=<instance>`); run it after
   touching tolerances or scaling.  Paste its summary into the commit message
   or pull request.  For an intended change, run `python scripts/canary.py
   --update` and say why in the commit.  On a new machine, the first run (or
   `--update`) writes the baseline; run `--update` on the commit before your
   change to have something to compare with.
3. **Warnings.**  `SdpModel::solve` warns when one-off work takes more than a
   quarter of a solve and over a second (`SolveProfile::one_off_warning`).
   Such a warning is worth an issue unless the cause is known.
4. **Tolerances.**  A threshold that decides structure (a rank, a kernel,
   a dropped entry, consistency) is relative to the data it tests: the
   summands of a sum, the norm of the matrix, the size of the right-hand
   sides.  An absolute 1e-12 next to unnormalized data is a bug waiting for
   a problem in other units.  Name the scale in a comment.
5. **Cross-solver comparison.**  Before a release, export a few larger
   problems (`write_sdpa`, `write_json`) and compare the native solver's
   times with CSDP and SCS through the scripts in `scripts/`.  The canary
   only compares ncalg with itself; the presolve slowdown above was found
   this way.

Not wanted: timing assertions in `cargo test` (they flake on a busy
machine), or a performance test per feature.

## GitHub mirror

```bash
python scripts/issues.py labels   # once: create the labels
python scripts/issues.py push     # open GitHub issues for files without one
python scripts/issues.py pull     # copy open/closed state back from GitHub
python scripts/issues.py sync     # push, then update text and state of mirrored issues
```

An issue that should stay local (for example one already reported upstream
by other means) gets `github: local` in its header: `push` (even when named
explicitly), `pull` and `close` then never touch GitHub for it, and the index
shows "local".

Issues filed on the website are labeled automatically from their Severity
and Area answers by `.github/workflows/issue-labels.yml`.  They do not get a
local file; that is fine, GitHub is the shared view.

Before committing a change under `issues/`, run `python scripts/issues.py
check`: it rejects malformed files and a stale `issues/README.md` (fix with
`python scripts/issues.py index`).
