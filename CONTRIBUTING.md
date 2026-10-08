# Contributing to medh5

medh5 is two things kept in step: a **format** — the normative
[specification](docs/spec/medh5-1.0.md), its JSON Schema and its conformance
corpus — and an **implementation**: one format engine in Rust, with three
frontends over it (the Rust crate, the Python package and the `medh5` command
line). Most of what follows is about keeping the format, the engine, its
frontends and the documentation that describes them from drifting apart.

## Setting up

You need a Rust toolchain (stable; the minimum is `rust-version` in
`Cargo.toml`), a C compiler and CMake — the engine builds HDF5 and C-Blosc2
from source and links them statically — and Python 3.10 or later.

```bash
git clone https://github.com/XwK-P/medh5.git
cd medh5
# The Python package, with the engine compiled into it by maturin (release
# profile). Re-run after changing Rust.
pip install -e ".[dev,torch,nifti,dicom,dicomseg,itk]"
pip install -r docs/requirements.txt        # to build the documentation site
pip install pre-commit && pre-commit install   # optional: ruff on every commit
```

MONAI is left out of that line on purpose: it is heavy and only a handful of
tests need it. They skip without it, and CI runs them in a job of their own:

```bash
pip install -e ".[monai]"
pytest tests/ -k "Monai or monai"
```

## The checks

Every change has to pass all of these; CI runs the same gates.

```bash
cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings \
  && cargo test --workspace \
  && ruff check . && ruff format --check . && mypy medh5 \
  && pytest tests/ --cov=medh5 --cov-fail-under=90 \
  && medh5 conformance run /tmp/corpus \
  && cargo run --release -p medh5-cli -- conformance run /tmp/corpus-native \
  && mkdocs build --strict
```

- **Rust** — `rustfmt` (`rustfmt.toml`), `clippy` with `-D warnings`, and the
  engine's tests: unit tests beside the code, the conformance corpus in
  `crates/medh5/tests/`, and the crate README's example as a doctest.
- **Lint and format** — `ruff`, pinned in the `dev` extra and in
  `.pre-commit-config.yaml`; bump the two together.
- **Types** — `mypy --strict` over `medh5/`. The engine's Python types are
  `medh5/_core.pyi`; `tests/project/test_typing.py` runs `stubtest` against the
  built module, so a binding change without its stub fails.
- **Tests** — `tests/`, with a 90 % coverage floor: `format/`, `tools/`,
  `integrations/` and `project/`, each module named for what it tests.
- **Conformance** — the corpus is a shipped artifact that third-party
  implementations run, not a test fixture. It must stay green through both
  command lines.
- **Documentation** — `mkdocs build --strict` fails on any broken link or
  anchor, including links into the specification by clause.

CI also builds the wheel and the native binary for every platform the project
ships — Linux x86_64 and aarch64 (manylinux_2_28), macOS arm64 and x86_64,
Windows x64 — and runs the conformance corpus through each. It runs the suite
against the built wheel on Python 3.10–3.14, on macOS (for the `spawn` start
method), on Windows (for the atomic-replace paths) and at the NumPy floor;
checks the minimum Rust version; builds the sdist from source; and runs the
reference writer in `docs/examples/` to keep the specification's Appendix C.2
honest.

## Writing tests

- Name a test after the clause it holds: `test_S8_1_boxes_shift_by_half_a_voxel`,
  `s14_4_the_source_is_closed_before_the_replace`.
- Put a test with what it tests, not with the release that fixed it: the module
  for that part of the spec, tool or integration. A regression test cites its
  finding in its name; the changelog says which release fixed it.
- Build fixtures with the **public writer**, so every reader test is also a
  writer test: shared builders are in `tests/helpers.py`, fixtures in
  `tests/conftest.py`, and the small samples the regression tests use in
  `tests/kits.py`. Plant defects with `h5py`, a test dependency only.
- Engine internals with no Python door are held by Rust tests beside the code;
  the Python test keeps the observable outcome.
- Guard an optional dependency with `pytest.importorskip` — and make sure some
  CI job installs it. A skipped test is not a passing test.
- A bug fix comes with the test that would have caught it.

## Changing the format

`docs/spec/medh5-1.0.md` is **normative**: the code implements it, and when the
two disagree one of them is a bug. Decide which before changing either. Engine
modules map onto specification sections (`crates/medh5/README.md` lists them),
so a clause has one obvious home in the code.

- **A clause that turned out to be unimplementable**, ambiguous, or more than
  the implementation can honestly promise is corrected in the text *and*
  recorded as a row in Appendix C.1. A test counts the rows against the sentence
  that states how many there are.
- **Diagnostic codes are stable API.** They live in one table,
  `crates/medh5/data/codes.json`, which the validator, every frontend and the
  documentation read; a test asserts that it and the §15.2 table list the same
  codes. A code's meaning never changes, and a retired code is never reused. A
  new code needs a conformance case in `crates/medh5/src/conformance/build.rs`;
  a test fails until it has one.
- **Versioning follows §16**: a minor format version may add objects, kinds,
  profiles and codes; it may not change what an existing one means.
- The JSON Schema lives in one place, `crates/medh5/data/`, beside the code
  table: the engine embeds it, §2.4 names it, the documentation site publishes
  it, and `medh5 conformance publish` writes it into the suite.

## Writing documentation

The site is [MkDocs](https://www.mkdocs.org/) with the Material theme, laid out
as **tutorials** (learning), **how-to guides** (tasks), **reference** (what
everything is), **explanation** (why) and the **specification**. Put a page
where its reader will look for it, and link rather than repeat.

Documentation here is checked against the code, not proofread:

- **Every `python` block runs.** `tests/project/test_docs_python.py` executes
  each one against a real sample, with the names the pages use by convention
  (`s`, `w`, `ann`, `paths`, …) already bound. A block that genuinely cannot
  run — it needs a DICOM series, or it elides arguments with `...` — carries an
  `<!-- illustrative -->` comment on the line above its fence, where readers
  see it too. Most blocks must stay runnable.
- **Every documented CLI flag exists.** `tests/project/test_docs_examples.py`
  parses each `medh5 …` line in a fenced block against the command line's
  grammar.
- **Corrected claims stay corrected.** The same file keeps a list of statements
  that were once wrong in the documentation; add to it when you fix a claim
  that could plausibly come back.
- **Counts are checked.** Where a page states the size of the conformance
  corpus or its breakdown, a test compares it with the corpus.
- **Measurements say how they were taken.** A number on a page comes from a
  stated run on a stated machine; re-measure rather than carry one forward, and
  read a new window for every patch — the same window twice is HDF5's chunk
  cache, not a read.
- **Some tables are generated.** The diagnostic codes, the cohort check codes
  and the sample-document schema are rendered at build time from
  `crates/medh5/data/codes.json`, the `CHECK_CODES` table in
  `crates/medh5/src/dataset/check.rs` and the JSON Schema by
  `hooks/mkdocs_hooks.py`. Edit the source, not the page; the page carries only
  a marker comment.
- **The Rust example is tested twice.** The engine crate's README is its crate
  documentation, so its example is a doctest; `docs/reference/rust.md` shows
  the same block, and a test compares the two.
- **Moving a page breaks links people already have.** Add the old path to
  `REDIRECTS` in `hooks/mkdocs_hooks.py`.
- `CHANGELOG.md` lives at the repository root and is pulled into the site by
  the same hook; link into the docs from it as `docs/…`, which is rewritten for
  the site.

Preview with `mkdocs serve`, which live-reloads on port 8000.

## Changelog and releases

Record user-visible changes under `## [Unreleased]` in `CHANGELOG.md`
([Keep a Changelog](https://keepachangelog.com/)). A change that can alter what
existing code produces or accepts goes under **Behaviour changes**, stated so a
reader can decide whether their pipeline is affected.

The version lives in one place, `[workspace.package] version` in `Cargo.toml`:
every crate inherits it, maturin stamps it on the wheel, and the engine writes
it into every file's `generator`. To release, set it (and `Cargo.lock` with
`cargo update -w`), move the unreleased notes under the new version, and push a
`vX.Y.Z` tag. `.github/workflows/release.yml` then:

1. runs the full CI on the tagged commit, which builds every wheel, the sdist
   and every binary;
2. checks the tag against the workspace version, every platform's wheel and
   binary against the tag, and extracts the version's section of
   `CHANGELOG.md` — a tag with no section fails here, before anything is
   published;
3. publishes the wheels and the sdist to PyPI through Trusted Publishing, and
   the crates to crates.io (`medh5-sys`, `medh5`, `medh5-cli`, in that order,
   with the `CARGO_REGISTRY_TOKEN` secret);
4. creates the GitHub Release, with the changelog section as its notes and the
   distributions, the binaries and their `SHA256SUMS` attached;
5. pushes the Homebrew formula for the binaries to the tap
   (`<owner>/homebrew-medh5`, or the `HOMEBREW_TAP` variable) with the
   `HOMEBREW_TAP_TOKEN` secret. Pre-releases skip this step.

Without `CARGO_REGISTRY_TOKEN` or `HOMEBREW_TAP_TOKEN` the step that needs it
warns and does nothing, so a fork can release to PyPI alone. The format version
(`medh5.FORMAT_VERSION`) is separate and changes only with the specification.

GitHub Actions are pinned by commit SHA; Dependabot proposes the updates as pull
requests.
