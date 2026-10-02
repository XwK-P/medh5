# Contributing to medh5

medh5 is two things kept in step: a **format** — the normative
[specification](docs/spec/medh5-1.0.md), its JSON Schema and its conformance
corpus — and a **package** that implements it. Most of what follows is about
keeping the two, and the documentation that describes them, from drifting apart.

## Setting up

```bash
git clone https://github.com/XwK-P/medh5.git
cd medh5
pip install -e ".[dev,torch,nifti,dicom,dicomseg,itk,interp]"
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
ruff check . && ruff format --check . && mypy medh5 \
  && pytest tests/ --cov=medh5 --cov-fail-under=90 \
  && medh5 conformance run /tmp/corpus \
  && mkdocs build --strict
```

- **Lint and format** — `ruff`, pinned in the `dev` extra and in
  `.pre-commit-config.yaml`; bump the two together.
- **Types** — `mypy --strict` over `medh5/`.
- **Tests** — `tests/v1/`, with a 90 % coverage floor.
- **Conformance** — the corpus is a shipped artifact that third-party
  implementations run, not a test fixture. It must stay green.
- **Documentation** — `mkdocs build --strict` fails on any broken link or
  anchor, including links into the specification by clause.

CI also runs the suite on Python 3.10–3.14, on macOS (for the `spawn` start
method), on Windows (for the atomic-replace paths), at the minimum dependency
versions `pyproject.toml` declares, and runs the reference writer in
`docs/examples/` to keep the specification's Appendix C.2 honest.

## Writing tests

- Name a test after the clause it holds: `test_S8_1_boxes_shift_by_half_a_voxel`.
- Build fixtures with the **public writer**, so every reader test is also a
  writer test.
- Guard an optional dependency with `pytest.importorskip` — and make sure some
  CI job installs it. A skipped test is not a passing test.
- A bug fix comes with the test that would have caught it.

## Changing the format

`docs/spec/medh5-1.0.md` is **normative**: the code implements it, and when the
two disagree one of them is a bug. Decide which before changing either.

- **A clause that turned out to be unimplementable**, ambiguous, or more than
  the implementation can honestly promise is corrected in the text *and*
  recorded as a row in Appendix C.1. A test counts the rows against the sentence
  that states how many there are.
- **Diagnostic codes are stable API.** The §15.2 table and `medh5/errors.py`
  must list the same codes (a test asserts it), a code's meaning never changes,
  and a retired code is never reused. A new code needs a conformance case in
  `medh5/conformance/corpus.py`; a test fails until it has one.
- **Versioning follows §16**: a minor format version may add objects, kinds,
  profiles and codes; it may not change what an existing one means.
- `schemas/medh5-sample-1.0.schema.json` and its packaged copy under
  `medh5/schemas/` must stay identical; a test compares them.

## Writing documentation

The site is [MkDocs](https://www.mkdocs.org/) with the Material theme, laid out
as **tutorials** (learning), **how-to guides** (tasks), **reference** (what
everything is), **explanation** (why) and the **specification**. Put a page
where its reader will look for it, and link rather than repeat.

Documentation here is checked against the code, not proofread:

- **Every `python` block runs.** `tests/v1/test_docs_python.py` executes each
  one against a real sample, with the names the pages use by convention (`s`,
  `w`, `ann`, `paths`, …) already bound. A block that genuinely cannot run —
  it needs a DICOM series, or it elides arguments with `...` — carries an
  `<!-- illustrative -->` comment on the line above its fence, where readers
  see it too. Most blocks must stay runnable.
- **Every documented CLI flag exists.** `tests/v1/test_docs_examples.py` parses
  each `medh5 …` line in a fenced block against the real argument parser.
- **Corrected claims stay corrected.** The same file keeps a list of statements
  that were once wrong in the documentation; add to it when you fix a claim
  that could plausibly come back.
- **Counts are checked.** Where a page states the size of the conformance
  corpus or its breakdown, a test compares it with the corpus.
- **Some tables are generated.** The diagnostic codes, the cohort check codes
  and the sample-document schema are rendered at build time from
  `medh5/errors.py`, `medh5/dataset/check.py` and the JSON Schema by
  `hooks/mkdocs_hooks.py`. Edit the source, not the page; the page carries only
  a marker comment.
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

The package version lives in one place, `medh5/__about__.py`. To release, set it,
move the unreleased notes under the new version, and push a `vX.Y.Z` tag:
`.github/workflows/release.yml` runs the full CI on the tagged commit, checks the
tag against the version, builds and checks the distributions, publishes to PyPI
through Trusted Publishing, and then creates the GitHub Release: its notes are
the version's section of `CHANGELOG.md`, and it carries the same distributions.
A tag whose version has no section there fails before anything is published.
The format version (`medh5.FORMAT_VERSION`) is separate and changes only with
the specification.

GitHub Actions are pinned by commit SHA; Dependabot proposes the updates as pull
requests.
